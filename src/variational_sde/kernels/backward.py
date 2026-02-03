from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from variational_sde.kernels._utils import next_power_of_two
from variational_sde.inference.constants import DIAG_MIN
from variational_sde.kernels.constants import (
    GATE_N,
    GATE_R,
    GATE_Z,
    NUM_GATES,
)
from variational_sde.kernels.helpers import (
    atomic_add_rzn_bias,
    project_input_rzn_with_grad,
    project_scalar_rzn_with_grad,
)
from variational_sde.kernels.weights import SavedActivations, SDEWeights


@triton.jit
def tanh_bwd(tanh_out: tl.tensor) -> tl.tensor:
    return 1.0 - tanh_out * tanh_out


@triton.jit
def sigmoid_bwd(sig_out: tl.tensor) -> tl.tensor:
    return sig_out * (1.0 - sig_out)


@triton.jit
def gru_cell_standard_bwd(
    d_h: tl.tensor,
    h_input: tl.tensor,
    h_recurrent: tl.tensor,
    r: tl.tensor,
    z: tl.tensor,
    n: tl.tensor,
    n_hh: tl.tensor,
    W_ih_ptr: tl.tensor,
    W_hh_ptr: tl.tensor,
    grad_W_ih_ptr: tl.tensor,
    grad_W_hh_ptr: tl.tensor,
    grad_b_ih_ptr: tl.tensor,
    grad_b_hh_ptr: tl.tensor,
    h_range: tl.tensor,
    h_mask: tl.tensor,
    j_range: tl.tensor,
    offs_W_hh: tl.tensor,
    mask_2d: tl.tensor,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:
    d_n = (1.0 - z) * d_h
    d_z = (h_recurrent - n) * d_h
    d_h_recurrent_from_z = z * d_h

    d_n_pre = d_n * tanh_bwd(n)
    d_z_pre = d_z * sigmoid_bwd(z)
    d_n_hh = d_n_pre * r
    d_r = d_n_pre * n_hh
    d_r_pre = d_r * sigmoid_bwd(r)

    W_ih_r = tl.load(
        W_ih_ptr + offs_W_hh + GATE_R * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)
    W_ih_z = tl.load(
        W_ih_ptr + offs_W_hh + GATE_Z * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)
    W_ih_n = tl.load(
        W_ih_ptr + offs_W_hh + GATE_N * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)

    d_r_pre_row = d_r_pre[None, :]
    d_z_pre_row = d_z_pre[None, :]
    d_n_pre_row = d_n_pre[None, :]

    d_h_input = (
        tl.sum(W_ih_r * d_r_pre_row, axis=1)
        + tl.sum(W_ih_z * d_z_pre_row, axis=1)
        + tl.sum(W_ih_n * d_n_pre_row, axis=1)
    )

    W_hh_r = tl.load(
        W_hh_ptr + offs_W_hh + GATE_R * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)
    W_hh_z = tl.load(
        W_hh_ptr + offs_W_hh + GATE_Z * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)
    W_hh_n = tl.load(
        W_hh_ptr + offs_W_hh + GATE_N * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)

    d_n_hh_row = d_n_hh[None, :]
    d_h_recurrent_from_W_hh = (
        tl.sum(W_hh_r * d_r_pre_row, axis=1)
        + tl.sum(W_hh_z * d_z_pre_row, axis=1)
        + tl.sum(W_hh_n * d_n_hh_row, axis=1)
    )
    d_h_recurrent = d_h_recurrent_from_z + d_h_recurrent_from_W_hh

    h_input_col = h_input[:, None]
    tl.atomic_add(
        grad_W_ih_ptr + offs_W_hh + GATE_R * HIDDEN,
        h_input_col * d_r_pre_row,
        mask=mask_2d,
    )
    tl.atomic_add(
        grad_W_ih_ptr + offs_W_hh + GATE_Z * HIDDEN,
        h_input_col * d_z_pre_row,
        mask=mask_2d,
    )
    tl.atomic_add(
        grad_W_ih_ptr + offs_W_hh + GATE_N * HIDDEN,
        h_input_col * d_n_pre_row,
        mask=mask_2d,
    )

    h_recurrent_col = h_recurrent[:, None]
    tl.atomic_add(
        grad_W_hh_ptr + offs_W_hh + GATE_R * HIDDEN,
        h_recurrent_col * d_r_pre_row,
        mask=mask_2d,
    )
    tl.atomic_add(
        grad_W_hh_ptr + offs_W_hh + GATE_Z * HIDDEN,
        h_recurrent_col * d_z_pre_row,
        mask=mask_2d,
    )
    tl.atomic_add(
        grad_W_hh_ptr + offs_W_hh + GATE_N * HIDDEN,
        h_recurrent_col * d_n_hh_row,
        mask=mask_2d,
    )

    atomic_add_rzn_bias(
        grad_b_ih_ptr,
        grad_b_hh_ptr,
        d_r_pre,
        d_z_pre,
        d_n_pre,
        d_n_hh,
        h_range,
        h_mask,
        HIDDEN,
    )

    return d_h_input, d_h_recurrent


@triton.jit
def sde_bwd_kernel(
    grad_diffusion_paths_ptr: tl.tensor,
    grad_transition_means_ptr: tl.tensor,
    grad_transition_cholesky_ptr: tl.tensor,
    diffusion_paths_ptr: tl.tensor,
    transition_cholesky_raw_ptr: tl.tensor,
    eps_ptr: tl.tensor,
    context_ptr: tl.tensor,
    sde_param_ptr: tl.tensor,
    h_l0_ptr: tl.tensor,
    r_l0_ptr: tl.tensor,
    z_l0_ptr: tl.tensor,
    n_l0_ptr: tl.tensor,
    n_hh_l0_ptr: tl.tensor,
    h_stack_ptr: tl.tensor,
    r_stack_ptr: tl.tensor,
    z_stack_ptr: tl.tensor,
    n_stack_ptr: tl.tensor,
    n_hh_stack_ptr: tl.tensor,
    W_ih_l0_ptr: tl.tensor,
    W_hh_l0_ptr: tl.tensor,
    W_ih_stack_ptr: tl.tensor,
    W_hh_stack_ptr: tl.tensor,
    out_weight_ptr: tl.tensor,
    grad_x0_ptr: tl.tensor,
    grad_context_ptr: tl.tensor,
    grad_sde_param_ptr: tl.tensor,
    grad_W_ih_l0_ptr: tl.tensor,
    grad_W_hh_l0_ptr: tl.tensor,
    grad_b_ih_l0_ptr: tl.tensor,
    grad_b_hh_l0_ptr: tl.tensor,
    grad_W_ih_stack_ptr: tl.tensor,
    grad_W_hh_stack_ptr: tl.tensor,
    grad_b_ih_stack_ptr: tl.tensor,
    grad_b_hh_stack_ptr: tl.tensor,
    grad_out_weight_ptr: tl.tensor,
    grad_out_bias_ptr: tl.tensor,
    n_steps: int,
    time_step: float,
    sqrt_time_step: float,
    context_stride: int,
    DIAG_MIN: tl.constexpr,
    HIDDEN: tl.constexpr,
    CONTEXT: tl.constexpr,
    PARAM: tl.constexpr,
    STATE: tl.constexpr,
    N_TRIL: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
) -> None:
    batch_idx = tl.program_id(0)
    h_range = tl.arange(0, BLOCK_H)
    h_mask = h_range < HIDDEN
    s_range = tl.arange(0, BLOCK_S)
    s_mask = s_range < STATE

    act_base_l0 = batch_idx * n_steps * HIDDEN
    act_stack_base = (
        batch_idx * (NUM_LAYERS - 1) * n_steps * HIDDEN if NUM_LAYERS > 1 else 0
    )
    sde_param_base = batch_idx * PARAM
    diffusion_path_base = batch_idx * (n_steps + 1) * STATE

    t_off = (STATE + CONTEXT) * NUM_GATES * HIDDEN

    d_x_t = tl.zeros((BLOCK_S,), dtype=tl.float32)
    d_h_l0 = tl.zeros((BLOCK_H,), dtype=tl.float32)

    d_h_stack_l1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    d_h_stack_l2 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    d_h_stack_l3 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    d_h_stack_l4 = tl.zeros((BLOCK_H,), dtype=tl.float32)

    accum_b_ih_l0_r = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accum_b_ih_l0_z = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accum_b_ih_l0_n = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accum_b_hh_l0_r = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accum_b_hh_l0_z = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accum_b_hh_l0_n = tl.zeros((BLOCK_H,), dtype=tl.float32)

    j_range = tl.arange(0, BLOCK_H)
    offs_W_hh = j_range[:, None] * NUM_GATES * HIDDEN + h_range[None, :]
    mask_2d = (j_range[:, None] < HIDDEN) & (h_range[None, :] < HIDDEN)

    W_hh_l0_r = tl.load(
        W_hh_l0_ptr + offs_W_hh + GATE_R * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)
    W_hh_l0_z = tl.load(
        W_hh_l0_ptr + offs_W_hh + GATE_Z * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)
    W_hh_l0_n = tl.load(
        W_hh_l0_ptr + offs_W_hh + GATE_N * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)

    W_ih_layer_stride = HIDDEN * NUM_GATES * HIDDEN
    W_hh_layer_stride = HIDDEN * NUM_GATES * HIDDEN
    b_layer_stride = NUM_GATES * HIDDEN
    h_layer_stride = n_steps * HIDDEN

    for step_rev in range(n_steps):
        step = n_steps - 1 - step_rev
        step_base_l0 = act_base_l0 + step * HIDDEN
        ctx_base = batch_idx * context_stride + step * CONTEXT

        diffusion_path_step_base = diffusion_path_base + (step + 1) * STATE
        d_path = tl.load(
            grad_diffusion_paths_ptr + diffusion_path_step_base + s_range,
            mask=s_mask,
            other=0.0,
        ).to(tl.float32)

        eps_base = batch_idx * n_steps * STATE + step * STATE
        eps_n = tl.load(eps_ptr + eps_base + s_range, mask=s_mask, other=0.0).to(
            tl.float32
        )

        d_mu_ext = tl.load(
            grad_transition_means_ptr + eps_base + s_range, mask=s_mask, other=0.0
        ).to(tl.float32)

        d_x_t += d_path
        d_mu = d_x_t * time_step + d_mu_ext

        if NUM_LAYERS == 1:
            h_final = tl.load(
                h_l0_ptr + step_base_l0 + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)
        else:
            final_layer_idx = NUM_LAYERS - 2
            step_base_final = (
                act_stack_base + final_layer_idx * h_layer_stride + step * HIDDEN
            )
            h_final = tl.load(
                h_stack_ptr + step_base_final + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)

        d_h_final = tl.zeros((BLOCK_H,), dtype=tl.float32)

        for s in range(STATE):
            d_mu_s = tl.sum(tl.where(s_range == s, d_mu, 0.0))
            out_w_mu_s = tl.load(
                out_weight_ptr + s * HIDDEN + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)
            tl.atomic_add(grad_out_bias_ptr + s, d_mu_s)
            tl.atomic_add(
                grad_out_weight_ptr + s * HIDDEN + h_range,
                d_mu_s * h_final,
                mask=h_mask,
            )
            d_h_final += d_mu_s * out_w_mu_s

        transition_cholesky_raw_step_base = batch_idx * n_steps * N_TRIL + step * N_TRIL
        grad_transition_cholesky_step_base = (
            batch_idx * n_steps * STATE * STATE + step * STATE * STATE
        )
        tril_idx = 0
        for row in range(STATE):
            d_x_row = tl.sum(tl.where(s_range == row, d_x_t, 0.0))
            for col in range(row + 1):
                eps_col = tl.sum(tl.where(s_range == col, eps_n, 0.0))
                d_L_ext_ij = tl.load(
                    grad_transition_cholesky_ptr
                    + grad_transition_cholesky_step_base
                    + row * STATE
                    + col
                )
                d_L_ij = d_x_row * eps_col * sqrt_time_step + d_L_ext_ij
                transition_cholesky_raw_ij = tl.load(
                    transition_cholesky_raw_ptr
                    + transition_cholesky_raw_step_base
                    + tril_idx
                )
                is_diag = row == col
                pass_through = (transition_cholesky_raw_ij >= DIAG_MIN) | (d_L_ij < 0.0)
                d_transition_cholesky_raw_ij = tl.where(
                    is_diag, tl.where(pass_through, d_L_ij, 0.0), d_L_ij
                )
                out_w_L = tl.load(
                    out_weight_ptr + (STATE + tril_idx) * HIDDEN + h_range,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                tl.atomic_add(
                    grad_out_bias_ptr + STATE + tril_idx, d_transition_cholesky_raw_ij
                )
                tl.atomic_add(
                    grad_out_weight_ptr + (STATE + tril_idx) * HIDDEN + h_range,
                    d_transition_cholesky_raw_ij * h_final,
                    mask=h_mask,
                )
                d_h_final += d_transition_cholesky_raw_ij * out_w_L
                tril_idx += 1

        d_h_current = d_h_final

        for layer_rev in tl.static_range(NUM_LAYERS - 1):
            layer = NUM_LAYERS - 1 - layer_rev
            layer_idx = layer - 1
            step_base_stack = (
                act_stack_base + layer_idx * h_layer_stride + step * HIDDEN
            )

            if layer == 1:
                d_h_current += d_h_stack_l1
            elif layer == 2:
                d_h_current += d_h_stack_l2
            elif layer == 3:
                d_h_current += d_h_stack_l3
            elif layer == 4:
                d_h_current += d_h_stack_l4

            r = tl.load(
                r_stack_ptr + step_base_stack + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)
            z = tl.load(
                z_stack_ptr + step_base_stack + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)
            n = tl.load(
                n_stack_ptr + step_base_stack + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)
            n_hh = tl.load(
                n_hh_stack_ptr + step_base_stack + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)

            if step > 0:
                h_recurrent = tl.load(
                    h_stack_ptr
                    + act_stack_base
                    + layer_idx * h_layer_stride
                    + (step - 1) * HIDDEN
                    + h_range,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                h_recurrent = tl.zeros((BLOCK_H,), dtype=tl.float32)

            if layer == 1:
                h_input = tl.load(
                    h_l0_ptr + step_base_l0 + h_range, mask=h_mask, other=0.0
                ).to(tl.float32)
            else:
                prev_layer_idx = layer_idx - 1
                h_input = tl.load(
                    h_stack_ptr
                    + act_stack_base
                    + prev_layer_idx * h_layer_stride
                    + step * HIDDEN
                    + h_range,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)

            W_ih_layer_ptr = W_ih_stack_ptr + layer_idx * W_ih_layer_stride
            W_hh_layer_ptr = W_hh_stack_ptr + layer_idx * W_hh_layer_stride
            grad_W_ih_layer_ptr = grad_W_ih_stack_ptr + layer_idx * W_ih_layer_stride
            grad_W_hh_layer_ptr = grad_W_hh_stack_ptr + layer_idx * W_hh_layer_stride
            grad_b_ih_layer_ptr = grad_b_ih_stack_ptr + layer_idx * b_layer_stride
            grad_b_hh_layer_ptr = grad_b_hh_stack_ptr + layer_idx * b_layer_stride

            d_h_input, d_h_recurrent = gru_cell_standard_bwd(
                d_h_current,
                h_input,
                h_recurrent,
                r,
                z,
                n,
                n_hh,
                W_ih_layer_ptr,
                W_hh_layer_ptr,
                grad_W_ih_layer_ptr,
                grad_W_hh_layer_ptr,
                grad_b_ih_layer_ptr,
                grad_b_hh_layer_ptr,
                h_range,
                h_mask,
                j_range,
                offs_W_hh,
                mask_2d,
                HIDDEN,
                BLOCK_H,
            )

            if layer == 1:
                d_h_stack_l1 = d_h_recurrent
            elif layer == 2:
                d_h_stack_l2 = d_h_recurrent
            elif layer == 3:
                d_h_stack_l3 = d_h_recurrent
            elif layer == 4:
                d_h_stack_l4 = d_h_recurrent

            d_h_current = d_h_input

        if NUM_LAYERS > 1:
            d_h_l0 += d_h_current
        else:
            d_h_l0 += d_h_final

        r_l0 = tl.load(r_l0_ptr + step_base_l0 + h_range, mask=h_mask, other=0.0).to(
            tl.float32
        )
        z_l0 = tl.load(z_l0_ptr + step_base_l0 + h_range, mask=h_mask, other=0.0).to(
            tl.float32
        )
        n_l0 = tl.load(n_l0_ptr + step_base_l0 + h_range, mask=h_mask, other=0.0).to(
            tl.float32
        )
        n_hh_l0 = tl.load(
            n_hh_l0_ptr + step_base_l0 + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)

        if step > 0:
            h_l0_old = tl.load(
                h_l0_ptr + act_base_l0 + (step - 1) * HIDDEN + h_range,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            h_l0_old = tl.zeros((BLOCK_H,), dtype=tl.float32)

        d_h_l0_old = z_l0 * d_h_l0
        d_n_l0 = (1.0 - z_l0) * d_h_l0
        d_z_l0 = (h_l0_old - n_l0) * d_h_l0

        d_n_l0_pre = d_n_l0 * tanh_bwd(n_l0)
        d_r_l0_pre = d_n_l0_pre * n_hh_l0 * sigmoid_bwd(r_l0)
        d_z_l0_pre = d_z_l0 * sigmoid_bwd(z_l0)
        d_n_hh_l0 = d_n_l0_pre * r_l0

        x_t = tl.load(
            diffusion_paths_ptr + diffusion_path_base + step * STATE + s_range,
            mask=s_mask,
            other=0.0,
        ).to(tl.float32)

        d_x_from_state = project_input_rzn_with_grad(
            x_t,
            W_ih_l0_ptr,
            grad_W_ih_l0_ptr,
            0,
            d_r_l0_pre,
            d_z_l0_pre,
            d_n_l0_pre,
            STATE,
            s_range,
            h_range,
            h_mask,
            HIDDEN,
            BLOCK_S,
        )
        d_x_t += d_x_from_state

        for p in range(PARAM):
            sde_param = tl.load(sde_param_ptr + sde_param_base + p).to(tl.float32)
            p_off = t_off + p * NUM_GATES * HIDDEN
            W_tp_r = tl.load(
                W_ih_l0_ptr + p_off + GATE_R * HIDDEN + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)
            W_tp_z = tl.load(
                W_ih_l0_ptr + p_off + GATE_Z * HIDDEN + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)
            W_tp_n = tl.load(
                W_ih_l0_ptr + p_off + GATE_N * HIDDEN + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)

            d_sde_param = (
                tl.sum(d_r_l0_pre * W_tp_r)
                + tl.sum(d_z_l0_pre * W_tp_z)
                + tl.sum(d_n_l0_pre * W_tp_n)
            )
            grad_sde_param = tl.load(grad_sde_param_ptr + sde_param_base + p)
            tl.store(
                grad_sde_param_ptr + sde_param_base + p, grad_sde_param + d_sde_param
            )

            tl.atomic_add(
                grad_W_ih_l0_ptr + p_off + GATE_R * HIDDEN + h_range,
                sde_param * d_r_l0_pre,
                mask=h_mask,
            )
            tl.atomic_add(
                grad_W_ih_l0_ptr + p_off + GATE_Z * HIDDEN + h_range,
                sde_param * d_z_l0_pre,
                mask=h_mask,
            )
            tl.atomic_add(
                grad_W_ih_l0_ptr + p_off + GATE_N * HIDDEN + h_range,
                sde_param * d_n_l0_pre,
                mask=h_mask,
            )

        project_scalar_rzn_with_grad(
            context_ptr,
            ctx_base,
            grad_context_ptr,
            W_ih_l0_ptr,
            grad_W_ih_l0_ptr,
            STATE * NUM_GATES * HIDDEN,
            d_r_l0_pre,
            d_z_l0_pre,
            d_n_l0_pre,
            CONTEXT,
            h_range,
            h_mask,
            HIDDEN,
        )

        d_r_l0_pre_row = d_r_l0_pre[None, :]
        d_z_l0_pre_row = d_z_l0_pre[None, :]
        d_n_hh_l0_row = d_n_hh_l0[None, :]
        d_h_l0_old += (
            tl.sum(W_hh_l0_r * d_r_l0_pre_row, axis=1)
            + tl.sum(W_hh_l0_z * d_z_l0_pre_row, axis=1)
            + tl.sum(W_hh_l0_n * d_n_hh_l0_row, axis=1)
        )

        h_l0_old_col = h_l0_old[:, None]
        tl.atomic_add(
            grad_W_hh_l0_ptr + offs_W_hh + GATE_R * HIDDEN,
            h_l0_old_col * d_r_l0_pre_row,
            mask=mask_2d,
        )
        tl.atomic_add(
            grad_W_hh_l0_ptr + offs_W_hh + GATE_Z * HIDDEN,
            h_l0_old_col * d_z_l0_pre_row,
            mask=mask_2d,
        )
        tl.atomic_add(
            grad_W_hh_l0_ptr + offs_W_hh + GATE_N * HIDDEN,
            h_l0_old_col * d_n_hh_l0_row,
            mask=mask_2d,
        )

        accum_b_ih_l0_r += d_r_l0_pre
        accum_b_ih_l0_z += d_z_l0_pre
        accum_b_ih_l0_n += d_n_l0_pre
        accum_b_hh_l0_r += d_r_l0_pre
        accum_b_hh_l0_z += d_z_l0_pre
        accum_b_hh_l0_n += d_n_hh_l0

        d_h_l0 = d_h_l0_old

    tl.atomic_add(
        grad_b_ih_l0_ptr + GATE_R * HIDDEN + h_range, accum_b_ih_l0_r, mask=h_mask
    )
    tl.atomic_add(
        grad_b_ih_l0_ptr + GATE_Z * HIDDEN + h_range, accum_b_ih_l0_z, mask=h_mask
    )
    tl.atomic_add(
        grad_b_ih_l0_ptr + GATE_N * HIDDEN + h_range, accum_b_ih_l0_n, mask=h_mask
    )
    tl.atomic_add(
        grad_b_hh_l0_ptr + GATE_R * HIDDEN + h_range, accum_b_hh_l0_r, mask=h_mask
    )
    tl.atomic_add(
        grad_b_hh_l0_ptr + GATE_Z * HIDDEN + h_range, accum_b_hh_l0_z, mask=h_mask
    )
    tl.atomic_add(
        grad_b_hh_l0_ptr + GATE_N * HIDDEN + h_range, accum_b_hh_l0_n, mask=h_mask
    )

    d_x0_path = tl.load(
        grad_diffusion_paths_ptr + diffusion_path_base + s_range, mask=s_mask, other=0.0
    ).to(tl.float32)
    d_x_t += d_x0_path
    tl.store(grad_x0_ptr + batch_idx * STATE + s_range, d_x_t, mask=s_mask)


def launch_bwd(
    grad_diffusion_paths: Tensor,
    grad_transition_means: Tensor,
    grad_transition_cholesky: Tensor,
    context: Tensor,
    sde_parameters: Tensor,
    eps: Tensor,
    saved: SavedActivations,
    weights: SDEWeights,
    time_step: float,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    batch_size = context.shape[0]
    n_steps = context.shape[1]
    device = context.device

    hidden_dim = weights.hidden_dim
    context_dim = weights.context_dim
    sde_param_dim = weights.sde_param_dim
    state_dim = weights.state_dim
    num_layers = weights.num_layers
    n_tril = state_dim * (state_dim + 1) // 2

    sqrt_time_step = math.sqrt(time_step)
    context_stride = n_steps * context_dim
    block_h = next_power_of_two(hidden_dim)
    block_s = next_power_of_two(state_dim)

    grad_x0 = torch.zeros(batch_size, state_dim, device=device, dtype=torch.float32)
    grad_context = torch.zeros_like(context, dtype=torch.float32)
    grad_sde_parameters = torch.zeros_like(sde_parameters, dtype=torch.float32)

    W_ih_l0_shape = weights.W_ih_l0.shape
    W_hh_l0_shape = weights.W_hh_l0.shape

    grad_W_ih_l0 = torch.zeros(W_ih_l0_shape, device=device, dtype=torch.float32)
    grad_W_hh_l0 = torch.zeros(W_hh_l0_shape, device=device, dtype=torch.float32)
    grad_b_ih_l0 = torch.zeros(
        NUM_GATES * hidden_dim, device=device, dtype=torch.float32
    )
    grad_b_hh_l0 = torch.zeros(
        NUM_GATES * hidden_dim, device=device, dtype=torch.float32
    )

    if num_layers > 1:
        grad_W_ih_stack = torch.zeros_like(weights.W_ih_stack, dtype=torch.float32)
        grad_W_hh_stack = torch.zeros_like(weights.W_hh_stack, dtype=torch.float32)
        grad_b_ih_stack = torch.zeros_like(weights.b_ih_stack, dtype=torch.float32)
        grad_b_hh_stack = torch.zeros_like(weights.b_hh_stack, dtype=torch.float32)
    else:
        grad_W_ih_stack = torch.empty(0, device=device, dtype=torch.float32)
        grad_W_hh_stack = torch.empty(0, device=device, dtype=torch.float32)
        grad_b_ih_stack = torch.empty(0, device=device, dtype=torch.float32)
        grad_b_hh_stack = torch.empty(0, device=device, dtype=torch.float32)

    grad_out_weight = torch.zeros(
        weights.out_weight.shape, device=device, dtype=torch.float32
    )
    grad_out_bias = torch.zeros(state_dim + n_tril, device=device, dtype=torch.float32)

    sde_bwd_kernel[(batch_size,)](
        grad_diffusion_paths.view(-1).contiguous(),
        grad_transition_means.view(-1).contiguous(),
        grad_transition_cholesky.view(-1).contiguous(),
        saved.diffusion_paths.view(-1).contiguous(),
        saved.transition_cholesky_raw.view(-1).contiguous(),
        eps.view(batch_size, -1).contiguous().float(),
        context.contiguous().float(),
        sde_parameters.contiguous().float(),
        saved.h_l0.view(-1).contiguous(),
        saved.r_l0.view(-1).contiguous(),
        saved.z_l0.view(-1).contiguous(),
        saved.n_l0.view(-1).contiguous(),
        saved.n_hh_l0.view(-1).contiguous(),
        saved.h_stack.view(-1).contiguous()
        if saved.h_stack.numel() > 0
        else saved.h_stack,
        saved.r_stack.view(-1).contiguous()
        if saved.r_stack.numel() > 0
        else saved.r_stack,
        saved.z_stack.view(-1).contiguous()
        if saved.z_stack.numel() > 0
        else saved.z_stack,
        saved.n_stack.view(-1).contiguous()
        if saved.n_stack.numel() > 0
        else saved.n_stack,
        saved.n_hh_stack.view(-1).contiguous()
        if saved.n_hh_stack.numel() > 0
        else saved.n_hh_stack,
        weights.W_ih_l0.float(),
        weights.W_hh_l0.float(),
        weights.W_ih_stack.view(-1).float()
        if weights.W_ih_stack.numel() > 0
        else weights.W_ih_stack,
        weights.W_hh_stack.view(-1).float()
        if weights.W_hh_stack.numel() > 0
        else weights.W_hh_stack,
        weights.out_weight.float(),
        grad_x0.view(-1),
        grad_context.view(batch_size, n_steps * context_dim),
        grad_sde_parameters,
        grad_W_ih_l0.view(-1),
        grad_W_hh_l0.view(-1),
        grad_b_ih_l0,
        grad_b_hh_l0,
        grad_W_ih_stack.view(-1) if grad_W_ih_stack.numel() > 0 else grad_W_ih_stack,
        grad_W_hh_stack.view(-1) if grad_W_hh_stack.numel() > 0 else grad_W_hh_stack,
        grad_b_ih_stack.view(-1) if grad_b_ih_stack.numel() > 0 else grad_b_ih_stack,
        grad_b_hh_stack.view(-1) if grad_b_hh_stack.numel() > 0 else grad_b_hh_stack,
        grad_out_weight.view(-1),
        grad_out_bias,
        n_steps,
        time_step,
        sqrt_time_step,
        context_stride,
        DIAG_MIN=DIAG_MIN,
        HIDDEN=hidden_dim,
        CONTEXT=context_dim,
        PARAM=sde_param_dim,
        STATE=state_dim,
        N_TRIL=n_tril,
        BLOCK_H=block_h,
        BLOCK_S=block_s,
        NUM_LAYERS=num_layers,
    )

    return (
        grad_x0,
        grad_context,
        grad_sde_parameters,
        grad_W_ih_l0.T.contiguous(),
        grad_W_hh_l0.T.contiguous(),
        grad_b_ih_l0,
        grad_b_hh_l0,
        grad_W_ih_stack.transpose(-2, -1).contiguous()
        if grad_W_ih_stack.numel() > 0
        else grad_W_ih_stack,
        grad_W_hh_stack.transpose(-2, -1).contiguous()
        if grad_W_hh_stack.numel() > 0
        else grad_W_hh_stack,
        grad_b_ih_stack,
        grad_b_hh_stack,
        grad_out_weight,
        grad_out_bias,
    )
