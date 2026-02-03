from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from variational_sde.kernels._utils import next_power_of_two
from variational_sde.inference.constants import DIAG_MIN
from variational_sde.kernels.constants import (
    GATE_N,
    GATE_R,
    GATE_Z,
    NUM_GATES,
)
from variational_sde.kernels.gru_ops import load_l0_biases
from variational_sde.kernels.helpers import (
    project_input_rzn,
    project_scalar_rzn,
    store_gru_activations,
)
from variational_sde.kernels.weights import SavedActivations, SDEWeights


@triton.jit
def tanh_fwd(x: tl.tensor) -> tl.tensor:
    return libdevice.tanh(x)


@triton.jit
def gru_cell_standard(
    h_input: tl.tensor,
    h_recurrent: tl.tensor,
    W_ih_ptr: tl.tensor,
    W_hh_ptr: tl.tensor,
    b_ih_ptr: tl.tensor,
    b_hh_ptr: tl.tensor,
    h_range: tl.tensor,
    h_mask: tl.tensor,
    j_range: tl.tensor,
    offs_W_hh: tl.tensor,
    mask_2d: tl.tensor,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    b_ir, b_iz, b_in, b_hr, b_hz, b_hn = load_l0_biases(
        b_ih_ptr, b_hh_ptr, h_range, h_mask, HIDDEN
    )

    W_ih_r = tl.load(
        W_ih_ptr + offs_W_hh + GATE_R * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)
    W_ih_z = tl.load(
        W_ih_ptr + offs_W_hh + GATE_Z * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)
    W_ih_n = tl.load(
        W_ih_ptr + offs_W_hh + GATE_N * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)

    h_input_col = h_input[:, None]
    r_ih = b_ir + tl.sum(h_input_col * W_ih_r, axis=0)
    z_ih = b_iz + tl.sum(h_input_col * W_ih_z, axis=0)
    n_ih = b_in + tl.sum(h_input_col * W_ih_n, axis=0)

    W_hh_r = tl.load(
        W_hh_ptr + offs_W_hh + GATE_R * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)
    W_hh_z = tl.load(
        W_hh_ptr + offs_W_hh + GATE_Z * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)
    W_hh_n = tl.load(
        W_hh_ptr + offs_W_hh + GATE_N * HIDDEN, mask=mask_2d, other=0.0
    ).to(tl.float32)

    h_rec_col = h_recurrent[:, None]
    r_hh = b_hr + tl.sum(h_rec_col * W_hh_r, axis=0)
    z_hh = b_hz + tl.sum(h_rec_col * W_hh_z, axis=0)
    n_hh = b_hn + tl.sum(h_rec_col * W_hh_n, axis=0)

    r = tl.sigmoid(r_ih + r_hh)
    z = tl.sigmoid(z_ih + z_hh)
    n = tanh_fwd(n_ih + r * n_hh)
    h_new = (1.0 - z) * n + z * h_recurrent

    return h_new, r, z, n, n_hh


@triton.jit
def sde_fwd_kernel(
    x_ptr: tl.tensor,
    context_ptr: tl.tensor,
    sde_param_ptr: tl.tensor,
    eps_ptr: tl.tensor,
    W_ih_l0_ptr: tl.tensor,
    W_hh_l0_ptr: tl.tensor,
    b_ih_l0_ptr: tl.tensor,
    b_hh_l0_ptr: tl.tensor,
    W_ih_stack_ptr: tl.tensor,
    W_hh_stack_ptr: tl.tensor,
    b_ih_stack_ptr: tl.tensor,
    b_hh_stack_ptr: tl.tensor,
    out_weight_ptr: tl.tensor,
    out_bias_ptr: tl.tensor,
    diffusion_paths_ptr: tl.tensor,
    transition_means_ptr: tl.tensor,
    transition_cholesky_ptr: tl.tensor,
    transition_cholesky_raw_ptr: tl.tensor,
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
    h_recurrent_stack_ptr: tl.tensor,
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
    SAVE_ACTIVATIONS: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
) -> None:
    batch_idx = tl.program_id(0)
    h_range = tl.arange(0, BLOCK_H)
    h_mask = h_range < HIDDEN
    s_range = tl.arange(0, BLOCK_S)
    s_mask = s_range < STATE

    h_l0 = tl.zeros((BLOCK_H,), dtype=tl.float32)

    x_t = tl.load(x_ptr + batch_idx * STATE + s_range, mask=s_mask, other=0.0).to(
        tl.float32
    )

    sde_param_base = batch_idx * PARAM
    diffusion_path_base = batch_idx * (n_steps + 1) * STATE
    tl.store(diffusion_paths_ptr + diffusion_path_base + s_range, x_t, mask=s_mask)

    b_ir0, b_iz0, b_in0, b_hr0, b_hz0, b_hn0 = load_l0_biases(
        b_ih_l0_ptr, b_hh_l0_ptr, h_range, h_mask, HIDDEN
    )

    sde_param_r = tl.zeros((BLOCK_H,), dtype=tl.float32)
    sde_param_z = tl.zeros((BLOCK_H,), dtype=tl.float32)
    sde_param_n = tl.zeros((BLOCK_H,), dtype=tl.float32)
    t_off = (STATE + CONTEXT) * NUM_GATES * HIDDEN
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
        sde_param_r += sde_param * W_tp_r
        sde_param_z += sde_param * W_tp_z
        sde_param_n += sde_param * W_tp_n

    if SAVE_ACTIVATIONS:
        act_base_l0 = batch_idx * n_steps * HIDDEN
        act_stack_base = (
            batch_idx * (NUM_LAYERS - 1) * n_steps * HIDDEN if NUM_LAYERS > 1 else 0
        )

    j_range = tl.arange(0, BLOCK_H)
    offs_W_hh = j_range[:, None] * NUM_GATES * HIDDEN + h_range[None, :]
    mask_2d = (j_range[:, None] < HIDDEN) & (h_range[None, :] < HIDDEN)

    W_ih_layer_stride = HIDDEN * NUM_GATES * HIDDEN
    W_hh_layer_stride = HIDDEN * NUM_GATES * HIDDEN
    b_layer_stride = NUM_GATES * HIDDEN
    h_layer_stride = n_steps * HIDDEN

    for step in range(n_steps):
        ctx_base = batch_idx * context_stride + step * CONTEXT

        r_ih = b_ir0 + sde_param_r
        z_ih = b_iz0 + sde_param_z
        n_ih = b_in0 + sde_param_n

        r_s, z_s, n_s = project_input_rzn(
            x_t, W_ih_l0_ptr, 0, STATE, s_range, h_range, h_mask, HIDDEN, BLOCK_H
        )
        r_ih += r_s
        z_ih += z_s
        n_ih += n_s

        r_c, z_c, n_c = project_scalar_rzn(
            context_ptr,
            ctx_base,
            W_ih_l0_ptr,
            STATE * NUM_GATES * HIDDEN,
            CONTEXT,
            h_range,
            h_mask,
            HIDDEN,
            BLOCK_H,
        )
        r_ih += r_c
        z_ih += z_c
        n_ih += n_c

        W_r0 = tl.load(
            W_hh_l0_ptr + offs_W_hh + GATE_R * HIDDEN, mask=mask_2d, other=0.0
        ).to(tl.float32)
        W_z0 = tl.load(
            W_hh_l0_ptr + offs_W_hh + GATE_Z * HIDDEN, mask=mask_2d, other=0.0
        ).to(tl.float32)
        W_n0 = tl.load(
            W_hh_l0_ptr + offs_W_hh + GATE_N * HIDDEN, mask=mask_2d, other=0.0
        ).to(tl.float32)
        h_l0_col = h_l0[:, None]
        r_hh = b_hr0 + tl.sum(h_l0_col * W_r0, axis=0)
        z_hh = b_hz0 + tl.sum(h_l0_col * W_z0, axis=0)
        n_hh_l0 = b_hn0 + tl.sum(h_l0_col * W_n0, axis=0)

        r_l0 = tl.sigmoid(r_ih + r_hh)
        z_l0 = tl.sigmoid(z_ih + z_hh)
        n_l0 = tanh_fwd(n_ih + r_l0 * n_hh_l0)
        h_l0 = (1.0 - z_l0) * n_l0 + z_l0 * h_l0

        if SAVE_ACTIVATIONS:
            step_base_l0 = act_base_l0 + step * HIDDEN
            store_gru_activations(
                h_l0_ptr,
                r_l0_ptr,
                z_l0_ptr,
                n_l0_ptr,
                n_hh_l0_ptr,
                step_base_l0,
                h_l0,
                r_l0,
                z_l0,
                n_l0,
                n_hh_l0,
                h_range,
                h_mask,
            )

        h_current = h_l0

        for layer in tl.static_range(1, NUM_LAYERS):
            layer_idx = layer - 1
            W_ih_layer_ptr = W_ih_stack_ptr + layer_idx * W_ih_layer_stride
            W_hh_layer_ptr = W_hh_stack_ptr + layer_idx * W_hh_layer_stride
            b_ih_layer_ptr = b_ih_stack_ptr + layer_idx * b_layer_stride
            b_hh_layer_ptr = b_hh_stack_ptr + layer_idx * b_layer_stride

            h_rec_base = batch_idx * (NUM_LAYERS - 1) * HIDDEN + layer_idx * HIDDEN
            h_recurrent = tl.load(
                h_recurrent_stack_ptr + h_rec_base + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)

            h_new, r, z, n, n_hh = gru_cell_standard(
                h_current,
                h_recurrent,
                W_ih_layer_ptr,
                W_hh_layer_ptr,
                b_ih_layer_ptr,
                b_hh_layer_ptr,
                h_range,
                h_mask,
                j_range,
                offs_W_hh,
                mask_2d,
                HIDDEN,
                BLOCK_H,
            )

            tl.store(h_recurrent_stack_ptr + h_rec_base + h_range, h_new, mask=h_mask)

            if SAVE_ACTIVATIONS:
                step_base_stack = (
                    act_stack_base + layer_idx * h_layer_stride + step * HIDDEN
                )
                store_gru_activations(
                    h_stack_ptr,
                    r_stack_ptr,
                    z_stack_ptr,
                    n_stack_ptr,
                    n_hh_stack_ptr,
                    step_base_stack,
                    h_new,
                    r,
                    z,
                    n,
                    n_hh,
                    h_range,
                    h_mask,
                )

            h_current = h_new

        h_final = h_current

        mu = tl.zeros((BLOCK_S,), dtype=tl.float32)
        for s in range(STATE):
            out_w_mu_s = tl.load(
                out_weight_ptr + s * HIDDEN + h_range, mask=h_mask, other=0.0
            ).to(tl.float32)
            out_bias_mu_s = tl.load(out_bias_ptr + s).to(tl.float32)
            mu_s = out_bias_mu_s + tl.sum(h_final * out_w_mu_s)
            mu = tl.where(s_range == s, mu_s, mu)

        eps_base = batch_idx * n_steps * STATE + step * STATE
        eps_n = tl.load(eps_ptr + eps_base + s_range, mask=s_mask, other=0.0).to(
            tl.float32
        )

        L_times_eps = tl.zeros((BLOCK_S,), dtype=tl.float32)
        L_step_base = batch_idx * n_steps * STATE * STATE + step * STATE * STATE
        if SAVE_ACTIVATIONS:
            transition_cholesky_raw_step_base = (
                batch_idx * n_steps * N_TRIL + step * N_TRIL
            )
        tril_idx = 0
        for row in range(STATE):
            row_sum = tl.sum(tl.zeros((1,), dtype=tl.float32))
            eps_row_base = L_step_base + row * STATE
            for col in range(row + 1):
                out_w_L = tl.load(
                    out_weight_ptr + (STATE + tril_idx) * HIDDEN + h_range,
                    mask=h_mask,
                    other=0.0,
                ).to(tl.float32)
                out_bias_L = tl.load(out_bias_ptr + STATE + tril_idx).to(tl.float32)
                transition_cholesky_raw_ij = out_bias_L + tl.sum(h_final * out_w_L)
                is_diag = row == col
                L_ij = tl.where(
                    is_diag,
                    tl.maximum(transition_cholesky_raw_ij, DIAG_MIN),
                    transition_cholesky_raw_ij,
                )
                eps_col = tl.sum(tl.where(s_range == col, eps_n, 0.0))
                row_sum += L_ij * eps_col
                tl.store(transition_cholesky_ptr + eps_row_base + col, L_ij)
                if SAVE_ACTIVATIONS:
                    tl.store(
                        transition_cholesky_raw_ptr
                        + transition_cholesky_raw_step_base
                        + tril_idx,
                        transition_cholesky_raw_ij,
                    )
                tril_idx += 1
            L_times_eps = tl.where(s_range == row, row_sum, L_times_eps)

        x_t = x_t + mu * time_step + L_times_eps * sqrt_time_step

        diffusion_path_step_base = diffusion_path_base + (step + 1) * STATE
        tl.store(
            diffusion_paths_ptr + diffusion_path_step_base + s_range, x_t, mask=s_mask
        )
        tl.store(
            transition_means_ptr + batch_idx * n_steps * STATE + step * STATE + s_range,
            mu,
            mask=s_mask,
        )


def launch_fwd(
    x0: Tensor,
    context: Tensor,
    sde_parameters: Tensor,
    eps: Tensor,
    weights: SDEWeights,
    time_step: float,
    save_activations: bool,
) -> tuple[Tensor, Tensor, Tensor, SavedActivations | None]:
    batch_size = x0.shape[0]
    n_steps = context.shape[1]
    device = x0.device

    hidden_dim = weights.hidden_dim
    context_dim = weights.context_dim
    sde_param_dim = weights.sde_param_dim
    state_dim = weights.state_dim
    num_layers = weights.num_layers
    n_tril = state_dim * (state_dim + 1) // 2

    diffusion_paths = torch.empty(
        batch_size, n_steps + 1, state_dim, device=device, dtype=torch.float32
    )
    transition_means = torch.empty(
        batch_size, n_steps, state_dim, device=device, dtype=torch.float32
    )
    transition_cholesky = torch.zeros(
        batch_size, n_steps, state_dim, state_dim, device=device, dtype=torch.float32
    )

    h_recurrent_stack = torch.zeros(
        batch_size,
        max(num_layers - 1, 1),
        hidden_dim,
        device=device,
        dtype=torch.float32,
    )

    if save_activations:
        transition_cholesky_raw = torch.empty(
            batch_size, n_steps, n_tril, device=device, dtype=torch.float32
        )
        h_l0 = torch.empty(
            batch_size, n_steps, hidden_dim, device=device, dtype=torch.float32
        )
        r_l0 = torch.empty(
            batch_size, n_steps, hidden_dim, device=device, dtype=torch.float32
        )
        z_l0 = torch.empty(
            batch_size, n_steps, hidden_dim, device=device, dtype=torch.float32
        )
        n_l0 = torch.empty(
            batch_size, n_steps, hidden_dim, device=device, dtype=torch.float32
        )
        n_hh_l0 = torch.empty(
            batch_size, n_steps, hidden_dim, device=device, dtype=torch.float32
        )

        if num_layers > 1:
            h_stack = torch.empty(
                batch_size,
                num_layers - 1,
                n_steps,
                hidden_dim,
                device=device,
                dtype=torch.float32,
            )
            r_stack = torch.empty(
                batch_size,
                num_layers - 1,
                n_steps,
                hidden_dim,
                device=device,
                dtype=torch.float32,
            )
            z_stack = torch.empty(
                batch_size,
                num_layers - 1,
                n_steps,
                hidden_dim,
                device=device,
                dtype=torch.float32,
            )
            n_stack = torch.empty(
                batch_size,
                num_layers - 1,
                n_steps,
                hidden_dim,
                device=device,
                dtype=torch.float32,
            )
            n_hh_stack = torch.empty(
                batch_size,
                num_layers - 1,
                n_steps,
                hidden_dim,
                device=device,
                dtype=torch.float32,
            )
        else:
            h_stack = torch.empty(0, device=device, dtype=torch.float32)
            r_stack = torch.empty(0, device=device, dtype=torch.float32)
            z_stack = torch.empty(0, device=device, dtype=torch.float32)
            n_stack = torch.empty(0, device=device, dtype=torch.float32)
            n_hh_stack = torch.empty(0, device=device, dtype=torch.float32)
    else:
        dummy = torch.empty(0, device=device, dtype=torch.float32)
        transition_cholesky_raw = h_l0 = r_l0 = z_l0 = n_l0 = n_hh_l0 = dummy
        h_stack = r_stack = z_stack = n_stack = n_hh_stack = dummy

    sqrt_time_step = math.sqrt(time_step)
    context_stride = n_steps * context_dim
    block_h = next_power_of_two(hidden_dim)
    block_s = next_power_of_two(state_dim)

    sde_fwd_kernel[(batch_size,)](
        x0.view(-1).contiguous().float(),
        context.contiguous().float(),
        sde_parameters.contiguous().float(),
        eps.view(batch_size, -1).contiguous().float(),
        weights.W_ih_l0,
        weights.W_hh_l0,
        weights.b_ih_l0,
        weights.b_hh_l0,
        weights.W_ih_stack.view(-1)
        if weights.W_ih_stack.numel() > 0
        else weights.W_ih_stack,
        weights.W_hh_stack.view(-1)
        if weights.W_hh_stack.numel() > 0
        else weights.W_hh_stack,
        weights.b_ih_stack.view(-1)
        if weights.b_ih_stack.numel() > 0
        else weights.b_ih_stack,
        weights.b_hh_stack.view(-1)
        if weights.b_hh_stack.numel() > 0
        else weights.b_hh_stack,
        weights.out_weight,
        weights.out_bias,
        diffusion_paths.view(-1),
        transition_means.view(-1),
        transition_cholesky.view(-1),
        transition_cholesky_raw.view(-1),
        h_l0.view(-1),
        r_l0.view(-1),
        z_l0.view(-1),
        n_l0.view(-1),
        n_hh_l0.view(-1),
        h_stack.view(-1),
        r_stack.view(-1),
        z_stack.view(-1),
        n_stack.view(-1),
        n_hh_stack.view(-1),
        h_recurrent_stack.view(-1),
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
        SAVE_ACTIVATIONS=save_activations,
        NUM_LAYERS=num_layers,
    )

    saved = None
    if save_activations:
        saved = SavedActivations(
            diffusion_paths,
            transition_cholesky_raw,
            h_l0,
            r_l0,
            z_l0,
            n_l0,
            n_hh_l0,
            h_stack,
            r_stack,
            z_stack,
            n_stack,
            n_hh_stack,
        )
    return diffusion_paths, transition_means, transition_cholesky, saved
