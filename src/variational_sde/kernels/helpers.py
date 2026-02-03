from __future__ import annotations

import triton
import triton.language as tl

from variational_sde.kernels.constants import GATE_N, GATE_R, GATE_Z, NUM_GATES


@triton.jit
def project_input_rzn(
    input_vec: tl.tensor,
    W_ptr: tl.tensor,
    base_offset: int,
    dim: tl.constexpr,
    dim_range: tl.tensor,
    h_range: tl.tensor,
    h_mask: tl.tensor,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
    r_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    z_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    n_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for i in range(dim):
        val = tl.sum(tl.where(dim_range == i, input_vec, 0.0))
        i_base = base_offset + i * NUM_GATES * HIDDEN
        W_r = tl.load(
            W_ptr + i_base + GATE_R * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)
        W_z = tl.load(
            W_ptr + i_base + GATE_Z * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)
        W_n = tl.load(
            W_ptr + i_base + GATE_N * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)
        r_acc += val * W_r
        z_acc += val * W_z
        n_acc += val * W_n
    return r_acc, z_acc, n_acc


@triton.jit
def project_scalar_rzn(
    scalar_ptr: tl.tensor,
    scalar_base: int,
    W_ptr: tl.tensor,
    weight_base_offset: int,
    dim: tl.constexpr,
    h_range: tl.tensor,
    h_mask: tl.tensor,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
    r_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    z_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    n_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for i in range(dim):
        val = tl.load(scalar_ptr + scalar_base + i).to(tl.float32)
        i_base = weight_base_offset + i * NUM_GATES * HIDDEN
        W_r = tl.load(
            W_ptr + i_base + GATE_R * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)
        W_z = tl.load(
            W_ptr + i_base + GATE_Z * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)
        W_n = tl.load(
            W_ptr + i_base + GATE_N * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)
        r_acc += val * W_r
        z_acc += val * W_z
        n_acc += val * W_n
    return r_acc, z_acc, n_acc


@triton.jit
def atomic_add_rzn_bias(
    grad_b_ih_ptr: tl.tensor,
    grad_b_hh_ptr: tl.tensor,
    d_r_pre: tl.tensor,
    d_z_pre: tl.tensor,
    d_n_pre: tl.tensor,
    d_n_hh: tl.tensor,
    h_range: tl.tensor,
    h_mask: tl.tensor,
    HIDDEN: tl.constexpr,
) -> None:
    tl.atomic_add(grad_b_ih_ptr + GATE_R * HIDDEN + h_range, d_r_pre, mask=h_mask)
    tl.atomic_add(grad_b_ih_ptr + GATE_Z * HIDDEN + h_range, d_z_pre, mask=h_mask)
    tl.atomic_add(grad_b_ih_ptr + GATE_N * HIDDEN + h_range, d_n_pre, mask=h_mask)
    tl.atomic_add(grad_b_hh_ptr + GATE_R * HIDDEN + h_range, d_r_pre, mask=h_mask)
    tl.atomic_add(grad_b_hh_ptr + GATE_Z * HIDDEN + h_range, d_z_pre, mask=h_mask)
    tl.atomic_add(grad_b_hh_ptr + GATE_N * HIDDEN + h_range, d_n_hh, mask=h_mask)


@triton.jit
def store_gru_activations(
    h_ptr: tl.tensor,
    r_ptr: tl.tensor,
    z_ptr: tl.tensor,
    n_ptr: tl.tensor,
    n_hh_ptr: tl.tensor,
    step_base: int,
    h: tl.tensor,
    r: tl.tensor,
    z: tl.tensor,
    n: tl.tensor,
    n_hh: tl.tensor,
    h_range: tl.tensor,
    h_mask: tl.tensor,
) -> None:
    tl.store(h_ptr + step_base + h_range, h, mask=h_mask)
    tl.store(r_ptr + step_base + h_range, r, mask=h_mask)
    tl.store(z_ptr + step_base + h_range, z, mask=h_mask)
    tl.store(n_ptr + step_base + h_range, n, mask=h_mask)
    tl.store(n_hh_ptr + step_base + h_range, n_hh, mask=h_mask)


@triton.jit
def project_input_rzn_with_grad(
    input_vec: tl.tensor,
    W_ptr: tl.tensor,
    grad_W_ptr: tl.tensor,
    base_offset: int,
    d_r_pre: tl.tensor,
    d_z_pre: tl.tensor,
    d_n_pre: tl.tensor,
    dim: tl.constexpr,
    dim_range: tl.tensor,
    h_range: tl.tensor,
    h_mask: tl.tensor,
    HIDDEN: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
) -> tl.tensor:
    d_input = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
    for i in range(dim):
        val = tl.sum(tl.where(dim_range == i, input_vec, 0.0))
        i_base = base_offset + i * NUM_GATES * HIDDEN
        W_r = tl.load(
            W_ptr + i_base + GATE_R * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)
        W_z = tl.load(
            W_ptr + i_base + GATE_Z * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)
        W_n = tl.load(
            W_ptr + i_base + GATE_N * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)

        d_input_i = (
            tl.sum(d_r_pre * W_r) + tl.sum(d_z_pre * W_z) + tl.sum(d_n_pre * W_n)
        )
        d_input = tl.where(dim_range == i, d_input + d_input_i, d_input)

        tl.atomic_add(
            grad_W_ptr + i_base + GATE_R * HIDDEN + h_range, val * d_r_pre, mask=h_mask
        )
        tl.atomic_add(
            grad_W_ptr + i_base + GATE_Z * HIDDEN + h_range, val * d_z_pre, mask=h_mask
        )
        tl.atomic_add(
            grad_W_ptr + i_base + GATE_N * HIDDEN + h_range, val * d_n_pre, mask=h_mask
        )
    return d_input


@triton.jit
def project_scalar_rzn_with_grad(
    scalar_ptr: tl.tensor,
    scalar_base: int,
    grad_scalar_ptr: tl.tensor,
    W_ptr: tl.tensor,
    grad_W_ptr: tl.tensor,
    weight_base_offset: int,
    d_r_pre: tl.tensor,
    d_z_pre: tl.tensor,
    d_n_pre: tl.tensor,
    dim: tl.constexpr,
    h_range: tl.tensor,
    h_mask: tl.tensor,
    HIDDEN: tl.constexpr,
) -> None:
    for i in range(dim):
        val = tl.load(scalar_ptr + scalar_base + i).to(tl.float32)
        i_base = weight_base_offset + i * NUM_GATES * HIDDEN
        W_r = tl.load(
            W_ptr + i_base + GATE_R * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)
        W_z = tl.load(
            W_ptr + i_base + GATE_Z * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)
        W_n = tl.load(
            W_ptr + i_base + GATE_N * HIDDEN + h_range, mask=h_mask, other=0.0
        ).to(tl.float32)

        d_scalar_i = (
            tl.sum(d_r_pre * W_r) + tl.sum(d_z_pre * W_z) + tl.sum(d_n_pre * W_n)
        )
        tl.store(grad_scalar_ptr + scalar_base + i, d_scalar_i)

        tl.atomic_add(
            grad_W_ptr + i_base + GATE_R * HIDDEN + h_range, val * d_r_pre, mask=h_mask
        )
        tl.atomic_add(
            grad_W_ptr + i_base + GATE_Z * HIDDEN + h_range, val * d_z_pre, mask=h_mask
        )
        tl.atomic_add(
            grad_W_ptr + i_base + GATE_N * HIDDEN + h_range, val * d_n_pre, mask=h_mask
        )
