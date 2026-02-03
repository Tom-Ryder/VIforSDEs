from __future__ import annotations

import triton
import triton.language as tl

from variational_sde.kernels.constants import GATE_N, GATE_R, GATE_Z


@triton.jit
def load_l0_biases(
    b_ih_ptr: tl.tensor,
    b_hh_ptr: tl.tensor,
    h_range: tl.tensor,
    h_mask: tl.tensor,
    HIDDEN: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    b_ir = tl.load(b_ih_ptr + GATE_R * HIDDEN + h_range, mask=h_mask, other=0.0).to(
        tl.float32
    )
    b_iz = tl.load(b_ih_ptr + GATE_Z * HIDDEN + h_range, mask=h_mask, other=0.0).to(
        tl.float32
    )
    b_in = tl.load(b_ih_ptr + GATE_N * HIDDEN + h_range, mask=h_mask, other=0.0).to(
        tl.float32
    )
    b_hr = tl.load(b_hh_ptr + GATE_R * HIDDEN + h_range, mask=h_mask, other=0.0).to(
        tl.float32
    )
    b_hz = tl.load(b_hh_ptr + GATE_Z * HIDDEN + h_range, mask=h_mask, other=0.0).to(
        tl.float32
    )
    b_hn = tl.load(b_hh_ptr + GATE_N * HIDDEN + h_range, mask=h_mask, other=0.0).to(
        tl.float32
    )
    return b_ir, b_iz, b_in, b_hr, b_hz, b_hn
