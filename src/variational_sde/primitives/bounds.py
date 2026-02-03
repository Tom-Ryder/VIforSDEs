from __future__ import annotations

from typing import Protocol, cast

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx


class LowerBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, bound: Tensor) -> Tensor:
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor, None]:
        saved = cast(_LowerBoundCtx, ctx).saved_tensors
        x, bound = saved
        pass_through_if = (x >= bound) | (grad_output < 0)
        return pass_through_if * grad_output, None


class _LowerBoundCtx(Protocol):
    saved_tensors: tuple[Tensor, Tensor]


def lower_bound(x: Tensor, bound: float | Tensor) -> Tensor:
    bound_tensor = torch.as_tensor(bound, dtype=x.dtype, device=x.device)
    result: Tensor = LowerBoundFunction.apply(x, bound_tensor)
    return result
