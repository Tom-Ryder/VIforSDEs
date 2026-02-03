from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


class StateSpace:
    def __init__(self, dim: int, positive_dims: list[int] | None = None) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        resolved = positive_dims or []
        if any(d < 0 or d >= dim for d in resolved):
            raise ValueError(f"positive_dims must be in [0, {dim}), got {resolved}")
        if len(resolved) != len(set(resolved)):
            raise ValueError(f"positive_dims must be unique, got {resolved}")
        self.dim = dim
        self.positive_dims = resolved

    def to_state(self, z: Tensor) -> Tensor:
        if not self.positive_dims:
            return z
        x = z.clone()
        x[..., self.positive_dims] = F.softplus(z[..., self.positive_dims])
        return x

    def to_latent(self, x: Tensor) -> Tensor:
        if not self.positive_dims:
            return x
        z = x.clone()
        x_pos = x[..., self.positive_dims].clamp(min=1e-6)
        z[..., self.positive_dims] = x_pos + torch.log(-torch.expm1(-x_pos))
        return z

    def log_jacobian(self, z: Tensor) -> Tensor:
        if not self.positive_dims:
            return torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        return F.logsigmoid(z[..., self.positive_dims]).sum(dim=-1)
