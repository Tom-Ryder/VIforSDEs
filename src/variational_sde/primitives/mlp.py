from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from variational_sde.primitives import initializer


class SwiGLUActivation(nn.Module):
    def __init__(self, dim: int | None = None, *, channel_last: bool = True) -> None:
        super().__init__()
        if dim is None:
            dim = -1 if channel_last else 1
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size(self.dim)
        if size % 2:
            raise ValueError(
                f"SwiGLUActivation expects an even split along dim {self.dim}, got {size}"
            )
        left, right = torch.chunk(x, 2, dim=self.dim)
        return F.silu(left) * right


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        *,
        bias: bool = True,
        activation_dim: int | None = None,
        channel_last: bool = True,
        policy: initializer.InitPolicy = initializer.DEFAULT_INIT_POLICY,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim * 2, bias=bias)
        self.activation = SwiGLUActivation(
            dim=activation_dim, channel_last=channel_last
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.in_dim, bias=bias)

        policy.mlp_in(self.input_proj)
        policy.mlp_out(self.output_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden: torch.Tensor = self.input_proj(x)
        gated = self.activation(hidden)
        out: torch.Tensor = self.output_proj(gated)
        return out
