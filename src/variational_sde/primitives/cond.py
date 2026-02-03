from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from variational_sde.primitives import initializer


@dataclass
class CondBranch:
    scale: torch.Tensor
    shift: torch.Tensor
    gate_value: torch.Tensor

    def affine(self, tensor: torch.Tensor) -> torch.Tensor:
        return (1 + self.scale) * tensor + self.shift

    def gate(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.gate_value


class CondModulator(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int,
        *,
        branches: int = 1,
    ) -> None:
        super().__init__()
        if branches <= 0:
            raise ValueError("branches must be positive")
        self.branch_count = branches
        self.chunk_count = 3 * branches
        linear = nn.Linear(cond_dim, hidden_dim * self.chunk_count)
        initializer.zero_weights(linear)
        self.net = nn.Sequential(nn.SiLU(), linear)

    def forward(self, *, cond: torch.Tensor) -> tuple[CondBranch, ...]:
        updates = self.net(cond)
        chunked = torch.chunk(updates, self.chunk_count, dim=-1)
        outputs: list[CondBranch] = []
        for idx in range(self.branch_count):
            start = idx * 3
            outputs.append(CondBranch(
                scale=chunked[start],
                shift=chunked[start + 1],
                gate_value=chunked[start + 2],
            ))
        return tuple(outputs)


class CondMixin(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int,
        *,
        branches: int = 1,
    ) -> None:
        super().__init__()
        self._cond_modulator = CondModulator(
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            branches=branches,
        )

    def cond_params(self, *, cond: torch.Tensor) -> tuple[CondBranch, ...]:
        result: tuple[CondBranch, ...] = self._cond_modulator(cond=cond)
        return result
