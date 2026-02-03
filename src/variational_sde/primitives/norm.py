from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn


class RMS(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, requires_grad: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms * self.weight.to(x.dtype)
        return x.to(orig_dtype)


class NormConfig(Protocol):
    def build(self, *, dim: int) -> nn.Module: ...


@dataclass(frozen=True)
class LayerNormConfig:
    eps: float = 1e-5
    affine: bool = True

    def build(self, *, dim: int) -> nn.Module:
        return nn.LayerNorm(dim, eps=self.eps, elementwise_affine=self.affine)
