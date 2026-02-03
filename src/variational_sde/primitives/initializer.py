from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn

TRUNC_STD: Final[float] = 0.02


def _weight(module: nn.Module) -> nn.Parameter:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, nn.Parameter):
        raise AttributeError(f"{module.__class__.__name__} has no trainable weight")
    return weight


def _zero_bias(module: nn.Module) -> None:
    bias = getattr(module, "bias", None)
    if isinstance(bias, torch.Tensor):
        torch.nn.init.zeros_(bias)


def _init_transformer_linear(module: nn.Module) -> None:
    torch.nn.init.trunc_normal_(_weight(module), mean=0.0, std=TRUNC_STD)
    _zero_bias(module)


def zero_weights(module: nn.Module) -> None:
    torch.nn.init.zeros_(_weight(module))
    _zero_bias(module)


@dataclass(frozen=True)
class InitPolicy:
    attn_in: Callable[[nn.Module], None] = _init_transformer_linear
    attn_out: Callable[[nn.Module], None] = _init_transformer_linear
    mlp_in: Callable[[nn.Module], None] = _init_transformer_linear
    mlp_out: Callable[[nn.Module], None] = _init_transformer_linear
    linear: Callable[[nn.Module], None] = _init_transformer_linear


DEFAULT_INIT_POLICY = InitPolicy()
