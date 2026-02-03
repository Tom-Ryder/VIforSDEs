from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t[..., None] * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def precompute_freq_cis(
    dim: int,
    end: int = 1000,
    theta: float = 10000.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even")
    compute_device = torch.device(device) if device is not None else None
    compute_dtype = torch.float32
    idx = torch.arange(0, dim, 2, dtype=compute_dtype, device=compute_device)
    inv_freq = theta ** (-idx / dim)
    positions = torch.arange(end, dtype=compute_dtype, device=compute_device)
    angles = torch.outer(positions, inv_freq)
    freqs = torch.polar(torch.ones_like(angles), angles)
    if dtype is None:
        return freqs
    if dtype == torch.float64:
        return freqs.to(torch.complex128)
    if dtype == torch.float32:
        return freqs.to(torch.complex64)
    return freqs


def apply_rope_1d(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    seq_len = tensor.shape[-2]
    rot_dim = freqs.shape[-1] * 2
    rot, remainder = tensor[..., :rot_dim], tensor[..., rot_dim:]
    half = rot.shape[-1] // 2
    real = rot[..., :half]
    imag = rot[..., half:]
    rot_complex = torch.complex(real.to(freqs.real.dtype), imag.to(freqs.real.dtype))
    if seq_len > freqs.shape[0]:
        raise ValueError("requested sequence length exceeds precomputed frequencies")
    freq_slice = freqs[:seq_len].to(device=rot_complex.device)
    while freq_slice.ndim < rot_complex.ndim:
        freq_slice = freq_slice.unsqueeze(0)
    freq_slice = freq_slice.expand(rot_complex.shape)
    rotated = rot_complex * freq_slice
    rotated_real = torch.cat((rotated.real, rotated.imag), dim=-1).to(tensor.dtype)
    return torch.cat((rotated_real, remainder), dim=-1)


@dataclass(frozen=True)
class RotarySpec:
    rotary_freqs: torch.Tensor

    @classmethod
    def from_freqs(cls: type[RotarySpec], freqs: torch.Tensor) -> RotarySpec:
        return cls(rotary_freqs=freqs)
