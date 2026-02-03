from __future__ import annotations

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from variational_sde.primitives import initializer
from variational_sde.primitives.embeddings import RotarySpec, apply_rope_1d
from variational_sde.primitives.norm import RMS


def _apply_rotary(
    tensor: torch.Tensor,
    *,
    num_heads: int,
    rotary_freqs: torch.Tensor | None,
) -> torch.Tensor:
    if rotary_freqs is None:
        return tensor
    flat = rearrange(tensor, "b s h d -> (b h) s d")
    flat = apply_rope_1d(flat, rotary_freqs)
    return rearrange(flat, "(b h) s d -> b s h d", h=num_heads)


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        qk_norm_eps: float = 1e-6,
        qk_norm: bool = True,
        bias: bool = True,
        gate: bool = True,
        residual_v: bool = False,
        policy: initializer.InitPolicy = initializer.DEFAULT_INIT_POLICY,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        policy.attn_in(self.qkv_proj)
        policy.attn_out(self.out_proj)

        self._use_gate = gate
        if gate:
            self.gate_proj = nn.Linear(embed_dim, self.head_dim, bias=True)
            initializer.zero_weights(self.gate_proj)

        self._use_residual_v = residual_v
        if residual_v:
            self.v_residual_lambda = nn.Parameter(torch.tensor(0.5))

        self.q_norm: nn.Module = (
            RMS(self.head_dim, eps=qk_norm_eps, requires_grad=False)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm: nn.Module = (
            RMS(self.head_dim, eps=qk_norm_eps, requires_grad=False)
            if qk_norm
            else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        rotary: RotarySpec | None = None,
        v0: torch.Tensor | None = None,
        return_value: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        rotary_freqs = rotary.rotary_freqs if rotary is not None else None
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = _apply_rotary(q, num_heads=self.num_heads, rotary_freqs=rotary_freqs)
        k = _apply_rotary(k, num_heads=self.num_heads, rotary_freqs=rotary_freqs)

        if self._use_residual_v:
            if v0 is not None:
                if v0.shape != v.shape:
                    raise ValueError(
                        f"v0 shape {v0.shape} must match value heads {v.shape}"
                    )
                v = self.v_residual_lambda * v + (1.0 - self.v_residual_lambda) * v0

        v_out = v
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        if self._use_gate:
            gate_scores = torch.sigmoid(self.gate_proj(hidden_states))
            gate_scores = rearrange(gate_scores, "b s d -> b 1 s d")
            attn_output = attn_output * gate_scores

        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        out: torch.Tensor = self.out_proj(attn_output)
        if self._use_residual_v or return_value:
            return out, v_out
        return out
