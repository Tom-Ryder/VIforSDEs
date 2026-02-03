from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from variational_sde.primitives import initializer
from variational_sde.primitives.attn import Attention
from variational_sde.primitives.cond import CondBranch, CondMixin
from variational_sde.primitives.embeddings import RotarySpec
from variational_sde.primitives.mlp import SwiGLU
from variational_sde.primitives.norm import LayerNormConfig, NormConfig


@dataclass(frozen=True)
class SiTConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    cond_dim: int
    num_heads: int
    depth: int
    mlp_hidden_dim: int
    bias: bool = True
    attn_gate: bool = True
    attn_residual_v: bool = True
    use_qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    attn_norm: NormConfig = LayerNormConfig(affine=False)
    mlp_norm: NormConfig = LayerNormConfig(affine=False)
    policy: initializer.InitPolicy = initializer.DEFAULT_INIT_POLICY


class SiTBlock(CondMixin):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        cond_dim: int,
        bias: bool = True,
        attn_gate: bool = True,
        attn_residual_v: bool = False,
        return_value_state: bool = False,
        use_qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        attn_norm: NormConfig = LayerNormConfig(affine=False),
        mlp_norm: NormConfig = LayerNormConfig(affine=False),
        policy: initializer.InitPolicy = initializer.DEFAULT_INIT_POLICY,
    ) -> None:
        super().__init__(
            cond_dim=cond_dim,
            hidden_dim=dim,
            branches=2,
        )
        self.dim = dim
        self.num_heads = num_heads
        self.attn_residual_v = attn_residual_v
        self._return_value_state = return_value_state
        self.self_attn = Attention(
            embed_dim=dim,
            num_heads=num_heads,
            bias=bias,
            gate=attn_gate,
            qk_norm=use_qk_norm,
            qk_norm_eps=qk_norm_eps,
            residual_v=attn_residual_v,
            policy=policy,
        )
        self.mlp = SwiGLU(
            in_dim=dim, hidden_dim=mlp_hidden_dim, bias=bias, policy=policy
        )
        self.attn_norm = attn_norm.build(dim=dim)
        self.mlp_norm = mlp_norm.build(dim=dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cond: torch.Tensor,
        rotary: RotarySpec | None = None,
        v0: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_branch, mlp_branch = self.cond_params(cond=cond)
        attn_result, value_state = self._apply_attention(
            hidden_states,
            branch=attn_branch,
            rotary=rotary,
            v0=v0,
        )
        hidden_states = attn_result
        hidden_states = self._apply_mlp(hidden_states, branch=mlp_branch)
        if value_state is not None:
            return hidden_states, value_state
        return hidden_states

    def _apply_attention(
        self,
        tensor: torch.Tensor,
        *,
        branch: CondBranch,
        rotary: RotarySpec | None,
        v0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = self.attn_norm(tensor)
        modulated = branch.affine(normed)
        attn_result = self.self_attn(
            modulated,
            rotary=rotary,
            v0=v0,
            return_value=self.attn_residual_v or self._return_value_state,
        )
        if self.attn_residual_v or self._return_value_state:
            attn_out, value_state = attn_result
        else:
            attn_out = attn_result
            value_state = None
        gated = branch.gate(attn_out)
        return tensor + gated, value_state

    def _apply_mlp(self, tensor: torch.Tensor, *, branch: CondBranch) -> torch.Tensor:
        normed = self.mlp_norm(tensor)
        modulated = branch.affine(normed)
        mlp_out = self.mlp(modulated)
        gated = branch.gate(mlp_out)
        return tensor + gated


class SiT(nn.Module):
    def __init__(self, config: SiTConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                SiTBlock(
                    dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    mlp_hidden_dim=config.mlp_hidden_dim,
                    cond_dim=config.cond_dim,
                    bias=config.bias,
                    attn_gate=config.attn_gate,
                    attn_residual_v=config.attn_residual_v and idx > 0,
                    return_value_state=config.attn_residual_v,
                    use_qk_norm=config.use_qk_norm,
                    qk_norm_eps=config.qk_norm_eps,
                    attn_norm=config.attn_norm,
                    mlp_norm=config.mlp_norm,
                    policy=config.policy,
                )
                for idx in range(config.depth)
            ]
        )
        self.input_proj = nn.Linear(config.in_dim, config.hidden_dim, bias=config.bias)
        self.output_proj = nn.Linear(
            config.hidden_dim, config.out_dim, bias=config.bias
        )
        config.policy.linear(self.input_proj)
        config.policy.linear(self.output_proj)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cond: torch.Tensor,
        rotary: RotarySpec | None = None,
    ) -> torch.Tensor:
        tokens = x.contiguous()
        tokens = self.input_proj(tokens)
        cached_v: torch.Tensor | None = None
        for block in self.blocks:
            out = block(
                tokens,
                cond=cond,
                rotary=rotary,
                v0=cached_v,
            )
            if isinstance(out, tuple):
                tokens, block_value = out
                if cached_v is None:
                    cached_v = block_value
            else:
                tokens = out
        result: torch.Tensor = self.output_proj(tokens)
        return result
