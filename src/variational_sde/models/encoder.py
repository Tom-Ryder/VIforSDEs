from __future__ import annotations

import torch
from einops import repeat
from torch import Tensor, nn

from variational_sde.config import EncoderConfig
from variational_sde.primitives.embeddings import (
    RotarySpec,
    SinusoidalEmbedding,
    precompute_freq_cis,
)
from variational_sde.primitives.sit import SiT, SiTConfig


class ObservationContextEncoder(nn.Module):
    rope_freqs: Tensor

    def __init__(
        self,
        observation_dim: int,
        sde_param_dim: int,
        config: EncoderConfig,
    ) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.hidden_dim = hidden_dim
        self.num_heads = config.num_heads

        self.obs_proj = nn.Linear(observation_dim, hidden_dim)
        self.bridge_token = nn.Parameter(torch.randn(hidden_dim))
        self.time_embed = SinusoidalEmbedding(hidden_dim)

        self.sde_param_proj = nn.Sequential(
            nn.Linear(sde_param_dim, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
        )

        self.register_buffer(
            "rope_freqs", precompute_freq_cis(hidden_dim // config.num_heads, end=2048)
        )

        self.sit = SiT(
            SiTConfig(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                cond_dim=config.cond_dim,
                num_heads=config.num_heads,
                depth=config.depth,
                mlp_hidden_dim=int(hidden_dim * config.mlp_ratio),
            )
        )

    def forward(
        self,
        obs_values: Tensor,
        obs_times: Tensor,
        sde_parameters: Tensor,
        time_horizon: float,
        time_step: float,
    ) -> Tensor:
        batch = sde_parameters.shape[0]
        device = sde_parameters.device
        dtype = sde_parameters.dtype

        n_steps = int(round(time_horizon / time_step)) + 1
        grid_times = torch.linspace(
            0, time_horizon, n_steps, device=device, dtype=dtype
        )

        h = repeat(self.bridge_token.to(dtype), "d -> t d", t=n_steps).clone()

        t_indices = torch.round(obs_times / time_step).long().clamp(max=n_steps - 1)
        h[t_indices] = self.obs_proj(obs_values).to(dtype)

        time_emb = self.time_embed(grid_times).to(dtype)
        h = h + time_emb

        h = repeat(h, "t d -> b t d", b=batch)

        cond = self.sde_param_proj(sde_parameters)
        cond = repeat(cond, "b d -> b t d", t=n_steps)

        rope_freqs = self.rope_freqs
        if n_steps > rope_freqs.shape[0]:
            rope_freqs = precompute_freq_cis(
                self.hidden_dim // self.num_heads,
                end=n_steps,
                device=device,
                dtype=rope_freqs.dtype,
            )
        rotary = RotarySpec.from_freqs(rope_freqs[:n_steps])

        context: Tensor = self.sit(h, cond=cond, rotary=rotary)
        return context
