from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from variational_sde.core.sde import SDE


def euler_maruyama(
    sde: SDE,
    x0: Tensor,
    theta: Tensor,
    time_horizon: float,
    dt: float,
    positive_dims: Sequence[int] = (),
    noise: Tensor | None = None,
) -> Tensor:
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if time_horizon <= 0:
        raise ValueError(f"time_horizon must be positive, got {time_horizon}")
    n_steps = round(time_horizon / dt)
    sqrt_dt = dt**0.5
    batch, state_dim = x0.shape

    if noise is None:
        noise = torch.randn(batch, n_steps, state_dim, device=x0.device, dtype=x0.dtype)

    trajectory = torch.empty(
        batch, n_steps + 1, state_dim, device=x0.device, dtype=x0.dtype
    )
    trajectory[:, 0] = x0
    x = x0.clone()

    for step in range(n_steps):
        drift = sde.drift(x, theta)
        diffusion = sde.diffusion(x, theta)
        x = x + drift * dt + torch.einsum("bij,bj->bi", diffusion, noise[:, step]) * sqrt_dt
        if positive_dims:
            x[:, list(positive_dims)] = x[:, list(positive_dims)].clamp(min=1e-6)
        trajectory[:, step + 1] = x

    return trajectory
