from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from variational_sde.core.observations import Observations
from variational_sde.inference.state_space import StateSpace
from variational_sde.inference.types import DiffusionPathSample


class EncoderProtocol(Protocol):
    def __call__(
        self,
        obs_values: Tensor,
        obs_times: Tensor,
        sde_parameters: Tensor,
        time_horizon: float,
        time_step: float,
    ) -> Tensor: ...


class HeadProtocol(Protocol):
    def sample_diffusion_paths(
        self,
        x0: Tensor,
        context: Tensor,
        sde_parameters: Tensor,
        standard_noise: Tensor,
        time_step: float,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


def sample_diffusion_paths(
    encoder: EncoderProtocol,
    head: HeadProtocol,
    observations: Observations,
    sde_parameters: Tensor,
    x0: Tensor,
    time_horizon: float,
    time_step: float,
    state_space: StateSpace,
) -> DiffusionPathSample:
    batch_size, state_dim = x0.shape
    device, dtype = x0.device, x0.dtype

    context = encoder(
        observations.values,
        observations.times,
        sde_parameters,
        time_horizon,
        time_step,
    )
    n_steps = context.shape[1] - 1

    noise = torch.randn(batch_size, n_steps, state_dim, device=device, dtype=dtype)
    z0 = state_space.to_latent(x0)

    paths, transition_means, transition_cholesky = head.sample_diffusion_paths(
        z0, context[:, :-1], sde_parameters, noise, time_step
    )

    return DiffusionPathSample(
        z=paths,
        transition_means=transition_means,
        transition_cholesky=transition_cholesky,
        state_space=state_space,
    )
