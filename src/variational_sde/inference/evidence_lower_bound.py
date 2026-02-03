from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.distributions import MultivariateNormal

from variational_sde.core.observations import ObservationLikelihood, Observations
from variational_sde.core.priors import Prior
from variational_sde.core.sde import SDE
from variational_sde.inference.types import (
    DiffusionPathSample,
    EvidenceLowerBoundComponents,
    EvidenceLowerBoundResult,
)
from variational_sde.models.sde_parameter_posterior import SDEParameterPosterior


def compute_evidence_lower_bound(
    sde: SDE,
    observations: Observations,
    observation_likelihood: ObservationLikelihood,
    prior: Prior,
    sde_parameter_posterior: SDEParameterPosterior,
    sde_parameters: Tensor,
    sample: DiffusionPathSample,
    time_step: float,
) -> EvidenceLowerBoundResult:
    z = sample.z
    x = sample.x
    batch, n_steps = z.shape[0], z.shape[1] - 1
    sqrt_dt = time_step**0.5

    z_t, z_next = z[:, :-1], z[:, 1:]
    x_t, x_next = x[:, :-1], x[:, 1:]

    x_t_flat = rearrange(x_t, "b t d -> (b t) d")
    theta_flat = repeat(sde_parameters, "b d -> (b t) d", t=n_steps)
    drift = rearrange(sde.drift(x_t_flat, theta_flat), "(b t) d -> b t d", b=batch)
    diffusion = rearrange(sde.diffusion(x_t_flat, theta_flat), "(b t) d e -> b t d e", b=batch)

    sde_mu = x_t + drift * time_step
    sde_L = diffusion * sqrt_dt
    sde_log_prob = _gaussian_log_prob(x_next, sde_mu, sde_L)

    gen_mu = z_t + sample.transition_means * time_step
    gen_L = sample.transition_cholesky * sqrt_dt
    gen_log_prob = _gaussian_log_prob(z_next, gen_mu, gen_L)

    jacobian = sample.log_jacobian()

    obs_idx = torch.clamp(torch.round(observations.times / time_step).long(), max=n_steps)
    obs_log_prob = observation_likelihood.log_prob(
        repeat(observations.values, "t d -> b t d", b=batch),
        x[:, obs_idx],
    ).sum(dim=-1)

    prior_log_prob = prior.log_prob(sde_parameters)
    if prior_log_prob.ndim > 1:
        prior_log_prob = prior_log_prob.sum(-1)
    posterior_log_prob = sde_parameter_posterior.log_prob(sde_parameters)

    elbo = obs_log_prob + sde_log_prob - gen_log_prob + jacobian + prior_log_prob - posterior_log_prob

    return EvidenceLowerBoundResult(
        evidence_lower_bound=elbo.mean(),
        components=EvidenceLowerBoundComponents(
            observation_log_prob=obs_log_prob.mean(),
            sde_log_prob=sde_log_prob.mean(),
            generative_log_prob=gen_log_prob.mean(),
            prior_log_prob=prior_log_prob.mean(),
            posterior_log_prob=posterior_log_prob.mean(),
        ),
    )


def _gaussian_log_prob(x: Tensor, mu: Tensor, L: Tensor) -> Tensor:
    flat_x = rearrange(x, "b t d -> (b t) d")
    flat_mu = rearrange(mu, "b t d -> (b t) d")
    flat_L = rearrange(L, "b t d e -> (b t) d e")
    dist = MultivariateNormal(loc=flat_mu, scale_tril=flat_L)
    log_probs: Tensor = dist.log_prob(flat_x)
    return log_probs.reshape(x.shape[0], -1).sum(dim=-1)
