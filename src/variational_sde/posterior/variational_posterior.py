from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from einops import repeat
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from variational_sde.core.observations import Observations
from variational_sde.core.priors import Prior
from variational_sde.inference.diffusion_path_sampler import sample_diffusion_paths
from variational_sde.inference.exponential_moving_average import (
    ExponentialMovingAverage,
)
from variational_sde.inference.state_space import StateSpace
from variational_sde.inference.types import DiffusionPathSample
from variational_sde.models.variational_sde_posterior import VariationalSDEPosterior
from variational_sde.visualization import plot_posterior

QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class VariationalPosteriorSamples:
    sde_parameters: Tensor
    diffusion_paths: Tensor


@dataclass(frozen=True, slots=True)
class Quantiles:
    q05: Tensor
    q25: Tensor
    q50: Tensor
    q75: Tensor
    q95: Tensor


@dataclass
class VariationalPosteriorSummary:
    sde_parameter_mean: Tensor
    sde_parameter_std: Tensor
    sde_parameter_quantiles: Quantiles
    diffusion_path_mean: Tensor
    diffusion_path_std: Tensor


@dataclass
class InferenceDiagnostics:
    evidence_lower_bound_history: list[float]
    final_evidence_lower_bound: float
    n_iterations: int


class VariationalPosteriorCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_state: dict[str, Tensor]
    ema_state: dict[str, Tensor]
    time_horizon: float
    time_step: float
    state_positive_dims: list[int]
    evidence_lower_bound_history: list[float]


class VariationalPosterior:
    def __init__(
        self,
        model: VariationalSDEPosterior,
        exponential_moving_average: ExponentialMovingAverage,
        prior: Prior,
        observations: Observations,
        time_horizon: float,
        time_step: float,
        state_space: StateSpace,
        evidence_lower_bound_history: list[float],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.exponential_moving_average = exponential_moving_average
        self.prior = prior
        self.observations = Observations(
            times=observations.times.to(device), values=observations.values.to(device)
        )
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.state_space = state_space
        self.evidence_lower_bound_history = evidence_lower_bound_history
        self.device = device

    @torch.no_grad()
    def sample(self, n: int) -> VariationalPosteriorSamples:
        self.model.eval()
        with self.exponential_moving_average.apply():
            sde_parameters = self.model.sde_parameter_posterior.rsample(n)
            x0 = repeat(self.observations.values[0], "d -> n d", n=n)
            result: DiffusionPathSample = sample_diffusion_paths(
                self.model.encoder,
                self.model.head,
                self.observations,
                sde_parameters,
                x0,
                self.time_horizon,
                self.time_step,
                self.state_space,
            )

        return VariationalPosteriorSamples(
            sde_parameters=sde_parameters,
            diffusion_paths=result.x,
        )

    def summary(self, n_samples: int = 1000) -> VariationalPosteriorSummary:
        samples = self.sample(n_samples)
        sde_parameters = samples.sde_parameters
        diffusion_paths = samples.diffusion_paths

        q = torch.quantile(
            sde_parameters,
            torch.tensor(
                QUANTILE_LEVELS, device=self.device, dtype=sde_parameters.dtype
            ),
            dim=0,
        )

        quantiles = Quantiles(q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4])
        return VariationalPosteriorSummary(
            sde_parameter_mean=sde_parameters.mean(dim=0),
            sde_parameter_std=sde_parameters.std(dim=0),
            sde_parameter_quantiles=quantiles,
            diffusion_path_mean=diffusion_paths.mean(dim=0),
            diffusion_path_std=diffusion_paths.std(dim=0),
        )

    def diagnostics(self) -> InferenceDiagnostics:
        return InferenceDiagnostics(
            evidence_lower_bound_history=self.evidence_lower_bound_history,
            final_evidence_lower_bound=self.evidence_lower_bound_history[-1]
            if self.evidence_lower_bound_history
            else float("nan"),
            n_iterations=len(self.evidence_lower_bound_history),
        )

    def plot(self, n_trajectories: int = 50, show: bool = True) -> Figure:
        samples = self.sample(n_trajectories)
        return plot_posterior(samples, self.observations, self.time_horizon, show)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "ema_state": self.exponential_moving_average.state_dict(),
                "time_horizon": self.time_horizon,
                "time_step": self.time_step,
                "state_positive_dims": self.state_space.positive_dims,
                "evidence_lower_bound_history": self.evidence_lower_bound_history,
            },
            Path(path),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        model: VariationalSDEPosterior,
        prior: Prior,
        observations: Observations,
        device: torch.device,
    ) -> VariationalPosterior:
        checkpoint = VariationalPosteriorCheckpoint.model_validate(
            torch.load(Path(path), map_location=device, weights_only=True)
        )
        model.load_state_dict(checkpoint.model_state)
        exponential_moving_average = ExponentialMovingAverage(model)
        exponential_moving_average.load_state_dict(checkpoint.ema_state)

        state_dim = observations.values.shape[-1]
        state_space = StateSpace(state_dim, checkpoint.state_positive_dims)

        return cls(
            model=model,
            exponential_moving_average=exponential_moving_average,
            prior=prior,
            observations=observations,
            time_horizon=checkpoint.time_horizon,
            time_step=checkpoint.time_step,
            state_space=state_space,
            evidence_lower_bound_history=checkpoint.evidence_lower_bound_history,
            device=device,
        )
