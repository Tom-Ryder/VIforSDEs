from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.distributed as dist
from torch import nn

if TYPE_CHECKING:
    from variational_sde.accelerate import Accelerator

from variational_sde.accelerate import suppress_torch_compile_output
from variational_sde.console import Console
from einops import repeat

from variational_sde.config import EncoderConfig, HeadConfig, TrainingConfig, PretrainConfig
from variational_sde.core.euler_maruyama import euler_maruyama
from variational_sde.core.observations import ObservationLikelihood, Observations
from variational_sde.core.priors import Prior
from variational_sde.core.sde import SDE
from variational_sde.inference.constants import LOSS_EMA_DECAY
from variational_sde.inference.diffusion_path_sampler import sample_diffusion_paths
from variational_sde.inference.evidence_lower_bound import compute_evidence_lower_bound
from variational_sde.inference.exponential_moving_average import (
    ExponentialMovingAverage,
)
from variational_sde.inference.state_space import StateSpace
from variational_sde.inference.training_context import TrainingContext
from variational_sde.inference.types import EvidenceLowerBoundResult
from variational_sde.models.variational_sde_posterior import VariationalSDEPosterior


@dataclass
class TrainingState:
    step: int
    evidence_lower_bound_history: list[float]
    best_evidence_lower_bound: float
    model: VariationalSDEPosterior
    exponential_moving_average: ExponentialMovingAverage


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    elbo_result: EvidenceLowerBoundResult
    grad_norm: float


class VariationalInferenceTrainer:
    def __init__(
        self,
        sde: SDE,
        observations: Observations,
        observation_likelihood: ObservationLikelihood,
        prior: Prior,
        time_horizon: float,
        config: TrainingConfig,
        encoder_config: EncoderConfig,
        head_config: HeadConfig,
        state_positive_dims: list[int],
        sde_param_positive_dims: list[int],
        device: torch.device | str = "cuda",
        mixed_precision: bool = True,
        console: Console | None = None,
        param_names: list[str] | None = None,
        accelerator: Accelerator | None = None,
        sde_param_init_mean: torch.Tensor | None = None,
    ) -> None:
        self.sde = sde
        self.param_names = param_names
        self.observation_likelihood = observation_likelihood
        self.prior = prior
        self.time_horizon = time_horizon
        self.config = config
        self.state_space = StateSpace(sde.state_dim, state_positive_dims)
        self.sde_param_positive_dims = sde_param_positive_dims
        self.console = console if console is not None else Console()

        self.ctx = TrainingContext.create(
            observations=observations,
            state_dim=sde.state_dim,
            sde_param_dim=sde.sde_param_dim,
            config=config,
            encoder_config=encoder_config,
            head_config=head_config,
            sde_param_positive_dims=sde_param_positive_dims,
            device=device,
            mixed_precision=mixed_precision,
            accelerator=accelerator,
            sde_param_init_mean=sde_param_init_mean,
        )

        self.step = 0
        self.evidence_lower_bound_history: list[float] = []
        self.best_evidence_lower_bound = float("-inf")

    @property
    def device(self) -> torch.device:
        return self.ctx.device

    def train(
        self,
        callback: Callable[[int, float], None] | None = None,
    ) -> TrainingState:
        self.ctx.trainable_model().train()

        if self.ctx.is_main:
            self.console.config_panel(self.config)

        model = self.ctx.unwrap_model()
        loss_ema = 0.0

        with (
            suppress_torch_compile_output(),
            self.console.training_progress(
                self.config.n_iterations,
                update_interval=10,
                param_names=self.param_names,
            ) as progress,
        ):
            for step in range(self.config.n_iterations):
                self.step = step
                step_result = self._train_step(model)
                elbo = step_result.elbo_result.evidence_lower_bound.item()

                self.ctx.ema.update()

                if self.ctx.is_distributed:
                    elbo_tensor = torch.tensor(elbo, device=self.ctx.device)
                    dist.all_reduce(elbo_tensor, op=dist.ReduceOp.AVG)
                    elbo = elbo_tensor.item()

                loss_ema = (
                    LOSS_EMA_DECAY * loss_ema + (1 - LOSS_EMA_DECAY) * (-elbo)
                    if step > 0
                    else -elbo
                )
                smoothed_loss = loss_ema / (1 - LOSS_EMA_DECAY ** (step + 1))

                self.evidence_lower_bound_history.append(elbo)
                if elbo > self.best_evidence_lower_bound:
                    self.best_evidence_lower_bound = elbo

                if self.ctx.is_main:
                    progress.update(
                        step=step,
                        loss=smoothed_loss,
                        elbo=elbo,
                        best_elbo=self.best_evidence_lower_bound,
                        components=step_result.elbo_result.components,
                        grad_norm=step_result.grad_norm,
                        param_means=model.sde_parameter_posterior.expected_value,
                    )

                if callback is not None and self.ctx.is_main:
                    callback(step, elbo)

        return TrainingState(
            step=self.step,
            evidence_lower_bound_history=self.evidence_lower_bound_history,
            best_evidence_lower_bound=self.best_evidence_lower_bound,
            model=self.ctx.model,
            exponential_moving_average=self.ctx.ema,
        )

    def _train_step(self, model: VariationalSDEPosterior) -> TrainStepResult:
        self.ctx.optimizer.zero_grad()

        sde_parameters = model.sde_parameter_posterior.rsample(self.config.batch_size)

        with torch.autocast(
            device_type=self.ctx.device.type,
            dtype=self.config.amp_dtype.value,
            enabled=self.ctx.scaler.is_enabled(),
        ):
            sample = sample_diffusion_paths(
                model.encoder,
                model.head,
                self.ctx.observations,
                sde_parameters,
                self.ctx.x0_buffer,
                self.time_horizon,
                self.config.time_step,
                self.state_space,
            )
            result = compute_evidence_lower_bound(
                self.sde,
                self.ctx.observations,
                self.observation_likelihood,
                self.prior,
                model.sde_parameter_posterior,
                sde_parameters,
                sample,
                self.config.time_step,
            )

        scaled_loss: torch.Tensor = self.ctx.scaler.scale(-result.evidence_lower_bound)
        scaled_loss.backward()
        self.ctx.scaler.unscale_(self.ctx.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.ctx.model.parameters(), self.config.grad_clip_norm
        )
        self.ctx.scaler.step(self.ctx.optimizer)
        self.ctx.scaler.update()

        return TrainStepResult(elbo_result=result, grad_norm=grad_norm.item())

    def pretrain_sde_parameters(self, config: PretrainConfig | None = None) -> torch.Tensor:
        cfg = config or PretrainConfig()
        d = self.sde.sde_param_dim

        mu = nn.Parameter(torch.zeros(d, device=self.device))
        mu.data[self.sde_param_positive_dims] = 0.0
        unconstrained = [i for i in range(d) if i not in self.sde_param_positive_dims]
        if unconstrained:
            mu.data[unconstrained] = cfg.init_scale * torch.randn(len(unconstrained), device=self.device)
        log_sigma = nn.Parameter(torch.zeros(d, device=self.device))
        opt = torch.optim.Adam([mu, log_sigma], lr=cfg.learning_rate)

        best_mu = mu.detach().clone()
        best_mse = float("inf")

        with self.console.pretrain_progress(cfg.n_iterations) as progress:
            for step in range(cfg.n_iterations):
                opt.zero_grad()

                sigma = log_sigma.exp()
                eps = torch.randn(cfg.batch_size, d, device=self.device)
                log_theta = mu + sigma * eps
                theta = self._to_constrained_theta_batch(log_theta)

                mse = self._pretrain_mse_batch(theta)

                if torch.isfinite(mse) and mse.item() < best_mse:
                    best_mu = mu.detach().clone()
                    best_mse = mse.item()

                if torch.isfinite(mse):
                    mse.backward()
                    nn.utils.clip_grad_norm_([mu, log_sigma], 1.0)
                    opt.step()

                progress.update(step, mse.item(), best_mse, sigma.median().item())

        return best_mu

    def _to_constrained_theta_batch(self, log_theta: torch.Tensor) -> torch.Tensor:
        theta = log_theta.clone()
        theta[:, self.sde_param_positive_dims] = log_theta[:, self.sde_param_positive_dims].exp()
        return theta

    def _pretrain_mse_batch(self, theta: torch.Tensor) -> torch.Tensor:
        n = theta.shape[0]
        x0 = repeat(self.ctx.observations.values[0], 'd -> n d', n=n)
        paths = euler_maruyama(
            self.sde, x0, theta, self.time_horizon, self.config.time_step, self.state_space.positive_dims
        )
        obs_idx = (self.ctx.observations.times / self.config.time_step).round().long()
        return ((paths[:, obs_idx] - self.ctx.observations.values) ** 2).mean()

    def cleanup(self) -> None:
        self.ctx.cleanup()
