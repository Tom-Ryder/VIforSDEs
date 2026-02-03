from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor

from variational_sde.config import EncoderConfig, HeadConfig, PretrainConfig, TrainingConfig
from variational_sde.core.observations import Observations, ObservationLikelihood
from variational_sde.core.priors import Prior
from variational_sde.core.sde import SDE
from variational_sde.inference.state_space import StateSpace
from variational_sde.inference.trainer import VariationalInferenceTrainer
from variational_sde.posterior.variational_posterior import VariationalPosterior

if TYPE_CHECKING:
    from variational_sde.accelerate import Accelerator
    from variational_sde.console import Console


@dataclass(frozen=True)
class InferenceConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    state_positive_dims: list[int] = field(default_factory=list)
    sde_param_positive_dims: list[int] = field(default_factory=list)
    device: str | torch.device = "cuda"
    mixed_precision: bool = True
    param_names: list[str] | None = None
    accelerator: Accelerator | None = None
    sde_param_init_mean: Tensor | None = None
    pretrain: bool | PretrainConfig = False
    console: Console | None = None


class _InferenceInputs(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    observations: Observations
    time_horizon: float
    time_step: float
    state_dim: int
    sde_param_dim: int
    state_positive_dims: list[int]
    sde_param_positive_dims: list[int]
    prior: Prior

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        if self.time_horizon <= 0:
            raise ValueError("time_horizon must be positive")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        times = self.observations.times
        if times.numel() == 0:
            raise ValueError("observations must be non-empty")
        ratio = self.time_horizon / self.time_step
        n_steps = round(ratio)
        if not math.isclose(ratio, n_steps, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("time_horizon must be an integer multiple of time_step")
        tol = max(1e-6, 1e-4 * self.time_step)
        if torch.any(torch.abs(times[0]) > tol).item():
            raise ValueError("first observation time must be 0")
        aligned = torch.round(times / self.time_step) * self.time_step
        if torch.any(torch.abs(aligned - times) > tol).item():
            raise ValueError("observation times must align to time_step grid")
        if torch.any(times < 0).item() or torch.any(times > self.time_horizon).item():
            raise ValueError("observation times must be within [0, time_horizon]")
        if len(set(self.state_positive_dims)) != len(self.state_positive_dims):
            raise ValueError("state_positive_dims must be unique")
        if len(set(self.sde_param_positive_dims)) != len(self.sde_param_positive_dims):
            raise ValueError("sde_param_positive_dims must be unique")
        if any(d < 0 or d >= self.state_dim for d in self.state_positive_dims):
            raise ValueError("state_positive_dims must be within [0, state_dim)")
        if any(d < 0 or d >= self.sde_param_dim for d in self.sde_param_positive_dims):
            raise ValueError(
                "sde_param_positive_dims must be within [0, sde_param_dim)"
            )
        if self.prior.dim != self.sde_param_dim:
            raise ValueError("prior dim must match sde_param_dim")
        return self


def infer(
    sde: SDE,
    observations: Observations,
    observation_likelihood: ObservationLikelihood,
    prior: Prior,
    time_horizon: float,
    config: InferenceConfig | None = None,
) -> VariationalPosterior:
    cfg = config or InferenceConfig()
    device = cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"

    inputs = _InferenceInputs(
        observations=observations,
        time_horizon=time_horizon,
        time_step=cfg.training.time_step,
        state_dim=sde.state_dim,
        sde_param_dim=sde.sde_param_dim,
        state_positive_dims=list(cfg.state_positive_dims),
        sde_param_positive_dims=list(cfg.sde_param_positive_dims),
        prior=prior,
    )

    trainer = VariationalInferenceTrainer(
        sde=sde,
        observations=inputs.observations,
        observation_likelihood=observation_likelihood,
        prior=prior,
        time_horizon=inputs.time_horizon,
        config=cfg.training,
        encoder_config=cfg.encoder,
        head_config=cfg.head,
        state_positive_dims=inputs.state_positive_dims,
        sde_param_positive_dims=inputs.sde_param_positive_dims,
        device=device,
        mixed_precision=cfg.mixed_precision,
        console=cfg.console,
        param_names=cfg.param_names,
        accelerator=cfg.accelerator,
        sde_param_init_mean=cfg.sde_param_init_mean,
    )

    if cfg.pretrain and cfg.sde_param_init_mean is None:
        pretrain_config = cfg.pretrain if isinstance(cfg.pretrain, PretrainConfig) else None
        pretrained_mean = trainer.pretrain_sde_parameters(pretrain_config)
        trainer.ctx.model.sde_parameter_posterior.mean.data.copy_(pretrained_mean)

    try:
        state = trainer.train()
    finally:
        trainer.cleanup()

    state_space = StateSpace(sde.state_dim, inputs.state_positive_dims)

    return VariationalPosterior(
        model=state.model,
        exponential_moving_average=state.exponential_moving_average,
        prior=prior,
        observations=inputs.observations,
        time_horizon=inputs.time_horizon,
        time_step=cfg.training.time_step,
        state_space=state_space,
        evidence_lower_bound_history=state.evidence_lower_bound_history,
        device=trainer.device,
    )
