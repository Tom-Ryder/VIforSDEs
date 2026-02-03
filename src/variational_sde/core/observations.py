from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor
from typing_extensions import Self


class Observations(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    times: Tensor
    values: Tensor

    @model_validator(mode="after")
    def validate_shapes_and_order(self) -> Self:
        if self.times.ndim != 1:
            raise ValueError("times must be 1D tensor")
        if self.values.ndim != 2:
            raise ValueError("values must be 2D tensor [T_obs, obs_dim]")
        if self.times.shape[0] != self.values.shape[0]:
            raise ValueError(
                f"times and values must have same first dimension: "
                f"got {self.times.shape[0]} vs {self.values.shape[0]}"
            )
        if not torch.all(self.times[1:] >= self.times[:-1]):
            raise ValueError("times must be sorted in non-decreasing order")
        return self


@runtime_checkable
class ObservationLikelihood(Protocol):
    def log_prob(self, observations: Tensor, state: Tensor) -> Tensor: ...


class GaussianObservationLikelihood(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    variance: float
    obs_matrix: Tensor | None = None

    @field_validator("variance")
    @classmethod
    def validate_variance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("variance must be positive")
        return v

    def log_prob(self, observations: Tensor, state: Tensor) -> Tensor:
        if self.obs_matrix is not None:
            if self.obs_matrix.ndim != 2:
                raise ValueError("obs_matrix must be 2D [obs_dim, state_dim]")
            if self.obs_matrix.shape[0] != observations.shape[-1]:
                raise ValueError("obs_matrix first dim must match observations")
            if self.obs_matrix.shape[1] != state.shape[-1]:
                raise ValueError("obs_matrix second dim must match state")
            predicted = torch.einsum("od,...d->...o", self.obs_matrix, state)
        else:
            predicted = state

        if observations.shape != predicted.shape:
            raise ValueError(
                f"observation shape {observations.shape} does not match predicted shape {predicted.shape}"
            )

        var = self.variance
        diff = observations - predicted
        log_prob = -0.5 * (diff**2) / var
        log_prob = log_prob - 0.5 * math.log(2 * math.pi * var)

        return log_prob.sum(dim=-1)
