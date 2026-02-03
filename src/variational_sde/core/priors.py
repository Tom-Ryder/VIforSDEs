from __future__ import annotations

from enum import Enum, auto

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor
from torch.distributions import (
    Independent,
)
from torch.distributions import (
    LogNormal as TorchLogNormal,
)
from torch.distributions import (
    Normal as TorchNormal,
)


class PriorType(Enum):
    NORMAL = auto()
    LOG_NORMAL = auto()


class Prior(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    type: PriorType
    mean: float
    std: float
    dim: int

    @field_validator("dim")
    @classmethod
    def dim_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("dim must be positive")
        return v

    @field_validator("std")
    @classmethod
    def std_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("std must be positive")
        return v

    def _dist(
        self, device: torch.device | None = None
    ) -> Independent[TorchNormal] | Independent[TorchLogNormal]:
        loc = torch.full((self.dim,), self.mean, device=device)
        scale = torch.full((self.dim,), self.std, device=device)
        if self.type == PriorType.LOG_NORMAL:
            return Independent(TorchLogNormal(loc=loc, scale=scale), 1)
        return Independent(TorchNormal(loc=loc, scale=scale), 1)

    def sample(self, n: int) -> Tensor:
        return self._dist().sample((n,))

    def log_prob(self, sde_parameters: Tensor) -> Tensor:
        dist = self._dist(sde_parameters.device)
        return dist.log_prob(sde_parameters)  # type: ignore[no-any-return]
