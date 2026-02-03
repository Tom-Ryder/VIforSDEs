from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from variational_sde.inference.state_space import StateSpace


@dataclass(frozen=True, slots=True)
class DiffusionPathSample:
    z: Tensor
    transition_means: Tensor
    transition_cholesky: Tensor
    state_space: StateSpace

    @property
    def x(self) -> Tensor:
        return self.state_space.to_state(self.z)

    def log_jacobian(self) -> Tensor:
        return self.state_space.log_jacobian(self.z[:, 1:]).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class EvidenceLowerBoundComponents:
    observation_log_prob: Tensor
    sde_log_prob: Tensor
    generative_log_prob: Tensor
    prior_log_prob: Tensor
    posterior_log_prob: Tensor


@dataclass(frozen=True, slots=True)
class EvidenceLowerBoundResult:
    evidence_lower_bound: Tensor
    components: EvidenceLowerBoundComponents
