from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class SDE(Protocol):
    state_dim: int
    sde_param_dim: int

    def drift(self, x: Tensor, sde_parameters: Tensor) -> Tensor: ...
    def diffusion(self, x: Tensor, sde_parameters: Tensor) -> Tensor: ...


class FunctionalSDE:
    def __init__(
        self,
        drift_fn: Callable[[Tensor, Tensor], Tensor],
        diffusion_fn: Callable[[Tensor, Tensor], Tensor],
        state_dim: int,
        sde_param_dim: int,
    ) -> None:
        self._drift_fn = drift_fn
        self._diffusion_fn = diffusion_fn
        self.state_dim = state_dim
        self.sde_param_dim = sde_param_dim

    def drift(self, x: Tensor, sde_parameters: Tensor) -> Tensor:
        return self._drift_fn(x, sde_parameters)

    def diffusion(self, x: Tensor, sde_parameters: Tensor) -> Tensor:
        return self._diffusion_fn(x, sde_parameters)


def make_sde(
    drift: Callable[[Tensor, Tensor], Tensor],
    diffusion: Callable[[Tensor, Tensor], Tensor],
    state_dim: int,
    sde_param_dim: int,
) -> SDE:
    return FunctionalSDE(
        drift_fn=drift,
        diffusion_fn=diffusion,
        state_dim=state_dim,
        sde_param_dim=sde_param_dim,
    )
