from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import LogNormal, Normal


class SDEParameterPosterior(nn.Module):
    positive_mask: Tensor

    def __init__(
        self,
        sde_param_dim: int,
        sde_param_positive_dims: list[int],
        init_mean: Tensor | None = None,
        init_std: float = 1.0,
    ) -> None:
        super().__init__()
        if sde_param_dim < 1:
            raise ValueError(f"sde_param_dim must be >= 1, got {sde_param_dim}")
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}")
        if any(d < 0 or d >= sde_param_dim for d in sde_param_positive_dims):
            raise ValueError(f"sde_param_positive_dims must be in [0, {sde_param_dim})")
        self.sde_param_dim = sde_param_dim
        if init_mean is not None:
            self.mean = nn.Parameter(init_mean.clone())
        else:
            self.mean = nn.Parameter(torch.zeros(sde_param_dim))
        self.log_std = nn.Parameter(torch.full((sde_param_dim,), math.log(init_std)))
        mask = torch.zeros(sde_param_dim, dtype=torch.bool)
        mask[sde_param_positive_dims] = True
        self.register_buffer("positive_mask", mask)

    def rsample(self, n: int) -> Tensor:
        std = self.log_std.exp()
        eps = torch.randn(
            n, self.sde_param_dim, device=self.mean.device, dtype=self.mean.dtype
        )
        sde_parameters = self.mean + std * eps
        sde_parameters = torch.where(
            self.positive_mask, sde_parameters.exp(), sde_parameters
        )
        return sde_parameters

    def log_prob(self, sde_parameters: Tensor) -> Tensor:
        std = self.log_std.exp()
        batch = sde_parameters.shape[0]
        result = torch.zeros(batch, device=sde_parameters.device, dtype=sde_parameters.dtype)
        pos = self.positive_mask
        neg = ~pos
        if pos.any():
            pos_dist = LogNormal(self.mean[pos], std[pos])
            result = result + pos_dist.log_prob(sde_parameters[:, pos]).sum(-1)
        if neg.any():
            neg_dist = Normal(self.mean[neg], std[neg])
            result = result + neg_dist.log_prob(sde_parameters[:, neg]).sum(-1)
        return result

    @property
    def expected_value(self) -> Tensor:
        std = self.log_std.exp()
        result = self.mean.clone()
        result[self.positive_mask] = (
            self.mean[self.positive_mask] + 0.5 * std[self.positive_mask] ** 2
        ).exp()
        return result
