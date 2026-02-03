from __future__ import annotations

from torch import Tensor, nn

from variational_sde.config import EncoderConfig, HeadConfig
from variational_sde.models.encoder import ObservationContextEncoder
from variational_sde.models.head import DiffusionTransitionHead
from variational_sde.models.sde_parameter_posterior import SDEParameterPosterior


class VariationalSDEPosterior(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        sde_param_dim: int,
        encoder_config: EncoderConfig,
        head_config: HeadConfig,
        sde_param_positive_dims: list[int],
        sde_param_init_mean: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ObservationContextEncoder(
            observation_dim=observation_dim,
            sde_param_dim=sde_param_dim,
            config=encoder_config,
        )
        self.head = DiffusionTransitionHead(
            state_dim=state_dim,
            context_dim=encoder_config.hidden_dim,
            sde_param_dim=sde_param_dim,
            config=head_config,
        )
        self.sde_parameter_posterior = SDEParameterPosterior(
            sde_param_dim, sde_param_positive_dims, init_mean=sde_param_init_mean
        )
