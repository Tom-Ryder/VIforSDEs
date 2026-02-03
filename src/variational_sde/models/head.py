from __future__ import annotations

from typing import cast

import torch
from einops import rearrange
from torch import Tensor, nn

from variational_sde.config import HeadConfig
from variational_sde.inference.constants import DIAG_MIN
from variational_sde.kernels.autograd import (
    _SDEFunction,
    sample_diffusion_paths as kernel_sample_diffusion_paths,
)
from variational_sde.kernels.constants import MAX_LAYERS
from variational_sde.kernels.weights import SDEWeights
from variational_sde.primitives.bounds import lower_bound


class DiffusionTransitionHead(nn.Module):
    _tril_rows: Tensor
    _tril_cols: Tensor
    _diag_mask: Tensor

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        sde_param_dim: int,
        config: HeadConfig,
    ) -> None:
        super().__init__()
        if config.num_layers < 1 or config.num_layers > MAX_LAYERS:
            raise ValueError(
                f"num_layers must be in [1, {MAX_LAYERS}], got {config.num_layers}"
            )
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.sde_param_dim = sde_param_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.n_tril = state_dim * (state_dim + 1) // 2

        rows, cols = torch.tril_indices(state_dim, state_dim)
        self.register_buffer("_tril_rows", rows)
        self.register_buffer("_tril_cols", cols)
        self.register_buffer("_diag_mask", rows == cols)

        input_dim = state_dim + context_dim + sde_param_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
        )

        self.out_proj = nn.Linear(config.hidden_dim, state_dim + self.n_tril)
        self._init_out_proj()

    def _init_out_proj(self) -> None:
        with torch.no_grad():
            self.out_proj.weight.zero_()
            self.out_proj.bias.zero_()
            for k in range(self.state_dim):
                diag_idx = self.state_dim + k * (k + 3) // 2
                self.out_proj.bias[diag_idx] = 1.0

    def forward(
        self,
        x_t: Tensor,
        context_t: Tensor,
        sde_parameters: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        inp = torch.cat([x_t, context_t, sde_parameters], dim=-1)
        inp = rearrange(inp, "b d -> b 1 d")

        out, hidden = self.gru(inp, hidden)
        out = rearrange(out, "b 1 d -> b d")

        params = self.out_proj(out)
        mu = params[..., : self.state_dim]
        tril_params = params[..., self.state_dim :]

        L = self._tril_from_params(tril_params)
        return mu, L, hidden

    def _tril_from_params(self, params: Tensor) -> Tensor:
        batch = params.shape[0]
        d = self.state_dim
        off_diag_mask = ~self._diag_mask
        off_diag = params[:, off_diag_mask]
        diag = lower_bound(params[:, self._diag_mask], DIAG_MIN)
        L = torch.zeros(batch, d, d, device=params.device, dtype=params.dtype)
        L[:, self._tril_rows[off_diag_mask], self._tril_cols[off_diag_mask]] = off_diag
        L[:, self._tril_rows[self._diag_mask], self._tril_cols[self._diag_mask]] = diag
        return L

    def init_hidden(
        self, batch: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tensor:
        return torch.zeros(
            self.num_layers, batch, self.hidden_dim, device=device, dtype=dtype
        )

    def _extract_gru_weights(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # GRU attributes typed as Tensor | None but guaranteed non-None with bias=True (default)
        W_ih_l0: Tensor = self.gru.weight_ih_l0  # type: ignore[assignment]
        W_hh_l0: Tensor = self.gru.weight_hh_l0  # type: ignore[assignment]
        b_ih_l0: Tensor = self.gru.bias_ih_l0  # type: ignore[assignment]
        b_hh_l0: Tensor = self.gru.bias_hh_l0  # type: ignore[assignment]

        if self.num_layers > 1:
            W_ih_stack = torch.stack(
                [
                    getattr(self.gru, f"weight_ih_l{k}")
                    for k in range(1, self.num_layers)
                ]
            )
            W_hh_stack = torch.stack(
                [
                    getattr(self.gru, f"weight_hh_l{k}")
                    for k in range(1, self.num_layers)
                ]
            )
            b_ih_stack = torch.stack(
                [getattr(self.gru, f"bias_ih_l{k}") for k in range(1, self.num_layers)]
            )
            b_hh_stack = torch.stack(
                [getattr(self.gru, f"bias_hh_l{k}") for k in range(1, self.num_layers)]
            )
        else:
            device, dtype = W_ih_l0.device, W_ih_l0.dtype
            W_ih_stack = torch.empty(
                0, 3 * self.hidden_dim, self.hidden_dim, device=device, dtype=dtype
            )
            W_hh_stack = torch.empty(
                0, 3 * self.hidden_dim, self.hidden_dim, device=device, dtype=dtype
            )
            b_ih_stack = torch.empty(0, 3 * self.hidden_dim, device=device, dtype=dtype)
            b_hh_stack = torch.empty(0, 3 * self.hidden_dim, device=device, dtype=dtype)

        return (
            W_ih_l0,
            W_hh_l0,
            b_ih_l0,
            b_hh_l0,
            W_ih_stack,
            W_hh_stack,
            b_ih_stack,
            b_hh_stack,
        )

    def sample_diffusion_paths(
        self,
        x0: Tensor,
        context: Tensor,
        sde_parameters: Tensor,
        standard_noise: Tensor,
        time_step: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.training:
            (
                W_ih_l0,
                W_hh_l0,
                b_ih_l0,
                b_hh_l0,
                W_ih_stack,
                W_hh_stack,
                b_ih_stack,
                b_hh_stack,
            ) = self._extract_gru_weights()
            return cast(
                tuple[Tensor, Tensor, Tensor],
                _SDEFunction.apply(
                    x0,
                    context,
                    sde_parameters,
                    standard_noise,
                    time_step,
                    self.hidden_dim,
                    self.context_dim,
                    self.sde_param_dim,
                    self.state_dim,
                    self.num_layers,
                    W_ih_l0,
                    W_hh_l0,
                    b_ih_l0,
                    b_hh_l0,
                    W_ih_stack,
                    W_hh_stack,
                    b_ih_stack,
                    b_hh_stack,
                    self.out_proj.weight,
                    self.out_proj.bias,
                ),
            )
        weights = SDEWeights.from_modules(
            self.gru,
            self.out_proj,
            self.context_dim,
            self.sde_param_dim,
            self.state_dim,
        )
        return kernel_sample_diffusion_paths(
            x0, context, sde_parameters, standard_noise, weights, time_step
        )
