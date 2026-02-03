from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from variational_sde.kernels.constants import MAX_LAYERS


class SavedActivations(NamedTuple):
    diffusion_paths: Tensor
    transition_cholesky_raw: Tensor
    h_l0: Tensor
    r_l0: Tensor
    z_l0: Tensor
    n_l0: Tensor
    n_hh_l0: Tensor
    h_stack: Tensor
    r_stack: Tensor
    z_stack: Tensor
    n_stack: Tensor
    n_hh_stack: Tensor


class SDEWeights:
    __slots__ = (
        "W_ih_l0",
        "W_hh_l0",
        "b_ih_l0",
        "b_hh_l0",
        "W_ih_stack",
        "W_hh_stack",
        "b_ih_stack",
        "b_hh_stack",
        "out_weight",
        "out_bias",
        "hidden_dim",
        "context_dim",
        "sde_param_dim",
        "state_dim",
        "num_layers",
    )

    def __init__(
        self,
        W_ih_l0: Tensor,
        W_hh_l0: Tensor,
        b_ih_l0: Tensor,
        b_hh_l0: Tensor,
        W_ih_stack: Tensor,
        W_hh_stack: Tensor,
        b_ih_stack: Tensor,
        b_hh_stack: Tensor,
        out_weight: Tensor,
        out_bias: Tensor,
        hidden_dim: int,
        context_dim: int,
        sde_param_dim: int,
        state_dim: int,
        num_layers: int,
    ) -> None:
        self.W_ih_l0 = W_ih_l0
        self.W_hh_l0 = W_hh_l0
        self.b_ih_l0 = b_ih_l0
        self.b_hh_l0 = b_hh_l0
        self.W_ih_stack = W_ih_stack
        self.W_hh_stack = W_hh_stack
        self.b_ih_stack = b_ih_stack
        self.b_hh_stack = b_hh_stack
        self.out_weight = out_weight
        self.out_bias = out_bias
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.sde_param_dim = sde_param_dim
        self.state_dim = state_dim
        self.num_layers = num_layers

    @classmethod
    def from_modules(
        cls,
        gru: nn.GRU,
        out_proj: nn.Linear,
        context_dim: int,
        sde_param_dim: int,
        state_dim: int,
    ) -> SDEWeights:
        num_layers = gru.num_layers
        if num_layers > MAX_LAYERS:
            raise ValueError(f"num_layers must be <= {MAX_LAYERS}, got {num_layers}")

        hidden_dim = gru.hidden_size
        # GRU attributes typed as Tensor | None but guaranteed non-None with bias=True (default)
        weight_ih_l0: Tensor = gru.weight_ih_l0  # type: ignore[assignment]
        weight_hh_l0: Tensor = gru.weight_hh_l0  # type: ignore[assignment]
        bias_ih_l0: Tensor = gru.bias_ih_l0  # type: ignore[assignment]
        bias_hh_l0: Tensor = gru.bias_hh_l0  # type: ignore[assignment]
        device = weight_ih_l0.device
        dtype = weight_ih_l0.dtype

        W_ih_l0: Tensor = weight_ih_l0.detach().T.contiguous()
        W_hh_l0: Tensor = weight_hh_l0.detach().T.contiguous()
        b_ih_l0: Tensor = bias_ih_l0.detach().contiguous()
        b_hh_l0: Tensor = bias_hh_l0.detach().contiguous()

        if num_layers > 1:
            W_ih_list: list[Tensor] = []
            W_hh_list: list[Tensor] = []
            b_ih_list: list[Tensor] = []
            b_hh_list: list[Tensor] = []
            for k in range(1, num_layers):
                w_ih: Tensor = getattr(gru, f"weight_ih_l{k}")
                w_hh: Tensor = getattr(gru, f"weight_hh_l{k}")
                b_ih: Tensor = getattr(gru, f"bias_ih_l{k}")
                b_hh: Tensor = getattr(gru, f"bias_hh_l{k}")
                W_ih_list.append(w_ih.detach().T.contiguous())
                W_hh_list.append(w_hh.detach().T.contiguous())
                b_ih_list.append(b_ih.detach().contiguous())
                b_hh_list.append(b_hh.detach().contiguous())
            W_ih_stack = torch.stack(W_ih_list, dim=0)
            W_hh_stack = torch.stack(W_hh_list, dim=0)
            b_ih_stack = torch.stack(b_ih_list, dim=0)
            b_hh_stack = torch.stack(b_hh_list, dim=0)
        else:
            W_ih_stack = torch.empty(
                0, hidden_dim, 3 * hidden_dim, device=device, dtype=dtype
            )
            W_hh_stack = torch.empty(
                0, hidden_dim, 3 * hidden_dim, device=device, dtype=dtype
            )
            b_ih_stack = torch.empty(0, 3 * hidden_dim, device=device, dtype=dtype)
            b_hh_stack = torch.empty(0, 3 * hidden_dim, device=device, dtype=dtype)

        out_weight: Tensor = out_proj.weight.detach().contiguous()
        bias = out_proj.bias
        assert bias is not None
        out_bias: Tensor = bias.detach().contiguous()

        return cls(
            W_ih_l0,
            W_hh_l0,
            b_ih_l0,
            b_hh_l0,
            W_ih_stack,
            W_hh_stack,
            b_ih_stack,
            b_hh_stack,
            out_weight,
            out_bias,
            hidden_dim,
            context_dim,
            sde_param_dim,
            state_dim,
            num_layers,
        )

    @classmethod
    def from_tensors(
        cls,
        W_ih_l0: Tensor,
        W_hh_l0: Tensor,
        b_ih_l0: Tensor,
        b_hh_l0: Tensor,
        W_ih_stack: Tensor,
        W_hh_stack: Tensor,
        b_ih_stack: Tensor,
        b_hh_stack: Tensor,
        out_weight: Tensor,
        out_bias: Tensor,
        hidden_dim: int,
        context_dim: int,
        sde_param_dim: int,
        state_dim: int,
        num_layers: int,
    ) -> SDEWeights:
        return cls(
            W_ih_l0.T.contiguous(),
            W_hh_l0.T.contiguous(),
            b_ih_l0.contiguous(),
            b_hh_l0.contiguous(),
            W_ih_stack.transpose(-2, -1).contiguous()
            if W_ih_stack.numel() > 0
            else W_ih_stack,
            W_hh_stack.transpose(-2, -1).contiguous()
            if W_hh_stack.numel() > 0
            else W_hh_stack,
            b_ih_stack.contiguous(),
            b_hh_stack.contiguous(),
            out_weight.contiguous(),
            out_bias.contiguous(),
            hidden_dim,
            context_dim,
            sde_param_dim,
            state_dim,
            num_layers,
        )
