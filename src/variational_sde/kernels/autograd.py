from __future__ import annotations

from typing import Protocol, cast

import torch
from torch import Tensor

from variational_sde.kernels.backward import launch_bwd
from variational_sde.kernels.forward import launch_fwd
from variational_sde.kernels.weights import SDEWeights, SavedActivations


class _SDECtx(Protocol):
    saved_tensors: tuple[Tensor, ...]
    saved_activations: SavedActivations
    time_step: float
    x0_dtype: torch.dtype
    context_dtype: torch.dtype
    sde_param_dtype: torch.dtype
    hidden_dim: int
    context_dim: int
    sde_param_dim: int
    state_dim: int
    num_layers: int


def _ctx(ctx: torch.autograd.function.FunctionCtx) -> _SDECtx:
    return cast(_SDECtx, ctx)


def _to_dtype(tensor: Tensor, dtype: torch.dtype) -> Tensor:
    return tensor.to(dtype) if tensor.dtype != dtype else tensor


class _SDEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x0: Tensor,
        context: Tensor,
        sde_parameters: Tensor,
        standard_noise: Tensor,
        time_step: float,
        hidden_dim: int,
        context_dim: int,
        sde_param_dim: int,
        state_dim: int,
        num_layers: int,
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        x0_dtype = x0.dtype

        weights = SDEWeights.from_tensors(
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
        diffusion_paths, transition_means, transition_cholesky, saved = launch_fwd(
            x0.float(),
            context.float(),
            sde_parameters.float(),
            standard_noise.float(),
            weights,
            time_step,
            save_activations=True,
        )
        assert saved is not None

        ctx.save_for_backward(
            context.detach(),
            sde_parameters.detach(),
            standard_noise.detach(),
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
        )
        _ctx(ctx).saved_activations = saved
        _ctx(ctx).time_step = time_step
        _ctx(ctx).x0_dtype = x0_dtype
        _ctx(ctx).context_dtype = context.dtype
        _ctx(ctx).sde_param_dtype = sde_parameters.dtype
        _ctx(ctx).hidden_dim = hidden_dim
        _ctx(ctx).context_dim = context_dim
        _ctx(ctx).sde_param_dim = sde_param_dim
        _ctx(ctx).state_dim = state_dim
        _ctx(ctx).num_layers = num_layers

        if x0_dtype != torch.float32:
            return (
                diffusion_paths.to(x0_dtype),
                transition_means.to(x0_dtype),
                transition_cholesky.to(x0_dtype),
            )
        return diffusion_paths, transition_means, transition_cholesky

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_diffusion_paths: Tensor,
        grad_transition_means: Tensor,
        grad_transition_cholesky: Tensor,
    ) -> tuple[
        Tensor,  # grad_x0
        Tensor,  # grad_context
        Tensor,  # grad_sde_parameters
        None,  # eps
        None,  # time_step
        None,  # hidden_dim
        None,  # context_dim
        None,  # sde_param_dim
        None,  # state_dim
        None,  # num_layers
        Tensor,  # grad_W_ih_l0
        Tensor,  # grad_W_hh_l0
        Tensor,  # grad_b_ih_l0
        Tensor,  # grad_b_hh_l0
        Tensor,  # grad_W_ih_stack
        Tensor,  # grad_W_hh_stack
        Tensor,  # grad_b_ih_stack
        Tensor,  # grad_b_hh_stack
        Tensor,  # grad_out_weight
        Tensor,  # grad_out_bias
    ]:
        (
            context,
            sde_parameters,
            standard_noise,
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
        ) = _ctx(ctx).saved_tensors

        weights = SDEWeights.from_tensors(
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
            _ctx(ctx).hidden_dim,
            _ctx(ctx).context_dim,
            _ctx(ctx).sde_param_dim,
            _ctx(ctx).state_dim,
            _ctx(ctx).num_layers,
        )

        grad_diffusion_paths_f32 = _to_dtype(grad_diffusion_paths, torch.float32)
        grad_transition_means_f32 = _to_dtype(grad_transition_means, torch.float32)
        grad_transition_cholesky_f32 = _to_dtype(
            grad_transition_cholesky, torch.float32
        )

        grads = launch_bwd(
            grad_diffusion_paths_f32,
            grad_transition_means_f32,
            grad_transition_cholesky_f32,
            context,
            sde_parameters,
            standard_noise,
            _ctx(ctx).saved_activations,
            weights,
            _ctx(ctx).time_step,
        )

        grad_x0 = _to_dtype(grads[0], _ctx(ctx).x0_dtype)
        grad_context = _to_dtype(grads[1], _ctx(ctx).context_dtype)
        grad_sde_parameters = _to_dtype(grads[2], _ctx(ctx).sde_param_dtype)

        grad_W_ih_l0 = _to_dtype(grads[3], W_ih_l0.dtype)
        grad_W_hh_l0 = _to_dtype(grads[4], W_hh_l0.dtype)
        grad_b_ih_l0 = _to_dtype(grads[5], b_ih_l0.dtype)
        grad_b_hh_l0 = _to_dtype(grads[6], b_hh_l0.dtype)
        grad_W_ih_stack = _to_dtype(grads[7], W_ih_stack.dtype)
        grad_W_hh_stack = _to_dtype(grads[8], W_hh_stack.dtype)
        grad_b_ih_stack = _to_dtype(grads[9], b_ih_stack.dtype)
        grad_b_hh_stack = _to_dtype(grads[10], b_hh_stack.dtype)
        grad_out_weight = _to_dtype(grads[11], out_weight.dtype)
        grad_out_bias = _to_dtype(grads[12], out_bias.dtype)

        return (
            grad_x0,
            grad_context,
            grad_sde_parameters,
            None,  # eps
            None,  # time_step
            None,  # hidden_dim
            None,  # context_dim
            None,  # sde_param_dim
            None,  # state_dim
            None,  # num_layers
            grad_W_ih_l0,
            grad_W_hh_l0,
            grad_b_ih_l0,
            grad_b_hh_l0,
            grad_W_ih_stack,
            grad_W_hh_stack,
            grad_b_ih_stack,
            grad_b_hh_stack,
            grad_out_weight,
            grad_out_bias,
        )


def sample_diffusion_paths(
    x0: Tensor,
    context: Tensor,
    sde_parameters: Tensor,
    standard_noise: Tensor,
    weights: SDEWeights,
    time_step: float,
) -> tuple[Tensor, Tensor, Tensor]:
    input_dtype = x0.dtype
    diffusion_paths, transition_means, transition_cholesky, _ = launch_fwd(
        x0.float(),
        context.float(),
        sde_parameters.float(),
        standard_noise.float(),
        weights,
        time_step,
        save_activations=False,
    )
    if input_dtype != torch.float32:
        return (
            diffusion_paths.to(input_dtype),
            transition_means.to(input_dtype),
            transition_cholesky.to(input_dtype),
        )
    return diffusion_paths, transition_means, transition_cholesky
