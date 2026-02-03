from __future__ import annotations

import torch
from torch import Tensor

from variational_sde.accelerate import Accelerator, CompileMode
from variational_sde.config import EncoderConfig, HeadConfig, PretrainConfig, TrainingConfig
from variational_sde.console import Console
from variational_sde.core.observations import (
    GaussianObservationLikelihood,
    Observations,
)
from variational_sde.core.priors import Prior, PriorType
from variational_sde.core.sde import SDE
from variational_sde.infer import InferenceConfig, infer


class LotkaVolterra(SDE):
    state_dim = 2
    sde_param_dim = 3

    def drift(self, x: Tensor, sde_parameters: Tensor) -> Tensor:
        u, v = x[..., 0], x[..., 1]
        t1 = sde_parameters[..., 0]
        t2 = sde_parameters[..., 1]
        t3 = sde_parameters[..., 2]
        du = t1 * u - t2 * u * v
        dv = t2 * u * v - t3 * v
        return torch.stack([du, dv], dim=-1)

    def diffusion(self, x: Tensor, sde_parameters: Tensor) -> Tensor:
        u, v = x[..., 0], x[..., 1]
        t1 = sde_parameters[..., 0]
        t2 = sde_parameters[..., 1]
        t3 = sde_parameters[..., 2]
        uv = u * v
        b11 = t1 * u + t2 * uv
        b12 = -t2 * uv
        b22 = t3 * v + t2 * uv
        L00 = torch.sqrt(b11.clamp(min=1e-6))
        L10 = b12 / L00.clamp(min=1e-6)
        L11 = torch.sqrt((b22 - L10**2).clamp(min=1e-6))
        zeros = torch.zeros_like(L00)
        row0 = torch.stack([L00, zeros], dim=-1)
        row1 = torch.stack([L10, L11], dim=-1)
        return torch.stack([row0, row1], dim=-2)


def main() -> None:
    console = Console()
    param_names = ["θ₁", "θ₂", "θ₃"]

    observations = Observations(
        times=torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0]),
        values=torch.tensor(
            [
                [71.0, 79.0],
                [47.61225908, 447.20971405],
                [80.53119269, 50.26254069],
                [23.10087379, 339.40432691],
                [158.05238324, 66.79611979],
            ]
        ),
    )

    prior = Prior(type=PriorType.LOG_NORMAL, mean=0.0, std=1.5, dim=3)

    posterior = infer(
        sde=LotkaVolterra(),
        observations=observations,
        observation_likelihood=GaussianObservationLikelihood(variance=1.0),
        prior=prior,
        time_horizon=40.0,
        config=InferenceConfig(
            training=TrainingConfig(
                time_step=0.1,
                batch_size=24,
                n_iterations=30000,
                learning_rate=1e-4,
                sde_param_lr=1e-3,
                grad_clip_norm=1.0,
            ),
            encoder=EncoderConfig(
                hidden_dim=256,
                num_heads=4,
                depth=8,
            ),
            head=HeadConfig(
                hidden_dim=64,
                num_layers=2,
            ),
            state_positive_dims=[0, 1],
            sde_param_positive_dims=[0, 1, 2],
            console=console,
            param_names=param_names,
            accelerator=Accelerator(compile=True, compile_mode=CompileMode.MAX_AUTOTUNE),
            pretrain=PretrainConfig(),
        ),
    )

    summary = posterior.summary(n_samples=500)
    diag = posterior.diagnostics()
    console.summary_table(summary, diag, param_names=["θ₁", "θ₂", "θ₃"])

    posterior.plot(n_trajectories=30, show=True)
    posterior.save("lotka_volterra_posterior.pt")


if __name__ == "__main__":
    main()
