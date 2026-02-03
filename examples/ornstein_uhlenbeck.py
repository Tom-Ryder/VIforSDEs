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


class OrnsteinUhlenbeck(SDE):
    state_dim = 1
    sde_param_dim = 3

    def drift(self, x: Tensor, sde_parameters: Tensor) -> Tensor:
        kappa = sde_parameters[..., 0:1]
        mu = sde_parameters[..., 1:2]
        return kappa * (mu - x)

    def diffusion(self, x: Tensor, sde_parameters: Tensor) -> Tensor:
        sigma = sde_parameters[..., 2:3]
        batch = x.shape[0]
        return sigma.view(batch, 1, 1)


def main() -> None:
    console = Console()
    param_names = ["κ", "μ", "σ"]

    observations = Observations(
        times=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        values=torch.tensor(
            [
                [2.0],
                [1.5],
                [0.8],
                [1.2],
                [0.9],
                [1.1],
            ]
        ),
    )

    prior = Prior(type=PriorType.NORMAL, mean=0.0, std=1.0, dim=3)

    posterior = infer(
        sde=OrnsteinUhlenbeck(),
        observations=observations,
        observation_likelihood=GaussianObservationLikelihood(variance=0.1),
        prior=prior,
        time_horizon=5.0,
        config=InferenceConfig(
            training=TrainingConfig(
                time_step=0.05,
                batch_size=128,
                n_iterations=20000,
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
            sde_param_positive_dims=[0, 2],
            console=console,
            param_names=param_names,
            accelerator=Accelerator(compile=True, compile_mode=CompileMode.MAX_AUTOTUNE),
            pretrain=PretrainConfig(),
        ),
    )

    summary = posterior.summary(n_samples=500)
    diag = posterior.diagnostics()
    console.summary_table(summary, diag, param_names=["κ", "μ", "σ"])

    posterior.plot(n_trajectories=30, show=True)
    posterior.save("ou_posterior.pt")


if __name__ == "__main__":
    main()
