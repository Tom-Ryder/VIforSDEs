from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

if TYPE_CHECKING:
    from variational_sde.core.observations import Observations
    from variational_sde.posterior.variational_posterior import (
        VariationalPosteriorSamples,
    )


def plot_posterior(
    samples: VariationalPosteriorSamples,
    observations: Observations,
    time_horizon: float,
    show: bool = True,
) -> Figure:
    diffusion_paths: NDArray[np.floating] = samples.diffusion_paths.cpu().numpy()
    sde_parameters: NDArray[np.floating] = samples.sde_parameters.cpu().numpy()
    times: NDArray[np.floating] = torch.linspace(
        0, time_horizon, diffusion_paths.shape[1]
    ).numpy()

    n_trajectories = diffusion_paths.shape[0]
    state_dim = diffusion_paths.shape[2]
    param_dim = sde_parameters.shape[1]

    fig, axes = plt.subplots(
        1,
        state_dim + param_dim,
        figsize=(4 * (state_dim + param_dim), 4),
        squeeze=False,
    )
    ax_list: list[Axes] = list(axes[0])

    for d in range(state_dim):
        for i in range(n_trajectories):
            ax_list[d].plot(times, diffusion_paths[i, :, d], alpha=0.3, color="C0")
        ax_list[d].scatter(
            observations.times.cpu().numpy(),
            observations.values[:, d].cpu().numpy(),
            color="red",
            s=50,
            zorder=5,
        )
        ax_list[d].set_xlabel("Time")
        ax_list[d].set_ylabel(f"State {d}")

    for p in range(param_dim):
        ax_list[state_dim + p].hist(
            sde_parameters[:, p], bins=30, density=True, alpha=0.7, color="C1"
        )
        ax_list[state_dim + p].set_xlabel(f"param_{p}")
        ax_list[state_dim + p].set_ylabel("Density")

    plt.tight_layout()
    if show:
        plt.show()

    return fig
