# Variational Inference for Stochastic Differential Equations

PyTorch implementation of [Black-box Variational Inference for Stochastic Differential Equations](https://arxiv.org/abs/1802.03335) (ICML 2018).

The method performs Bayesian inference for SDEs by jointly learning the posterior over latent diffusion paths and SDE parameters. A neural network approximates the posterior of conditioned diffusion paths, learning Gaussian state transitions that bridge between observations—automating the bridge constructs that MCMC methods require manual tuning for.

## Installation

```bash
uv sync
```

Requires Python 3.11+, PyTorch 2.5+, Triton 3.2+ (Linux, for GPU kernels).

## Usage

Define an SDE, provide observations, and run inference:

```python
import torch
from torch import Tensor

from variational_sde.core.observations import GaussianObservationLikelihood, Observations
from variational_sde.core.priors import Prior, PriorType
from variational_sde.core.sde import SDE
from variational_sde.infer import InferenceConfig, infer

class OrnsteinUhlenbeck(SDE):
    state_dim = 1
    sde_param_dim = 3  # κ (mean reversion), μ (long-term mean), σ (volatility)

    def drift(self, x: Tensor, sde_parameters: Tensor) -> Tensor:
        kappa, mu = sde_parameters[..., 0:1], sde_parameters[..., 1:2]
        return kappa * (mu - x)

    def diffusion(self, x: Tensor, sde_parameters: Tensor) -> Tensor:
        sigma = sde_parameters[..., 2:3]
        return sigma.view(-1, 1, 1)

observations = Observations(
    times=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    values=torch.tensor([[2.0], [1.5], [0.8], [1.2], [0.9], [1.1]]),
)

posterior = infer(
    sde=OrnsteinUhlenbeck(),
    observations=observations,
    observation_likelihood=GaussianObservationLikelihood(variance=0.1),
    prior=Prior(type=PriorType.NORMAL, mean=0.0, std=1.0, dim=3),
    time_horizon=5.0,
)

summary = posterior.summary(n_samples=500)
posterior.plot(n_trajectories=30, show=True)
posterior.save("ou_posterior.pt")
```

See `examples/` for complete working examples including Ornstein-Uhlenbeck and Lotka-Volterra systems.

## Architecture

```
src/variational_sde/
├── core/           # SDE protocol, observations, priors, Euler-Maruyama solver
├── models/         # VariationalSDEPosterior, SiT encoder, GRU head
├── inference/      # Trainer, ELBO computation, path sampling, state space transforms
├── kernels/        # Fused Triton forward/backward GRU kernels
├── primitives/     # Transformer blocks, attention, rotary embeddings, SwiGLU
└── posterior/      # Posterior sampling, summaries, checkpointing
```

**Model components:**

| Component | Description |
|-----------|-------------|
| `SDEParameterPosterior` | Learnable diagonal Gaussian over θ with log-normal support for positive parameters |
| `ObservationContextEncoder` | SiT transformer with RoPE that encodes observations + θ into context for path generation |
| `DiffusionTransitionHead` | Stacked GRU outputting per-step Gaussian transitions (mean + Cholesky factor) |

**ELBO decomposition:**

```
ELBO = E_q[log p(y|x)] + E_q[log p(x|θ)] - E_q[log q(x|y,θ)] + log p(θ) - log q(θ) + jacobian
         └─────────────┘   └────────────┘   └───────────────┘   └────────────────────┘
          observation       SDE dynamics      variational         parameter KL
          likelihood        (true model)      path density        divergence
```

## Implementation Details

**Triton kernels.** The GRU forward and backward passes are fused into custom Triton kernels (`src/variational_sde/kernels/`). This eliminates Python overhead and memory bandwidth bottlenecks for the sequential path sampling, which dominates training time.

**State space transforms.** For SDEs with positive state dimensions (e.g., population models), a softplus transform maps between unconstrained latent space z and constrained state space x. The log-Jacobian correction is included in the ELBO.

**Mixed precision.** Training uses BFloat16/Float16 with gradient scaling. The Triton kernels operate in FP32 for numerical stability in the recurrent computation.

**Distributed training.** Supports PyTorch DDP for multi-GPU training. EMA weights are synchronized across ranks.

## Configuration

```python
from variational_sde.config import TrainingConfig, EncoderConfig, HeadConfig

config = InferenceConfig(
    training=TrainingConfig(
        time_step=0.05,          # Euler-Maruyama discretization
        batch_size=128,
        n_iterations=20000,
        learning_rate=1e-4,      # encoder/head learning rate
        sde_param_lr=1e-3,       # parameter posterior learning rate
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
    sde_param_positive_dims=[0, 2],  # κ, σ > 0
)
```

## Development

```bash
uv run pytest tests/              # run tests
uv run mypy src/                  # type check
uv run python examples/ornstein_uhlenbeck.py
```

## Reference

```bibtex
@inproceedings{ryder2018black,
  title={Black-box Variational Inference for Stochastic Differential Equations},
  author={Ryder, Tom and Golightly, Andrew and McGough, A. Stephen and Prangle, Dennis},
  booktitle={International Conference on Machine Learning},
  pages={4423--4432},
  year={2018},
  organization={PMLR}
}
```

## License

MIT
