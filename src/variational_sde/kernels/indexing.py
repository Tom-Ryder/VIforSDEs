from __future__ import annotations

from dataclasses import dataclass

from variational_sde.kernels.constants import NUM_GATES_INT


@dataclass(frozen=True)
class GRULayout:
    hidden_dim: int

    def gate_offset(self, gate: int) -> int:
        return gate * self.hidden_dim

    def bias_offset(self, gate: int) -> int:
        return gate * self.hidden_dim

    def weight_row_stride(self) -> int:
        return NUM_GATES_INT * self.hidden_dim


@dataclass(frozen=True)
class ActivationLayout:
    batch_size: int
    num_layers: int
    n_steps: int
    hidden_dim: int

    def l0_offset(self, batch: int, step: int) -> int:
        return batch * self.n_steps * self.hidden_dim + step * self.hidden_dim

    def stack_offset(self, batch: int, layer: int, step: int) -> int:
        layer_idx = layer - 1
        return (
            batch * (self.num_layers - 1) * self.n_steps * self.hidden_dim
            + layer_idx * self.n_steps * self.hidden_dim
            + step * self.hidden_dim
        )
