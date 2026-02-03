from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Iterator

import torch
from torch import Tensor, nn

from variational_sde.inference.constants import DEFAULT_EMA_DECAY


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = DEFAULT_EMA_DECAY) -> None:
        self.model = model
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        self._init_shadow()

    def _init_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            self.shadow[name].lerp_(param.detach(), 1 - self.decay)

    @contextmanager
    def apply(self) -> Iterator[None]:
        backup: dict[str, Tensor] = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(backup[name])

    def state_dict(self) -> dict[str, Tensor]:
        return deepcopy(self.shadow)

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.shadow = deepcopy(state)
