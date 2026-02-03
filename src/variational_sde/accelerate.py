from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, TypeVar

import torch
import torch._inductor.config as inductor_config
from torch import nn

T = TypeVar("T", bound=nn.Module)


class CompileMode(Enum):
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"


@contextmanager
def suppress_torch_compile_output() -> Iterator[None]:
    old_verbose = inductor_config.verbose_progress
    inductor_config.verbose_progress = False

    loggers = [
        logging.getLogger("torch._inductor"),
        logging.getLogger("torch._dynamo"),
        logging.getLogger("torch.fx"),
    ]
    old_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="torch")
        try:
            yield
        finally:
            inductor_config.verbose_progress = old_verbose
            for logger, level in zip(loggers, old_levels):
                logger.setLevel(level)


@dataclass(frozen=True)
class Accelerator:
    compile: bool = False
    compile_mode: CompileMode = CompileMode.REDUCE_OVERHEAD
    compile_fullgraph: bool = False
    compile_dynamic: bool | None = None

    def optimize(self, module: T) -> T:
        if self.compile:
            return torch.compile(  # type: ignore[return-value]
                module,
                mode=self.compile_mode.value,
                fullgraph=self.compile_fullgraph,
                dynamic=self.compile_dynamic,
            )
        return module
