from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Self

import torch
import yaml
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class YamlConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls(**data)


class AmpDtype(Enum):
    FLOAT16 = torch.float16
    BFLOAT16 = torch.bfloat16


class TrainingConfig(YamlConfig):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    time_step: float = 0.1
    batch_size: int = 50
    n_iterations: int = 25000
    learning_rate: float = 1e-4
    sde_param_lr: float = 1e-3
    grad_clip_norm: float = 1.0
    amp_dtype: AmpDtype = AmpDtype.BFLOAT16

    @field_validator("time_step", "learning_rate", "sde_param_lr", "grad_clip_norm")
    @classmethod
    def validate_positive_floats(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("value must be positive")
        return v

    @field_validator("batch_size", "n_iterations")
    @classmethod
    def validate_positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be positive")
        return v


class EncoderConfig(YamlConfig):
    hidden_dim: int = 128
    cond_dim: int = 128
    num_heads: int = 4
    depth: int = 4
    mlp_ratio: float = 8 / 3

    @field_validator("hidden_dim", "cond_dim", "num_heads", "depth")
    @classmethod
    def validate_positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be positive")
        return v

    @field_validator("mlp_ratio")
    @classmethod
    def validate_positive_ratio(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("mlp_ratio must be positive")
        return v

    @field_validator("hidden_dim")
    @classmethod
    def validate_head_divisible(cls, v: int, info: ValidationInfo) -> int:
        num_heads = info.data.get("num_heads")
        if isinstance(num_heads, int) and num_heads > 0 and v % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        return v


class HeadConfig(YamlConfig):
    hidden_dim: int = 64
    num_layers: int = 2

    @field_validator("hidden_dim", "num_layers")
    @classmethod
    def validate_positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be positive")
        return v


class PretrainConfig(YamlConfig):
    n_iterations: int = 1000
    batch_size: int = 4096
    learning_rate: float = 0.02
    init_scale: float = 2.0

    @field_validator("n_iterations", "batch_size")
    @classmethod
    def validate_positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be positive")
        return v

    @field_validator("learning_rate", "init_scale")
    @classmethod
    def validate_positive_floats(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("value must be positive")
        return v
