from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.nn.parallel import DistributedDataParallel as DDP

from variational_sde.config import EncoderConfig, HeadConfig, TrainingConfig
from variational_sde.core.observations import Observations
from variational_sde.inference.exponential_moving_average import (
    ExponentialMovingAverage,
)
from variational_sde.models.variational_sde_posterior import VariationalSDEPosterior

if TYPE_CHECKING:
    from variational_sde.accelerate import Accelerator


@dataclass
class TrainingContext:
    model: VariationalSDEPosterior
    model_ddp: DDP | None
    optimizer: torch.optim.AdamW
    scaler: GradScaler
    ema: ExponentialMovingAverage
    observations: Observations
    x0_buffer: torch.Tensor
    device: torch.device
    is_distributed: bool
    is_main: bool
    local_rank: int

    def trainable_model(self) -> VariationalSDEPosterior | DDP:
        return self.model_ddp if self.model_ddp is not None else self.model

    def unwrap_model(self) -> VariationalSDEPosterior:
        model = self.trainable_model()
        return model if isinstance(model, VariationalSDEPosterior) else model.module

    @classmethod
    def create(
        cls,
        observations: Observations,
        state_dim: int,
        sde_param_dim: int,
        config: TrainingConfig,
        encoder_config: EncoderConfig,
        head_config: HeadConfig,
        sde_param_positive_dims: list[int],
        device: torch.device | str,
        mixed_precision: bool,
        accelerator: Accelerator | None,
        sde_param_init_mean: torch.Tensor | None = None,
    ) -> TrainingContext:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_distributed = world_size > 1
        is_main = local_rank == 0

        if is_distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            resolved_device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(resolved_device)
        else:
            resolved_device = (
                torch.device(device) if isinstance(device, str) else device
            )

        model = VariationalSDEPosterior(
            observation_dim=observations.values.shape[-1],
            state_dim=state_dim,
            sde_param_dim=sde_param_dim,
            encoder_config=encoder_config,
            head_config=head_config,
            sde_param_positive_dims=sde_param_positive_dims,
            sde_param_init_mean=sde_param_init_mean,
        ).to(resolved_device)

        if accelerator is not None:
            model.encoder.sit = accelerator.optimize(model.encoder.sit)

        ema = ExponentialMovingAverage(model)

        model_ddp: DDP | None = None
        if is_distributed:
            model_ddp = DDP(model, device_ids=[local_rank])

        sde_param_params = list(model.sde_parameter_posterior.parameters())
        sde_param_ids = {id(p) for p in sde_param_params}
        other_params = [p for p in model.parameters() if id(p) not in sde_param_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": config.learning_rate},
                {"params": sde_param_params, "lr": config.sde_param_lr},
            ]
        )

        use_mixed_precision = mixed_precision and torch.cuda.is_available()
        scaler = GradScaler("cuda", enabled=use_mixed_precision)

        device_observations = Observations(
            times=observations.times.to(resolved_device),
            values=observations.values.to(resolved_device),
        )

        x0_buffer = (
            device_observations.values[0]
            .unsqueeze(0)
            .expand(config.batch_size, -1)
            .contiguous()
        )

        return cls(
            model=model,
            model_ddp=model_ddp,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            observations=device_observations,
            x0_buffer=x0_buffer,
            device=resolved_device,
            is_distributed=is_distributed,
            is_main=is_main,
            local_rank=local_rank,
        )

    def cleanup(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
