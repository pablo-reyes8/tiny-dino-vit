from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.paralel.distributed import DistributedContext


def wrap_data_parallel(
    model: nn.Module,
    ctx: DistributedContext,
    *,
    sync_batchnorm: bool = False,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """
    Wrap a model for DDP when distributed is enabled, otherwise DataParallel if possible.
    """
    if ctx.enabled:
        model = model.to(ctx.device)
        if sync_batchnorm and ctx.device.type == "cuda":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            output_device=ctx.local_rank if ctx.device.type == "cuda" else None,
            find_unused_parameters=find_unused_parameters,
        )

    if torch.cuda.device_count() > 1:
        return nn.DataParallel(model).to(ctx.device)

    return model.to(ctx.device)


def build_distributed_loader(
    dataset,
    *,
    ctx: DistributedContext,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    persistent_workers: Optional[bool] = None,
    worker_init_fn=None,
) -> DataLoader:
    """
    Build a DataLoader with DistributedSampler when running under torchrun.
    """
    sampler = None
    if ctx.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False

    if persistent_workers is None:
        persistent_workers = bool(num_workers > 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
    )
