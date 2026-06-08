from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    backend: str
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def is_distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_distributed_context() -> DistributedContext:
    enabled = is_distributed_available()
    rank = dist.get_rank() if enabled else int(os.getenv("RANK", "0"))
    world_size = dist.get_world_size() if enabled else int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return DistributedContext(
        enabled=enabled,
        backend=dist.get_backend() if enabled else "none",
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def setup_distributed(
    backend: Optional[str] = None,
    init_method: str = "env://",
) -> DistributedContext:
    """
    Initialize torch.distributed from torchrun environment variables.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = backend or "nccl"
    else:
        device = torch.device("cpu")
        backend = backend or "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)

    ctx = get_distributed_context()
    return DistributedContext(
        enabled=world_size > 1,
        backend=backend,
        rank=ctx.rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return get_distributed_context().rank == 0


def barrier() -> None:
    if is_distributed_available():
        dist.barrier()
