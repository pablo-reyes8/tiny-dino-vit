"""Configure mixed-precision autocast and gradient-scaling utilities."""

from __future__ import annotations

import inspect
import random
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional, Union, Tuple, List
import numpy as np
import torch

# ============================================================
# Dtype mapping
# ============================================================

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "none": torch.float32,}


# ============================================================
# Device helpers
# ============================================================

def resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """
    Resolve device string into a torch.device.

    Supports:
        "auto", "cuda", "cpu", "mps"
    """
    if isinstance(device, torch.device):
        requested = device

    elif device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    else:
        requested = torch.device(device)

    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but CUDA is not available.")

    if requested.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but MPS is not available.")

    return requested


def normalize_device_type(device: Union[str, torch.device]) -> str:
    """
    Returns device type as string: 'cuda', 'cpu', 'mps', etc.
    """
    return torch.device(device).type


# ============================================================
#  AMP dtype helpers
# ============================================================

def resolve_amp_dtype(
    amp_dtype: str = "bf16",
    device: Union[str, torch.device] = "cuda",
) -> torch.dtype:
    """
    Converts user AMP dtype string into torch dtype.
    """
    amp_dtype = amp_dtype.lower()

    if amp_dtype not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported amp_dtype={amp_dtype}. "
            f"Expected one of {sorted(DTYPE_MAP.keys())}."
        )

    return DTYPE_MAP[amp_dtype]


def cuda_supports_bf16() -> bool:
    """
    Checks whether current CUDA device supports bfloat16.
    """
    if not torch.cuda.is_available():
        return False

    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            pass

    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def get_effective_amp_dtype(
    amp_dtype: str = "bf16",
    device: Union[str, torch.device] = "cuda",
    fallback_bf16_to_fp16: bool = True,
) -> Optional[torch.dtype]:
    """
    Returns the actual dtype that should be used inside autocast.

    Returns None when AMP should be disabled.
    """
    device_type = normalize_device_type(device)
    requested_dtype = resolve_amp_dtype(amp_dtype, device=device)

    if requested_dtype == torch.float32:
        return None

    if device_type == "cuda":
        if not torch.cuda.is_available():
            return None

        if requested_dtype == torch.bfloat16:
            if cuda_supports_bf16():
                return torch.bfloat16

            return torch.float16 if fallback_bf16_to_fp16 else None

        if requested_dtype == torch.float16:
            return torch.float16

        return None

    # CPU autocast works mainly with bf16.
    if device_type == "cpu":
        if requested_dtype == torch.bfloat16:
            return torch.bfloat16
        return None

    # MPS autocast support is limited/unstable depending on PyTorch version.
    return None


# ============================================================
# GradScaler helpers
# ============================================================

def should_use_grad_scaler(
    device: Union[str, torch.device] = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    fallback_bf16_to_fp16: bool = True,
) -> bool:
    """
    GradScaler is needed for fp16 CUDA training.

    It is not needed for:
        - bf16
        - fp32
        - CPU
        - disabled AMP
    """
    if not amp_enabled:
        return False

    if normalize_device_type(device) != "cuda":
        return False

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=device,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    return effective_dtype == torch.float16


def make_grad_scaler(
    device: Union[str, torch.device] = "cuda",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    fallback_bf16_to_fp16: bool = True,
):
    """
    Creates a GradScaler only when needed.

    Returns:
        torch GradScaler or None.
    """
    enabled = should_use_grad_scaler(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    if not enabled:
        return None

    device_type = normalize_device_type(device)

    # New PyTorch API.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            sig = inspect.signature(torch.amp.GradScaler)

            if "device" in sig.parameters:
                return torch.amp.GradScaler(device=device_type, enabled=True)

            return torch.amp.GradScaler(device_type, enabled=True)

        except Exception:
            pass

    # Legacy CUDA API.
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)

    return None


# ============================================================
# Autocast context
# ============================================================

@contextmanager
def autocast_ctx(
    device: Union[str, torch.device] = "cuda",
    enabled: bool = True,
    amp_dtype: str = "bf16",
    cache_enabled: bool = True,
    fallback_bf16_to_fp16: bool = True,
):
    """
    Safe autocast context for DINO student/teacher forward passes.

    Usage:
        with autocast_ctx(**precision["autocast_kwargs"]):
            out = model(x)

    Important:
        Do NOT wrap the yield in try/except. If model forward fails,
        contextlib can otherwise raise:
            RuntimeError: generator didn't stop after throw()
    """
    if not enabled:
        with nullcontext():
            yield
        return

    device_type = normalize_device_type(device)

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=device,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    if effective_dtype is None:
        with nullcontext():
            yield
        return

    if not hasattr(torch, "amp") or not hasattr(torch.amp, "autocast"):
        with nullcontext():
            yield
        return

    if device_type in {"cuda", "cpu"}:
        ctx = torch.amp.autocast(
            device_type=device_type,
            dtype=effective_dtype,
            cache_enabled=cache_enabled,
        )

        with ctx:
            yield

        return

    with nullcontext():
        yield


# ============================================================
# Setup precision
# ============================================================

def setup_device_and_precision(
    device: Union[str, torch.device] = "auto",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    cache_enabled: bool = True,
    fallback_bf16_to_fp16: bool = True,
) -> Dict[str, Any]:
    """
    Creates a precision/device dictionary for training.

    Designed to be stored inside bundle_dino["precision"].

    Returns
    -------
    precision: dict
        {
            "device": torch.device,
            "device_type": str,
            "amp_enabled": bool,
            "amp_dtype_requested": str,
            "amp_dtype_effective": torch.dtype | None,
            "use_grad_scaler": bool,
            "scaler": GradScaler | None,
            "cache_enabled": bool,
            "fallback_bf16_to_fp16": bool,
            "autocast_kwargs": dict,
        }
    """
    resolved_device = resolve_device(device)
    device_type = normalize_device_type(resolved_device)

    effective_dtype = get_effective_amp_dtype(
        amp_dtype=amp_dtype,
        device=resolved_device,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    final_amp_enabled = bool(amp_enabled and effective_dtype is not None)

    scaler = make_grad_scaler(
        device=resolved_device,
        amp_enabled=final_amp_enabled,
        amp_dtype=amp_dtype,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    autocast_kwargs = {
        "device": resolved_device,
        "enabled": final_amp_enabled,
        "amp_dtype": amp_dtype,
        "cache_enabled": cache_enabled,
        "fallback_bf16_to_fp16": fallback_bf16_to_fp16,
    }

    return {
        "device": resolved_device,
        "device_type": device_type,
        "amp_enabled": final_amp_enabled,
        "amp_dtype_requested": amp_dtype,
        "amp_dtype_effective": effective_dtype,
        "use_grad_scaler": scaler is not None,
        "scaler": scaler,
        "cache_enabled": cache_enabled,
        "fallback_bf16_to_fp16": fallback_bf16_to_fp16,
        "autocast_kwargs": autocast_kwargs,
    }


# ============================================================
# Recursive batch movement
# ============================================================

def move_batch_to_device(
    batch: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:
    """
    Recursively move tensors in a batch to device.

    Supports:
        - Tensor
        - dict
        - tuple
        - list

    This is DINO-compatible because a DINO batch is usually a list:
        [
            global_crop_1,
            global_crop_2,
            local_crop_1,
            ...
        ]
    """
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=non_blocking)

    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device, non_blocking=non_blocking)
            for key, value in batch.items()
        }

    if isinstance(batch, tuple):
        return tuple(
            move_batch_to_device(value, device, non_blocking=non_blocking)
            for value in batch
        )

    if isinstance(batch, list):
        return [
            move_batch_to_device(value, device, non_blocking=non_blocking)
            for value in batch
        ]

    return batch


# ============================================================
# DINO-specific multi-crop movement
# ============================================================

def move_dino_crops_to_device(
    crops: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    device: torch.device,
    non_blocking: bool = True,
) -> List[torch.Tensor]:
    """
    Move DINO multi-crop batch to device.

    Expected input:
        crops = [
            global_crop_1,
            global_crop_2,
            local_crop_1,
            local_crop_2,
            ...
        ]

    Returns:
        List[Tensor] on device.
    """
    if not isinstance(crops, (list, tuple)):
        raise TypeError(
            f"DINO crops must be list or tuple, got {type(crops)}"
        )

    if len(crops) == 0:
        raise ValueError("DINO crops list is empty.")

    moved_crops = []

    for idx, crop in enumerate(crops):
        if not torch.is_tensor(crop):
            raise TypeError(
                f"Expected crop {idx} to be torch.Tensor, got {type(crop)}"
            )

        moved_crops.append(
            crop.to(device=device, non_blocking=non_blocking)
        )

    return moved_crops


def split_dino_crops(
    crops: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    num_global_crops: int = 2,
) -> Dict[str, List[torch.Tensor]]:
    """
    Split DINO multi-crop batch into teacher and student crops.

    Teacher:
        sees only global crops.

    Student:
        sees global + local crops.

    Expected input order:
        [
            global_crop_1,
            global_crop_2,
            local_crop_1,
            ...
        ]

    Returns:
        {
            "teacher_crops": List[Tensor],
            "student_crops": List[Tensor],
        }
    """
    if not isinstance(crops, (list, tuple)):
        raise TypeError(
            f"DINO crops must be list or tuple, got {type(crops)}"
        )

    if num_global_crops <= 0:
        raise ValueError(
            f"num_global_crops must be positive, got {num_global_crops}"
        )

    if len(crops) < num_global_crops:
        raise ValueError(
            f"Received {len(crops)} crops, but num_global_crops={num_global_crops}."
        )

    crops = list(crops)

    return {
        "teacher_crops": crops[:num_global_crops],
        "student_crops": crops,
    }


# ============================================================
# Convenience helper for DINO training step input
# ============================================================

def prepare_dino_batch_for_device(
    batch: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    device: torch.device,
    num_global_crops: int = 2,
    non_blocking: bool = True,
) -> Dict[str, List[torch.Tensor]]:
    """
    Moves DINO multi-crop batch to device and splits it.

    Usage:
        crop_dict = prepare_dino_batch_for_device(
            batch=batch,
            device=precision["device"],
            num_global_crops=2,
        )

        teacher_crops = crop_dict["teacher_crops"]
        student_crops = crop_dict["student_crops"]
    """
    crops = move_dino_crops_to_device(
        crops=batch,
        device=device,
        non_blocking=non_blocking,
    )

    return split_dino_crops(
        crops=crops,
        num_global_crops=num_global_crops,
    )
