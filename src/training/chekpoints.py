# ============================================================
# DINO CHECKPOINT UTILITIES
# Efficient resume-safe checkpointing for student/teacher DINO
# ============================================================

from __future__ import annotations

import os
import re
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Small helpers
# ============================================================

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Handles DataParallel/DDP-style wrappers.
    """
    return model.module if hasattr(model, "module") else model


def safe_to_serializable(obj: Any) -> Any:
    """
    Best-effort conversion for configs/metadata into JSON-safe objects.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (list, tuple)):
        return [safe_to_serializable(x) for x in obj]

    if isinstance(obj, dict):
        return {
            str(k): safe_to_serializable(v)
            for k, v in obj.items()
        }

    if hasattr(obj, "__dict__"):
        return {
            str(k): safe_to_serializable(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }

    return str(obj)


def get_rng_state() -> Dict[str, Any]:
    """
    Capture random states so training can be resumed more faithfully.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """
    Restore Python, NumPy, PyTorch CPU and CUDA RNG states safely.
    """
    if not state:
        return

    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])

    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])

    if "torch" in state and state["torch"] is not None:
        torch_state = state["torch"]

        if not torch.is_tensor(torch_state):
            torch_state = torch.tensor(torch_state, dtype=torch.uint8)

        torch_state = torch_state.detach().cpu().to(dtype=torch.uint8)
        torch.set_rng_state(torch_state)

    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = state["cuda"]

        fixed_cuda_states = []

        for s in cuda_states:
            if not torch.is_tensor(s):
                s = torch.tensor(s, dtype=torch.uint8)

            s = s.detach().cpu().to(dtype=torch.uint8)
            fixed_cuda_states.append(s)

        torch.cuda.set_rng_state_all(fixed_cuda_states)


def atomic_torch_save(obj: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Atomic save:
        1. save to temporary file
        2. rename to final path

    Avoids corrupting the target checkpoint if the process crashes mid-save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def extract_step_from_name(path: Union[str, Path]) -> int:
    """
    Extract step number from filenames like:
        dino_checkpoint_step_00000100.pt
        checkpoint_step_100.pt
    """
    name = Path(path).name
    matches = re.findall(r"step[_-](\d+)", name)

    if not matches:
        return -1

    return int(matches[-1])


def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 3,
    pattern: str = "dino_checkpoint_step_*.pt",
) -> None:
    """
    Keep only the last N step checkpoints.

    Does not delete:
        latest.pt
        best.pt
        dino_checkpoint_best_*.pt
    """
    checkpoint_dir = Path(checkpoint_dir)

    if keep_last_n <= 0:
        return

    candidates = list(checkpoint_dir.glob(pattern))

    if len(candidates) <= keep_last_n:
        return

    candidates = sorted(
        candidates,
        key=lambda p: (extract_step_from_name(p), p.stat().st_mtime),
    )

    to_delete = candidates[:-keep_last_n]

    for path in to_delete:
        try:
            path.unlink()
        except Exception:
            pass

        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            try:
                sidecar.unlink()
            except Exception:
                pass


# ============================================================
# Save DINO checkpoint
# ============================================================

def save_dino_checkpoint(
    checkpoint_dir: Union[str, Path],
    student: nn.Module,
    teacher: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    best_metric: Optional[float] = None,
    config: Optional[Any] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    save_rng_state: bool = True,
    keep_last_n: Optional[int] = None,
    tag: Optional[str] = None,
    save_latest: bool = True,
    save_json_sidecar: bool = True,
) -> Path:
    """
    Save a full DINO training checkpoint.

    Saves:
        - student state_dict
        - teacher state_dict
        - optimizer state_dict, if provided
        - scheduler state_dict, if provided
        - scaler state_dict, if provided
        - epoch / step / best_metric
        - config / extra_state
        - RNG state, if requested

    In DINO, the teacher is the EMA model, so it is saved explicitly.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        if tag is not None:
            filename = f"dino_checkpoint_{tag}_step_{step:08d}.pt"
        else:
            filename = f"dino_checkpoint_step_{step:08d}.pt"

    ckpt_path = checkpoint_dir / filename

    raw_student = unwrap_model(student)
    raw_teacher = unwrap_model(teacher)

    checkpoint = {
        "checkpoint_type": "dino",
        "student_state_dict": raw_student.state_dict(),
        "teacher_state_dict": raw_teacher.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "best_metric": best_metric,
        "config": safe_to_serializable(config),
        "extra_state": extra_state or {},
        "rng_state": get_rng_state() if save_rng_state else None,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = (
            scheduler.state_dict()
            if hasattr(scheduler, "state_dict")
            else None
        )

    if scaler is not None:
        checkpoint["scaler_state_dict"] = (
            scaler.state_dict()
            if hasattr(scaler, "state_dict")
            else None
        )

    atomic_torch_save(checkpoint, ckpt_path)

    if save_json_sidecar:
        metadata_path = ckpt_path.with_suffix(".json")

        metadata = {
            "checkpoint": ckpt_path.name,
            "checkpoint_type": "dino",
            "epoch": int(epoch),
            "step": int(step),
            "best_metric": best_metric,
            "tag": tag,
            "has_student": True,
            "has_teacher": True,
            "has_optimizer": optimizer is not None,
            "has_scheduler": scheduler is not None,
            "has_scaler": scaler is not None,
            "has_rng_state": save_rng_state,
            "config": safe_to_serializable(config),
            "extra_state": safe_to_serializable(extra_state or {}),
        }

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    if save_latest:
        latest_path = checkpoint_dir / "latest.pt"
        try:
            shutil.copyfile(ckpt_path, latest_path)
        except Exception:
            pass

    if keep_last_n is not None and keep_last_n > 0:
        cleanup_old_checkpoints(
            checkpoint_dir=checkpoint_dir,
            keep_last_n=keep_last_n,
            pattern="dino_checkpoint_step_*.pt",
        )

    return ckpt_path

# ============================================================
# Load DINO checkpoint
# ============================================================

def load_dino_checkpoint(
    checkpoint_path: Union[str, Path],
    student: Optional[nn.Module] = None,
    teacher: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict_student: bool = True,
    strict_teacher: bool = True,
    load_student: bool = True,
    load_teacher: bool = True,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    load_scaler: bool = True,
    load_rng_state: bool = True,
) -> Dict[str, Any]:
    """
    Load DINO checkpoint into student/teacher/optimizer/scheduler/scaler.

    Returns:
        resume_state dict with:
            epoch
            step
            best_metric
            config
            extra_state
            missing/unexpected keys
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    student_missing = []
    student_unexpected = []
    teacher_missing = []
    teacher_unexpected = []

    if student is not None and load_student:
        if "student_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'student_state_dict'.")

        raw_student = unwrap_model(student)

        result = raw_student.load_state_dict(
            checkpoint["student_state_dict"],
            strict=strict_student,
        )

        student_missing = list(result.missing_keys)
        student_unexpected = list(result.unexpected_keys)

    if teacher is not None and load_teacher:
        if "teacher_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'teacher_state_dict'.")

        raw_teacher = unwrap_model(teacher)

        result = raw_teacher.load_state_dict(
            checkpoint["teacher_state_dict"],
            strict=strict_teacher,
        )

        teacher_missing = list(result.missing_keys)
        teacher_unexpected = list(result.unexpected_keys)

    if (
        optimizer is not None
        and load_optimizer
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler is not None
        and load_scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if (
        scaler is not None
        and load_scaler
        and "scaler_state_dict" in checkpoint
        and checkpoint["scaler_state_dict"] is not None
    ):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if load_rng_state:
        set_rng_state(checkpoint.get("rng_state", None))

    return {
        "checkpoint_type": checkpoint.get("checkpoint_type", None),
        "epoch": int(checkpoint.get("epoch", 0)),
        "step": int(checkpoint.get("step", 0)),
        "best_metric": checkpoint.get("best_metric", None),
        "config": checkpoint.get("config", None),
        "extra_state": checkpoint.get("extra_state", {}),
        "checkpoint_path": str(checkpoint_path),
        "student_missing_keys": student_missing,
        "student_unexpected_keys": student_unexpected,
        "teacher_missing_keys": teacher_missing,
        "teacher_unexpected_keys": teacher_unexpected,
        "loaded_student": bool(student is not None and load_student),
        "loaded_teacher": bool(teacher is not None and load_teacher),
        "loaded_optimizer": bool(
            optimizer is not None
            and load_optimizer
            and "optimizer_state_dict" in checkpoint
        ),
        "loaded_scheduler": bool(
            scheduler is not None
            and load_scheduler
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
        ),
        "loaded_scaler": bool(
            scaler is not None
            and load_scaler
            and "scaler_state_dict" in checkpoint
            and checkpoint["scaler_state_dict"] is not None
        ),
    }

def get_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    prefer_latest_pointer: bool = True,
) -> Optional[Path]:
    """
    Finds the latest DINO checkpoint in checkpoint_dir.

    Priority:
        1. latest.pt if it exists and prefer_latest_pointer=True
        2. highest step among dino_checkpoint_step_*.pt
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    latest_path = checkpoint_dir / "latest.pt"

    if prefer_latest_pointer and latest_path.exists():
        return latest_path

    candidates = list(checkpoint_dir.glob("dino_checkpoint_step_*.pt"))

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda p: (extract_step_from_name(p), p.stat().st_mtime),
    )

    return candidates[-1]


def checkpoint_exists(checkpoint_path: Union[str, Path]) -> bool:
    """
    Small convenience helper.
    """
    return Path(checkpoint_path).exists()

# ============================================================
# Best checkpoint helper
# ============================================================

def maybe_save_best_dino_checkpoint(
    checkpoint_dir: Union[str, Path],
    student: nn.Module,
    teacher: nn.Module,
    metric: float,
    best_metric: Optional[float],
    mode: str = "min",
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    config: Optional[Any] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    save_rng_state: bool = True,
) -> tuple[Optional[Path], float, bool]:
    """
    Save best DINO checkpoint if metric improves.

    Parameters
    ----------
    mode:
        "min" means lower metric is better.
        "max" means higher metric is better.

    Returns
    -------
    path:
        Saved checkpoint path or None.

    best_metric:
        Updated best metric.

    improved:
        Whether metric improved.
    """
    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    metric = float(metric)

    if best_metric is None:
        improved = True
    elif mode == "min":
        improved = metric < best_metric
    else:
        improved = metric > best_metric

    if not improved:
        return None, float(best_metric), False

    new_best = metric

    path = save_dino_checkpoint(
        checkpoint_dir=checkpoint_dir,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        step=step,
        best_metric=new_best,
        config=config,
        extra_state=extra_state,
        filename="best.pt",
        save_rng_state=save_rng_state,
        keep_last_n=None,
        tag="best",
        save_latest=False,
        save_json_sidecar=True,
    )

    return path, new_best, True