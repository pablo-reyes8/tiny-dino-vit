# ============================================================
# DINO PRINT HELPERS
# ============================================================

from __future__ import annotations

import os
import json
import shutil
import time
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn



def dino_rule(width: int = 110, ch: str = "─") -> str:
    return ch * width


def dino_title(title: str, width: int = 110, ch: str = "═") -> None:
    print("\n" + ch * width)
    print(title)
    print(ch * width)


def dino_section(title: str, width: int = 110) -> None:
    print("\n" + title)
    print(dino_rule(width=width, ch="─"))


def fmt_hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def fmt_num(x, digits: int = 4, sci_low: float = 1e-3, sci_high: float = 1e4) -> str:
    try:
        x = float(x)
    except Exception:
        return "—"

    if not math.isfinite(x):
        return "—"

    ax = abs(x)

    if ax == 0:
        return "0"

    if ax < sci_low or ax >= sci_high:
        return f"{x:.2e}"

    return f"{x:.{digits}f}"


def fmt_lr(x) -> str:
    try:
        return f"{float(x):.2e}"
    except Exception:
        return "—"


def prefixed_stats(prefix: str, stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if stats is None:
        return {}

    return {f"{prefix}_{k}": v for k, v in stats.items()}


def is_better_metric(
    current: float,
    best: Optional[float],
    mode: str = "min",
) -> bool:
    if best is None:
        return True

    if mode == "min":
        return current < best

    if mode == "max":
        return current > best

    raise ValueError(f"monitor_mode must be 'min' or 'max', got {mode}")


def resolve_dino_monitor_value(
    monitor_name: str,
    train_stats: Dict[str, Any],
) -> float:
    if monitor_name.startswith("train_"):
        key = monitor_name[len("train_"):]
    else:
        key = monitor_name

    if key not in train_stats:
        raise KeyError(
            f"Monitor key '{key}' not found in train_stats. "
            f"Available keys: {sorted(train_stats.keys())[:80]}"
        )

    return float(train_stats[key])


def get_main_lr_from_optimizer(optimizer: Any) -> float:
    if hasattr(optimizer, "param_groups") and optimizer.param_groups:
        return float(optimizer.param_groups[0]["lr"])

    return float("nan")


def print_dino_run_header(
    *,
    run_name: str,
    bundle_dino: Dict[str, Any],
    device,
    precision: Dict[str, Any],
    optimizer_type: str,
    epochs: int,
    start_epoch: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    monitor_name: str,
    monitor_mode: str,
    best_metric,
    grad_clip,
    grad_accum_steps: int,
    use_gram_loss: bool,
    gram_loss_weight: float,
    teacher_offload: bool,
    ckpt_dir: str,
) -> None:
    model_info = bundle_dino.get("model_info", {})

    dino_title(f"DINO run: {run_name}")

    print(
        f"Device    : {device} | AMP: {precision['amp_enabled']} "
        f"({precision['amp_dtype_requested']} -> {precision['amp_dtype_effective']})"
    )

    print(
        f"Optimizer : {optimizer_type} | grad_clip: {grad_clip} | "
        f"grad_accum_steps: {grad_accum_steps}"
    )

    print(
        f"Schedule  : epochs={epochs} | start_epoch={start_epoch} | "
        f"global_step={global_step} | total_steps={total_steps} | warmup_steps={warmup_steps}"
    )

    print(
        f"Monitor   : {monitor_name} ({monitor_mode}) | best_metric={best_metric}"
    )

    print(
        f"DINO      : gram_loss={use_gram_loss} | gram_weight={gram_loss_weight} | "
        f"teacher_offload={teacher_offload}"
    )

    if model_info:
        backbone = model_info.get("backbone", {})
        head = model_info.get("head", {})

        print(
            f"Backbone  : {backbone.get('type', 'ViT')} | "
            f"img={backbone.get('img_size', '?')} | patch={backbone.get('patch_size', '?')} | "
            f"dim={backbone.get('embed_dim', '?')} | depth={backbone.get('depth', '?')} | "
            f"heads={backbone.get('num_heads', '?')} | pos={backbone.get('pos_embed_type', '?')}"
        )

        print(
            f"Head      : out_dim={head.get('out_dim', '?')} | "
            f"hidden={head.get('hidden_dim', '?')} | bottleneck={head.get('bottleneck_dim', '?')} | "
            f"layers={head.get('num_layers', '?')}"
        )

        print(
            f"Params    : student={model_info.get('student_total_params', 0):,} | "
            f"trainable={model_info.get('student_trainable_params', 0):,} | "
            f"teacher_trainable={model_info.get('teacher_trainable_params', 0):,}"
        )

    print(f"Checkpoints: {ckpt_dir}")
    print(dino_rule())


def print_dino_epoch_summary(
    *,
    epoch: int,
    global_step: int,
    sec: float,
    optimizer,
    train_stats: Dict[str, Any],
    monitor_name: str,
    current_metric: float,
    best_metric: float,
    improved: bool,
) -> None:
    dino_section(f"Epoch {epoch:03d} summary")

    print(
        f"step={global_step} | time={fmt_hms(sec)} | "
        f"lr={fmt_lr(get_main_lr_from_optimizer(optimizer))}"
    )

    keys = [
        "loss",
        "loss_dino",
        "loss_gram",
        "grad_norm",
        "teacher_temp",
        "teacher_momentum",
        "teacher_entropy_norm",
        "student_entropy_norm",
        "student_cls_dim_std_mean",
        "cuda_mem_alloc_gb",
        "n_optimizer_steps",
    ]

    parts = []

    for key in keys:
        if key in train_stats:
            parts.append(f"{key}={fmt_num(train_stats[key])}")

    print("train -> " + " | ".join(parts))

    print(
        f"monitor -> {monitor_name}={current_metric:.6f} | "
        f"best={best_metric:.6f} | improved={improved}"
    )

    print(dino_rule())


# ============================================================
# TRAIN DINO
# High-level orchestration
# ============================================================



def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_record = {}

    for k, v in record.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                safe_record[k] = float(v.detach().cpu().item())
            else:
                safe_record[k] = str(tuple(v.shape))

        elif isinstance(v, (int, float, str, bool)) or v is None:
            safe_record[k] = v

        else:
            try:
                json.dumps(v)
                safe_record[k] = v
            except Exception:
                safe_record[k] = str(v)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def copy_file_to_dir_fixed(
    src_path: str | Path,
    dst_dir: str | Path,
    fixed_name: str,
) -> None:
    try:
        if not dst_dir:
            return

        src_path = Path(src_path)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_path = dst_dir / fixed_name

        if dst_path.exists():
            dst_path.unlink()

        shutil.copy2(src_path, dst_path)

        print(f"└─ [COPY] {src_path.name} → {dst_path}")

    except Exception as e:
        print(f"└─ [COPY] ERROR: {e}")


def build_dino_optimizer(
    *,
    student: nn.Module,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.04,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> tuple[torch.optim.Optimizer, Dict[str, Any]]:
    """
    Minimal AdamW optimizer for DINO student.

    The teacher must not be passed to the optimizer.
    """
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    opt_info = {
        "optimizer_type": "adamw",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "betas": betas,
        "eps": eps,
        "num_param_groups": len(optimizer.param_groups),
    }

    return optimizer, opt_info
