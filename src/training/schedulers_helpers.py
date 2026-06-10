"""Build reusable scalar schedules for DINO training hyperparameters."""

# ============================================================
# DINO SCHEDULERS
# Step-based schedules for:
#   - learning rate
#   - weight decay
#   - teacher EMA momentum
#   - teacher temperature
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ============================================================
# Generic schedule functions
# ============================================================

def cosine_schedule_value(
    step: int,
    total_steps: int,
    start_value: float,
    end_value: float,
) -> float:
    """
    Cosine interpolation from start_value to end_value.

    step = 0           -> start_value
    step = total_steps -> end_value
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")

    step = min(max(int(step), 0), int(total_steps))

    progress = step / total_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    value = end_value + (start_value - end_value) * cosine

    return float(value)


def linear_schedule_value(
    step: int,
    total_steps: int,
    start_value: float,
    end_value: float,
) -> float:
    """
    Linear interpolation from start_value to end_value.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")

    step = min(max(int(step), 0), int(total_steps))

    progress = step / total_steps
    value = start_value + progress * (end_value - start_value)

    return float(value)


def warmup_cosine_schedule_value(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_value: float,
    final_value: float,
    start_warmup_value: float = 0.0,
) -> float:
    """
    Linear warmup followed by cosine decay.

    During warmup:
        start_warmup_value -> base_value

    After warmup:
        base_value -> final_value
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")

    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

    step = min(max(int(step), 0), int(total_steps))

    if warmup_steps > 0 and step < warmup_steps:
        return linear_schedule_value(
            step=step,
            total_steps=warmup_steps,
            start_value=start_warmup_value,
            end_value=base_value,
        )

    decay_steps = max(1, total_steps - warmup_steps)
    decay_step = max(0, step - warmup_steps)

    return cosine_schedule_value(
        step=decay_step,
        total_steps=decay_steps,
        start_value=base_value,
        end_value=final_value,
    )


def warmup_constant_schedule_value(
    step: int,
    warmup_steps: int,
    warmup_start_value: float,
    target_value: float,
) -> float:
    """
    Linear warmup to a constant value.

    Used for teacher temperature:
        warmup_teacher_temp -> teacher_temp
    """
    if warmup_steps <= 0:
        return float(target_value)

    step = max(int(step), 0)

    if step >= warmup_steps:
        return float(target_value)

    return linear_schedule_value(
        step=step,
        total_steps=warmup_steps,
        start_value=warmup_start_value,
        end_value=target_value,
    )

# ============================================================
# 2. Learning-rate scheduler
# ============================================================

class WarmupCosineLR:
    """
    Step-based linear warmup + cosine LR decay.

    DINO usage:
        optimizer.step()
        lr_scheduler.step()

    Supports standard torch optimizers with param_groups.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 0.0,
        start_warmup_lr: float = 0.0,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")

        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {min_lr}")

        if start_warmup_lr < 0:
            raise ValueError(f"start_warmup_lr must be >= 0, got {start_warmup_lr}")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.start_warmup_lr = float(start_warmup_lr)

        self.step_num = 0

        self.base_lrs = [
            float(group["lr"])
            for group in optimizer.param_groups
        ]

    def _compute_lr(
        self,
        base_lr: float,
        step: int,
    ) -> float:
        return warmup_cosine_schedule_value(
            step=step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            base_value=base_lr,
            final_value=self.min_lr,
            start_warmup_value=self.start_warmup_lr,
        )

    def _set_lr(self, step: int) -> None:
        for idx, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self._compute_lr(
                base_lr=self.base_lrs[idx],
                step=step,
            )

    def step(self) -> None:
        self.step_num += 1
        self._set_lr(self.step_num)

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        self.step_num = int(step)
        self._set_lr(self.step_num)

    def get_last_lr(self) -> List[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def get_lr_dict(self) -> Dict[str, Any]:
        lrs = self.get_last_lr()

        return {
            "step": int(self.step_num),
            "lr": float(lrs[0]) if lrs else None,
            "lrs": lrs,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_num": int(self.step_num),
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps),
            "min_lr": float(self.min_lr),
            "start_warmup_lr": float(self.start_warmup_lr),
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return

        self.step_num = int(state_dict.get("step_num", 0))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.start_warmup_lr = float(
            state_dict.get("start_warmup_lr", self.start_warmup_lr)
        )

        loaded_base_lrs = state_dict.get("base_lrs", None)

        if (
            isinstance(loaded_base_lrs, (list, tuple))
            and len(loaded_base_lrs) == len(self.optimizer.param_groups)
        ):
            self.base_lrs = [float(x) for x in loaded_base_lrs]

        self._set_lr(self.step_num)
