from src.training.schedulers_helpers import * 

# ============================================================
# Weight decay scheduler
# ============================================================

class CosineWeightDecay:
    """
    Step-based cosine weight decay scheduler.

    DINO-style:
        weight_decay_start -> weight_decay_end

    Usually:
        0.04 -> 0.4

    Only modifies param groups where:
        group.get("apply_weight_decay", True) is True
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        weight_decay_start: float = 0.04,
        weight_decay_end: float = 0.4,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")

        if weight_decay_start < 0 or weight_decay_end < 0:
            raise ValueError("weight decay values must be >= 0")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.weight_decay_start = float(weight_decay_start)
        self.weight_decay_end = float(weight_decay_end)

        self.step_num = 0

    def _compute_weight_decay(self, step: int) -> float:
        return cosine_schedule_value(
            step=step,
            total_steps=self.total_steps,
            start_value=self.weight_decay_start,
            end_value=self.weight_decay_end,
        )

    def _set_weight_decay(self, step: int) -> None:
        wd = self._compute_weight_decay(step)

        for group in self.optimizer.param_groups:
            if group.get("apply_weight_decay", True):
                group["weight_decay"] = wd

    def step(self) -> None:
        self.step_num += 1
        self._set_weight_decay(self.step_num)

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        self.step_num = int(step)
        self._set_weight_decay(self.step_num)

    def get_last_weight_decay(self) -> float:
        return float(
            self._compute_weight_decay(self.step_num)
        )

    def get_weight_decay_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step_num),
            "weight_decay": self.get_last_weight_decay(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_num": int(self.step_num),
            "total_steps": int(self.total_steps),
            "weight_decay_start": float(self.weight_decay_start),
            "weight_decay_end": float(self.weight_decay_end),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return

        self.step_num = int(state_dict.get("step_num", 0))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.weight_decay_start = float(
            state_dict.get("weight_decay_start", self.weight_decay_start)
        )
        self.weight_decay_end = float(
            state_dict.get("weight_decay_end", self.weight_decay_end)
        )

        self._set_weight_decay(self.step_num)


# ============================================================
# Teacher EMA momentum scheduler
# ============================================================

class CosineTeacherMomentum:
    """
    DINO teacher EMA momentum scheduler.

    DINO-style:
        teacher_momentum_start -> teacher_momentum_end

    Usually:
        0.996 -> 1.0

    This scheduler does not update the teacher by itself.
    It only returns the momentum to use in EMA update.
    """

    def __init__(
        self,
        total_steps: int,
        momentum_start: float = 0.996,
        momentum_end: float = 1.0,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")

        if not 0.0 <= momentum_start <= 1.0:
            raise ValueError(f"momentum_start must be in [0, 1], got {momentum_start}")

        if not 0.0 <= momentum_end <= 1.0:
            raise ValueError(f"momentum_end must be in [0, 1], got {momentum_end}")

        if momentum_start > momentum_end:
            raise ValueError(
                f"momentum_start should be <= momentum_end. "
                f"Got {momentum_start} > {momentum_end}."
            )

        self.total_steps = int(total_steps)
        self.momentum_start = float(momentum_start)
        self.momentum_end = float(momentum_end)

        self.step_num = 0

    def _compute_momentum(self, step: int) -> float:
        return cosine_schedule_value(
            step=step,
            total_steps=self.total_steps,
            start_value=self.momentum_start,
            end_value=self.momentum_end,
        )

    def step(self) -> float:
        self.step_num += 1
        return self.get_momentum()

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        self.step_num = int(step)

    def get_momentum(self) -> float:
        return float(
            self._compute_momentum(self.step_num)
        )

    def get_momentum_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step_num),
            "teacher_momentum": self.get_momentum(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_num": int(self.step_num),
            "total_steps": int(self.total_steps),
            "momentum_start": float(self.momentum_start),
            "momentum_end": float(self.momentum_end),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return

        self.step_num = int(state_dict.get("step_num", 0))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.momentum_start = float(
            state_dict.get("momentum_start", self.momentum_start)
        )
        self.momentum_end = float(
            state_dict.get("momentum_end", self.momentum_end)
        )


# ============================================================
# Teacher temperature scheduler
# ============================================================

class TeacherTemperatureScheduler:
    """
    DINO teacher temperature scheduler.

    Usually:
        warmup_teacher_temp = 0.04
        teacher_temp = 0.07
        warmup_teacher_temp_steps = some warmup period

    After warmup, teacher_temp stays constant.
    """

    def __init__(
        self,
        warmup_steps: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
    ):
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        if warmup_teacher_temp <= 0:
            raise ValueError(
                f"warmup_teacher_temp must be > 0, got {warmup_teacher_temp}"
            )

        if teacher_temp <= 0:
            raise ValueError(f"teacher_temp must be > 0, got {teacher_temp}")

        self.warmup_steps = int(warmup_steps)
        self.warmup_teacher_temp = float(warmup_teacher_temp)
        self.teacher_temp = float(teacher_temp)

        self.step_num = 0

    def _compute_temperature(self, step: int) -> float:
        return warmup_constant_schedule_value(
            step=step,
            warmup_steps=self.warmup_steps,
            warmup_start_value=self.warmup_teacher_temp,
            target_value=self.teacher_temp,
        )

    def step(self) -> float:
        self.step_num += 1
        return self.get_temperature()

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        self.step_num = int(step)

    def get_temperature(self) -> float:
        return float(
            self._compute_temperature(self.step_num)
        )

    def get_temperature_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step_num),
            "teacher_temp": self.get_temperature(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_num": int(self.step_num),
            "warmup_steps": int(self.warmup_steps),
            "warmup_teacher_temp": float(self.warmup_teacher_temp),
            "teacher_temp": float(self.teacher_temp),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return

        self.step_num = int(state_dict.get("step_num", 0))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.warmup_teacher_temp = float(
            state_dict.get("warmup_teacher_temp", self.warmup_teacher_temp)
        )
        self.teacher_temp = float(
            state_dict.get("teacher_temp", self.teacher_temp)
        )


@dataclass
class DINOSchedulerValues:
    lr: Optional[float]
    weight_decay: Optional[float]
    teacher_momentum: float
    teacher_temp: float


class DINOSchedulerBundle:
    """
    Wrapper for all DINO schedules.

    Contains:
        - LR scheduler
        - weight decay scheduler
        - teacher momentum scheduler
        - teacher temperature scheduler

    Usage:
        values = schedulers.step()

        momentum = values.teacher_momentum
        teacher_temp = values.teacher_temp

        update_teacher_ema(student, teacher, momentum)
        dino_loss(..., teacher_temp=teacher_temp)
    """

    def __init__(
        self,
        lr_scheduler: Optional[WarmupCosineLR],
        weight_decay_scheduler: Optional[CosineWeightDecay],
        teacher_momentum_scheduler: CosineTeacherMomentum,
        teacher_temp_scheduler: TeacherTemperatureScheduler,
    ):
        self.lr_scheduler = lr_scheduler
        self.weight_decay_scheduler = weight_decay_scheduler
        self.teacher_momentum_scheduler = teacher_momentum_scheduler
        self.teacher_temp_scheduler = teacher_temp_scheduler

        self.step_num = 0

    def step(self) -> DINOSchedulerValues:
        """
        Advances all schedulers by one step.

        Recommended order in training:
            optimizer.step()
            schedulers.step()
            update_teacher_ema(..., momentum=schedulers current momentum)
        """
        self.step_num += 1

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.weight_decay_scheduler is not None:
            self.weight_decay_scheduler.step()

        teacher_momentum = self.teacher_momentum_scheduler.step()
        teacher_temp = self.teacher_temp_scheduler.step()

        lr = None
        if self.lr_scheduler is not None:
            lrs = self.lr_scheduler.get_last_lr()
            lr = float(lrs[0]) if lrs else None

        weight_decay = None
        if self.weight_decay_scheduler is not None:
            weight_decay = self.weight_decay_scheduler.get_last_weight_decay()

        return DINOSchedulerValues(
            lr=lr,
            weight_decay=weight_decay,
            teacher_momentum=teacher_momentum,
            teacher_temp=teacher_temp,
        )

    def set_step(self, step: int) -> None:
        """
        Resume-safe step setter.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        self.step_num = int(step)

        if self.lr_scheduler is not None:
            self.lr_scheduler.set_step(step)

        if self.weight_decay_scheduler is not None:
            self.weight_decay_scheduler.set_step(step)

        self.teacher_momentum_scheduler.set_step(step)
        self.teacher_temp_scheduler.set_step(step)

    def get_values(self) -> DINOSchedulerValues:
        """
        Returns current values without advancing step.
        """
        lr = None
        if self.lr_scheduler is not None:
            lrs = self.lr_scheduler.get_last_lr()
            lr = float(lrs[0]) if lrs else None

        weight_decay = None
        if self.weight_decay_scheduler is not None:
            weight_decay = self.weight_decay_scheduler.get_last_weight_decay()

        return DINOSchedulerValues(
            lr=lr,
            weight_decay=weight_decay,
            teacher_momentum=self.teacher_momentum_scheduler.get_momentum(),
            teacher_temp=self.teacher_temp_scheduler.get_temperature(),
        )

    def get_log_dict(self) -> Dict[str, Any]:
        values = self.get_values()

        return {
            "scheduler_step": int(self.step_num),
            "lr": values.lr,
            "weight_decay": values.weight_decay,
            "teacher_momentum": values.teacher_momentum,
            "teacher_temp": values.teacher_temp,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_num": int(self.step_num),
            "lr_scheduler": (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            ),
            "weight_decay_scheduler": (
                self.weight_decay_scheduler.state_dict()
                if self.weight_decay_scheduler is not None
                else None
            ),
            "teacher_momentum_scheduler": self.teacher_momentum_scheduler.state_dict(),
            "teacher_temp_scheduler": self.teacher_temp_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return

        self.step_num = int(state_dict.get("step_num", 0))

        if self.lr_scheduler is not None and state_dict.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        if (
            self.weight_decay_scheduler is not None
            and state_dict.get("weight_decay_scheduler") is not None
        ):
            self.weight_decay_scheduler.load_state_dict(
                state_dict["weight_decay_scheduler"]
            )

        self.teacher_momentum_scheduler.load_state_dict(
            state_dict.get("teacher_momentum_scheduler", {})
        )

        self.teacher_temp_scheduler.load_state_dict(
            state_dict.get("teacher_temp_scheduler", {})
        )


def build_dino_schedulers(
    optimizer,
    CFG: Dict[str, Any],
) -> DINOSchedulerBundle:
    """
    Build DINO schedulers from CFG.

    Expected CFG structure:

    CFG = {
        "train": {
            "epochs": 100,
            "steps_per_epoch": len(train_loader),
        },
        "optim": {
            "lr": 5e-4,
            "min_lr": 1e-6,
            "warmup_epochs": 10,
            "weight_decay_start": 0.04,
            "weight_decay_end": 0.4,
        },
        "dino": {
            "teacher_momentum_start": 0.996,
            "teacher_momentum_end": 1.0,
            "warmup_teacher_temp": 0.04,
            "teacher_temp": 0.07,
            "warmup_teacher_temp_epochs": 30,
        }
    }
    """

    train_cfg = CFG["train"]
    optim_cfg = CFG["optim"]
    dino_cfg = CFG["dino"]

    epochs = int(train_cfg["epochs"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])

    total_steps = epochs * steps_per_epoch

    warmup_epochs = int(optim_cfg.get("warmup_epochs", 0))
    warmup_steps = warmup_epochs * steps_per_epoch

    warmup_teacher_temp_epochs = int(
        dino_cfg.get("warmup_teacher_temp_epochs", 0)
    )
    warmup_teacher_temp_steps = warmup_teacher_temp_epochs * steps_per_epoch

    lr_scheduler = WarmupCosineLR(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=float(optim_cfg.get("min_lr", 0.0)),
        start_warmup_lr=float(optim_cfg.get("start_warmup_lr", 0.0)),
    )

    weight_decay_scheduler = CosineWeightDecay(
        optimizer=optimizer,
        total_steps=total_steps,
        weight_decay_start=float(optim_cfg.get("weight_decay_start", 0.04)),
        weight_decay_end=float(optim_cfg.get("weight_decay_end", 0.4)),
    )

    teacher_momentum_scheduler = CosineTeacherMomentum(
        total_steps=total_steps,
        momentum_start=float(dino_cfg.get("teacher_momentum_start", 0.996)),
        momentum_end=float(dino_cfg.get("teacher_momentum_end", 1.0)),
    )

    teacher_temp_scheduler = TeacherTemperatureScheduler(
        warmup_steps=warmup_teacher_temp_steps,
        warmup_teacher_temp=float(dino_cfg.get("warmup_teacher_temp", 0.04)),
        teacher_temp=float(dino_cfg.get("teacher_temp", 0.07)),
    )

    return DINOSchedulerBundle(
        lr_scheduler=lr_scheduler,
        weight_decay_scheduler=weight_decay_scheduler,
        teacher_momentum_scheduler=teacher_momentum_scheduler,
        teacher_temp_scheduler=teacher_temp_scheduler,
    )