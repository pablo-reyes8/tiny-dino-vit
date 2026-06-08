# ============================================================
# DINO TEACHER EMA UTILITIES
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


# ============================================================
#  Model unwrap
# ============================================================

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Returns the underlying model if wrapped in DataParallel/DDP.
    """
    return model.module if hasattr(model, "module") else model


# ============================================================
# Freeze teacher
# ============================================================

def freeze_model(model: nn.Module) -> nn.Module:
    """
    Disable gradients for all model parameters.
    """
    model = unwrap_model(model)

    for param in model.parameters():
        param.requires_grad = False

    return model


def freeze_teacher(teacher: nn.Module) -> nn.Module:
    """
    DINO teacher should never be optimized by backprop.
    It is updated only through EMA from the student.
    """
    teacher = unwrap_model(teacher)
    teacher.eval()
    freeze_model(teacher)
    return teacher


# ============================================================
# Initialize teacher from student
# ============================================================

@torch.no_grad()
def initialize_teacher_from_student(
    student: nn.Module,
    teacher: nn.Module,
    strict: bool = True,
    freeze: bool = True,
) -> nn.Module:
    """
    Hard-copy student weights into teacher.

    This should be called once before DINO training starts.

    Parameters
    ----------
    student:
        Student model.

    teacher:
        Teacher model.

    strict:
        Passed to load_state_dict.

    freeze:
        If True, disables gradients on teacher.
    """
    student = unwrap_model(student)
    teacher = unwrap_model(teacher)

    teacher.load_state_dict(
        student.state_dict(),
        strict=strict,
    )

    if freeze:
        freeze_teacher(teacher)

    return teacher

# ============================================================
# DINO teacher EMA update
# ============================================================

@torch.no_grad()
def update_teacher_ema(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float,
    update_buffers: bool = True,
) -> None:
    """
    Update DINO teacher using EMA of student.

    Formula:
        teacher = momentum * teacher + (1 - momentum) * student

    In DINO:
        - student is optimized by backprop.
        - teacher is frozen.
        - teacher is updated only with this EMA rule.

    Parameters
    ----------
    student:
        Student model.

    teacher:
        Teacher model.

    momentum:
        EMA momentum. Usually scheduled from 0.996 to 1.0.

    update_buffers:
        If True, copies buffers from student to teacher.
        For a plain ViT this usually matters little, but it is safer.
    """
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"momentum must be in [0, 1], got {momentum}")

    student = unwrap_model(student)
    teacher = unwrap_model(teacher)

    student_params = dict(student.named_parameters())
    teacher_params = dict(teacher.named_parameters())

    if student_params.keys() != teacher_params.keys():
        missing_in_teacher = student_params.keys() - teacher_params.keys()
        missing_in_student = teacher_params.keys() - student_params.keys()

        raise RuntimeError(
            "Student and teacher parameter names do not match.\n"
            f"Missing in teacher: {list(missing_in_teacher)[:10]}\n"
            f"Missing in student: {list(missing_in_student)[:10]}"
        )

    for name, teacher_param in teacher_params.items():
        student_param = student_params[name].detach()

        if teacher_param.device != student_param.device:
            student_param = student_param.to(
                device=teacher_param.device,
                non_blocking=True,
            )

        if teacher_param.dtype != student_param.dtype:
            student_param = student_param.to(dtype=teacher_param.dtype)

        teacher_param.data.mul_(momentum).add_(
            student_param.data,
            alpha=1.0 - momentum,
        )

    if update_buffers:
        student_buffers = dict(student.named_buffers())
        teacher_buffers = dict(teacher.named_buffers())

        for name, teacher_buffer in teacher_buffers.items():
            if name not in student_buffers:
                continue

            student_buffer = student_buffers[name].detach()

            if teacher_buffer.device != student_buffer.device:
                student_buffer = student_buffer.to(
                    device=teacher_buffer.device,
                    non_blocking=True,
                )

            if teacher_buffer.dtype != student_buffer.dtype:
                student_buffer = student_buffer.to(dtype=teacher_buffer.dtype)

            teacher_buffer.copy_(student_buffer)


# ============================================================
#  DINO EMA health checks
# ============================================================

@torch.no_grad()
def dino_teacher_student_distance(
    student: nn.Module,
    teacher: nn.Module,
    max_params: Optional[int] = None,
) -> Dict[str, float]:
    """
    Computes basic distance diagnostics between student and teacher.

    Useful for checking that EMA is working.

    Returns
    -------
    {
        "student_norm": float,
        "teacher_norm": float,
        "relative_distance": float,
    }
    """
    student = unwrap_model(student)
    teacher = unwrap_model(teacher)

    student_chunks = []
    teacher_chunks = []

    total_params = 0

    teacher_params = dict(teacher.named_parameters())

    for name, student_param in student.named_parameters():
        if name not in teacher_params:
            continue

        s = student_param.detach().float().cpu().reshape(-1)
        t = teacher_params[name].detach().float().cpu().reshape(-1)

        student_chunks.append(s)
        teacher_chunks.append(t)

        total_params += s.numel()

        if max_params is not None and total_params >= max_params:
            break

    if len(student_chunks) == 0:
        return {
            "student_norm": 0.0,
            "teacher_norm": 0.0,
            "relative_distance": float("inf"),
        }

    student_flat = torch.cat(student_chunks)
    teacher_flat = torch.cat(teacher_chunks)

    student_norm = student_flat.norm().item()
    teacher_norm = teacher_flat.norm().item()

    relative_distance = (
        (student_flat - teacher_flat).norm().item()
        / (student_norm + 1e-8)
    )

    return {
        "student_norm": float(student_norm),
        "teacher_norm": float(teacher_norm),
        "relative_distance": float(relative_distance),
    }


def assert_teacher_is_frozen(teacher: nn.Module) -> None:
    """
    Raises an error if any teacher parameter requires grad.
    """
    teacher = unwrap_model(teacher)

    trainable = [
        name for name, param in teacher.named_parameters()
        if param.requires_grad
    ]

    if len(trainable) > 0:
        raise RuntimeError(
            f"Teacher has trainable parameters. First examples: {trainable[:10]}"
        )