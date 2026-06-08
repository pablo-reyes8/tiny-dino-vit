# ============================================================
# DINO METRICS UTILITIES
# Metrics for monitoring DINO training health
# ============================================================

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_float(x: Any) -> Optional[float]:
    """
    Safely convert tensors/numbers to Python float.
    """
    if x is None:
        return None

    if torch.is_tensor(x):
        if x.numel() == 0:
            return None
        return float(x.detach().float().mean().cpu().item())

    if isinstance(x, (int, float)):
        return float(x)

    return None


def is_finite_number(x: Any) -> bool:
    """
    Check whether a scalar/tensor is finite.
    """
    if x is None:
        return False

    if torch.is_tensor(x):
        return bool(torch.isfinite(x.detach()).all().item())

    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def detach_float_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a metrics dictionary into plain floats when possible.
    """
    out = {}

    for key, value in metrics.items():
        value_float = to_float(value)

        if value_float is not None:
            out[key] = value_float

    return out


class SmoothedValue:
    """
    Track a scalar with windowed and global averages.
    """

    def __init__(self, window_size: int = 20):
        self.deque = deque(maxlen=int(window_size))
        self.total = 0.0
        self.count = 0

    def update(self, value: Any, n: int = 1) -> None:
        value_float = to_float(value)
        if value_float is None or not math.isfinite(value_float):
            return

        n = max(1, int(n))
        self.deque.append(value_float)
        self.total += value_float * n
        self.count += n

    @property
    def median(self) -> float:
        if len(self.deque) == 0:
            return 0.0
        return float(torch.tensor(list(self.deque)).median().item())

    @property
    def avg(self) -> float:
        if len(self.deque) == 0:
            return 0.0
        return float(sum(self.deque) / len(self.deque))

    @property
    def global_avg(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.total / self.count)

    @property
    def latest(self) -> float:
        if len(self.deque) == 0:
            return 0.0
        return float(self.deque[-1])


class MetricLogger:
    """
    Lightweight metric accumulator used by the training loop.
    """

    def __init__(self, window_size: int = 20):
        self.meters = defaultdict(lambda: SmoothedValue(window_size=window_size))

    def update(self, n: int = 1, **metrics: Any) -> None:
        for key, value in metrics.items():
            self.meters[key].update(value, n=n)

    def get_global_avg_dict(self) -> Dict[str, float]:
        return {
            key: meter.global_avg
            for key, meter in self.meters.items()
        }

    def get_latest_dict(self) -> Dict[str, float]:
        return {
            key: meter.latest
            for key, meter in self.meters.items()
        }


def to_float(x: Any) -> Optional[float]:
    """
    Safely convert tensors/numbers to Python float.
    """
    if x is None:
        return None

    if torch.is_tensor(x):
        if x.numel() == 0:
            return None
        return float(x.detach().float().mean().cpu().item())

    if isinstance(x, (int, float)):
        return float(x)

    return None


def is_finite_number(x: Any) -> bool:
    """
    Check whether a scalar/tensor is finite.
    """
    if x is None:
        return False

    if torch.is_tensor(x):
        return bool(torch.isfinite(x.detach()).all().item())

    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def detach_float_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a metrics dictionary into plain floats when possible.
    """
    out = {}

    for key, value in metrics.items():
        value_float = to_float(value)

        if value_float is not None:
            out[key] = value_float

    return out

# ============================================================
# Probability / logits metrics
# ============================================================

@torch.no_grad()
def probability_stats(
    probs: torch.Tensor,
    prefix: str,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Statistics for probability distributions.

    Expected:
        probs: [B, K]

    Useful for DINO head outputs after softmax.
    """
    p = probs.detach().float().clamp_min(eps)

    entropy = -(p * p.log()).sum(dim=-1)
    max_prob = p.max(dim=-1).values

    num_classes = p.shape[-1]
    normalized_entropy = entropy / math.log(num_classes)

    return {
        f"{prefix}_entropy": entropy.mean().item(),
        f"{prefix}_entropy_norm": normalized_entropy.mean().item(),
        f"{prefix}_max_prob": max_prob.mean().item(),
        f"{prefix}_prob_std": p.std(unbiased=False).item(),
    }


@torch.no_grad()
def logits_stats(
    logits: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    """
    Diagnostics for logits before softmax.
    """
    z = logits.detach().float()

    return {
        f"{prefix}_logit_mean": z.mean().item(),
        f"{prefix}_logit_std": z.std(unbiased=False).item(),
        f"{prefix}_logit_abs_mean": z.abs().mean().item(),
        f"{prefix}_logit_max": z.max().item(),
        f"{prefix}_logit_min": z.min().item(),
    }


@torch.no_grad()
def dino_softmax_metrics(
    student_logits: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    teacher_logits: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    teacher_center: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute DINO distribution health metrics.

    This function does NOT compute loss.
    It only checks entropy, max probability and logits scale.

    Inputs can be:
        - Tensor [B, K]
        - list of Tensor [B, K]
    """
    metrics: Dict[str, float] = {}

    if student_logits is not None:
        if isinstance(student_logits, list):
            s_logits = torch.cat([x.detach() for x in student_logits], dim=0)
        else:
            s_logits = student_logits.detach()

        s_probs = F.softmax(s_logits.float() / student_temp, dim=-1)

        metrics.update(logits_stats(s_logits, prefix="student"))
        metrics.update(probability_stats(s_probs, prefix="student"))

    if teacher_logits is not None:
        if isinstance(teacher_logits, list):
            t_logits = torch.cat([x.detach() for x in teacher_logits], dim=0)
        else:
            t_logits = teacher_logits.detach()

        if teacher_center is not None:
            center = teacher_center.detach().to(
                device=t_logits.device,
                dtype=t_logits.dtype,
            )
            t_logits_centered = t_logits - center
        else:
            t_logits_centered = t_logits

        t_probs = F.softmax(t_logits_centered.float() / teacher_temp, dim=-1)

        metrics.update(logits_stats(t_logits, prefix="teacher"))
        metrics.update(probability_stats(t_probs, prefix="teacher"))

        if teacher_center is not None:
            metrics["teacher_center_norm"] = teacher_center.detach().float().norm().item()
            metrics["teacher_center_mean"] = teacher_center.detach().float().mean().item()
            metrics["teacher_center_std"] = teacher_center.detach().float().std(unbiased=False).item()

    return metrics

# ============================================================
# DINO feature metrics from model outputs
# ============================================================

@torch.no_grad()
def feature_stats(
    features: torch.Tensor,
    prefix: str,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Basic representation health stats for [B, D] or [B, N, D] features.
    """
    x = features.detach().float()

    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim != 2:
        x = x.reshape(x.shape[0], -1)

    if x.numel() == 0 or x.shape[0] == 0:
        return {}

    dim_std = x.std(dim=0, unbiased=False)
    norms = x.norm(dim=-1)

    metrics = {
        f"{prefix}_mean": x.mean().item(),
        f"{prefix}_std": x.std(unbiased=False).item(),
        f"{prefix}_abs_mean": x.abs().mean().item(),
        f"{prefix}_norm_mean": norms.mean().item(),
        f"{prefix}_norm_std": norms.std(unbiased=False).item(),
        f"{prefix}_dim_std_mean": dim_std.mean().item(),
        f"{prefix}_dim_std_min": dim_std.min().item(),
        f"{prefix}_collapsed_dim_frac": (dim_std < eps).float().mean().item(),
    }

    if x.shape[0] > 1:
        x_norm = F.normalize(x, dim=-1, eps=eps)
        cosine = x_norm @ x_norm.t()
        offdiag_mask = ~torch.eye(
            cosine.shape[0],
            dtype=torch.bool,
            device=cosine.device,
        )
        offdiag = cosine[offdiag_mask]
        metrics[f"{prefix}_mean_offdiag_cosine"] = offdiag.mean().item()
        metrics[f"{prefix}_abs_offdiag_cosine"] = offdiag.abs().mean().item()

    return metrics


@torch.no_grad()
def dino_feature_metrics(
    student_outputs: Optional[List[Dict[str, Any]]] = None,
    teacher_outputs: Optional[List[Dict[str, Any]]] = None,
    max_tokens_for_pairwise: int = 2048,
) -> Dict[str, float]:
    """
    Compute representation health metrics from DINO model outputs.

    Expected each output dict may contain:
        out["cls"]      -> [B, D]
        out["patches"]  -> [B, N, D]

    This function does NOT run forward.
    It only consumes already-computed outputs.
    """
    metrics: Dict[str, float] = {}

    if student_outputs is not None and len(student_outputs) > 0:
        student_cls = [
            out["cls"].detach()
            for out in student_outputs
            if "cls" in out and torch.is_tensor(out["cls"])
        ]

        if len(student_cls) > 0:
            student_cls = torch.cat(student_cls, dim=0)
            metrics.update(feature_stats(student_cls, prefix="student_cls"))

        student_patches = [
            out["patches"].detach()
            for out in student_outputs
            if "patches" in out and torch.is_tensor(out["patches"])
        ]

        if len(student_patches) > 0:
            patches = torch.cat(
                [p.reshape(-1, p.shape[-1]) for p in student_patches],
                dim=0,
            )

            if patches.shape[0] > max_tokens_for_pairwise:
                patches = patches[:max_tokens_for_pairwise]

            metrics.update(feature_stats(patches, prefix="student_patch"))

    if teacher_outputs is not None and len(teacher_outputs) > 0:
        teacher_cls = [
            out["cls"].detach()
            for out in teacher_outputs
            if "cls" in out and torch.is_tensor(out["cls"])
        ]

        if len(teacher_cls) > 0:
            teacher_cls = torch.cat(teacher_cls, dim=0)
            metrics.update(feature_stats(teacher_cls, prefix="teacher_cls"))

        teacher_patches = [
            out["patches"].detach()
            for out in teacher_outputs
            if "patches" in out and torch.is_tensor(out["patches"])
        ]

        if len(teacher_patches) > 0:
            patches = torch.cat(
                [p.reshape(-1, p.shape[-1]) for p in teacher_patches],
                dim=0,
            )

            if patches.shape[0] > max_tokens_for_pairwise:
                patches = patches[:max_tokens_for_pairwise]

            metrics.update(feature_stats(patches, prefix="teacher_patch"))

    return metrics


# ============================================================
#  Gradient / parameter metrics
# ============================================================

@torch.no_grad()
def compute_grad_norm(
    model: nn.Module,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> float:
    """
    Compute total gradient norm over parameters with gradients.
    """
    parameters = [
        p for p in model.parameters()
        if p.grad is not None
    ]

    if len(parameters) == 0:
        return 0.0

    device = parameters[0].grad.device

    norms = torch.stack([
        torch.norm(p.grad.detach(), norm_type).to(device)
        for p in parameters
    ])

    total_norm = torch.norm(norms, norm_type)

    if error_if_nonfinite and not torch.isfinite(total_norm):
        raise RuntimeError(f"Non-finite gradient norm: {total_norm}")

    return float(total_norm.item())


@torch.no_grad()
def compute_param_norm(
    model: nn.Module,
    norm_type: float = 2.0,
    trainable_only: bool = True,
) -> float:
    """
    Compute total parameter norm.
    """
    parameters = []

    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        parameters.append(p.detach())

    if len(parameters) == 0:
        return 0.0

    device = parameters[0].device

    norms = torch.stack([
        torch.norm(p, norm_type).to(device)
        for p in parameters
    ])

    total_norm = torch.norm(norms, norm_type)

    return float(total_norm.item())


@torch.no_grad()
def count_nonfinite_grads(model: nn.Module) -> int:
    """
    Count parameters whose gradients contain NaN or Inf.
    """
    count = 0

    for p in model.parameters():
        if p.grad is None:
            continue

        if not torch.isfinite(p.grad).all():
            count += 1

    return int(count)

# ============================================================
# Teacher-student distance metrics
# ============================================================

@torch.no_grad()
def teacher_student_param_distance(
    student: nn.Module,
    teacher: nn.Module,
    max_params: Optional[int] = 1_000_000,
) -> Dict[str, float]:
    """
    Approximate relative parameter distance between student and teacher.

    Useful for checking EMA behavior.

    If distance is exactly 0 for many steps, teacher may not be updating
    or student may not be learning.

    If distance explodes, EMA/update may be broken.
    """
    student_params = dict(student.named_parameters())
    teacher_params = dict(teacher.named_parameters())

    chunks_student = []
    chunks_teacher = []

    total = 0

    for name, s_param in student_params.items():
        if name not in teacher_params:
            continue

        s = s_param.detach().float().cpu().reshape(-1)
        t = teacher_params[name].detach().float().cpu().reshape(-1)

        chunks_student.append(s)
        chunks_teacher.append(t)

        total += s.numel()

        if max_params is not None and total >= max_params:
            break

    if len(chunks_student) == 0:
        return {
            "teacher_student_rel_dist": float("inf"),
            "student_param_norm_sample": 0.0,
            "teacher_param_norm_sample": 0.0,
        }

    s_flat = torch.cat(chunks_student)
    t_flat = torch.cat(chunks_teacher)

    s_norm = s_flat.norm().item()
    t_norm = t_flat.norm().item()

    rel_dist = (s_flat - t_flat).norm().item() / (s_norm + 1e-8)

    return {
        "teacher_student_rel_dist": float(rel_dist),
        "student_param_norm_sample": float(s_norm),
        "teacher_param_norm_sample": float(t_norm),
    }

def cuda_memory_metrics(
    device: Optional[Union[str, torch.device]] = None,
    prefix: str = "cuda",
) -> Dict[str, float]:
    """
    CUDA memory diagnostics in GB.

    Safe on CPU: returns empty dict.
    """
    if not torch.cuda.is_available():
        return {}

    if device is None:
        device = torch.cuda.current_device()

    device = torch.device(device)

    if device.type != "cuda":
        return {}

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    return {
        f"{prefix}_mem_alloc_gb": float(allocated),
        f"{prefix}_mem_reserved_gb": float(reserved),
        f"{prefix}_max_mem_alloc_gb": float(max_allocated),
        f"{prefix}_max_mem_reserved_gb": float(max_reserved),
    }


def reset_cuda_peak_memory(device: Optional[Union[str, torch.device]] = None) -> None:
    """
    Reset CUDA peak memory stats.
    """
    if not torch.cuda.is_available():
        return

    if device is None:
        torch.cuda.reset_peak_memory_stats()
    else:
        torch.cuda.reset_peak_memory_stats(device)


def loss_metrics(
    loss: Optional[Any] = None,
    dino_loss: Optional[Any] = None,
    gram_loss: Optional[Any] = None,
    extra_losses: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Convert loss tensors into printable metric scalars.

    This function does NOT compute losses.
    It only receives already-computed losses.
    """
    metrics: Dict[str, float] = {}

    if loss is not None:
        metrics["loss"] = to_float(loss)

    if dino_loss is not None:
        metrics["loss_dino"] = to_float(dino_loss)

    if gram_loss is not None:
        metrics["loss_gram"] = to_float(gram_loss)

    if extra_losses is not None:
        for key, value in extra_losses.items():
            metrics[f"loss_{key}"] = to_float(value)

    return {
        key: value
        for key, value in metrics.items()
        if value is not None
    }


def optimizer_metrics(optimizer: Any) -> Dict[str, float]:
    """
    Extract LR and weight decay stats from a torch optimizer-like object.
    """
    if optimizer is None:
        return {}

    if hasattr(optimizer, "adamw") and hasattr(optimizer, "muon"):
        metrics: Dict[str, float] = {}
        metrics.update(optimizer_metrics(optimizer.adamw))
        muon_metrics = optimizer_metrics(optimizer.muon)
        metrics.update({f"muon_{k}": v for k, v in muon_metrics.items()})
        return metrics

    param_groups = getattr(optimizer, "param_groups", None)
    if not param_groups:
        return {}

    lrs = [float(group.get("lr", 0.0)) for group in param_groups]
    weight_decays = [float(group.get("weight_decay", 0.0)) for group in param_groups]

    metrics = {
        "lr": lrs[0],
        "lr_mean": float(sum(lrs) / len(lrs)),
        "lr_min": float(min(lrs)),
        "lr_max": float(max(lrs)),
        "weight_decay": weight_decays[0],
        "weight_decay_mean": float(sum(weight_decays) / len(weight_decays)),
        "weight_decay_min": float(min(weight_decays)),
        "weight_decay_max": float(max(weight_decays)),
    }

    return metrics


def scheduler_value_metrics(scheduler_values: Optional[Any]) -> Dict[str, float]:
    """
    Convert scheduler dataclass/dict/object values into flat float metrics.
    """
    if scheduler_values is None:
        return {}

    if isinstance(scheduler_values, dict):
        items = scheduler_values.items()
    else:
        names = [
            "lr",
            "weight_decay",
            "teacher_momentum",
            "teacher_temp",
            "step",
            "scheduler_step",
        ]
        items = (
            (name, getattr(scheduler_values, name))
            for name in names
            if hasattr(scheduler_values, name)
        )

    metrics: Dict[str, float] = {}
    for key, value in items:
        value_float = to_float(value)
        if value_float is not None:
            metrics[str(key)] = value_float

    return metrics


