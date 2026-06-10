"""Collect per-step DINO metrics into structured dictionaries."""

from src.training.metrics import *  

# ============================================================
# Main collector for one DINO step
# ============================================================

@torch.no_grad()
def collect_dino_step_metrics(
    *,
    loss: Optional[Any] = None,
    dino_loss: Optional[Any] = None,
    gram_loss: Optional[Any] = None,
    extra_losses: Optional[Dict[str, Any]] = None,
    student_outputs: Optional[List[Dict[str, Any]]] = None,
    teacher_outputs: Optional[List[Dict[str, Any]]] = None,
    student_logits: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    teacher_logits: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    teacher_center: Optional[torch.Tensor] = None,
    student: Optional[nn.Module] = None,
    teacher: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler_values: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    include_feature_metrics: bool = True,
    include_param_distance: bool = False,
    include_memory: bool = True,
) -> Dict[str, float]:
    """
    Collect useful DINO training metrics from already-computed objects.

    This function does NOT:
        - run forward
        - compute loss
        - call backward
        - step optimizer
        - update teacher
    """
    metrics: Dict[str, float] = {}

    metrics.update(
        loss_metrics(
            loss=loss,
            dino_loss=dino_loss,
            gram_loss=gram_loss,
            extra_losses=extra_losses,
        )
    )

    if student_logits is not None or teacher_logits is not None:
        metrics.update(
            dino_softmax_metrics(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_temp=student_temp,
                teacher_temp=teacher_temp,
                teacher_center=teacher_center,
            )
        )

    if include_feature_metrics:
        metrics.update(
            dino_feature_metrics(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
            )
        )

    if student is not None:
        metrics["grad_norm"] = compute_grad_norm(student)
        metrics["student_param_norm"] = compute_param_norm(student, trainable_only=True)
        metrics["nonfinite_grad_params"] = float(count_nonfinite_grads(student))

    if student is not None and teacher is not None and include_param_distance:
        metrics.update(
            teacher_student_param_distance(
                student=student,
                teacher=teacher,
                max_params=1_000_000,
            )
        )

    if optimizer is not None:
        metrics.update(optimizer_metrics(optimizer))

    metrics.update(
        scheduler_value_metrics(scheduler_values)
    )

    if include_memory:
        metrics.update(
            cuda_memory_metrics(device=device)
        )

    return detach_float_dict(metrics)


DEFAULT_DINO_PRINT_KEYS = [
    "loss",
    "loss_dino",
    "loss_gram",
    "lr",
    "weight_decay",
    "teacher_momentum",
    "teacher_temp",
    "student_entropy_norm",
    "teacher_entropy_norm",
    "student_max_prob",
    "teacher_max_prob",
    "student_cls_dim_std_mean",
    "teacher_cls_dim_std_mean",
    "student_cls_mean_offdiag_cosine",
    "grad_norm",
    "cuda_mem_alloc_gb",
]


def format_metric_dict(
    metrics: Dict[str, Any],
    keys: Optional[List[str]] = None,
    precision: int = 4,
    skip_missing: bool = True,
) -> str:
    """
    Format metrics into a compact printable string.
    """
    if keys is None:
        keys = list(metrics.keys())

    parts = []

    for key in keys:
        if key not in metrics:
            if skip_missing:
                continue
            parts.append(f"{key}: NA")
            continue

        value = to_float(metrics[key])

        if value is None:
            continue

        if abs(value) >= 1e4 or (abs(value) < 1e-3 and value != 0.0):
            parts.append(f"{key}: {value:.{precision}e}")
        else:
            parts.append(f"{key}: {value:.{precision}f}")

    return " | ".join(parts)


def format_dino_train_print(
    epoch: int,
    step: int,
    total_steps: Optional[int],
    metrics: Dict[str, Any],
    keys: Optional[List[str]] = None,
    precision: int = 4,
) -> str:
    """
    Standard DINO training print string.
    """
    if keys is None:
        keys = DEFAULT_DINO_PRINT_KEYS

    if total_steps is None:
        header = f"Epoch {epoch} | Step {step}"
    else:
        header = f"Epoch {epoch} | Step {step}/{total_steps}"

    body = format_metric_dict(
        metrics=metrics,
        keys=keys,
        precision=precision,
        skip_missing=True,
    )

    if body:
        return f"{header} | {body}"

    return header
