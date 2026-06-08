# ============================================================
# TRAIN ONE EPOCH FOR DINO
# ============================================================

from __future__ import annotations

import gc
import time
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn

from src.training.main_metrics import * 
from src.training.autocast import *  
from src.training.ema import * 



def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _extract_logits_from_output(
    output: Dict[str, Any],
    head: Optional[nn.Module] = None,
    token_key: str = "cls",
) -> torch.Tensor:
    """
    Extract DINO logits from a model output.

    Priority:
        1. output["logits"]
        2. head(output[token_key])
        3. output[token_key] if no head is provided

    This lets the trainer work with:
        - backbone + separate DINO head
        - model that already returns logits
        - debugging mode using CLS directly
    """
    if "logits" in output and torch.is_tensor(output["logits"]):
        return output["logits"]

    if token_key not in output:
        raise KeyError(
            f"Output does not contain '{token_key}' or 'logits'. "
            f"Available keys: {list(output.keys())}"
        )

    token = output[token_key]

    if head is not None:
        return head(token)

    return token


def _call_dino_model(
    model: nn.Module,
    crop: torch.Tensor,
    head: Optional[nn.Module] = None,
    return_all_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Forward for one crop.

    Expected model interface:
        model(
            crop,
            return_features=True,
            return_all_tokens=True,
        )

    Returns dict containing at least:
        "cls"
        "patches"
        "grid_size"

    Adds:
        "logits"
    """
    output = model(
        crop,
        return_features=True,
        return_all_tokens=return_all_tokens,
        return_attention=False,
    )

    if not isinstance(output, dict):
        raise TypeError(
            f"DINO model must return dict when return_features=True. Got {type(output)}."
        )

    logits = _extract_logits_from_output(
        output=output,
        head=head,
        token_key="cls",
    )

    output["logits"] = logits

    return output


def _normalize_loss_output(loss_output: Any) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Accepts:
        loss_tensor
        {"loss": tensor, ...}

    Returns:
        loss, extra_loss_dict
    """
    if torch.is_tensor(loss_output):
        return loss_output, {}

    if isinstance(loss_output, dict):
        if "loss" not in loss_output:
            raise KeyError("Loss dict must contain key 'loss'.")

        loss = loss_output["loss"]

        extras = {
            k: v
            for k, v in loss_output.items()
            if k != "loss" and torch.is_tensor(v)
        }

        return loss, extras

    raise TypeError(
        f"Loss function must return Tensor or dict with key 'loss'. Got {type(loss_output)}."
    )


def _call_dino_loss(
    dino_loss_fn,
    *,
    student_outputs: List[Dict[str, Any]],
    teacher_outputs: List[Dict[str, Any]],
    student_temp: float,
    teacher_temp: float,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Flexible DINO loss caller.

    Preferred loss signature:
        dino_loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_temp=student_temp,
            teacher_temp=teacher_temp,
        )

    where:
        student_logits: list[Tensor]
        teacher_logits: list[Tensor]
    """
    student_logits = [out["logits"] for out in student_outputs]
    teacher_logits = [out["logits"] for out in teacher_outputs]

    try:
        loss_output = dino_loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_temp=student_temp,
            teacher_temp=teacher_temp,
        )
    except TypeError:
        loss_output = dino_loss_fn(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            student_temp=student_temp,
            teacher_temp=teacher_temp,
        )

    return _normalize_loss_output(loss_output)


def _call_gram_loss(
    gram_loss_fn,
    *,
    student_outputs: List[Dict[str, Any]],
    teacher_outputs: List[Dict[str, Any]],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Flexible Gram loss caller.

    Preferred signature:
        gram_loss_fn(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
        )

    It should return Tensor or {"loss": Tensor, ...}.
    """
    loss_output = gram_loss_fn(
        student_outputs=student_outputs,
        teacher_outputs=teacher_outputs,
    )

    return _normalize_loss_output(loss_output)


def _optimizer_unscale_grads(
    scaler: Optional[Any],
    optimizer: Any,
) -> None:
    if scaler is None:
        return

    if hasattr(optimizer, "muon") and hasattr(optimizer, "adamw"):
        scaler.unscale_(optimizer.muon)
        scaler.unscale_(optimizer.adamw)
    else:
        scaler.unscale_(optimizer)


def _optimizer_step_with_scaler(
    scaler: Any,
    optimizer: Any,
) -> bool:
    """
    Returns whether optimizer step likely happened.
    """
    old_scale = scaler.get_scale()

    if hasattr(optimizer, "muon") and hasattr(optimizer, "adamw"):
        scaler.step(optimizer.muon)
        scaler.step(optimizer.adamw)
    else:
        scaler.step(optimizer)

    scaler.update()

    new_scale = scaler.get_scale()

    return new_scale >= old_scale


def _move_module_to_device(
    module: Optional[nn.Module],
    device: torch.device | str,
) -> Optional[nn.Module]:
    if module is None:
        return None

    module.to(device)
    return module


def _maybe_empty_cuda_cache(
    enabled: bool = False,
) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_module_train_mode(
    module: Optional[nn.Module],
    train: bool,
) -> None:
    if module is None:
        return

    if train:
        module.train()
    else:
        module.eval()


def _freeze_module(
    module: Optional[nn.Module],
) -> None:
    if module is None:
        return

    module.eval()

    for p in module.parameters():
        p.requires_grad = False


def _validate_gram_pairs(
    student_outputs: List[Dict[str, Any]],
    teacher_outputs: List[Dict[str, Any]],
) -> None:
    """
    Sanity check for Gram loss.

    For direct patch-token Gram matching, student and teacher outputs must have:
        - same number of views
        - same patch grid per paired view
        - "patches" key available
    """
    if len(student_outputs) != len(teacher_outputs):
        raise ValueError(
            f"Gram loss expects same number of student/teacher outputs. "
            f"Got {len(student_outputs)} student and {len(teacher_outputs)} teacher."
        )

    for i, (s_out, t_out) in enumerate(zip(student_outputs, teacher_outputs)):
        if "patches" not in s_out:
            raise KeyError(f"student_outputs[{i}] does not contain key 'patches'.")

        if "patches" not in t_out:
            raise KeyError(f"teacher_outputs[{i}] does not contain key 'patches'.")

        if s_out["patches"].shape[1] != t_out["patches"].shape[1]:
            raise ValueError(
                f"Gram patch-token mismatch at pair {i}: "
                f"student patches={s_out['patches'].shape}, "
                f"teacher patches={t_out['patches'].shape}. "
                f"Use gram_on_global_crops_only=True or make local/global crop sizes compatible."
            )

        if "grid_size" in s_out and "grid_size" in t_out:
            if tuple(s_out["grid_size"]) != tuple(t_out["grid_size"]):
                raise ValueError(
                    f"Gram grid mismatch at pair {i}: "
                    f"student grid={s_out['grid_size']}, "
                    f"teacher grid={t_out['grid_size']}."
                )


def train_one_epoch(
    *,
    bundle_dino: Dict[str, Any],
    dataloader,
    optimizer: torch.optim.Optimizer,
    dino_loss_fn: Any,
    precision: Dict[str, Any],
    scheduler: Optional[Any] = None,
    gram_loss_fn: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    grad_clip: Optional[float] = 1.0,
    grad_accum_steps: int = 1,
    max_batches: Optional[int] = None,
    log_every: int = 10,
    metrics_every: int = 10,
    num_global_crops: int = 2,
    student_temp: float = 0.1,
    default_teacher_temp: float = 0.04,
    default_teacher_momentum: float = 0.996,
    use_gram_loss: bool = False,
    gram_loss_weight: float = 0.05,
    gram_on_global_crops_only: bool = True,
    validate_gram_shapes: bool = True,
    teacher_offload: bool = False,
    exclusive_gpu_model: bool = False,
    empty_cache_after_offload: bool = False,
    log_mem: bool = True,
    include_feature_metrics: bool = True,
    include_param_distance: bool = False,
    on_oom: str = "skip",
    is_main_process: bool = True,
) -> Tuple[Dict[str, float], int]:
    """
    Train one epoch for DINO.

    Expected bundle_dino:
        {
            "student": DINOModel,
            "teacher": DINOModel,
            "config": CFG,
            "model_info": metadata,
            "device": device,
        }

    DINO logic:
        - Teacher sees only global crops.
        - Student sees global + local crops.
        - Teacher is frozen and updated by EMA after optimizer.step().
        - Optional Gram loss uses patch tokens.

    Memory modes
    ------------
    teacher_offload=False:
        student and teacher live on GPU.
        Fastest, highest VRAM.

    teacher_offload=True:
        teacher lives on CPU outside teacher forward.
        For teacher forward:
            teacher CPU -> GPU -> forward -> CPU.
        Student remains on GPU.

    teacher_offload=True + exclusive_gpu_model=True:
        tries to keep only one ViT on GPU at a time:
            student GPU -> CPU
            teacher CPU -> GPU
            teacher forward
            teacher GPU -> CPU
            student CPU -> GPU
            student forward/backward
        This is slower but can reduce peak model-weight VRAM.

    Gram loss
    ---------
    By default, Gram loss is computed only on global crops:
        student_outputs[:num_global_crops] vs teacher_outputs

    This avoids patch-grid mismatches between local crops and global teacher crops.
    """

    # --------------------------------------------------------
    # 0. Validate arguments
    # --------------------------------------------------------

    if use_gram_loss and gram_loss_fn is None:
        raise ValueError("use_gram_loss=True but gram_loss_fn is None.")

    if exclusive_gpu_model and not teacher_offload:
        teacher_offload = True

    if exclusive_gpu_model and grad_accum_steps > 1 and is_main_process:
        print(
            "[WARN] exclusive_gpu_model=True with grad_accum_steps > 1. "
            "This can work, but it repeatedly moves accumulated gradients CPU<->GPU "
            "and will be slow."
        )

    # --------------------------------------------------------
    # 1. Resolve models / device
    # --------------------------------------------------------

    student = bundle_dino["student"]
    teacher = bundle_dino["teacher"]

    # Legacy support if you still keep external heads in some experiment.
    student_head = bundle_dino.get("student_head", None)
    teacher_head = bundle_dino.get("teacher_head", None)

    device = bundle_dino.get("device", precision.get("device", "cuda"))
    device = torch.device(device)

    cpu_device = torch.device("cpu")

    # Student starts on GPU.
    _move_module_to_device(student, device)
    _move_module_to_device(student_head, device)

    # Teacher either starts on GPU or CPU depending on offload.
    if teacher_offload:
        _move_module_to_device(teacher, cpu_device)
        _move_module_to_device(teacher_head, cpu_device)
    else:
        _move_module_to_device(teacher, device)
        _move_module_to_device(teacher_head, device)

    student.train()
    _set_module_train_mode(student_head, train=True)

    _freeze_module(teacher)
    _freeze_module(teacher_head)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    scaler = precision.get("scaler", None)

    metric_logger = MetricLogger(window_size=20)

    n_seen_batches = 0
    n_seen_samples = 0
    n_optimizer_steps = 0
    n_skipped_oom = 0

    t_epoch = time.time()

    if is_main_process:
        print(f"\nDINO train epoch {epoch}")
        print("─" * 120)

        if teacher_offload:
            print(
                f"[MEMORY] teacher_offload=True | "
                f"exclusive_gpu_model={exclusive_gpu_model} | "
                f"empty_cache_after_offload={empty_cache_after_offload}"
            )

    # --------------------------------------------------------
    # 3. Loop
    # --------------------------------------------------------

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        teacher_outputs = None
        student_outputs = None

        try:
            t0 = time.perf_counter()

            crop_dict = prepare_dino_batch_for_device(
                batch=batch,
                device=device,
                num_global_crops=num_global_crops,
                non_blocking=True,
            )

            teacher_crops = crop_dict["teacher_crops"]
            student_crops = crop_dict["student_crops"]

            batch_size = int(student_crops[0].shape[0])
            n_seen_samples += batch_size

            step_now = ((batch_idx + 1) % grad_accum_steps) == 0

            # ------------------------------------------------
            # Scheduler values for current step
            # ------------------------------------------------

            if scheduler is not None and hasattr(scheduler, "get_values"):
                scheduler_values = scheduler.get_values()

                teacher_temp = float(
                    getattr(
                        scheduler_values,
                        "teacher_temp",
                        default_teacher_temp,
                    )
                )

                teacher_momentum = float(
                    getattr(
                        scheduler_values,
                        "teacher_momentum",
                        default_teacher_momentum,
                    )
                )
            else:
                scheduler_values = None
                teacher_temp = float(default_teacher_temp)
                teacher_momentum = float(default_teacher_momentum)

            # ------------------------------------------------
            # Teacher forward: no grad, global crops only
            # ------------------------------------------------

            if exclusive_gpu_model:
                _move_module_to_device(student, cpu_device)
                _move_module_to_device(student_head, cpu_device)
                _maybe_empty_cuda_cache(enabled=empty_cache_after_offload)

            if teacher_offload:
                _move_module_to_device(teacher, device)
                _move_module_to_device(teacher_head, device)

            _freeze_module(teacher)
            _freeze_module(teacher_head)

            with torch.no_grad():
                with autocast_ctx(
                    device=device,
                    enabled=precision["amp_enabled"],
                    amp_dtype=precision["amp_dtype_requested"],
                    cache_enabled=precision["cache_enabled"],
                    fallback_bf16_to_fp16=precision["fallback_bf16_to_fp16"],
                ):
                    teacher_outputs = [
                        _call_dino_model(
                            model=teacher,
                            crop=crop,
                            head=teacher_head,
                            return_all_tokens=True,
                        )
                        for crop in teacher_crops
                    ]

            if teacher_offload:
                _move_module_to_device(teacher, cpu_device)
                _move_module_to_device(teacher_head, cpu_device)
                _maybe_empty_cuda_cache(enabled=empty_cache_after_offload)

            if exclusive_gpu_model:
                _move_module_to_device(student, device)
                _move_module_to_device(student_head, device)

            student.train()
            _set_module_train_mode(student_head, train=True)

            # ------------------------------------------------
            # Student forward: grad, global + local crops
            # ------------------------------------------------

            with autocast_ctx(
                device=device,
                enabled=precision["amp_enabled"],
                amp_dtype=precision["amp_dtype_requested"],
                cache_enabled=precision["cache_enabled"],
                fallback_bf16_to_fp16=precision["fallback_bf16_to_fp16"],
            ):
                student_outputs = [
                    _call_dino_model(
                        model=student,
                        crop=crop,
                        head=student_head,
                        return_all_tokens=True,
                    )
                    for crop in student_crops
                ]

                loss_dino, dino_extra_losses = _call_dino_loss(
                    dino_loss_fn=dino_loss_fn,
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    student_temp=student_temp,
                    teacher_temp=teacher_temp,
                )

                loss = loss_dino
                loss_gram = None
                gram_extra_losses = {}

                # --------------------------------------------
                # Optional DINOv3-style Gram loss
                # --------------------------------------------

                if use_gram_loss:
                    if gram_on_global_crops_only:
                        gram_student_outputs = student_outputs[:num_global_crops]
                        gram_teacher_outputs = teacher_outputs
                    else:
                        gram_student_outputs = student_outputs
                        gram_teacher_outputs = teacher_outputs

                    if validate_gram_shapes:
                        _validate_gram_pairs(
                            student_outputs=gram_student_outputs,
                            teacher_outputs=gram_teacher_outputs,
                        )

                    loss_gram, gram_extra_losses = _call_gram_loss(
                        gram_loss_fn=gram_loss_fn,
                        student_outputs=gram_student_outputs,
                        teacher_outputs=gram_teacher_outputs,
                    )

                    loss = loss + float(gram_loss_weight) * loss_gram

                loss_for_backward = loss / grad_accum_steps

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            if scaler is not None:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            optimizer_step_happened = False
            grad_norm_value = None

            # ------------------------------------------------
            # Optimizer step
            # ------------------------------------------------

            if step_now:
                if scaler is not None:
                    _optimizer_unscale_grads(scaler, optimizer)

                if grad_clip is not None and grad_clip > 0:
                    trainable_params = list(student.parameters())

                    if student_head is not None:
                        trainable_params += list(student_head.parameters())

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=float(grad_clip),
                    )

                    grad_norm_value = float(
                        grad_norm.detach().float().cpu().item()
                    )

                if scaler is not None:
                    optimizer_step_happened = _optimizer_step_with_scaler(
                        scaler=scaler,
                        optimizer=optimizer,
                    )
                else:
                    optimizer.step()
                    optimizer_step_happened = True

                if optimizer_step_happened:
                    # Advance schedules after optimizer step.
                    if scheduler is not None and hasattr(scheduler, "step"):
                        scheduler.step()

                    # ----------------------------------------
                    # EMA update:
                    # If teacher_offload=True, teacher may remain on CPU.
                    # update_teacher_ema must support cross-device copy:
                    # student GPU -> teacher CPU.
                    # ----------------------------------------

                    update_teacher_ema(
                        student=student,
                        teacher=teacher,
                        momentum=teacher_momentum,
                        update_buffers=True,
                    )

                    if student_head is not None and teacher_head is not None:
                        update_teacher_ema(
                            student=student_head,
                            teacher=teacher_head,
                            momentum=teacher_momentum,
                            update_buffers=True,
                        )

                    global_step += 1
                    n_optimizer_steps += 1

                optimizer.zero_grad(set_to_none=True)

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            n_seen_batches += 1

            should_collect_metrics = (
                metrics_every is not None
                and metrics_every > 0
                and (
                    batch_idx == 0
                    or (
                        step_now
                        and optimizer_step_happened
                        and global_step % metrics_every == 0
                    )
                )
            )

            should_log = (
                log_every is not None
                and log_every > 0
                and step_now
                and optimizer_step_happened
                and global_step % log_every == 0
            )

            extra_losses = {}
            extra_losses.update(
                {f"dino_{k}": v for k, v in dino_extra_losses.items()}
            )
            extra_losses.update(
                {f"gram_{k}": v for k, v in gram_extra_losses.items()}
            )

            if should_collect_metrics or should_log:
                student_logits = [out["logits"] for out in student_outputs]
                teacher_logits = [out["logits"] for out in teacher_outputs]

                metrics = collect_dino_step_metrics(
                    loss=loss,
                    dino_loss=loss_dino,
                    gram_loss=loss_gram,
                    extra_losses=extra_losses,
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    student_temp=student_temp,
                    teacher_temp=teacher_temp,
                    teacher_center=getattr(dino_loss_fn, "center", None),
                    student=student,
                    teacher=teacher,
                    optimizer=optimizer,
                    scheduler_values=scheduler_values,
                    device=device,
                    include_feature_metrics=include_feature_metrics,
                    include_param_distance=include_param_distance,
                    include_memory=log_mem,
                )
            else:
                metrics = {
                    "loss": float(loss.detach().float().cpu().item()),
                    "loss_dino": float(loss_dino.detach().float().cpu().item()),
                }

                if loss_gram is not None:
                    metrics["loss_gram"] = float(
                        loss_gram.detach().float().cpu().item()
                    )

                if grad_norm_value is not None:
                    metrics["grad_norm"] = float(grad_norm_value)

            if grad_norm_value is not None:
                metrics["grad_norm"] = float(grad_norm_value)

            metrics["batch_time_ms"] = float(
                (time.perf_counter() - t0) * 1000.0
            )

            metrics["teacher_offload"] = float(bool(teacher_offload))
            metrics["exclusive_gpu_model"] = float(bool(exclusive_gpu_model))

            metric_logger.update(n=batch_size, **metrics)

            if should_log and is_main_process:
                print(
                    format_dino_train_print(
                        epoch=epoch,
                        step=global_step,
                        total_steps=None,
                        metrics=metrics,
                    )
                )

            # ------------------------------------------------
            # Explicit cleanup of references
            # ------------------------------------------------

            del teacher_outputs
            del student_outputs

        except RuntimeError as e:
            is_oom = "out of memory" in str(e).lower()

            if is_oom and on_oom == "skip":
                n_skipped_oom += 1

                try:
                    del teacher_outputs
                    del student_outputs
                except Exception:
                    pass

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                optimizer.zero_grad(set_to_none=True)

                # Restore expected placement after OOM.
                _move_module_to_device(student, device)
                _set_module_train_mode(student, train=True)

                if teacher_offload:
                    _move_module_to_device(teacher, cpu_device)
                else:
                    _move_module_to_device(teacher, device)

                _freeze_module(teacher)

                if is_main_process:
                    print(
                        f"[WARN][OOM] Batch {batch_idx} skipped. "
                        f"Skipped OOM so far: {n_skipped_oom}"
                    )

                continue

            raise

    # --------------------------------------------------------
    #  End-of-epoch placement
    # --------------------------------------------------------

    _move_module_to_device(student, device)

    if teacher_offload:
        _move_module_to_device(teacher, cpu_device)
    else:
        _move_module_to_device(teacher, device)

    _freeze_module(teacher)

    bundle_dino["student"] = student
    bundle_dino["teacher"] = teacher

    # --------------------------------------------------------
    # Epoch stats
    # --------------------------------------------------------

    epoch_stats = metric_logger.get_global_avg_dict()

    epoch_stats["n_seen_batches"] = float(n_seen_batches)
    epoch_stats["n_seen_samples"] = float(n_seen_samples)
    epoch_stats["n_optimizer_steps"] = float(n_optimizer_steps)
    epoch_stats["n_skipped_oom"] = float(n_skipped_oom)
    epoch_stats["epoch_time_sec"] = float(time.time() - t_epoch)

    return epoch_stats, global_step