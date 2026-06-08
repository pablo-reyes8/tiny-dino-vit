
from src.training.high_level_training_helpers import * 
from src.training.seed import * 
from src.training.schedulers import * 
from src.training.train_one_epoch import * 
from src.training.chekpoints import * 
from src.model.loss import DINOGramLoss

def train_dino(
    *,
    bundle_dino: Dict[str, Any],
    train_loader,

    # Losses
    dino_loss_fn: Any,
    gram_loss_fn: Optional[Any] = None,

    # Device / precision
    seed: int = 42,
    deterministic: bool = False,
    device: str = "auto",
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    fallback_bf16_to_fp16: bool = True,

    # Optimizer
    optimizer: Optional[torch.optim.Optimizer] = None,
    optimizer_type: str = "adamw",
    learning_rate: float = 5e-4,
    min_learning_rate: float = 1e-6,
    weight_decay: float = 0.04,
    weight_decay_end: float = 0.4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,

    # Scheduler
    scheduler: Optional[Any] = None,
    total_steps: Optional[int] = None,
    warmup_epochs: int = 10,
    warmup_steps: Optional[int] = None,
    start_warmup_lr: float = 0.0,

    # DINO schedules
    teacher_momentum_start: float = 0.996,
    teacher_momentum_end: float = 1.0,
    warmup_teacher_temp: float = 0.04,
    teacher_temp: float = 0.07,
    warmup_teacher_temp_epochs: int = 30,

    # DINO temperatures / crops
    student_temp: float = 0.1,
    num_global_crops: int = 2,

    # Gram loss
    use_gram_loss: bool = False,
    gram_loss_weight: float = 0.05,

    # Teacher memory strategy
    teacher_offload: bool = False,
    empty_cache_after_teacher_offload: bool = False,

    # Training
    epochs: int = 100,
    start_epoch: int = 0,
    global_step: int = 0,
    grad_clip: Optional[float] = 1.0,
    grad_accum_steps: int = 1,
    max_batches_per_epoch: Optional[int] = None,
    log_every: int = 20,
    metrics_every: int = 20,
    log_mem: bool = True,
    include_feature_metrics: bool = True,
    include_param_distance: bool = False,
    on_oom: str = "skip",

    # Checkpoints
    ckpt_dir: str = "checkpoints/dino_tinyimagenet",
    run_name: str = "dino_tinyvit",
    save_every: int = 1,
    save_last: bool = True,
    keep_last_n_checkpoints: int = 3,
    monitor_name: str = "loss",
    monitor_mode: str = "min",
    best_metric: Optional[float] = None,
    resume_path: Optional[str] = None,
    strict_resume: bool = True,
    restore_rng_state: bool = False,

    # Metrics / mirroring
    metrics_jsonl_name: str = "metrics.jsonl",
    mirror_dir: Optional[str] = None,
    copy_fixed_to_mirror: bool = True,
    fixed_mirror_ckpt_name: str = "latest_dino.pt",
    fixed_mirror_metrics_name: str = "metrics_dino.jsonl",

    # Metadata
    config: Optional[Dict[str, Any]] = None,
    is_main_process: bool = True,
) -> Dict[str, Any]:
    """
    Full DINO training orchestration.

    Responsibilities:
        - seed / device / precision
        - optimizer creation
        - DINO schedulers
        - resume
        - train_one_epoch
        - checkpointing
        - metrics JSONL
        - optional teacher CPU offload

    Expected bundle_dino:
        {
            "student": DINOModel,
            "teacher": DINOModel,
            "config": CFG,
            "model_info": ...,
            "device": ...
        }
    """

    # ========================================================
    # Validate
    # ========================================================

    if optimizer_type not in {"adamw"}:
        raise ValueError(
            f"For now optimizer_type must be 'adamw', got {optimizer_type}."
        )

    if monitor_mode not in {"min", "max"}:
        raise ValueError(
            f"monitor_mode must be 'min' or 'max', got {monitor_mode}."
        )

    if use_gram_loss and gram_loss_fn is None:
        gram_loss_fn = DINOGramLoss()

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dir = str(ckpt_dir)
    metrics_path = Path(ckpt_dir) / metrics_jsonl_name

    # ========================================================
    # Seed / precision
    # ========================================================

    set_seed(seed, deterministic=deterministic)

    precision = setup_device_and_precision(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        fallback_bf16_to_fp16=fallback_bf16_to_fp16,
    )

    device_obj = precision["device"]

    bundle_dino["device"] = device_obj

    student = bundle_dino["student"]
    teacher = bundle_dino["teacher"]

    student = student.to(device_obj)

    if teacher_offload:
        teacher = teacher.to("cpu")
    else:
        teacher = teacher.to(device_obj)

    bundle_dino["student"] = student
    bundle_dino["teacher"] = teacher

    # ========================================================
    # Freeze teacher
    # ========================================================

    freeze_model(teacher)
    teacher.eval()

    # ========================================================
    # Optimizer
    # ========================================================

    if optimizer is None:
        optimizer, opt_info = build_dino_optimizer(
            student=student,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    else:
        opt_info = {
            "optimizer_type": "external",
            "num_param_groups": len(optimizer.param_groups),
        }

    # ========================================================
    # Total steps / schedulers
    # ========================================================

    if total_steps is None:
        steps_per_epoch = len(train_loader)

        if max_batches_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, int(max_batches_per_epoch))

        optim_steps_per_epoch = math.ceil(
            steps_per_epoch / max(1, int(grad_accum_steps))
        )

        total_steps = max(1, optim_steps_per_epoch * int(epochs))

    if warmup_steps is None:
        steps_per_epoch_for_warmup = len(train_loader)

        if max_batches_per_epoch is not None:
            steps_per_epoch_for_warmup = min(
                steps_per_epoch_for_warmup,
                int(max_batches_per_epoch),
            )

        optim_steps_per_epoch_for_warmup = math.ceil(
            steps_per_epoch_for_warmup / max(1, int(grad_accum_steps))
        )

        warmup_steps = int(warmup_epochs) * optim_steps_per_epoch_for_warmup

    if scheduler is None:
        CFG_scheduler = {
            "train": {
                "epochs": epochs,
                "steps_per_epoch": math.ceil(
                    min(len(train_loader), max_batches_per_epoch or len(train_loader))
                    / max(1, int(grad_accum_steps))
                ),
            },
            "optim": {
                "lr": learning_rate,
                "min_lr": min_learning_rate,
                "warmup_epochs": warmup_epochs,
                "start_warmup_lr": start_warmup_lr,
                "weight_decay_start": weight_decay,
                "weight_decay_end": weight_decay_end,
            },
            "dino": {
                "teacher_momentum_start": teacher_momentum_start,
                "teacher_momentum_end": teacher_momentum_end,
                "warmup_teacher_temp": warmup_teacher_temp,
                "teacher_temp": teacher_temp,
                "warmup_teacher_temp_epochs": warmup_teacher_temp_epochs,
            },
        }

        scheduler = build_dino_schedulers(
            optimizer=optimizer,
            CFG=CFG_scheduler,
        )

    # ========================================================
    # Resume
    # ========================================================

    if resume_path is not None and os.path.exists(resume_path):
        if is_main_process:
            dino_section("Resume checkpoint")

        state = load_dino_checkpoint(
            checkpoint_path=resume_path,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=precision.get("scaler", None),
            map_location="cpu",
            strict_student=strict_resume,
            strict_teacher=strict_resume,
            load_student=True,
            load_teacher=True,
            load_optimizer=True,
            load_scheduler=True,
            load_scaler=True,
            load_rng_state=restore_rng_state,
        )

        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["step"])
        best_metric = state["best_metric"]

        student = student.to(device_obj)

        if teacher_offload:
            teacher = teacher.to("cpu")
        else:
            teacher = teacher.to(device_obj)

        bundle_dino["student"] = student
        bundle_dino["teacher"] = teacher

        if is_main_process:
            print(f"[RESUME] path={resume_path}")
            print(
                f"[RESUME] start_epoch={start_epoch} | "
                f"global_step={global_step} | best_metric={best_metric}"
            )
            print(dino_rule())

    # ========================================================
    # Header
    # ========================================================

    if is_main_process:
        print_dino_run_header(
            run_name=run_name,
            bundle_dino=bundle_dino,
            device=device_obj,
            precision=precision,
            optimizer_type=optimizer_type,
            epochs=epochs,
            start_epoch=start_epoch,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            monitor_name=monitor_name,
            monitor_mode=monitor_mode,
            best_metric=best_metric,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
            use_gram_loss=use_gram_loss,
            gram_loss_weight=gram_loss_weight,
            teacher_offload=teacher_offload,
            ckpt_dir=ckpt_dir,
        )

    # ========================================================
    # Epoch loop
    # ========================================================

    total_time = 0.0
    train_stats = None
    combined_metrics = {}

    for epoch in range(start_epoch, epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if is_main_process:
            dino_title(f"Epoch {epoch:03d}/{epochs - 1:03d}", ch="─")

        t0 = time.time()

        # ----------------------------------------------------
        # Train one epoch
        # ----------------------------------------------------

        train_stats, global_step = train_one_epoch(
            bundle_dino=bundle_dino,
            dataloader=train_loader,
            optimizer=optimizer,
            dino_loss_fn=dino_loss_fn,
            gram_loss_fn=gram_loss_fn,
            precision=precision,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
            max_batches=max_batches_per_epoch,
            log_every=log_every,
            metrics_every=metrics_every,
            num_global_crops=num_global_crops,
            student_temp=student_temp,
            default_teacher_temp=teacher_temp,
            default_teacher_momentum=teacher_momentum_start,
            use_gram_loss=use_gram_loss,
            gram_loss_weight=gram_loss_weight,
            log_mem=log_mem,
            include_feature_metrics=include_feature_metrics,
            include_param_distance=include_param_distance,
            on_oom=on_oom,
            is_main_process=is_main_process,

            # These require the small patch in train_one_epoch.
            teacher_offload=teacher_offload,
            empty_cache_after_offload=empty_cache_after_teacher_offload,
        )

        # ----------------------------------------------------
        # Metrics / monitor
        # ----------------------------------------------------

        sec = time.time() - t0
        total_time += sec

        current_metric = resolve_dino_monitor_value(
            monitor_name=monitor_name,
            train_stats=train_stats,
        )

        improved = is_better_metric(
            current=current_metric,
            best=best_metric,
            mode=monitor_mode,
        )

        if improved:
            best_metric = current_metric

        # ----------------------------------------------------
        # Epoch summary
        # ----------------------------------------------------

        if is_main_process:
            print_dino_epoch_summary(
                epoch=epoch,
                global_step=global_step,
                sec=sec,
                optimizer=optimizer,
                train_stats=train_stats,
                monitor_name=monitor_name,
                current_metric=current_metric,
                best_metric=best_metric,
                improved=improved,
            )

        # ----------------------------------------------------
        # Metrics JSONL
        # ----------------------------------------------------

        combined_metrics = {}
        combined_metrics.update(prefixed_stats("train", train_stats))

        metrics_record = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "time_sec": float(sec),
            "monitor_name": monitor_name,
            "monitor_value": float(current_metric),
            "best_metric": float(best_metric) if best_metric is not None else None,
            "improved": bool(improved),
            "optimizer_type": optimizer_type,
            "teacher_offload": bool(teacher_offload),
            "use_gram_loss": bool(use_gram_loss),
            "gram_loss_weight": float(gram_loss_weight),
            **combined_metrics,
        }

        if is_main_process:
            append_jsonl(metrics_path, metrics_record)

        if is_main_process and mirror_dir:
            copy_file_to_dir_fixed(
                src_path=metrics_path,
                dst_dir=mirror_dir,
                fixed_name=fixed_mirror_metrics_name,
            )

        # ----------------------------------------------------
        # Checkpointing
        # ----------------------------------------------------

        if is_main_process:
            dino_section("Checkpointing")

        if is_main_process and improved:
            best_path = save_dino_checkpoint(
                checkpoint_dir=ckpt_dir,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=precision.get("scaler", None),
                epoch=epoch,
                step=global_step,
                best_metric=best_metric,
                config=config if config is not None else bundle_dino.get("config", None),
                extra_state={
                    "monitor_name": monitor_name,
                    "monitor_mode": monitor_mode,
                    "monitor_value": current_metric,
                    "train_stats": train_stats,
                    "optimizer_type": optimizer_type,
                    "opt_info": opt_info,
                    "model_info": bundle_dino.get("model_info", {}),
                    "teacher_offload": teacher_offload,
                    "use_gram_loss": use_gram_loss,
                    "gram_loss_weight": gram_loss_weight,
                },
                filename=f"{run_name}_best.pt",
                save_rng_state=True,
                keep_last_n=None,
                tag="best",
                save_latest=False,
                save_json_sidecar=True,
            )

            print(f"└─ [BEST] improved {monitor_name} -> {best_metric:.6f}")
            print(f"└─ [BEST] saved → {best_path}")

            if copy_fixed_to_mirror and mirror_dir:
                copy_file_to_dir_fixed(
                    src_path=best_path,
                    dst_dir=mirror_dir,
                    fixed_name=f"best_{fixed_mirror_ckpt_name}",
                )
        elif is_main_process:
            print("└─ [BEST] no improvement")

        should_save_epoch = (
            save_every is not None
            and save_every > 0
            and ((epoch % save_every == 0) or (epoch == epochs - 1))
        )

        if is_main_process and should_save_epoch:
            ckpt_path = save_dino_checkpoint(
                checkpoint_dir=ckpt_dir,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=precision.get("scaler", None),
                epoch=epoch,
                step=global_step,
                best_metric=best_metric,
                config=config if config is not None else bundle_dino.get("config", None),
                extra_state={
                    "monitor_name": monitor_name,
                    "monitor_mode": monitor_mode,
                    "monitor_value": current_metric,
                    "train_stats": train_stats,
                    "optimizer_type": optimizer_type,
                    "opt_info": opt_info,
                    "model_info": bundle_dino.get("model_info", {}),
                    "teacher_offload": teacher_offload,
                    "use_gram_loss": use_gram_loss,
                    "gram_loss_weight": gram_loss_weight,
                },
                filename=f"{run_name}_e{epoch:03d}.pt",
                save_rng_state=True,
                keep_last_n=keep_last_n_checkpoints,
                tag=None,
                save_latest=True,
                save_json_sidecar=True,
            )

            print(f"└─ [CKPT] saved → {ckpt_path}")

            if copy_fixed_to_mirror and mirror_dir:
                copy_file_to_dir_fixed(
                    src_path=ckpt_path,
                    dst_dir=mirror_dir,
                    fixed_name=fixed_mirror_ckpt_name,
                )
        elif is_main_process:
            print("└─ [CKPT] skipped by save_every")

    # ========================================================
    # Final checkpoint
    # ========================================================

    if is_main_process and save_last and train_stats is not None:
        dino_section("Final checkpoint")

        last_path = save_dino_checkpoint(
            checkpoint_dir=ckpt_dir,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=precision.get("scaler", None),
            epoch=epochs - 1,
            step=global_step,
            best_metric=best_metric,
            config=config if config is not None else bundle_dino.get("config", None),
            extra_state={
                "monitor_name": monitor_name,
                "monitor_mode": monitor_mode,
                "train_stats": train_stats,
                "optimizer_type": optimizer_type,
                "opt_info": opt_info,
                "model_info": bundle_dino.get("model_info", {}),
                "teacher_offload": teacher_offload,
                "use_gram_loss": use_gram_loss,
                "gram_loss_weight": gram_loss_weight,
            },
            filename=f"{run_name}_last_manual.pt",
            save_rng_state=True,
            keep_last_n=None,
            tag="last",
            save_latest=True,
            save_json_sidecar=True,
        )

        print(f"└─ [CKPT] final saved → {last_path}")

        if copy_fixed_to_mirror and mirror_dir:
            copy_file_to_dir_fixed(
                src_path=last_path,
                dst_dir=mirror_dir,
                fixed_name=fixed_mirror_ckpt_name,
            )

    # ========================================================
    # Done
    # ========================================================

    if is_main_process:
        dino_title("DINO training complete")
        print(f"Total time : {fmt_hms(total_time)}")
        print(f"Final step : {global_step}")
        print(f"Best metric: {best_metric}")
        print(f"Metrics    : {metrics_path}")
        print(f"Checkpoints: {ckpt_dir}")
        print(dino_rule())

    return {
        "bundle_dino": bundle_dino,
        "student": student,
        "teacher": teacher,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "precision": precision,
        "opt_info": opt_info,
        "global_step": global_step,
        "best_metric": best_metric,
        "last_train_stats": train_stats,
        "metrics_path": str(metrics_path),
        "checkpoint_dir": ckpt_dir,
    }
