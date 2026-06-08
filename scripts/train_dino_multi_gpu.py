from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU DINO training with torchrun + DDP.")
    parser.add_argument("--config", required=True, help="YAML or JSON config from configs/.")
    parser.add_argument("--backend", default=None, help="Distributed backend. Defaults to nccl on CUDA, gloo on CPU.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-process batch size.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--sync-batchnorm", action="store_true")
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--use-model-parallel", action="store_true", help="Experimental ViT block sharding instead of DDP.")
    parser.add_argument("--model-parallel-devices", default=None, help="Comma-separated devices, e.g. cuda:0,cuda:1.")
    return parser.parse_args()


def _training_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    train = config.get("train", {})
    dino = config.get("dino", {})
    optim = config.get("optim", {})
    checkpoint = config.get("checkpoint", {})

    return {
        "epochs": int(train.get("epochs", 100)),
        "warmup_epochs": int(train.get("warmup_epochs", 10)),
        "learning_rate": float(optim.get("learning_rate", optim.get("lr", 5e-4))),
        "min_learning_rate": float(optim.get("min_learning_rate", optim.get("min_lr", 1e-6))),
        "weight_decay": float(optim.get("weight_decay", 0.04)),
        "weight_decay_end": float(optim.get("weight_decay_end", 0.4)),
        "betas": tuple(optim.get("betas", (0.9, 0.95))),
        "eps": float(optim.get("eps", 1e-8)),
        "student_temp": float(dino.get("student_temp", 0.1)),
        "teacher_temp": float(dino.get("teacher_temp", 0.07)),
        "warmup_teacher_temp": float(dino.get("warmup_teacher_temp", 0.04)),
        "warmup_teacher_temp_epochs": int(dino.get("warmup_teacher_temp_epochs", 30)),
        "teacher_momentum_start": float(dino.get("teacher_momentum_start", 0.996)),
        "teacher_momentum_end": float(dino.get("teacher_momentum_end", 1.0)),
        "grad_clip": train.get("grad_clip", 1.0),
        "grad_accum_steps": int(train.get("grad_accum_steps", 1)),
        "max_batches_per_epoch": train.get("max_batches_per_epoch", None),
        "log_every": int(train.get("log_every", 20)),
        "metrics_every": int(train.get("metrics_every", 20)),
        "teacher_offload": bool(train.get("teacher_offload", False)),
        "empty_cache_after_teacher_offload": bool(train.get("empty_cache_after_offload", False)),
        "use_gram_loss": bool(dino.get("use_gram_loss", False)),
        "gram_loss_weight": float(dino.get("gram_loss_weight", 0.05)),
        "ckpt_dir": checkpoint.get("ckpt_dir", "checkpoints/dino_ddp"),
        "run_name": checkpoint.get("run_name", "dino_ddp"),
        "save_every": int(checkpoint.get("save_every", 1)),
        "keep_last_n_checkpoints": int(checkpoint.get("keep_last_n_checkpoints", 3)),
        "monitor_name": checkpoint.get("monitor_name", "loss"),
        "monitor_mode": checkpoint.get("monitor_mode", "min"),
    }


def main() -> None:
    args = parse_args()

    from data.data_config import merge_data_config
    from data.load_data import build_tinyimagenet_dataloaders, seed_worker
    from src.model.dino_full_model import build_dino_bundle
    from src.model.loss import DINOLoss, DINOGramLoss
    from src.paralel.config import load_yaml_or_json
    from src.paralel.data_parallel import build_distributed_loader, wrap_data_parallel
    from src.paralel.distributed import barrier, cleanup_distributed, is_main_process, setup_distributed
    from src.paralel.model_parallel import shard_vit_blocks
    from src.training.train_dino import train_dino

    ctx = setup_distributed(backend=args.backend)
    config = load_yaml_or_json(args.config)

    if config.get("device", "auto") == "auto":
        config["device"] = str(ctx.device)

    data_config = merge_data_config(config.get("data", {}))
    if "dino" in config:
        data_config["num_global_crops"] = int(config["dino"].get("num_global_crops", data_config["num_global_crops"]))
        data_config["num_local_crops"] = int(config["dino"].get("num_local_crops", data_config["num_local_crops"]))

    if args.batch_size is not None:
        data_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_config["num_workers"] = args.num_workers

    train_kwargs = _training_kwargs(config)
    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    if args.max_batches_per_epoch is not None:
        train_kwargs["max_batches_per_epoch"] = args.max_batches_per_epoch
    if args.ckpt_dir is not None:
        train_kwargs["ckpt_dir"] = args.ckpt_dir
    if args.run_name is not None:
        train_kwargs["run_name"] = args.run_name

    train_dataset, val_dataset, _, _ = build_tinyimagenet_dataloaders(config=data_config)
    train_loader = build_distributed_loader(
        train_dataset,
        ctx=ctx,
        batch_size=int(data_config["batch_size"]),
        shuffle=bool(data_config.get("shuffle_train", True)),
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", True)),
        drop_last=bool(data_config.get("drop_last_train", True)),
        persistent_workers=bool(data_config.get("persistent_workers", True)),
        worker_init_fn=seed_worker,
    )

    model_config = {
        "device": str(ctx.device),
        "model": config["model"],
        "dino_head": config["dino_head"],
        "data": data_config,
        "distributed": {
            "enabled": ctx.enabled,
            "backend": ctx.backend,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
        },
    }
    bundle = build_dino_bundle(model_config)

    if args.use_model_parallel:
        if ctx.world_size > 1:
            raise RuntimeError(
                "--use-model-parallel is intended for one process spanning multiple GPUs. "
                "Launch without torchrun or with --nproc_per_node=1. For multi-process training, use DDP."
            )
        devices = args.model_parallel_devices
        if devices is None:
            devices = ",".join(f"cuda:{idx}" for idx in range(torch.cuda.device_count()))
        device_list = [item.strip() for item in devices.split(",") if item.strip()]
        bundle["student"] = shard_vit_blocks(bundle["student"], device_list)
        bundle["teacher"] = shard_vit_blocks(bundle["teacher"], device_list)
    else:
        bundle["student"] = wrap_data_parallel(
            bundle["student"],
            ctx,
            sync_batchnorm=args.sync_batchnorm,
            find_unused_parameters=args.find_unused_parameters,
        )

    dino_loss = DINOLoss(out_dim=int(config["dino_head"].get("out_dim", 8192))).to(ctx.device)
    gram_loss = DINOGramLoss() if train_kwargs["use_gram_loss"] else None

    if is_main_process():
        print(f"[DDP] rank={ctx.rank} world_size={ctx.world_size} device={ctx.device}")
        print(f"[DDP] train examples per rank: {len(train_dataset)} | val examples: {len(val_dataset)}")

    try:
        result = train_dino(
            bundle_dino=bundle,
            train_loader=train_loader,
            dino_loss_fn=dino_loss,
            gram_loss_fn=gram_loss,
            config={**model_config, "train": train_kwargs, "dino": config.get("dino", {})},
            resume_path=args.resume_path,
            num_global_crops=int(data_config["num_global_crops"]),
            is_main_process=is_main_process(),
            **train_kwargs,
        )
        barrier()
        if is_main_process():
            print("Distributed training finished")
            print("Checkpoint dir:", result["checkpoint_dir"])
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
