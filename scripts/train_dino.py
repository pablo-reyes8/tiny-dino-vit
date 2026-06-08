from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def default_train_config() -> Dict[str, Any]:
    return {
        "device": "auto",
        "model": {
            "img_size": 64,
            "patch_size": 8,
            "in_chans": 3,
            "embed_dim": 192,
            "depth": 6,
            "num_heads": 3,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.1,
            "pos_embed_type": "learned",
            "use_cls_token": True,
            "num_register_tokens": 0,
            "strict_img_size": False,
        },
        "dino_head": {
            "out_dim": 8192,
            "hidden_dim": 2048,
            "bottleneck_dim": 256,
            "num_layers": 3,
            "use_bn": False,
            "norm_last_layer": True,
        },
        "data": {},
        "train": {
            "epochs": 100,
            "learning_rate": 5e-4,
            "min_learning_rate": 1e-6,
            "weight_decay": 0.04,
            "weight_decay_end": 0.4,
            "warmup_epochs": 10,
            "student_temp": 0.1,
            "teacher_temp": 0.07,
            "warmup_teacher_temp": 0.04,
            "warmup_teacher_temp_epochs": 30,
            "teacher_momentum_start": 0.996,
            "teacher_momentum_end": 1.0,
            "grad_clip": 1.0,
            "grad_accum_steps": 1,
            "max_batches_per_epoch": None,
            "use_gram_loss": False,
            "gram_loss_weight": 0.05,
            "ckpt_dir": "checkpoints/dino_tinyimagenet",
            "run_name": "dino_tinyvit",
            "save_every": 1,
            "log_every": 20,
            "metrics_every": 20,
        },
    }


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None) -> Dict[str, Any]:
    config = default_train_config()
    if path:
        from src.paralel.config import load_yaml_or_json

        deep_update(config, load_yaml_or_json(path))
    return config


def training_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    train = config.get("train", {})
    dino = config.get("dino", {})
    optim = config.get("optim", {})
    checkpoint = config.get("checkpoint", {})

    return {
        "epochs": int(train.get("epochs", 100)),
        "warmup_epochs": int(train.get("warmup_epochs", 10)),
        "learning_rate": float(train.get("learning_rate", optim.get("learning_rate", 5e-4))),
        "min_learning_rate": float(train.get("min_learning_rate", optim.get("min_learning_rate", 1e-6))),
        "weight_decay": float(train.get("weight_decay", optim.get("weight_decay", 0.04))),
        "weight_decay_end": float(train.get("weight_decay_end", optim.get("weight_decay_end", 0.4))),
        "betas": tuple(train.get("betas", optim.get("betas", (0.9, 0.95)))),
        "eps": float(train.get("eps", optim.get("eps", 1e-8))),
        "student_temp": float(train.get("student_temp", dino.get("student_temp", 0.1))),
        "teacher_temp": float(train.get("teacher_temp", dino.get("teacher_temp", 0.07))),
        "warmup_teacher_temp": float(train.get("warmup_teacher_temp", dino.get("warmup_teacher_temp", 0.04))),
        "warmup_teacher_temp_epochs": int(train.get("warmup_teacher_temp_epochs", dino.get("warmup_teacher_temp_epochs", 30))),
        "teacher_momentum_start": float(train.get("teacher_momentum_start", dino.get("teacher_momentum_start", 0.996))),
        "teacher_momentum_end": float(train.get("teacher_momentum_end", dino.get("teacher_momentum_end", 1.0))),
        "grad_clip": train.get("grad_clip", 1.0),
        "grad_accum_steps": int(train.get("grad_accum_steps", 1)),
        "max_batches_per_epoch": train.get("max_batches_per_epoch", None),
        "log_every": int(train.get("log_every", 20)),
        "metrics_every": int(train.get("metrics_every", 20)),
        "teacher_offload": bool(train.get("teacher_offload", False)),
        "empty_cache_after_teacher_offload": bool(train.get("empty_cache_after_offload", False)),
        "use_gram_loss": bool(train.get("use_gram_loss", dino.get("use_gram_loss", False))),
        "gram_loss_weight": float(train.get("gram_loss_weight", dino.get("gram_loss_weight", 0.05))),
        "ckpt_dir": train.get("ckpt_dir", checkpoint.get("ckpt_dir", "checkpoints/dino_tinyimagenet")),
        "run_name": train.get("run_name", checkpoint.get("run_name", "dino_tinyvit")),
        "save_every": int(train.get("save_every", checkpoint.get("save_every", 1))),
        "keep_last_n_checkpoints": int(train.get("keep_last_n_checkpoints", checkpoint.get("keep_last_n_checkpoints", 3))),
        "monitor_name": train.get("monitor_name", checkpoint.get("monitor_name", "loss")),
        "monitor_mode": train.get("monitor_mode", checkpoint.get("monitor_mode", "min")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a full DINO model.")
    parser.add_argument("--config", default=None, help="YAML or JSON config file.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-gram-loss", action="store_true")
    parser.add_argument("--resume-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from data.data_config import merge_data_config
    from data.load_data import build_tinyimagenet_dataloaders
    from src.model.dino_full_model import build_dino_bundle
    from src.model.loss import DINOLoss, DINOGramLoss
    from src.training.train_dino import train_dino

    config = load_config(args.config)

    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.max_batches_per_epoch is not None:
        config["train"]["max_batches_per_epoch"] = args.max_batches_per_epoch
    if args.ckpt_dir is not None:
        config["train"]["ckpt_dir"] = args.ckpt_dir
    if args.run_name is not None:
        config["train"]["run_name"] = args.run_name
    if args.device is not None:
        config["device"] = args.device
    if args.use_gram_loss:
        config.setdefault("dino", {})["use_gram_loss"] = True

    if config.get("device", "auto") == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    data_overrides = config.get("data", {})
    if args.batch_size is not None:
        data_overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_overrides["num_workers"] = args.num_workers

    data_config = merge_data_config(data_overrides)
    if "dino" in config:
        data_config["num_global_crops"] = int(config["dino"].get("num_global_crops", data_config["num_global_crops"]))
        data_config["num_local_crops"] = int(config["dino"].get("num_local_crops", data_config["num_local_crops"]))

    train_dataset, val_dataset, train_loader, val_loader = build_tinyimagenet_dataloaders(config=data_config)

    model_config = {
        "device": config["device"],
        "model": config["model"],
        "dino_head": config["dino_head"],
        "data": data_config,
    }
    bundle = build_dino_bundle(model_config)

    dino_loss = DINOLoss(out_dim=int(config["dino_head"].get("out_dim", 8192)))
    train_args = training_kwargs(config)
    gram_loss = DINOGramLoss() if train_args["use_gram_loss"] else None

    result = train_dino(
        bundle_dino=bundle,
        train_loader=train_loader,
        dino_loss_fn=dino_loss,
        gram_loss_fn=gram_loss,
        config={**model_config, "train": train_args, "dino": config.get("dino", {})},
        resume_path=args.resume_path,
        num_global_crops=int(data_config["num_global_crops"]),
        **train_args,
    )

    print("Training finished")
    print("Checkpoint dir:", result["checkpoint_dir"])
    print("Metrics path:", result["metrics_path"])
    print("Train dataset:", len(train_dataset), "Val dataset:", len(val_dataset))


if __name__ == "__main__":
    main()
