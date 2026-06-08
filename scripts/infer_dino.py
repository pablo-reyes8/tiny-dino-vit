from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import overlay_heatmap, overlay_mask, run_dino_inference, save_tensor_image
from src.model.dino_full_model import build_dino_bundle
from src.training.chekpoints import load_dino_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained DINO checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to DINO checkpoint .pt file.")
    parser.add_argument("--image", action="append", required=True, help="Image path. Can be passed multiple times.")
    parser.add_argument("--config", default=None, help="Optional model config JSON if checkpoint has no config.")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--segmentation", choices=["cls_similarity", "attention"], default="cls_similarity")
    return parser.parse_args()


def load_model_config(checkpoint_path: str, config_path: str | None, device: str):
    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config")
        if config is None:
            raise ValueError("Checkpoint has no config. Pass --config with model configuration JSON.")

    config = dict(config)
    if device != "auto":
        config["device"] = device
    return config


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_model_config(args.checkpoint, args.config, args.device)
    bundle = build_dino_bundle(config)

    load_dino_checkpoint(
        checkpoint_path=args.checkpoint,
        student=bundle["student"],
        teacher=bundle["teacher"],
        map_location="cpu",
        strict_student=False,
        strict_teacher=False,
        load_optimizer=False,
        load_scheduler=False,
        load_scaler=False,
        load_rng_state=False,
    )

    model = bundle[args.model]
    device = bundle["device"]
    model = model.to(device)

    result = run_dino_inference(
        model=model,
        images=args.image,
        image_size=args.image_size,
        device=device,
        segmentation=args.segmentation,
    )

    outputs = result["outputs"]
    score_map = result["score_map"]
    mask = result["mask"]

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2, sort_keys=True)

    for idx, image_path in enumerate(args.image):
        stem = Path(image_path).stem
        heatmap = overlay_heatmap(outputs["input"][idx], score_map[idx])
        mask_overlay = overlay_mask(outputs["input"][idx], mask[idx])
        save_tensor_image(heatmap, output_dir / f"{stem}_heatmap.png")
        save_tensor_image(mask_overlay, output_dir / f"{stem}_mask.png")

    print("Inference finished")
    print("Outputs:", output_dir)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()
