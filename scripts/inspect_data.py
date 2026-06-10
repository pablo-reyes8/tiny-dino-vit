"""Inspect DINO training data and optionally write dataset reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and document DINO training data.")
    parser.add_argument("--dataset-name", default=None, help="Hugging Face dataset name.")
    parser.add_argument("--dataset-revision", default=None, help="Optional Hugging Face revision.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--global-crop-size", type=int, default=None)
    parser.add_argument("--local-crop-size", type=int, default=None)
    parser.add_argument("--max-profile-examples", type=int, default=128)
    parser.add_argument("--reports-dir", default="data/reports")
    parser.add_argument("--write-report", action="store_true", help="Write JSON reports to reports-dir.")
    parser.add_argument("--skip-loader", action="store_true", help="Only inspect HF metadata/profile, skip DataLoader build.")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "global_crop_size": args.global_crop_size,
        "local_crop_size": args.local_crop_size,
        "reports_dir": args.reports_dir,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    from data.data_config import merge_data_config
    from data.dataops import (
        build_data_card,
        inspect_dataloader_batch,
        profile_hf_dataset,
        validate_data_config,
        write_json,
    )
    from data.load_data import build_tinyimagenet_dataloaders

    config = merge_data_config(build_overrides(args))
    validation = validate_data_config(config)

    if not validation["valid"]:
        raise SystemExit("Invalid data config: " + "; ".join(validation["errors"]))

    load_kwargs = {}
    if config.get("dataset_revision"):
        load_kwargs["revision"] = config["dataset_revision"]

    hf_dataset = load_dataset(config["dataset_name"], **load_kwargs)
    profile = profile_hf_dataset(
        hf_dataset,
        max_examples_per_split=args.max_profile_examples,
        quality_thresholds=config.get("quality", {}),
    )
    batch_report = None

    if not args.skip_loader:
        _, _, train_loader, val_loader = build_tinyimagenet_dataloaders(config=config)
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        batch_report = {
            "train": inspect_dataloader_batch(train_batch),
            "validation": inspect_dataloader_batch(val_batch),
        }

    data_card = build_data_card(config=config, profile=profile, validation=validation)

    print("Data config valid:", validation["valid"])
    print("Splits:", ", ".join(profile["splits"].keys()))
    for split_name, split_profile in profile["splits"].items():
        print(f"- {split_name}: {split_profile['num_rows']} rows")
        quality = split_profile.get("sampled_quality_summary", {})
        if quality.get("flag_counts"):
            print(f"  quality flags: {quality['flag_counts']}")

    if batch_report is not None:
        print("Train batch:", batch_report["train"])
        print("Validation batch:", batch_report["validation"])

    if args.write_report:
        reports_dir = Path(config["reports_dir"])
        write_json(reports_dir / "data_profile.json", profile)
        write_json(reports_dir / "data_card.json", data_card)
        if batch_report is not None:
            write_json(reports_dir / "batch_inspection.json", batch_report)
        print(f"Reports written to {reports_dir}")


if __name__ == "__main__":
    main()
