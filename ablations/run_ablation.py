#!/usr/bin/env python
"""Run DINO ablation studies.

Each variant is materialised as a standalone YAML config (base config deep-merged
with the variant override + an isolated checkpoint dir) and trained in its own
subprocess via scripts/train_dino.py. This keeps every run isolated (fresh
process, fresh CUDA context) and requires no changes to the training pipeline.

Examples
--------
    # List everything that can be run
    python ablations/run_ablation.py --list

    # Run all variants of a single ablation
    python ablations/run_ablation.py gram_loss

    # Run one specific variant
    python ablations/run_ablation.py gram_loss --variant weight_0p05

    # Run every ablation, but quick (short runs for a smoke comparison)
    python ablations/run_ablation.py --all --epochs 30 --max-batches-per-epoch 200

    # Just generate the configs and print the commands, run nothing
    python ablations/run_ablation.py --all --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Make `from _common import ...` work regardless of the current directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DEFAULT_REGISTRY,
    GENERATED_DIR,
    PROJECT_ROOT,
    deep_merge,
    dump_yaml,
    load_registry,
    load_yaml,
    resolve_path,
    variant_ckpt_dir,
    variant_identity,
)

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_dino.py"


def build_variant_config(
    *,
    base_config: dict,
    ablation: str,
    variant: str,
    variant_spec: dict,
    output_root: str,
) -> tuple[dict, Path, Path]:
    """Return (merged_config, generated_config_path, ckpt_dir)."""
    override = variant_spec.get("override", {})
    merged = deep_merge(base_config, override)

    run_name, _ = variant_identity(ablation, variant)
    ckpt_dir_rel = variant_ckpt_dir(output_root, ablation, variant)

    # Route checkpoints/metrics to an isolated directory. training_kwargs() reads
    # ckpt_dir/run_name from the `train` section first, then `checkpoint`; we set
    # both so the routing holds no matter which the base config used.
    merged.setdefault("checkpoint", {})
    merged["checkpoint"]["ckpt_dir"] = str(ckpt_dir_rel).replace("\\", "/")
    merged["checkpoint"]["run_name"] = run_name
    merged.setdefault("train", {})
    merged["train"]["ckpt_dir"] = str(ckpt_dir_rel).replace("\\", "/")
    merged["train"]["run_name"] = run_name

    gen_path = GENERATED_DIR / f"{ablation}__{variant}.yaml"
    return merged, gen_path, ckpt_dir_rel


def run_variant(
    *,
    ablation: str,
    variant: str,
    variant_spec: dict,
    base_config: dict,
    output_root: str,
    epochs: int | None,
    max_batches: int | None,
    device: str | None,
    dry_run: bool,
) -> int:
    merged, gen_path, ckpt_dir_rel = build_variant_config(
        base_config=base_config,
        ablation=ablation,
        variant=variant,
        variant_spec=variant_spec,
        output_root=output_root,
    )
    dump_yaml(merged, gen_path)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(gen_path),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if max_batches is not None:
        cmd += ["--max-batches-per-epoch", str(max_batches)]
    if device is not None:
        cmd += ["--device", device]

    print("=" * 100)
    print(f"[ABLATION] {ablation} :: {variant}")
    print(f"[CONFIG]   {gen_path.relative_to(PROJECT_ROOT)}")
    print(f"[CKPT]     {ckpt_dir_rel}")
    print(f"[CMD]      {' '.join(cmd)}")
    print("=" * 100)

    if dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        print(f"[ABLATION] {ablation}::{variant} FAILED (exit {completed.returncode})")
    return completed.returncode


def cmd_list(registry: dict) -> None:
    primary = registry.get("primary_ablation")
    print(f"Base config : {registry['base_config']}")
    print(f"Output root : {registry['output_root']}")
    print(f"Primary     : {primary}")
    print()
    for name, spec in registry["ablations"].items():
        star = "  <-- PRIMARY" if name == primary else ""
        print(f"- {name}{star}")
        print(f"    title : {spec.get('title', '')}")
        print(f"    axis  : {spec.get('axis', '')}")
        variants = [str(v) for v in spec.get("variants", {}).keys()]
        print(f"    variants ({len(variants)}): {', '.join(variants)}")
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DINO ablation studies.")
    p.add_argument("ablation", nargs="?", default=None, help="Ablation name to run.")
    p.add_argument("--all", action="store_true", help="Run every ablation in the registry.")
    p.add_argument("--variant", default=None, help="Run only this variant of the chosen ablation.")
    p.add_argument("--list", action="store_true", help="List ablations and exit.")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to ablations.yaml.")
    p.add_argument("--base-config", default=None, help="Override the registry base_config.")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs for every run.")
    p.add_argument("--max-batches-per-epoch", type=int, default=None, help="Cap batches/epoch for quick runs.")
    p.add_argument("--device", default=None, help="Force device (cuda/cpu).")
    p.add_argument("--dry-run", action="store_true", help="Generate configs and print commands only.")
    p.add_argument("--continue-on-error", action="store_true", help="Keep going if a run fails.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)

    if args.list:
        cmd_list(registry)
        return 0

    base_config_path = resolve_path(args.base_config or registry["base_config"])
    base_config = load_yaml(base_config_path)
    output_root = registry["output_root"]

    # Decide which (ablation, variant) pairs to run.
    work: list[tuple[str, str, dict]] = []
    if args.all:
        for ablation, spec in registry["ablations"].items():
            for variant, vspec in spec.get("variants", {}).items():
                work.append((ablation, variant, vspec))
    elif args.ablation:
        if args.ablation not in registry["ablations"]:
            print(f"Unknown ablation '{args.ablation}'. Use --list to see options.")
            return 2
        spec = registry["ablations"][args.ablation]
        variants = spec.get("variants", {})
        if args.variant:
            if args.variant not in variants:
                print(f"Unknown variant '{args.variant}' for ablation '{args.ablation}'.")
                return 2
            work.append((args.ablation, args.variant, variants[args.variant]))
        else:
            for variant, vspec in variants.items():
                work.append((args.ablation, variant, vspec))
    else:
        print("Nothing to do. Pass an ablation name, --all, or --list.")
        return 2

    print(f"Base config : {base_config_path.relative_to(PROJECT_ROOT)}")
    print(f"Runs queued : {len(work)}")
    if args.epochs is not None or args.max_batches_per_epoch is not None:
        print(f"Overrides   : epochs={args.epochs} max_batches={args.max_batches_per_epoch}")
    print()

    failures = 0
    for ablation, variant, vspec in work:
        rc = run_variant(
            ablation=ablation,
            variant=variant,
            variant_spec=vspec,
            base_config=base_config,
            output_root=output_root,
            epochs=args.epochs,
            max_batches=args.max_batches_per_epoch,
            device=args.device,
            dry_run=args.dry_run,
        )
        if rc != 0:
            failures += 1
            if not args.continue_on_error:
                print("Stopping (use --continue-on-error to keep going).")
                return rc

    print()
    print(f"Done. {len(work) - failures}/{len(work)} runs succeeded.")
    if not args.dry_run:
        print("Aggregate results with:  python ablations/collect_results.py")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
