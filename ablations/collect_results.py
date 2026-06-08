#!/usr/bin/env python
"""Aggregate ablation runs into a comparison report.

Reads each variant's ``metrics.jsonl`` (written by the training pipeline),
extracts the key self-supervised diagnostics, and exports a side-by-side
comparison as CSV, Markdown, and JSON under ``ablations/results/``.

Examples
--------
    python ablations/collect_results.py
    python ablations/collect_results.py --registry ablations/ablations.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows consoles default to cp1252, which cannot encode the report's Δ/✓/✗.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from _common import (  # noqa: E402
    DEFAULT_REGISTRY,
    PROJECT_ROOT,
    RESULTS_DIR,
    load_registry,
    resolve_path,
    variant_ckpt_dir,
)

# Columns pulled from the last epoch record unless noted otherwise.
# (display label, jsonl key, is_better_when)
METRIC_COLUMNS = [
    ("best_loss", "_best_loss", "min"),          # min monitor_value across epochs
    ("final_loss", "train_loss", "min"),
    ("loss_dino", "train_loss_dino", "min"),
    ("loss_gram", "train_loss_gram", "min"),
    ("teacher_H", "train_teacher_entropy_norm", "info"),
    ("student_H", "train_student_entropy_norm", "info"),
    ("cls_std", "train_student_cls_dim_std_mean", "max"),  # collapse indicator
]


def read_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not metrics_path.exists():
        return records
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarise_run(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce a list of per-epoch records to a single summary row."""
    if not records:
        return {"status": "no-metrics", "epochs": 0}

    last = records[-1]
    monitor_values = [r.get("monitor_value") for r in records if isinstance(r.get("monitor_value"), (int, float))]
    best_loss = min(monitor_values) if monitor_values else None
    total_time = sum(float(r.get("time_sec", 0.0)) for r in records)

    summary: Dict[str, Any] = {
        "status": "ok",
        "epochs": len(records),
        "total_time_sec": round(total_time, 1),
        "_best_loss": best_loss,
    }
    for _, key, _ in METRIC_COLUMNS:
        if key == "_best_loss":
            continue
        val = last.get(key)
        summary[key] = val
    return summary


def _fmt(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if abs(val) != 0 and (abs(val) < 1e-3 or abs(val) >= 1e4):
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)


def _delta(value: Optional[float], reference: Optional[float], better: str) -> str:
    if value is None or reference is None:
        return "—"
    diff = value - reference
    if abs(diff) < 1e-9:
        return "ref"
    arrow = ""
    if better in ("min", "max"):
        improved = (diff < 0) if better == "min" else (diff > 0)
        arrow = " ✓" if improved else " ✗"
    sign = "+" if diff >= 0 else ""
    return f"{sign}{_fmt(diff)}{arrow}"


def build_report(registry: Dict[str, Any]) -> Dict[str, Any]:
    output_root = registry["output_root"]
    report: Dict[str, Any] = {
        "base_config": registry["base_config"],
        "primary_ablation": registry.get("primary_ablation"),
        "ablations": {},
    }

    for ablation, spec in registry["ablations"].items():
        variants = spec.get("variants", {})
        reference = spec.get("reference") or (next(iter(variants), None))

        ablation_block: Dict[str, Any] = {
            "title": spec.get("title", ""),
            "axis": spec.get("axis", ""),
            "hypothesis": " ".join(spec.get("hypothesis", "").split()),
            "reference": reference,
            "runs": {},
        }

        for variant in variants:
            ckpt_dir = resolve_path(variant_ckpt_dir(output_root, ablation, variant))
            metrics_path = ckpt_dir / "metrics.jsonl"
            records = read_metrics(metrics_path)
            summary = summarise_run(records)
            summary["ckpt_dir"] = str(variant_ckpt_dir(output_root, ablation, variant)).replace("\\", "/")
            ablation_block["runs"][variant] = summary

        report["ablations"][ablation] = ablation_block

    return report


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# DINO Ablation Results")
    lines.append("")
    lines.append(f"Base config: `{report['base_config']}`")
    if report.get("primary_ablation"):
        lines.append(f"Primary ablation (central hypothesis): **{report['primary_ablation']}**")
    lines.append("")

    for ablation, block in report["ablations"].items():
        primary = " — PRIMARY" if ablation == report.get("primary_ablation") else ""
        lines.append(f"## {ablation}{primary}")
        lines.append("")
        lines.append(f"*{block['title']}* — axis: `{block['axis']}`")
        lines.append("")
        if block["hypothesis"]:
            lines.append(f"> **Hypothesis:** {block['hypothesis']}")
            lines.append("")

        ref = block["reference"]
        ref_run = block["runs"].get(ref, {})
        ref_best = ref_run.get("_best_loss")

        header = ["variant", "epochs", "best_loss", "Δbest", "loss_dino", "loss_gram", "cls_std", "student_H", "time(s)"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for variant, run in block["runs"].items():
            tag = f"**{variant}**" if variant == ref else variant
            if run.get("status") != "ok":
                lines.append(f"| {tag} | — | _{run.get('status', 'missing')}_ | — | — | — | — | — | — |")
                continue
            row = [
                tag,
                str(run.get("epochs", "—")),
                _fmt(run.get("_best_loss")),
                _delta(run.get("_best_loss"), ref_best, "min"),
                _fmt(run.get("train_loss_dino")),
                _fmt(run.get("train_loss_gram")),
                _fmt(run.get("train_student_cls_dim_std_mean")),
                _fmt(run.get("train_student_entropy_norm")),
                _fmt(run.get("total_time_sec")),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Legend: `best_loss` = lowest monitored loss across epochs (lower is better). "
                 "`Δbest` = difference vs the **reference** variant (✓ better / ✗ worse). "
                 "`cls_std` = mean per-dim std of student CLS features (higher = less collapse). "
                 "`student_H` = normalised student output entropy.")
    lines.append("")
    return "\n".join(lines)


def write_csv(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ablation", "variant", "is_reference", "status", "epochs",
        "best_loss", "final_loss", "loss_dino", "loss_gram",
        "teacher_H", "student_H", "cls_std", "total_time_sec", "ckpt_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ablation, block in report["ablations"].items():
            ref = block["reference"]
            for variant, run in block["runs"].items():
                writer.writerow({
                    "ablation": ablation,
                    "variant": variant,
                    "is_reference": variant == ref,
                    "status": run.get("status", "missing"),
                    "epochs": run.get("epochs", 0),
                    "best_loss": run.get("_best_loss"),
                    "final_loss": run.get("train_loss"),
                    "loss_dino": run.get("train_loss_dino"),
                    "loss_gram": run.get("train_loss_gram"),
                    "teacher_H": run.get("train_teacher_entropy_norm"),
                    "student_H": run.get("train_student_entropy_norm"),
                    "cls_std": run.get("train_student_cls_dim_std_mean"),
                    "total_time_sec": run.get("total_time_sec"),
                    "ckpt_dir": run.get("ckpt_dir"),
                })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate DINO ablation results.")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to ablations.yaml.")
    p.add_argument("--out-dir", default=str(RESULTS_DIR), help="Where to write summary files.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    report = build_report(registry)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_csv(report, csv_path)
    markdown = render_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print()
    for p in (json_path, csv_path, md_path):
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = p
        print(f"[WROTE] {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
