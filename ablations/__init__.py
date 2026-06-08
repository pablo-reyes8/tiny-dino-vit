"""DINO ablation harness.

Config-driven ablation studies that run entirely on top of the existing
training pipeline. Each ablation varies a single axis (defined in
``ablations.yaml``) by deep-merging a config override into the base config and
launching one isolated training run per variant.

Modules
-------
run_ablation.py    Generate per-variant configs and launch training runs.
run_all.py         Convenience wrapper: run every ablation sequentially.
collect_results.py Aggregate each run's metrics.jsonl into a comparison table.
"""
