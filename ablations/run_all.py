#!/usr/bin/env python
"""Run every ablation in the registry, then aggregate results.

Thin convenience wrapper around run_ablation.py (--all) followed by
collect_results.py. All extra CLI flags are forwarded to run_ablation.py.

Examples
--------
    python ablations/run_all.py
    python ablations/run_all.py --epochs 30 --max-batches-per-epoch 200
    python ablations/run_all.py --dry-run
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ABLATIONS_DIR = Path(__file__).resolve().parent


def _run_module(module_path: Path, argv: list[str]) -> int:
    saved = sys.argv
    sys.argv = [str(module_path)] + argv
    try:
        runpy.run_path(str(module_path), run_name="__main__")
        return 0
    except SystemExit as exc:  # scripts call SystemExit via main()
        code = exc.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    finally:
        sys.argv = saved


def main() -> int:
    extra = sys.argv[1:]
    dry_run = "--dry-run" in extra

    rc = _run_module(ABLATIONS_DIR / "run_ablation.py", ["--all", *extra])
    if rc not in (0, 1):  # 1 == some runs failed but harness ran fine
        return rc

    if not dry_run:
        print("\nAggregating results...\n")
        _run_module(ABLATIONS_DIR / "collect_results.py", [])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
