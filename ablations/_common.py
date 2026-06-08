"""Shared helpers for the ablation harness (registry loading, paths, merging)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Tuple

# ablations/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATIONS_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY = ABLATIONS_DIR / "ablations.yaml"
GENERATED_DIR = ABLATIONS_DIR / "_generated"
RESULTS_DIR = ABLATIONS_DIR / "results"


def _require_yaml():
    try:
        import yaml  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for the ablation harness: pip install pyyaml") from exc
    return yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    yaml = _require_yaml()
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top of {path}, got {type(data)}.")
    return data


def dump_yaml(data: Dict[str, Any], path: str | Path) -> None:
    yaml = _require_yaml()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a deep copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_registry(registry_path: str | Path = DEFAULT_REGISTRY) -> Dict[str, Any]:
    reg = load_yaml(registry_path)
    if "ablations" not in reg or not isinstance(reg["ablations"], dict):
        raise ValueError(f"Registry {registry_path} must contain an 'ablations' mapping.")
    reg.setdefault("base_config", "configs/set_up.yaml")
    reg.setdefault("output_root", "checkpoints/ablations")
    return reg


def resolve_path(maybe_relative: str | Path) -> Path:
    """Resolve a path relative to the project root if it is not absolute."""
    p = Path(maybe_relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def variant_identity(ablation: str, variant: str) -> Tuple[str, str]:
    """Return (run_name, relative ckpt_dir) for an ablation variant."""
    run_name = f"{ablation}__{variant}"
    return run_name, run_name


def variant_ckpt_dir(output_root: str | Path, ablation: str, variant: str) -> Path:
    """Relative checkpoint directory for a variant (project-root relative)."""
    return Path(output_root) / ablation / variant


def iter_variants(registry: Dict[str, Any]):
    """Yield (ablation_name, ablation_spec, variant_name, variant_spec)."""
    for ablation_name, spec in registry["ablations"].items():
        variants = spec.get("variants", {})
        for variant_name, variant_spec in variants.items():
            yield ablation_name, spec, variant_name, variant_spec
