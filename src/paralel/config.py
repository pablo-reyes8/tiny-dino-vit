"""Load and merge configuration files for parallel DINO training."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_or_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()

    with path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to load YAML configs: pip install pyyaml") from exc
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must load to a dictionary, got {type(data)} from {path}")
    return data


def merge_config(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(defaults)
    return deep_update(config, overrides)
