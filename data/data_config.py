from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


DEFAULT_DATA_CONFIG: Dict[str, Any] = {
    "dataset_name": "zh-plus/tiny-imagenet",
    "dataset_revision": None,
    "data_dir": "data",
    "artifacts_dir": "data/artifacts",
    "reports_dir": "data/reports",
    "seed": 42,
    "val_ratio": 0.1,
    "global_crop_size": 64,
    "local_crop_size": 32,
    "num_global_crops": 2,
    "num_local_crops": 4,
    "global_crop_scale": (0.5, 1.0),
    "local_crop_scale": (0.2, 0.5),
    "batch_size": 64,
    "num_workers": 2,
    "pin_memory": True,
    "persistent_workers": True,
    "shuffle_train": True,
    "drop_last_train": True,
    "drop_last_val": False,
    "quality": {
        "blur_variance_threshold": 0.002,
        "dark_mean_threshold": 0.08,
        "bright_mean_threshold": 0.92,
        "low_contrast_std_threshold": 0.05,
        "saturated_pixel_fraction_threshold": 0.2,
        "tiny_side_threshold": 32,
        "flag_top_k_examples": 25,
    },
}


def merge_data_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge user overrides into default data configuration.
    """
    config = deepcopy(DEFAULT_DATA_CONFIG)
    if overrides:
        config.update(overrides)
    return config
