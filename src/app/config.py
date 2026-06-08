from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class APISettings:
    app_name: str = "DINO Inference API"
    version: str = "0.1.0"
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    device: str = "auto"
    model_name: str = "teacher"
    default_image_size: Optional[int] = None
    max_upload_mb: int = 20
    max_feature_values: int = 256


def get_settings() -> APISettings:
    image_size = os.getenv("DINO_API_IMAGE_SIZE")
    return APISettings(
        checkpoint_path=os.getenv("DINO_CHECKPOINT_PATH"),
        config_path=os.getenv("DINO_CONFIG_PATH"),
        device=os.getenv("DINO_DEVICE", "auto"),
        model_name=os.getenv("DINO_MODEL", "teacher"),
        default_image_size=int(image_size) if image_size else None,
        max_upload_mb=int(os.getenv("DINO_API_MAX_UPLOAD_MB", "20")),
        max_feature_values=int(os.getenv("DINO_API_MAX_FEATURE_VALUES", "256")),
    )
