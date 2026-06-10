"""Decode, encode, and summarize images for API requests and responses."""

from __future__ import annotations

import base64
import io
from typing import Any

import torch


def decode_image_bytes(data: bytes):
    from PIL import Image

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image bytes. Expected a readable image file.") from exc


def decode_base64_image(payload: str):
    if "," in payload and payload.strip().lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return decode_image_bytes(base64.b64decode(payload, validate=True))
    except Exception as exc:
        raise ValueError("Invalid base64 image payload.") from exc


def tensor_png_base64(image: torch.Tensor) -> str:
    from src.inference.visualization import tensor_to_pil

    buffer = io.BytesIO()
    tensor_to_pil(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_size_hw(image: Any) -> list[int]:
    width, height = image.size
    return [int(height), int(width)]
