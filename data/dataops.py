from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import torch
import torch.nn.functional as F


@dataclass
class DataArtifact:
    name: str
    path: str
    artifact_type: str
    created_at: str
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Union[str, Path], payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate core DataOps invariants before ingestion/training.
    """
    errors = []
    warnings = []

    required = ["dataset_name", "seed", "batch_size", "global_crop_size", "local_crop_size"]
    for key in required:
        if key not in config:
            errors.append(f"Missing required config key: {key}")

    if int(config.get("batch_size", 1)) <= 0:
        errors.append("batch_size must be > 0")
    if int(config.get("global_crop_size", 1)) <= 0:
        errors.append("global_crop_size must be > 0")
    if int(config.get("local_crop_size", 1)) <= 0:
        errors.append("local_crop_size must be > 0")
    if int(config.get("num_global_crops", 0)) < 1:
        errors.append("num_global_crops must be >= 1")
    if int(config.get("num_local_crops", 0)) < 0:
        errors.append("num_local_crops must be >= 0")

    val_ratio = float(config.get("val_ratio", 0.1))
    if not 0.0 < val_ratio < 1.0:
        warnings.append("val_ratio should be in (0, 1) when manual validation split is needed")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def environment_fingerprint() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def image_to_float_tensor(image: Any) -> torch.Tensor:
    """
    Convert PIL-like image to [C, H, W] float tensor in [0, 1].
    """
    import numpy as np

    array = np.array(image.convert("RGB") if hasattr(image, "convert") else image, copy=True)
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float()
    if tensor.max() > 2.0:
        tensor = tensor / 255.0
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor.clamp(0.0, 1.0)


def laplacian_variance(image_tensor: torch.Tensor) -> float:
    """
    Blur proxy: variance of a Laplacian response on grayscale image.
    """
    gray = image_tensor.mean(dim=0, keepdim=True).unsqueeze(0)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    response = F.conv2d(gray, kernel, padding=1)
    return float(response.var(unbiased=False).item())


def assess_image_quality(
    image: Any,
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Dirty-data checks for image datasets: blur, brightness, contrast, saturation and size.
    """
    thresholds = thresholds or {}
    tensor = image_to_float_tensor(image)
    _, height, width = tensor.shape

    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item())
    blur_var = laplacian_variance(tensor)
    saturated_low = float((tensor <= 0.01).float().mean().item())
    saturated_high = float((tensor >= 0.99).float().mean().item())
    channel_delta = float((tensor.max(dim=0).values - tensor.min(dim=0).values).mean().item())

    blur_threshold = float(thresholds.get("blur_variance_threshold", 0.002))
    dark_threshold = float(thresholds.get("dark_mean_threshold", 0.08))
    bright_threshold = float(thresholds.get("bright_mean_threshold", 0.92))
    contrast_threshold = float(thresholds.get("low_contrast_std_threshold", 0.05))
    saturated_threshold = float(thresholds.get("saturated_pixel_fraction_threshold", 0.2))
    tiny_side_threshold = int(thresholds.get("tiny_side_threshold", 32))

    flags = []
    if blur_var < blur_threshold:
        flags.append("blurry")
    if mean < dark_threshold:
        flags.append("too_dark")
    if mean > bright_threshold:
        flags.append("too_bright")
    if std < contrast_threshold:
        flags.append("low_contrast")
    if max(saturated_low, saturated_high) > saturated_threshold:
        flags.append("saturated")
    if min(height, width) < tiny_side_threshold:
        flags.append("too_small")
    if channel_delta < 1e-4:
        flags.append("effectively_grayscale")

    return {
        "width": int(width),
        "height": int(height),
        "mean": mean,
        "std": std,
        "blur_laplacian_var": blur_var,
        "saturated_low_frac": saturated_low,
        "saturated_high_frac": saturated_high,
        "channel_delta_mean": channel_delta,
        "flags": flags,
    }


def summarize_quality(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    samples = list(samples)
    if not samples:
        return {"num_samples": 0, "flag_counts": {}, "metrics": {}}

    flag_counts: Dict[str, int] = {}
    metric_keys = [
        "mean",
        "std",
        "blur_laplacian_var",
        "saturated_low_frac",
        "saturated_high_frac",
        "channel_delta_mean",
        "width",
        "height",
    ]

    for sample in samples:
        for flag in sample.get("flags", []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    metrics = {}
    for key in metric_keys:
        values = torch.tensor([float(sample[key]) for sample in samples], dtype=torch.float32)
        metrics[key] = {
            "mean": float(values.mean().item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }

    return {
        "num_samples": len(samples),
        "flag_counts": flag_counts,
        "flag_rates": {key: value / len(samples) for key, value in flag_counts.items()},
        "metrics": metrics,
    }


def profile_hf_dataset(
    hf_dataset: Any,
    max_examples_per_split: int = 128,
    quality_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Profile a Hugging Face dataset object without materializing the full dataset.
    """
    profile: Dict[str, Any] = {
        "created_at": utc_now(),
        "splits": {},
    }

    for split_name, split in hf_dataset.items():
        n = len(split)
        sample_count = min(int(max_examples_per_split), n)
        labels = []
        image_modes = {}
        image_sizes = {}
        quality_samples = []
        suspect_examples = []
        flag_top_k = int((quality_thresholds or {}).get("flag_top_k_examples", 25))

        for idx in range(sample_count):
            sample = split[idx]
            if "label" in sample:
                labels.append(int(sample["label"]))
            if "image" in sample:
                image = sample["image"]
                image_modes[str(getattr(image, "mode", "unknown"))] = image_modes.get(str(getattr(image, "mode", "unknown")), 0) + 1
                image_sizes[str(getattr(image, "size", "unknown"))] = image_sizes.get(str(getattr(image, "size", "unknown")), 0) + 1
                quality = assess_image_quality(image, thresholds=quality_thresholds)
                quality["index"] = int(idx)
                quality_samples.append(quality)
                if quality["flags"] and len(suspect_examples) < flag_top_k:
                    suspect_examples.append({
                        "index": int(idx),
                        "flags": quality["flags"],
                        "width": quality["width"],
                        "height": quality["height"],
                        "mean": quality["mean"],
                        "std": quality["std"],
                        "blur_laplacian_var": quality["blur_laplacian_var"],
                    })

        label_hist = {}
        for label in labels:
            label_hist[str(label)] = label_hist.get(str(label), 0) + 1

        profile["splits"][split_name] = {
            "num_rows": int(n),
            "sampled_rows": int(sample_count),
            "features": list(getattr(split, "features", {}).keys()) if hasattr(split, "features") else None,
            "sampled_label_histogram": label_hist,
            "sampled_image_modes": image_modes,
            "sampled_image_sizes": image_sizes,
            "sampled_quality_summary": summarize_quality(quality_samples),
            "suspect_examples": suspect_examples,
        }

    return profile


def inspect_dataloader_batch(batch: Any) -> Dict[str, Any]:
    """
    Return shape/type diagnostics for a train or eval dataloader batch.
    """
    if isinstance(batch, list):
        return {
            "batch_type": "list",
            "num_crops": len(batch),
            "crops": [
                {
                    "index": idx,
                    "shape": list(crop.shape) if torch.is_tensor(crop) else None,
                    "dtype": str(crop.dtype) if torch.is_tensor(crop) else str(type(crop)),
                }
                for idx, crop in enumerate(batch)
            ],
        }

    if isinstance(batch, tuple):
        return {
            "batch_type": "tuple",
            "items": [inspect_dataloader_batch(item) for item in batch],
        }

    if torch.is_tensor(batch):
        return {
            "batch_type": "tensor",
            "shape": list(batch.shape),
            "dtype": str(batch.dtype),
            "min": float(batch.float().min().item()) if batch.numel() else None,
            "max": float(batch.float().max().item()) if batch.numel() else None,
            "mean": float(batch.float().mean().item()) if batch.numel() else None,
        }

    return {
        "batch_type": str(type(batch)),
        "repr": repr(batch)[:500],
    }


def document_artifact(
    path: Union[str, Path],
    name: Optional[str] = None,
    artifact_type: str = "file",
    metadata: Optional[Dict[str, Any]] = None,
) -> DataArtifact:
    path = Path(path)
    sha256 = file_sha256(path) if path.is_file() else None
    size_bytes = path.stat().st_size if path.exists() and path.is_file() else None
    return DataArtifact(
        name=name or path.name,
        path=str(path),
        artifact_type=artifact_type,
        created_at=utc_now(),
        sha256=sha256,
        size_bytes=size_bytes,
        metadata=metadata or {},
    )


def build_data_card(
    config: Dict[str, Any],
    profile: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Iterable[DataArtifact]] = None,
) -> Dict[str, Any]:
    """
    Build a machine-readable data card for governance and reproducibility.
    """
    return {
        "created_at": utc_now(),
        "dataset_name": config.get("dataset_name"),
        "dataset_revision": config.get("dataset_revision"),
        "intended_use": "DINO self-supervised pretraining and post-training evaluation",
        "governance": {
            "owner": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "pii_expected": False,
            "license_review_required": True,
            "reproducibility": "Config, profile, environment and artifacts are stored as JSON sidecars.",
        },
        "config": config,
        "validation": validation or validate_data_config(config),
        "profile": profile or {},
        "environment": environment_fingerprint(),
        "artifacts": [asdict(a) for a in artifacts] if artifacts else [],
    }
