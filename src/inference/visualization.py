from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.inference.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from src.inference.segmentation import normalize_map


def denormalize_tensor(
    image: torch.Tensor,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    """
    Undo ImageNet normalization for [C, H, W] or [B, C, H, W].
    """
    batched = image.ndim == 4
    if not batched:
        image = image.unsqueeze(0)

    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(1, -1, 1, 1)
    out = (image * std_t + mean_t).clamp(0.0, 1.0)
    return out if batched else out[0]


def tensor_to_pil(image: torch.Tensor):
    """
    Convert [C, H, W] tensor in [0, 1] to PIL image.
    """
    import numpy as np
    from PIL import Image

    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def colorize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Simple blue-red heatmap for [H, W] values in any range. Returns [3, H, W].
    """
    h = normalize_map(heatmap.unsqueeze(0))[0].detach().float()
    red = h
    green = 1.0 - (h - 0.5).abs() * 2.0
    blue = 1.0 - h
    return torch.stack([red, green.clamp_min(0.0), blue], dim=0).clamp(0.0, 1.0)


def overlay_heatmap(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float = 0.45,
    denormalize: bool = True,
) -> torch.Tensor:
    """
    Overlay [H, W] heatmap on [C, H, W] image. Returns [3, H, W].
    """
    if image.ndim == 4:
        image = image[0]

    if denormalize:
        image = denormalize_tensor(image)

    heatmap = heatmap.detach().float()
    if tuple(heatmap.shape[-2:]) != tuple(image.shape[-2:]):
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=tuple(image.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    color = colorize_heatmap(heatmap).to(image.device)
    return ((1.0 - float(alpha)) * image[:3] + float(alpha) * color).clamp(0.0, 1.0)


def overlay_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    color: Tuple[float, float, float] = (1.0, 0.1, 0.1),
    alpha: float = 0.45,
    denormalize: bool = True,
) -> torch.Tensor:
    """
    Overlay boolean/float mask on [C, H, W] image.
    """
    if image.ndim == 4:
        image = image[0]

    if denormalize:
        image = denormalize_tensor(image)

    mask = mask.detach().float()
    if tuple(mask.shape[-2:]) != tuple(image.shape[-2:]):
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=tuple(image.shape[-2:]),
            mode="nearest",
        )[0, 0]

    color_t = torch.tensor(color, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return torch.where(mask.bool().unsqueeze(0), (1.0 - alpha) * image[:3] + alpha * color_t, image[:3])


def save_tensor_image(image: torch.Tensor, path: Union[str, Path]) -> None:
    """
    Save [C, H, W] image tensor to disk.
    """
    tensor_to_pil(image).save(path)
