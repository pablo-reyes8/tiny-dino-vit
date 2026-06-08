from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path: Union[str, Path], mode: str = "RGB"):
    """
    Load an image with PIL.
    """
    from PIL import Image

    return Image.open(path).convert(mode)


def pil_to_tensor(image) -> torch.Tensor:
    """
    Convert a PIL image to float tensor [C, H, W] in [0, 1].
    """
    import numpy as np

    array = np.asarray(image)

    if array.ndim == 2:
        array = array[:, :, None]

    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float()
    return tensor / 255.0


def ensure_chw_float(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor layout/range to [C, H, W] float.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {tuple(image.shape)}.")

    if image.shape[0] not in {1, 3} and image.shape[-1] in {1, 3}:
        image = image.permute(2, 0, 1).contiguous()

    image = image.float()

    if image.max() > 2.0:
        image = image / 255.0

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    if image.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(image.shape)}.")

    return image.clamp(0.0, 1.0)


def resize_image_tensor(
    image: torch.Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resize [C, H, W] image tensor. If size is None, returns image unchanged.
    """
    if size is None:
        return image

    if isinstance(size, int):
        size = (size, size)

    image_4d = image.unsqueeze(0)
    resized = F.interpolate(
        image_4d,
        size=tuple(size),
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )
    return resized.squeeze(0)


def normalize_image_tensor(
    image: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Apply channel-wise normalization to [C, H, W].
    """
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return (image - mean_t) / std_t


def prepare_image(
    image: Union[str, Path, torch.Tensor, object],
    image_size: Optional[Union[int, Sequence[int]]] = None,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Prepare one image for DINO inference. Returns [C, H, W].
    """
    if isinstance(image, (str, Path)):
        image = load_image(image)

    if torch.is_tensor(image):
        tensor = ensure_chw_float(image)
    else:
        tensor = pil_to_tensor(image)
        tensor = ensure_chw_float(tensor)

    tensor = resize_image_tensor(tensor, image_size)

    if normalize:
        tensor = normalize_image_tensor(tensor, mean=mean, std=std)

    return tensor


def prepare_image_batch(
    images: Iterable[Union[str, Path, torch.Tensor, object]],
    image_size: Optional[Union[int, Sequence[int]]] = None,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    normalize: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Prepare many images into a [B, C, H, W] batch.
    """
    batch = torch.stack([
        prepare_image(
            image,
            image_size=image_size,
            mean=mean,
            std=std,
            normalize=normalize,
        )
        for image in images
    ])

    if device is not None:
        batch = batch.to(device)

    return batch
