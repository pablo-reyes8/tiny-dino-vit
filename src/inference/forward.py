from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from src.inference.preprocessing import prepare_image_batch


def infer_device(model: nn.Module) -> torch.device:
    """
    Infer model device from parameters or buffers.
    """
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    return torch.device("cpu")


@torch.no_grad()
def dino_forward(
    model: nn.Module,
    images: Union[torch.Tensor, Sequence[Any]],
    image_size: Optional[Union[int, Sequence[int]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_attention: bool = False,
    return_all_tokens: bool = True,
    normalize_input: bool = True,
) -> Dict[str, Any]:
    """
    Run a trained DINO model and return its feature dictionary.
    """
    was_training = model.training
    model.eval()

    if device is None:
        device = infer_device(model)
    device = torch.device(device)

    if torch.is_tensor(images):
        batch = images
        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
        batch = batch.to(device)
    else:
        batch = prepare_image_batch(
            images,
            image_size=image_size,
            normalize=normalize_input,
            device=device,
        )

    outputs = model(
        batch,
        return_features=True,
        return_all_tokens=return_all_tokens,
        return_attention=return_attention,
    )

    if was_training:
        model.train()

    if not isinstance(outputs, dict):
        raise TypeError(f"Expected model to return dict, got {type(outputs)}.")

    outputs["input"] = batch
    return outputs


def extract_cls_features(outputs: Dict[str, Any], normalize: bool = True) -> torch.Tensor:
    cls = outputs["cls"]
    return torch.nn.functional.normalize(cls, dim=-1) if normalize else cls


def extract_patch_tokens(outputs: Dict[str, Any], normalize: bool = True) -> torch.Tensor:
    patches = outputs["patches"]
    return torch.nn.functional.normalize(patches, dim=-1) if normalize else patches


def patch_feature_map(outputs: Dict[str, Any], normalize: bool = True) -> torch.Tensor:
    """
    Convert patch tokens [B, N, D] to feature map [B, D, Gh, Gw].
    """
    patches = extract_patch_tokens(outputs, normalize=normalize)
    grid_h, grid_w = outputs["grid_size"]
    return patches.transpose(1, 2).reshape(patches.shape[0], patches.shape[-1], grid_h, grid_w)


@torch.no_grad()
def predict_image(
    model: nn.Module,
    image: Any,
    image_size: Optional[Union[int, Sequence[int]]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for a single image.
    """
    return dino_forward(
        model=model,
        images=[image],
        image_size=image_size,
        device=device,
        return_attention=True,
        return_all_tokens=True,
    )
