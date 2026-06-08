from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch

from src.inference.forward import dino_forward
from src.inference.image_metrics import dino_image_metrics, segmentation_metrics
from src.inference.segmentation import (
    attention_cls_map,
    foreground_mask_from_cls,
    patch_similarity_map,
)


@torch.no_grad()
def run_dino_inference(
    model,
    images: Union[torch.Tensor, Sequence[Any]],
    image_size: Optional[Union[int, Sequence[int]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    segmentation: str = "cls_similarity",
) -> Dict[str, Any]:
    """
    End-to-end inference helper: forward + maps + masks + metrics.
    """
    outputs = dino_forward(
        model=model,
        images=images,
        image_size=image_size,
        device=device,
        return_attention=True,
        return_all_tokens=True,
    )

    if segmentation == "attention":
        score_map = attention_cls_map(outputs)
    elif segmentation == "cls_similarity":
        score_map = patch_similarity_map(outputs, query="cls")
    else:
        raise ValueError("segmentation must be 'cls_similarity' or 'attention'.")

    mask = foreground_mask_from_cls(outputs) if segmentation == "cls_similarity" else score_map >= score_map.flatten(1).quantile(0.6, dim=1).view(-1, 1, 1)

    metrics = dino_image_metrics(outputs)
    metrics.update(segmentation_metrics(mask=mask, score_map=score_map))

    return {
        "outputs": outputs,
        "score_map": score_map,
        "mask": mask,
        "metrics": metrics,
    }
