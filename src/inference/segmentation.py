from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _grid_size(outputs: Dict[str, Any]) -> Tuple[int, int]:
    grid_h, grid_w = outputs["grid_size"]
    return int(grid_h), int(grid_w)


def patch_similarity_map(
    outputs: Dict[str, Any],
    query: str = "cls",
    patch_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Build [B, Gh, Gw] cosine-similarity maps from patch tokens.
    """
    patches = F.normalize(outputs["patches"].float(), dim=-1)

    if query == "cls":
        query_tokens = F.normalize(outputs["cls"].float(), dim=-1).unsqueeze(1)
    elif query == "patch":
        if patch_index is None:
            patch_index = patches.shape[1] // 2
        query_tokens = patches[:, patch_index : patch_index + 1]
    elif query == "mean_patch":
        query_tokens = F.normalize(patches.mean(dim=1, keepdim=True), dim=-1)
    else:
        raise ValueError("query must be 'cls', 'patch' or 'mean_patch'.")

    scores = (patches * query_tokens).sum(dim=-1)
    grid_h, grid_w = _grid_size(outputs)
    return scores.reshape(scores.shape[0], grid_h, grid_w)


def attention_cls_map(
    outputs: Dict[str, Any],
    layer: int = -1,
    head: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract CLS-to-patch attention map [B, Gh, Gw].
    """
    attentions = outputs.get("attentions")
    if not attentions:
        raise KeyError("outputs does not contain attentions. Run inference with return_attention=True.")

    attn = attentions[layer].float()
    if head is None:
        attn = attn.mean(dim=1)
    else:
        attn = attn[:, int(head)]

    num_patches = outputs["patches"].shape[1]
    cls_to_patch = attn[:, 0, -num_patches:]
    grid_h, grid_w = _grid_size(outputs)
    return cls_to_patch.reshape(cls_to_patch.shape[0], grid_h, grid_w)


def normalize_map(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Min-max normalize each map in a batch to [0, 1].
    """
    flat = values.flatten(1)
    min_v = flat.min(dim=1).values.view(-1, 1, 1)
    max_v = flat.max(dim=1).values.view(-1, 1, 1)
    return (values - min_v) / (max_v - min_v + eps)


def foreground_mask_from_map(
    score_map: torch.Tensor,
    threshold: Optional[float] = None,
    quantile: float = 0.6,
) -> torch.Tensor:
    """
    Convert a [B, Gh, Gw] score map to a boolean foreground mask.
    """
    scores = normalize_map(score_map.float())
    if threshold is None:
        threshold = torch.quantile(scores.flatten(1), float(quantile), dim=1)
        threshold = threshold.view(-1, 1, 1)
    return scores >= threshold


def foreground_mask_from_cls(
    outputs: Dict[str, Any],
    threshold: Optional[float] = None,
    quantile: float = 0.6,
) -> torch.Tensor:
    """
    Segment likely foreground using CLS-to-patch similarity.
    """
    score_map = patch_similarity_map(outputs, query="cls")
    return foreground_mask_from_map(score_map, threshold=threshold, quantile=quantile)


def upsample_mask_or_map(
    values: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Upsample [B, H, W] masks/maps to image resolution.
    """
    values = values.float().unsqueeze(1)
    out = F.interpolate(
        values,
        size=size,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )
    return out[:, 0]


def kmeans_segment_patches(
    outputs: Dict[str, Any],
    num_clusters: int = 3,
    num_iters: int = 20,
    seed: int = 0,
) -> torch.Tensor:
    """
    Cluster patch tokens per image. Returns label maps [B, Gh, Gw].
    """
    patches = F.normalize(outputs["patches"].float(), dim=-1)
    batch_size, num_patches, _ = patches.shape
    grid_h, grid_w = _grid_size(outputs)
    labels_out = []

    generator = torch.Generator(device=patches.device)
    generator.manual_seed(int(seed))

    for batch_idx in range(batch_size):
        x = patches[batch_idx]
        perm = torch.randperm(num_patches, generator=generator, device=patches.device)
        centers = x[perm[:num_clusters]].clone()

        for _ in range(int(num_iters)):
            dist = torch.cdist(x, centers)
            labels = dist.argmin(dim=1)
            new_centers = []
            for cluster_idx in range(num_clusters):
                mask = labels == cluster_idx
                if mask.any():
                    new_centers.append(x[mask].mean(dim=0))
                else:
                    new_centers.append(centers[cluster_idx])
            centers = F.normalize(torch.stack(new_centers), dim=-1)

        labels_out.append(labels.reshape(grid_h, grid_w))

    return torch.stack(labels_out)
