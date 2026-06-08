from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def feature_quality_metrics(outputs: Dict[str, Any], eps: float = 1e-8) -> Dict[str, float]:
    """
    Metrics that summarize CLS and patch-token quality for inference batches.
    """
    metrics: Dict[str, float] = {}

    if "cls" in outputs:
        cls = outputs["cls"].detach().float()
        cls_norm = cls.norm(dim=-1)
        metrics.update({
            "cls_norm_mean": cls_norm.mean().item(),
            "cls_norm_std": cls_norm.std(unbiased=False).item(),
            "cls_dim_std_mean": cls.std(dim=0, unbiased=False).mean().item(),
        })

        if cls.shape[0] > 1:
            cls_cos = F.normalize(cls, dim=-1, eps=eps) @ F.normalize(cls, dim=-1, eps=eps).t()
            mask = ~torch.eye(cls.shape[0], dtype=torch.bool, device=cls.device)
            metrics["batch_cls_offdiag_cosine"] = cls_cos[mask].mean().item()

    if "patches" in outputs:
        patches = outputs["patches"].detach().float()
        patch_norm = patches.norm(dim=-1)
        metrics.update({
            "patch_norm_mean": patch_norm.mean().item(),
            "patch_norm_std": patch_norm.std(unbiased=False).item(),
            "patch_dim_std_mean": patches.reshape(-1, patches.shape[-1]).std(dim=0, unbiased=False).mean().item(),
        })

        patch_normed = F.normalize(patches, dim=-1, eps=eps)
        gram = patch_normed @ patch_normed.transpose(-1, -2)
        num_patches = patches.shape[1]
        offdiag = gram[:, ~torch.eye(num_patches, dtype=torch.bool, device=patches.device)]
        metrics.update({
            "patch_offdiag_cosine_mean": offdiag.mean().item(),
            "patch_offdiag_cosine_abs_mean": offdiag.abs().mean().item(),
        })

    return metrics


@torch.no_grad()
def attention_quality_metrics(
    outputs: Dict[str, Any],
    layer: int = -1,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Summarize attention sharpness/diversity for a selected layer.
    """
    attentions = outputs.get("attentions")
    if not attentions:
        return {}

    attn = attentions[layer].detach().float()
    probs = attn.clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1)

    return {
        "attention_entropy_mean": entropy.mean().item(),
        "attention_entropy_std": entropy.std(unbiased=False).item(),
        "attention_max_prob_mean": attn.max(dim=-1).values.mean().item(),
        "attention_min_prob_mean": attn.min(dim=-1).values.mean().item(),
    }


@torch.no_grad()
def segmentation_metrics(mask: torch.Tensor, score_map: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Simple diagnostics for generated masks/maps.
    """
    mask_f = mask.detach().float()
    metrics = {
        "mask_foreground_frac": mask_f.mean().item(),
        "mask_area_min": mask_f.flatten(1).mean(dim=1).min().item(),
        "mask_area_max": mask_f.flatten(1).mean(dim=1).max().item(),
    }

    if score_map is not None:
        scores = score_map.detach().float()
        metrics.update({
            "score_map_mean": scores.mean().item(),
            "score_map_std": scores.std(unbiased=False).item(),
            "score_map_min": scores.min().item(),
            "score_map_max": scores.max().item(),
        })

    return metrics


@torch.no_grad()
def dino_image_metrics(outputs: Dict[str, Any]) -> Dict[str, float]:
    """
    Collect post-training image inference metrics from one model output dict.
    """
    metrics = {}
    metrics.update(feature_quality_metrics(outputs))
    metrics.update(attention_quality_metrics(outputs))
    return metrics
