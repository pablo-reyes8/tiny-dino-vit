"""Expose the public DINO inference, segmentation, and visualization API."""

from src.inference.forward import (
    dino_forward,
    extract_cls_features,
    extract_patch_tokens,
    infer_device,
    patch_feature_map,
    predict_image,
)
from src.inference.image_metrics import (
    attention_quality_metrics,
    dino_image_metrics,
    feature_quality_metrics,
    segmentation_metrics,
)
from src.inference.pipeline import run_dino_inference
from src.inference.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ensure_chw_float,
    load_image,
    normalize_image_tensor,
    pil_to_tensor,
    prepare_image,
    prepare_image_batch,
    resize_image_tensor,
)
from src.inference.segmentation import (
    attention_cls_map,
    foreground_mask_from_cls,
    foreground_mask_from_map,
    kmeans_segment_patches,
    normalize_map,
    patch_similarity_map,
    upsample_mask_or_map,
)
from src.inference.visualization import (
    colorize_heatmap,
    denormalize_tensor,
    overlay_heatmap,
    overlay_mask,
    save_tensor_image,
    tensor_to_pil,
)

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "attention_cls_map",
    "attention_quality_metrics",
    "colorize_heatmap",
    "denormalize_tensor",
    "dino_forward",
    "dino_image_metrics",
    "ensure_chw_float",
    "extract_cls_features",
    "extract_patch_tokens",
    "feature_quality_metrics",
    "foreground_mask_from_cls",
    "foreground_mask_from_map",
    "infer_device",
    "kmeans_segment_patches",
    "load_image",
    "normalize_image_tensor",
    "normalize_map",
    "overlay_heatmap",
    "overlay_mask",
    "patch_feature_map",
    "patch_similarity_map",
    "pil_to_tensor",
    "predict_image",
    "prepare_image",
    "prepare_image_batch",
    "resize_image_tensor",
    "run_dino_inference",
    "save_tensor_image",
    "segmentation_metrics",
    "tensor_to_pil",
    "upsample_mask_or_map",
]
