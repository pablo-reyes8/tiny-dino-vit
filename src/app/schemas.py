from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SegmentationMode = Literal["cls_similarity", "attention"]
ReturnImageFormat = Literal["base64_png", "none"]


class APIStatus(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    checkpoint_path: Optional[str] = None
    device: str
    model_name: str
    version: str


class ModelMetadata(BaseModel):
    loaded: bool
    checkpoint_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None
    device: str
    model_name: str


class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image bytes. Data URLs are accepted.")
    filename: Optional[str] = None
    image_size: Optional[int] = None
    segmentation: SegmentationMode = "cls_similarity"
    return_overlay: bool = True
    return_mask: bool = True
    return_heatmap: bool = True
    return_features: bool = False
    max_feature_values: Optional[int] = None


class FeatureSummary(BaseModel):
    cls: Optional[List[float]] = None
    cls_dim: Optional[int] = None
    patch_dim: Optional[int] = None
    num_patches: Optional[int] = None
    grid_size: Optional[List[int]] = None


class ImageArtifactResponse(BaseModel):
    mime_type: str = "image/png"
    base64: str


class InferenceResponse(BaseModel):
    request_id: str
    filename: Optional[str] = None
    segmentation: SegmentationMode
    image_size: List[int]
    grid_size: List[int]
    metrics: Dict[str, float]
    features: Optional[FeatureSummary] = None
    overlay: Optional[ImageArtifactResponse] = None
    mask_overlay: Optional[ImageArtifactResponse] = None
    heatmap: Optional[ImageArtifactResponse] = None


class BatchInferenceResponse(BaseModel):
    request_id: str
    results: List[InferenceResponse]


class ErrorResponse(BaseModel):
    detail: str
