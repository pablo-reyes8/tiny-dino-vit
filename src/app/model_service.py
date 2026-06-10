"""Manage DINO model loading and request-level inference execution."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import torch

from src.app.config import APISettings
from src.app.image_io import image_size_hw, tensor_png_base64
from src.app.schemas import FeatureSummary, ImageArtifactResponse, InferenceResponse
from src.inference import (
    colorize_heatmap,
    overlay_heatmap,
    overlay_mask,
    run_dino_inference,
)
from src.model.dino_full_model import build_dino_bundle
from src.training.chekpoints import load_dino_checkpoint


class DINOModelService:
    """
    Runtime service for loading a DINO checkpoint and serving image inference.
    """

    def __init__(self, settings: APISettings):
        self.settings = settings
        self._lock = Lock()
        self._loaded = False
        self._bundle: Optional[Dict[str, Any]] = None
        self._model = None
        self._config: Optional[Dict[str, Any]] = None
        self._checkpoint_path = settings.checkpoint_path

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        if self._bundle is not None:
            return str(self._bundle.get("device", self.settings.device))
        return self.settings.device

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    def metadata(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "checkpoint_path": self._checkpoint_path,
            "config": self._config,
            "model_info": self._bundle.get("model_info") if self._bundle else None,
            "device": self.device,
            "model_name": self.model_name,
        }

    def load(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            if not self.settings.checkpoint_path:
                raise RuntimeError(
                    "DINO_CHECKPOINT_PATH is not configured. Set it before starting the API."
                )

            checkpoint_path = Path(self.settings.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            config = self._load_config(checkpoint_path)
            if self.settings.device != "auto":
                config["device"] = self.settings.device
            elif config.get("device", "auto") == "auto":
                config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

            bundle = build_dino_bundle(config)
            load_dino_checkpoint(
                checkpoint_path=checkpoint_path,
                student=bundle["student"],
                teacher=bundle["teacher"],
                map_location="cpu",
                strict_student=False,
                strict_teacher=False,
                load_optimizer=False,
                load_scheduler=False,
                load_scaler=False,
                load_rng_state=False,
            )

            model_name = self.settings.model_name
            if model_name not in {"student", "teacher"}:
                raise ValueError("DINO_MODEL must be 'student' or 'teacher'.")

            model = bundle[model_name].to(bundle["device"])
            model.eval()

            self._bundle = bundle
            self._model = model
            self._config = config
            self._loaded = True

    def _load_config(self, checkpoint_path: Path) -> Dict[str, Any]:
        if self.settings.config_path:
            with Path(self.settings.config_path).open("r", encoding="utf-8") as f:
                return json.load(f)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError(
                "Checkpoint does not contain a usable config. Provide DINO_CONFIG_PATH as JSON."
            )
        return config

    def infer_image(
        self,
        image: Any,
        *,
        filename: Optional[str] = None,
        image_size: Optional[int] = None,
        segmentation: str = "cls_similarity",
        return_overlay: bool = True,
        return_mask: bool = True,
        return_heatmap: bool = True,
        return_features: bool = False,
        max_feature_values: Optional[int] = None,
    ) -> InferenceResponse:
        self.load()
        assert self._model is not None
        assert self._bundle is not None

        request_id = str(uuid.uuid4())
        image_size = image_size if image_size is not None else self.settings.default_image_size

        result = run_dino_inference(
            model=self._model,
            images=[image],
            image_size=image_size,
            device=self._bundle["device"],
            segmentation=segmentation,
        )

        outputs = result["outputs"]
        score_map = result["score_map"][0]
        mask = result["mask"][0]
        input_image = outputs["input"][0]
        grid_h, grid_w = outputs["grid_size"]

        features = None
        if return_features:
            features = self._feature_summary(
                outputs,
                max_values=max_feature_values or self.settings.max_feature_values,
            )

        overlay = None
        if return_overlay:
            overlay = ImageArtifactResponse(
                base64=tensor_png_base64(overlay_heatmap(input_image, score_map))
            )

        mask_overlay = None
        if return_mask:
            mask_overlay = ImageArtifactResponse(
                base64=tensor_png_base64(overlay_mask(input_image, mask))
            )

        heatmap = None
        if return_heatmap:
            heatmap = ImageArtifactResponse(
                base64=tensor_png_base64(colorize_heatmap(score_map))
            )

        return InferenceResponse(
            request_id=request_id,
            filename=filename,
            segmentation=segmentation,
            image_size=image_size_hw(image),
            grid_size=[int(grid_h), int(grid_w)],
            metrics={key: float(value) for key, value in result["metrics"].items()},
            features=features,
            overlay=overlay,
            mask_overlay=mask_overlay,
            heatmap=heatmap,
        )

    def _feature_summary(self, outputs: Dict[str, Any], max_values: int) -> FeatureSummary:
        cls = outputs.get("cls")
        patches = outputs.get("patches")
        grid_h, grid_w = outputs["grid_size"]

        cls_values = None
        cls_dim = None
        if torch.is_tensor(cls):
            cls_flat = cls[0].detach().float().cpu()
            cls_dim = int(cls_flat.numel())
            cls_values = [float(x) for x in cls_flat[: max(0, int(max_values))].tolist()]

        return FeatureSummary(
            cls=cls_values,
            cls_dim=cls_dim,
            patch_dim=int(patches.shape[-1]) if torch.is_tensor(patches) else None,
            num_patches=int(patches.shape[1]) if torch.is_tensor(patches) else None,
            grid_size=[int(grid_h), int(grid_w)],
        )
