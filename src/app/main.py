from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from src.app.config import get_settings
from src.app.image_io import decode_base64_image, decode_image_bytes
from src.app.model_service import DINOModelService
from src.app.schemas import (
    APIStatus,
    Base64ImageRequest,
    BatchInferenceResponse,
    InferenceResponse,
    ModelMetadata,
)


settings = get_settings()
model_service = DINOModelService(settings=settings)


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Professional inference API for trained DINO models. "
            "It accepts image uploads or base64 image payloads and returns "
            "DINO segmentation maps, visual overlays, feature summaries and image metrics."
        ),
        contact={"name": "DINO Project"},
        license_info={"name": "Project license"},
        openapi_tags=[
            {"name": "system", "description": "Health, readiness and runtime metadata."},
            {"name": "inference", "description": "Image-to-image DINO inference endpoints."},
            {"name": "analysis", "description": "Focused segmentation/features/metrics endpoints."},
        ],
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(application)
    return application


def _ensure_upload_size(data: bytes) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large. Max upload size is {settings.max_upload_mb} MB.",
        )


async def _read_upload_image(file: UploadFile):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {file.content_type}. Expected image/*.",
        )

    data = await file.read()
    _ensure_upload_size(data)

    try:
        return decode_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Inference failed: {exc}")


def register_routes(application: FastAPI) -> None:
    @application.get("/health", response_model=APIStatus, tags=["system"])
    def health() -> APIStatus:
        return APIStatus(
            status="ok" if model_service.loaded or settings.checkpoint_path else "degraded",
            model_loaded=model_service.loaded,
            checkpoint_path=settings.checkpoint_path,
            device=model_service.device,
            model_name=settings.model_name,
            version=settings.version,
        )

    @application.get("/ready", response_model=APIStatus, tags=["system"])
    def ready() -> APIStatus:
        try:
            model_service.load()
        except Exception as exc:
            raise _service_error(exc)

        return APIStatus(
            status="ok",
            model_loaded=True,
            checkpoint_path=settings.checkpoint_path,
            device=model_service.device,
            model_name=settings.model_name,
            version=settings.version,
        )

    @application.get("/metadata", response_model=ModelMetadata, tags=["system"])
    def metadata() -> ModelMetadata:
        return ModelMetadata(**model_service.metadata())

    @application.post("/v1/infer", response_model=InferenceResponse, tags=["inference"])
    async def infer_upload(
        file: UploadFile = File(..., description="Image file sent as multipart/form-data."),
        image_size: Optional[int] = Form(None),
        segmentation: str = Form("cls_similarity"),
        return_overlay: bool = Form(True),
        return_mask: bool = Form(True),
        return_heatmap: bool = Form(True),
        return_features: bool = Form(False),
        max_feature_values: Optional[int] = Form(None),
    ) -> InferenceResponse:
        image = await _read_upload_image(file)
        try:
            return model_service.infer_image(
                image,
                filename=file.filename,
                image_size=image_size,
                segmentation=segmentation,
                return_overlay=return_overlay,
                return_mask=return_mask,
                return_heatmap=return_heatmap,
                return_features=return_features,
                max_feature_values=max_feature_values,
            )
        except Exception as exc:
            raise _service_error(exc)

    @application.post("/v1/infer/base64", response_model=InferenceResponse, tags=["inference"])
    def infer_base64(request: Base64ImageRequest) -> InferenceResponse:
        try:
            image = decode_base64_image(request.image_base64)
            return model_service.infer_image(
                image,
                filename=request.filename,
                image_size=request.image_size,
                segmentation=request.segmentation,
                return_overlay=request.return_overlay,
                return_mask=request.return_mask,
                return_heatmap=request.return_heatmap,
                return_features=request.return_features,
                max_feature_values=request.max_feature_values,
            )
        except Exception as exc:
            raise _service_error(exc)

    @application.post("/v1/infer/batch", response_model=BatchInferenceResponse, tags=["inference"])
    async def infer_batch_upload(
        files: List[UploadFile] = File(..., description="Multiple image files."),
        image_size: Optional[int] = Form(None),
        segmentation: str = Form("cls_similarity"),
        return_overlay: bool = Form(True),
        return_mask: bool = Form(True),
        return_heatmap: bool = Form(False),
        return_features: bool = Form(False),
    ) -> BatchInferenceResponse:
        import uuid

        request_id = str(uuid.uuid4())
        results = []

        for file in files:
            image = await _read_upload_image(file)
            try:
                results.append(
                    model_service.infer_image(
                        image,
                        filename=file.filename,
                        image_size=image_size,
                        segmentation=segmentation,
                        return_overlay=return_overlay,
                        return_mask=return_mask,
                        return_heatmap=return_heatmap,
                        return_features=return_features,
                    )
                )
            except Exception as exc:
                raise _service_error(exc)

        return BatchInferenceResponse(request_id=request_id, results=results)

    @application.post("/v1/segment", response_model=InferenceResponse, tags=["analysis"])
    async def segment_image(
        file: UploadFile = File(...),
        image_size: Optional[int] = Form(None),
        segmentation: str = Form("cls_similarity"),
    ) -> InferenceResponse:
        image = await _read_upload_image(file)
        try:
            return model_service.infer_image(
                image,
                filename=file.filename,
                image_size=image_size,
                segmentation=segmentation,
                return_overlay=True,
                return_mask=True,
                return_heatmap=True,
                return_features=False,
            )
        except Exception as exc:
            raise _service_error(exc)

    @application.post("/v1/features", response_model=InferenceResponse, tags=["analysis"])
    async def image_features(
        file: UploadFile = File(...),
        image_size: Optional[int] = Form(None),
        max_feature_values: Optional[int] = Form(None),
    ) -> InferenceResponse:
        image = await _read_upload_image(file)
        try:
            return model_service.infer_image(
                image,
                filename=file.filename,
                image_size=image_size,
                return_overlay=False,
                return_mask=False,
                return_heatmap=False,
                return_features=True,
                max_feature_values=max_feature_values,
            )
        except Exception as exc:
            raise _service_error(exc)

    @application.post("/v1/metrics", response_model=InferenceResponse, tags=["analysis"])
    async def image_metrics(
        file: UploadFile = File(...),
        image_size: Optional[int] = Form(None),
        segmentation: str = Form("cls_similarity"),
    ) -> InferenceResponse:
        image = await _read_upload_image(file)
        try:
            return model_service.infer_image(
                image,
                filename=file.filename,
                image_size=image_size,
                segmentation=segmentation,
                return_overlay=False,
                return_mask=False,
                return_heatmap=False,
                return_features=False,
            )
        except Exception as exc:
            raise _service_error(exc)


app = create_app()
