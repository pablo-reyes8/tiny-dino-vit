# DINO Inference API

FastAPI service for trained DINO checkpoints.

Run:

```bash
set DINO_CHECKPOINT_PATH=checkpoints/dino_tinyimagenet/latest.pt
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

Optional environment variables:

- `DINO_CONFIG_PATH`: JSON model config if the checkpoint does not contain config metadata.
- `DINO_DEVICE`: `auto`, `cpu` or `cuda`.
- `DINO_MODEL`: `teacher` or `student`.
- `DINO_API_IMAGE_SIZE`: optional resize size before inference.
- `DINO_API_MAX_UPLOAD_MB`: upload guardrail, default `20`.
- `DINO_API_MAX_FEATURE_VALUES`: maximum CLS values returned when `return_features=true`.

Main endpoints:

- `GET /health`: process health without forcing model load.
- `GET /ready`: loads checkpoint and reports readiness.
- `GET /metadata`: model/checkpoint/config metadata.
- `POST /v1/infer`: multipart image inference for frontend forms.
- `POST /v1/infer/base64`: JSON base64 image inference for backend-to-backend calls.
- `POST /v1/infer/batch`: multipart batch inference.
- `POST /v1/segment`: focused segmentation endpoint.
- `POST /v1/features`: focused feature endpoint.
- `POST /v1/metrics`: focused image/model metric endpoint.

Responses include DINO metrics and optional base64 PNG artifacts:

- `overlay`: heatmap over the input image.
- `mask_overlay`: foreground mask over the input image.
- `heatmap`: standalone DINO score heatmap.

OpenAPI docs are available at `/docs` once the server is running.
