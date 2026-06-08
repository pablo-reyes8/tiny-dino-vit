# Inference

## Overview

The inference pipeline takes a trained checkpoint and an input image and produces:

- A **CLS token** — the global image representation (one vector per image).
- **Patch tokens** — a spatial feature map (one vector per image patch).
- **Attention maps** — per-head self-attention from the last transformer block.
- A **segmentation mask** — binary foreground/background derived from attention or patch similarity.
- A **heatmap** — colorized visualization of patch-level activations.
- **Image metrics** — DINO-specific quality scores (entropy, coverage, etc.).

---

## Inference Pipeline

**File:** `src/inference/pipeline.py`  
**Function:** `run_dino_inference(model, image, ...)`

The pipeline is composed of three stages:

```
Input image (PIL / Tensor / path)
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │   resize → normalize → batch
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   dino_forward()    │   backbone + head → cls, patches, attentions
└─────────────────────┘
        │
        ├──→ cls_token          [1, D]
        ├──→ patch_tokens       [1, N, D]
        └──→ attention_maps     [1, H, N, N] × n_layers
                │
                ▼
┌─────────────────────┐
│   Segmentation      │   similarity map → binary mask → heatmap
└─────────────────────┘
        │
        ├──→ segmentation_mask  [H_img, W_img]  (bool)
        ├──→ heatmap_colored    [H_img, W_img, 3] (RGB)
        └──→ metrics            Dict[str, float]
```

---

## Forward Pass

**File:** `src/inference/forward.py`  
**Function:** `dino_forward(model, batch, return_attention=False)`

Runs the backbone with `return_features=True` and optionally captures attention maps from the last transformer block. Returns a dictionary:

```python
{
    "cls":         Tensor[B, D],
    "patches":     Tensor[B, N, D],
    "grid_size":   Tuple[int, int],
    "attentions":  List[Tensor[B, H, N+spec, N+spec]],  # if requested
}
```

The attention tensors include special tokens (CLS, registers) in both dimensions. When visualizing attention, the row corresponding to the CLS token is extracted: `attn[:, :, 0, spec:]` gives the CLS-to-patch attention across all heads, reshaped to `[H, Gh, Gw]`.

---

## Preprocessing

**File:** `src/inference/preprocessing.py`

Images are normalized to match the training distribution:

```python
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

The preprocessing module accepts PIL images, file paths, numpy arrays, and tensors. Images are resized to `img_size × img_size` with bicubic interpolation before being passed to the model.

---

## Segmentation

**File:** `src/inference/segmentation.py`

Three segmentation strategies are available:

### 1. CLS Similarity Map (`patch_similarity_map`)

For each patch token, compute the cosine similarity with the CLS token:

```
similarity[i] = cosine(patch_i, cls) = (patch_i · cls) / (||patch_i|| ||cls||)
```

High similarity means the patch is most relevant to the global image representation. Threshold at the median to produce a binary foreground mask.

### 2. Attention-Based Mask (`foreground_mask`)

Use the last-block CLS-to-patch attention averaged over heads:

```
attn_mean[i] = (1/H) * Σ_h attn_h[CLS, patch_i]
```

Threshold at the top-K percentile to segment the attended regions. This often highlights the primary object more precisely than the similarity map.

### 3. Multi-Head Attention Maps

Extract per-head attention maps to visualize what different attention heads specialize in. Some heads reliably track object boundaries, others track texture, others track background context. The inference module returns all heads so they can be analyzed individually.

---

## Visualization

**File:** `src/inference/visualization.py`

### `colorize_heatmap(values, colormap="bwr")`

Maps a 2D array of scalar values (e.g. similarity scores) to an RGB image using a blue-white-red colormap. Values near the mean are white; high values are red; low values are blue.

### `overlay_heatmap(image, heatmap, alpha=0.5)`

Blends the colorized heatmap over the original image with transparency `alpha`.

### `overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4)`

Overlays a binary segmentation mask as a semi-transparent colored region on the original image.

---

## Image Metrics

**File:** `src/inference/image_metrics.py`

DINO-specific metrics computed from the model outputs:

| Metric | Description |
|---|---|
| `entropy` | Shannon entropy of the CLS attention distribution. High entropy → attention is spread over many patches (diffuse). Low entropy → attention is focused (object-centric). |
| `coverage` | Fraction of patches with above-median similarity to the CLS token. |
| `patch_variance` | Average L2 norm variance across patch tokens. Low variance may indicate representation collapse. |
| `segmentation_iou` | (If ground-truth mask provided) Intersection over Union of the predicted mask. |
| `segmentation_precision` | Precision of the predicted mask against ground truth. |

---

## Command-Line Inference

```bash
# Full inference: features + segmentation + heatmap
python scripts/infer_dino.py \
    --checkpoint checkpoints/dino_vits8/dino_vits8_best.pt \
    --image path/to/image.jpg \
    --output outputs/

# Extract features only (no visualization)
python scripts/infer_dino.py \
    --checkpoint checkpoints/dino_vits8/dino_vits8_best.pt \
    --image path/to/image.jpg \
    --features-only

# Batch inference on a directory
python scripts/infer_dino.py \
    --checkpoint checkpoints/dino_vits8/dino_vits8_best.pt \
    --image-dir path/to/images/ \
    --output outputs/
```

Output files for each image:
- `{name}_heatmap.png` — colorized patch similarity heatmap
- `{name}_mask.png` — binary segmentation mask
- `{name}_overlay.png` — heatmap blended over original
- `{name}_features.npy` — numpy array of CLS and patch tokens

---

## FastAPI Server

**Files:** `src/app/`

The REST API wraps the inference pipeline and exposes it over HTTP. The model is loaded lazily on the first request and cached in memory for subsequent calls.

### Starting the server

```bash
# Directly with uvicorn
DINO_CHECKPOINT=checkpoints/dino_vits8/dino_vits8_best.pt \
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --workers 1

# With Docker
docker-compose up api
```

Interactive API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DINO_CHECKPOINT` | required | Path to the `.pt` checkpoint file |
| `DINO_DEVICE` | `"auto"` | `"cuda"`, `"cpu"`, or `"auto"` |
| `DINO_MAX_IMAGE_SIZE` | `1024` | Maximum input image dimension in pixels |
| `DINO_IMG_SIZE` | `224` | Model input size (must match checkpoint) |

### Endpoints

#### `GET /health`
Liveness probe. Always returns `{"status": "ok"}`.

#### `GET /ready`
Readiness probe. Returns `{"status": "ready", "model_loaded": true}` once the model is loaded.

#### `POST /v1/infer`

Full inference: returns CLS features, patch features, and segmentation mask.

**Request:**
```json
{
  "image_b64": "<base64-encoded image bytes>",
  "return_attention": false
}
```

**Response:**
```json
{
  "cls_token": [0.12, -0.03, ...],
  "patch_tokens": [[...], ...],
  "grid_size": [14, 14],
  "segmentation_mask_b64": "<base64-encoded PNG>",
  "heatmap_b64": "<base64-encoded PNG>",
  "metrics": {
    "entropy": 4.21,
    "coverage": 0.48
  }
}
```

#### `POST /v1/features`

Returns only the CLS and patch tokens without visualization. Suitable for feature extraction pipelines.

#### `POST /v1/segment`

Returns only the segmentation mask and heatmap PNGs (base64-encoded). Suitable for visual applications.

#### `POST /v1/metrics`

Returns only DINO image quality metrics without feature vectors.

### Python Client Example

```python
import httpx, base64, json

with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = httpx.post(
    "http://localhost:8000/v1/infer",
    json={"image_b64": img_b64},
    timeout=30.0,
)

result = response.json()
print(f"CLS token shape: {len(result['cls_token'])}")
print(f"Coverage: {result['metrics']['coverage']:.3f}")
```

---

## Loading a Checkpoint Manually

```python
import torch
from src.model.dino_full_model import build_dino_bundle
from src.inference.pipeline import run_dino_inference
from PIL import Image

# Load checkpoint
ckpt = torch.load("checkpoints/dino_vits8/dino_vits8_best.pt", map_location="cpu")
cfg  = ckpt["config"]

# Build model and load weights
bundle = build_dino_bundle(cfg)
bundle["student"].load_state_dict(ckpt["student_state_dict"])
model = bundle["student"].eval()

# Run inference
image = Image.open("image.jpg")
result = run_dino_inference(model, image)

print(result["cls"].shape)          # [1, 384]
print(result["patches"].shape)      # [1, 784, 384]
print(result["mask"].shape)         # [224, 224]
```
