# DINO — Self-Supervised Vision Transformer

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A clean, modular PyTorch implementation of **DINO** (Self-**DI**stillation with **NO** labels) — a self-supervised learning framework for Vision Transformers. The codebase adds Gram-matrix regularization inspired by DINOv3, a production-ready FastAPI inference server, and a full suite of training configs from ViT-Tiny up to ViT-Large.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Multi-GPU Training](#multi-gpu-training)
- [Inference](#inference)
- [API Server](#api-server)
- [Docker](#docker)
- [Tests](#tests)
- [Ablations](#ablations)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

DINO trains a Vision Transformer (ViT) student to match the output distribution of an exponential moving average (EMA) teacher — without any labels. The result is a backbone that produces rich, semantically meaningful features directly usable for downstream tasks such as:

- Linear probing / k-NN classification
- Semantic segmentation (from attention maps)
- Dense feature matching
- Zero-shot retrieval

This implementation follows the original [DINO paper](https://arxiv.org/abs/2104.14294) (Caron et al., 2021) and incorporates optional Gram-matrix loss from the DINOv3 line of work.

---

## Key Features

| Feature                           | Detail                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------ |
| **ViT backbone**            | Configurable depth / width / heads; learned, RoPE, or no positional embeddings |
| **Multi-crop augmentation** | 2 global (224 px) + N local (96 px) crops per image                            |
| **EMA teacher**             | Cosine momentum schedule; optional CPU offload for low-VRAM GPUs               |
| **Gram regularization**     | Patch-level Gram-matrix matching (DINOv3-style, optional)                      |
| **Mixed precision**         | Automatic bfloat16 / float16 / float32 selection                               |
| **Resumable training**      | Saves optimizer, scheduler, scaler, and RNG state                              |
| **Multi-GPU training**      | DDP data parallelism with `torchrun`; experimental ViT block sharding        |
| **FastAPI server**          | `/v1/infer`, `/v1/segment`, `/v1/features`, `/v1/metrics` endpoints    |
| **Data governance**         | Quality checks (blur, brightness, contrast), dataset profiling, data cards     |
| **YAML configs**            | Ready-made presets for Tiny → Small → Base → Large                          |

---

## Architecture

```
Input image
    │
    ▼
┌───────────────────────────────┐
│  Multi-crop augmentation      │  2 global crops + N local crops
└───────────────────────────────┘
    │
    ├──────────────── student (ViT) ──────────────────┐
    │                                                 │
    └──────────────── teacher (ViT, EMA) ─────────────┤
                                                      ▼
                                           ┌──────────────────┐
                                           │   DINO Head      │  MLP + L2-norm + WN
                                           └──────────────────┘
                                                      │
                                           ┌──────────────────┐
                                           │   DINO Loss      │  Cross-entropy over softmax
                                           │  + Gram Loss     │  Patch Gram matrix matching
                                           └──────────────────┘
```

**ViT backbone components:**

- `PatchEmbedding` — Conv2d projection of image patches
- `MultiHeadSelfAttention` — with optional 2D Rotary Position Embedding (RoPE)
- `TransformerEncoderBlock` — Pre-LN: Attention → FFN with DropPath
- `DINOHead` — bottleneck MLP → L2 norm → weight-normalised output layer

---

## Project Structure

```
dino/
├── configs/
│   ├── set_up.yaml          # ViT-Tiny/8 baseline (TinyImageNet, 64×64)
│   ├── vit_small.yaml       # ViT-Small/8  (~22 M params)
│   ├── vit_base.yaml        # ViT-Base/16  (~86 M params)
│   ├── vit_large.yaml       # ViT-Large/16 (~307 M params)
│   ├── multi_gpu.yaml       # DDP multi-GPU preset
│   └── debug.yaml           # Smoke-test config (2 epochs, 10 batches)
├── data/
│   ├── load_data.py         # TinyImageNet dataloader + multi-crop transforms
│   ├── data_config.py       # Default data config + merge helper
│   ├── dataops.py           # Quality checks, profiling, data cards
│   └── data_utils.py        # Visualization utilities
├── notebooks/
│   └── DINO_Implementation.ipynb
├── scripts/
│   ├── train_dino.py        # Training entry point
│   ├── infer_dino.py        # Inference entry point
│   ├── train_dino_multi_gpu.py # Multi-GPU DDP training entry point
│   └── inspect_data.py      # Dataset inspection + reporting
├── src/
│   ├── app/                 # FastAPI inference server
│   │   ├── main.py
│   │   ├── model_service.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   └── image_io.py
│   ├── model/               # ViT + DINO architecture
│   │   ├── vit_model.py
│   │   ├── attention.py
│   │   ├── transformer_block.py
│   │   ├── embeddings.py
│   │   ├── rope.py
│   │   ├── dino_head.py
│   │   ├── dino_full_model.py
│   │   └── loss/
│   │       ├── dino_loss.py
│   │       └── dino_gram_loss.py
│   ├── training/
│   │   ├── train_dino.py    # Training orchestrator
│   │   ├── train_one_epoch.py
│   │   ├── ema.py
│   │   ├── schedulers.py
│   │   ├── metrics.py
│   │   └── chekpoints.py
│   ├── paralel/             # DDP/data-parallel/model-parallel helpers
│   │   ├── distributed.py
│   │   ├── data_parallel.py
│   │   ├── model_parallel.py
│   │   └── config.py
│   └── inference/
│       ├── pipeline.py      # End-to-end inference
│       ├── forward.py
│       ├── segmentation.py
│       └── visualization.py
├── tests/
├── Dockerfile
├── Dockerfile.api
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-api.txt
└── .gitignore
```

---

## Installation

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA 11.8+ (optional but recommended).

### From source

```bash
git clone https://github.com/your-org/dino.git
cd dino
pip install -e .
```

### Training dependencies only

```bash
pip install -r requirements.txt
pip install -r requirements-parallel.txt  # YAML configs / multi-GPU CLI
```

### API server dependencies

```bash
pip install -r requirements-api.txt
```

### All extras (development + API)

```bash
pip install -e ".[api,dev]"
```

---

## Configuration

Five ready-made YAML presets live in `configs/`. Pick one based on available VRAM and dataset:

| Config             | Model        | Params | `img_size` | `patch_size` | Batch | Min VRAM |
| ------------------ | ------------ | ------ | ------------ | -------------- | ----- | -------- |
| `set_up.yaml`    | ViT-Tiny/8   | ~5 M   | 64           | 8              | 64    | 4 GB     |
| `vit_small.yaml` | ViT-Small/8  | ~22 M  | 224          | 8              | 128   | 12 GB    |
| `vit_base.yaml`  | ViT-Base/16  | ~86 M  | 224          | 16             | 64    | 24 GB    |
| `vit_large.yaml` | ViT-Large/16 | ~307 M | 224          | 16             | 32    | 40 GB    |
| `debug.yaml`     | micro        | <1 M   | 32           | 8              | 8     | CPU      |

All sections of a config can be overridden at runtime via the training script.

### Key parameters

```yaml
model:
  pos_embed_type: learned  # "learned" | "rope" | "none"
  num_register_tokens: 4   # improves dense prediction (DINOv2)

dino:
  use_gram_loss: true       # enable DINOv3-style Gram regularization
  gram_loss_weight: 0.05

train:
  teacher_offload: true     # offload teacher to CPU (saves ~50% VRAM)
  grad_accum_steps: 4       # simulate larger effective batch size
```

---

## Training

```bash
# ViT-Tiny on TinyImageNet (default, fits 4 GB VRAM)
python scripts/train_dino.py --config configs/set_up.yaml

# ViT-Small
python scripts/train_dino.py --config configs/vit_small.yaml

# Resume from a checkpoint
python scripts/train_dino.py --config configs/vit_small.yaml \
    --resume checkpoints/dino_vits8/dino_vits8_e050.pt

# Quick smoke test (CPU, ~1 minute)
python scripts/train_dino.py --config configs/debug.yaml
```

Checkpoints, metrics JSONL, and JSON sidecars are saved to `checkpoint.ckpt_dir` defined in the config.

---

## Multi-GPU Training

The project includes `src/paralel` utilities and a dedicated CLI for multi-GPU ViT training.

Preferred mode is DDP data parallelism with `torchrun`:

```bash
# 2 GPUs on one node
torchrun --standalone --nproc_per_node=2 scripts/train_dino_multi_gpu.py \
    --config configs/multi_gpu.yaml

# 4 GPUs, override per-GPU batch size
torchrun --standalone --nproc_per_node=4 scripts/train_dino_multi_gpu.py \
    --config configs/vit_small.yaml \
    --batch-size 64 \
    --ckpt-dir checkpoints/dino_vits8_ddp
```

DDP behavior:

- One process per GPU.
- `DistributedSampler` shards the training data across ranks.
- Student is wrapped in `DistributedDataParallel`.
- Teacher remains an EMA model local to each rank.
- Only rank 0 writes metrics and checkpoints.
- Batch size is per process, so effective batch is `batch_size * nproc_per_node * grad_accum_steps`.

Experimental model parallelism is also available for ViT block sharding across devices:

```bash
python scripts/train_dino_multi_gpu.py \
    --config configs/vit_large.yaml \
    --use-model-parallel \
    --model-parallel-devices cuda:0,cuda:1
```

Use model parallel mode only when a single ViT does not fit on one GPU. For normal training throughput, prefer DDP.

### Monitoring

Training emits a `metrics.jsonl` file in the checkpoint directory. Each line is a JSON record with epoch, step, loss, teacher temperature, momentum, and optional feature metrics.

```bash
# Tail live training metrics
tail -f checkpoints/dino_vits8/metrics.jsonl | python -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f'epoch={r[\"epoch\"]:03d} loss={r[\"train_loss\"]:.4f}')
"
```

---

## Inference

```bash
# Segment an image using trained checkpoint
python scripts/infer_dino.py \
    --checkpoint checkpoints/dino_vits8/dino_vits8_best.pt \
    --image path/to/image.jpg \
    --output outputs/

# Extract features only (no segmentation)
python scripts/infer_dino.py \
    --checkpoint checkpoints/dino_vits8/dino_vits8_best.pt \
    --image path/to/image.jpg \
    --features-only
```

The inference pipeline (`src/inference/pipeline.py`) returns:

- `cls_token` — global image representation
- `patch_tokens` — spatial feature map
- `attention_maps` — per-head self-attention
- `segmentation_mask` — foreground / background binary mask
- `heatmap` — colorized patch similarity map

---

## API Server

Start the FastAPI inference server:

```bash
# Using uvicorn directly
DINO_CHECKPOINT=checkpoints/dino_vits8/dino_vits8_best.pt \
uvicorn src.app.main:app --host 0.0.0.0 --port 8000

# Using Docker
docker-compose up api
```

### Endpoints

| Method   | Path             | Description                             |
| -------- | ---------------- | --------------------------------------- |
| `GET`  | `/health`      | Liveness probe                          |
| `GET`  | `/ready`       | Readiness probe (model loaded)          |
| `POST` | `/v1/infer`    | Full inference: features + segmentation |
| `POST` | `/v1/features` | CLS + patch token extraction            |
| `POST` | `/v1/segment`  | Segmentation mask / heatmap             |
| `POST` | `/v1/metrics`  | DINO image quality metrics              |

### Example request

```python
import httpx, base64

with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = httpx.post(
    "http://localhost:8000/v1/infer",
    json={"image_b64": img_b64}
)
print(response.json())
```

Interactive docs are available at `http://localhost:8000/docs`.

---

## Docker

### Training container

```bash
# Build
docker build -t dino-train .

# Run training on GPU
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    dino-train python scripts/train_dino.py --config configs/vit_small.yaml
```

### API container

```bash
# Build
docker build -f Dockerfile.api -t dino-api .

# Run (mount checkpoint)
docker run -p 8000:8000 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -e DINO_CHECKPOINT=/app/checkpoints/dino_vits8/dino_vits8_best.pt \
    dino-api
```

### docker-compose (both services)

```bash
# Start everything
docker-compose up

# Training only
docker-compose run train python scripts/train_dino.py --config configs/set_up.yaml

# API only
docker-compose up api
```

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific module
pytest tests/test_model_vit_model.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Ablations

A config-driven ablation harness lives in [`ablations/`](ablations/). Each study varies **one axis** while holding everything else fixed, runs each variant as an isolated training process, and aggregates the runs into a side-by-side comparison. It requires **no changes to the training pipeline** — every knob is already exposed by the `train_dino()` wrapper, so an ablation is just a config override plus an isolated checkpoint directory.

### Studies

| # | Ablation | Axis (config key) | Variants |
|---|---|---|---|
| 1 | **`gram_loss`** (primary) | `dino.use_gram_loss` / `dino.gram_loss_weight` | baseline, weight_0p05, weight_0p1 |
| 2 | `pos_embed` | `model.pos_embed_type` | learned, rope, none |
| 3 | `register_tokens` | `model.num_register_tokens` | reg_0, reg_4, reg_8 |
| 4 | `local_crops` | `dino.num_local_crops` | local_0, local_2, local_4, local_8 |
| 5 | `teacher_temp` | `dino.teacher_temp` | temp_0p02, temp_0p04, temp_0p07 |
| 6 | `head_out_dim` | `dino_head.out_dim` | dim_1024, dim_4096, dim_8192 |

All studies are declared in [`ablations/ablations.yaml`](ablations/ablations.yaml) — edit that file to add variants or new axes, no Python required.

### Primary hypothesis

The project's central hypothesis is that **DINOv3-style patch Gram regularization improves dense feature quality without destabilising training**. The `gram_loss` study is the one we use to answer it: it compares the standard DINO objective (`baseline`) against two Gram-loss weights, and we read the verdict off two columns in the comparison table — `best_loss` (should drop with Gram loss) and `cls_std`, the mean per-dimension std of student CLS features (should rise, indicating richer, less-collapsed features).

### Running

```bash
# List every study and its variants
python ablations/run_ablation.py --list

# Run all variants of the primary study (quick smoke comparison)
python ablations/run_ablation.py gram_loss --epochs 30 --max-batches-per-epoch 200

# Run a single variant
python ablations/run_ablation.py gram_loss --variant weight_0p05

# Run every study, then auto-aggregate results
python ablations/run_all.py --epochs 30 --max-batches-per-epoch 200

# Aggregate whatever runs currently exist into a comparison report
python ablations/collect_results.py
```

`collect_results.py` writes `summary.md`, `summary.csv`, and `summary.json` to `ablations/results/`, with a Δ column versus each study's reference variant. See [`ablations/README.md`](ablations/README.md) for the full guide.

---

## Documentation

In-depth technical documentation lives in the [`docs/`](docs/) directory.

| Document                                  | Description                                                                                            |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [docs/overview.md](docs/overview.md)         | What DINO is, the self-distillation concept, centering and sharpening, emergent properties             |
| [docs/architecture.md](docs/architecture.md) | PatchEmbedding, positional embeddings (learned / RoPE), attention, transformer blocks, DINOHead        |
| [docs/losses.md](docs/losses.md)             | DINO loss math, EMA centering, teacher temperature; Gram loss formulation and memory notes             |
| [docs/training.md](docs/training.md)         | Training pipeline, EMA teacher update, all three schedulers, AMP, gradient accumulation, checkpointing |
| [docs/data.md](docs/data.md)                 | Multi-crop augmentation pipeline, TinyImageNet, data governance and quality checks                     |
| [docs/inference.md](docs/inference.md)       | Segmentation strategies, visualization, FastAPI endpoint reference, manual checkpoint loading          |
| [docs/configs.md](docs/configs.md)           | Complete reference for every YAML parameter                                                            |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting issues, proposing features, and submitting pull requests.

---

## License

This project is released under the [MIT License](LICENSE).

---

## References

- Caron, M. et al. (2021). **Emerging Properties in Self-Supervised Vision Transformers.** [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- Oquab, M. et al. (2023). **DINOv2: Learning Robust Visual Features without Supervision.** [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Dosovitskiy, A. et al. (2020). **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.** [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
