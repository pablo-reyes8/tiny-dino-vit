# Training

## Quick Start

```bash
# Baseline ViT-Tiny on TinyImageNet (4 GB VRAM, ~1 h per 10 epochs on an RTX 3080)
python scripts/train_dino.py --config configs/set_up.yaml

# ViT-Small (12 GB VRAM)
python scripts/train_dino.py --config configs/vit_small.yaml

# Resume from a checkpoint
python scripts/train_dino.py --config configs/vit_small.yaml \
    --resume checkpoints/dino_vits8/dino_vits8_e050.pt

# Smoke test — completes in < 1 minute on CPU
python scripts/train_dino.py --config configs/debug.yaml
```

---

## Training Pipeline

The training orchestrator is `src/training/train_dino.py::train_dino()`. It is responsible for the full lifecycle of a run:

```
1. Seed / reproducibility
2. Device and precision setup (AMP)
3. Optimizer construction
4. LR, weight decay, teacher momentum, teacher temperature schedulers
5. Optional checkpoint resume
6. Epoch loop:
      a. train_one_epoch (forward, loss, backward, optimizer step)
      b. EMA teacher update
      c. Metric computation and JSONL logging
      d. Checkpoint saving (best / periodic)
7. Final checkpoint
```

---

## Multi-Crop Augmentation

The data loader produces a list of crops per image rather than a single tensor. Each batch is `List[Tensor[B, 3, H_i, W_i]]` of length `num_global_crops + num_local_crops`.

| Crop type | Count | Size | Scale range |
|---|---|---|---|
| Global | 2 | `global_crop_size` | `[0.5, 1.0]` |
| Local | 4–10 | `local_crop_size` | `[0.2, 0.5]` |

All crops pass through the student. Only global crops pass through the teacher. This asymmetry forces the student to learn representations that are consistent across scales: it must match the teacher's global-view output while processing smaller, lower-context local views.

The augmentation pipeline per crop:
1. RandomResizedCrop (with the scale range above)
2. RandomHorizontalFlip
3. ColorJitter (brightness, contrast, saturation, hue)
4. Optional GaussianBlur (stronger on global, weaker/absent on local crops)
5. Optional Solarize (randomly on one global crop)
6. Normalize with ImageNet mean/std: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`

---

## EMA Teacher Update

**File:** `src/training/ema.py`

After every optimizer step, the teacher weights are updated as an Exponential Moving Average of the student:

```
θ_teacher ← m * θ_teacher + (1 − m) * θ_student
```

The momentum `m` is cosine-scheduled from `teacher_momentum_start` (typically 0.996) to `teacher_momentum_end` (1.0) over the total number of training steps:

```
m(t) = teacher_momentum_end
       − (teacher_momentum_end − teacher_momentum_start)
         * (1 + cos(π * t / T)) / 2
```

At the start of training the teacher tracks the student quickly (low momentum). As training progresses, the teacher stabilizes, becoming a slowly evolving ensemble of all previous student snapshots. At `m = 1.0` the teacher stops moving entirely — this never actually happens but is approached asymptotically.

The update is `@torch.no_grad()` and operates in-place on the teacher's parameters. Buffers (e.g. `DINOLoss.center`) are also copied from student to teacher at each step.

### CPU Teacher Offload

When `teacher_offload: true`, the teacher is kept on CPU between forward passes. Before each forward pass the teacher is moved to GPU, the forward is computed, and the teacher is moved back to CPU. This halves the GPU memory footprint at the cost of CPU↔GPU transfer overhead (~50–200 ms per batch depending on model size and NVLink bandwidth).

Recommended only for ViT-Large or when GPU memory is under 24 GB.

---

## Schedulers

**File:** `src/training/schedulers.py`

Three quantities are scheduled over training:

### 1. Learning Rate

Cosine decay with linear warmup:

```
Phase 1 — warmup (first warmup_epochs epochs):
    lr(t) = start_warmup_lr + (learning_rate − start_warmup_lr) * t / warmup_steps

Phase 2 — cosine decay (remaining epochs):
    lr(t) = min_lr + (learning_rate − min_lr) * (1 + cos(π * t / T_decay)) / 2
```

where `T_decay` is the total steps after warmup.

### 2. Weight Decay

Cosine schedule from `weight_decay` (start) to `weight_decay_end`:

```
wd(t) = weight_decay_end
        − (weight_decay_end − weight_decay) * (1 + cos(π * t / T)) / 2
```

Weight decay increases over training because early in training loose regularization allows fast convergence, while late in training tighter regularization improves generalization.

### 3. Teacher Temperature

Linear warmup from `warmup_teacher_temp` to `teacher_temp` over `warmup_teacher_temp_epochs`:

```
τ_t(epoch) = warmup_teacher_temp
             + (teacher_temp − warmup_teacher_temp) * epoch / warmup_teacher_temp_epochs
```

After `warmup_teacher_temp_epochs`, the teacher temperature stays at `teacher_temp`.

---

## Mixed Precision (AMP)

**File:** `src/training/autocast.py`

Mixed precision is enabled by default and provides a 2–3× speedup on modern GPUs with minimal accuracy impact.

| Setting | dtype | When |
|---|---|---|
| `amp_dtype: "bf16"` | bfloat16 | A100, H100, RTX 3090+ (preferred: no overflow risk) |
| `amp_dtype: "fp16"` | float16 | Older GPUs (V100, RTX 20xx) — uses GradScaler |
| `amp_dtype: "fp32"` | float32 | CPU training or when AMP causes issues |
| `fallback_bf16_to_fp16: true` | float16 | Auto-fallback if bf16 not natively supported |

The GradScaler is only created for fp16 (not bf16, which does not overflow). Scaler state is included in checkpoints for seamless resume.

---

## Gradient Accumulation

Set `grad_accum_steps > 1` to simulate a larger effective batch size:

```python
effective_batch = batch_size * grad_accum_steps
```

The optimizer step and scheduler step happen only every `grad_accum_steps` mini-batches. The loss is divided by `grad_accum_steps` before the backward pass to normalize the gradient scale.

Example: `batch_size=32, grad_accum_steps=8` gives an effective batch of 256, which is the batch size used in the original DINO paper.

---

## Gradient Clipping

Global gradient norm clipping is applied before every optimizer step:

```python
nn.utils.clip_grad_norm_(student.parameters(), max_norm=grad_clip)
```

`grad_clip: 1.0` is the default for ViT-Tiny. Use `grad_clip: 3.0` for larger models; the larger models tend to have larger gradient norms early in training.

---

## Checkpointing

**File:** `src/training/chekpoints.py`

Checkpoints are saved to `checkpoint.ckpt_dir` as `.pt` files. Each checkpoint includes:

| Content | Description |
|---|---|
| `student_state_dict` | Student model weights |
| `teacher_state_dict` | Teacher model weights |
| `optimizer_state_dict` | Adam moments |
| `scheduler_state_dict` | LR / WD schedule state |
| `scaler_state_dict` | GradScaler state (fp16 only) |
| `epoch` | Last completed epoch |
| `step` | Global optimizer step count |
| `best_metric` | Best monitored metric value |
| `config` | Full training config dict |
| `rng_state` | Python / NumPy / PyTorch RNG states |
| `extra_state` | Monitor name/mode, optimizer type, model info |

A JSON sidecar (`.json`) is written alongside each `.pt` for quick inspection without loading the full checkpoint.

### Checkpoint naming

- `{run_name}_e{epoch:03d}.pt` — periodic save (every `save_every` epochs)
- `{run_name}_best.pt` — overwritten whenever the monitored metric improves
- `{run_name}_last_manual.pt` — written at end of training

The `keep_last_n_checkpoints` parameter automatically deletes older periodic checkpoints, keeping only the N most recent.

### Resuming

```bash
python scripts/train_dino.py --config configs/vit_small.yaml \
    --resume checkpoints/dino_vits8/dino_vits8_e100.pt
```

On resume, the following are restored: student/teacher weights, optimizer state, scheduler state, scaler state, `start_epoch`, `global_step`, and `best_metric`. Optionally, the full RNG state can be restored (`restore_rng_state: true`) for bitwise-reproducible continuation.

---

## Metrics and Logging

Each epoch appends one JSON record to `metrics.jsonl` in the checkpoint directory:

```json
{
  "epoch": 10,
  "global_step": 7820,
  "time_sec": 83.4,
  "monitor_name": "loss",
  "monitor_value": 3.142,
  "best_metric": 3.098,
  "improved": false,
  "train_loss": 3.142,
  "train_loss_dino": 3.051,
  "train_loss_gram": 0.091,
  "train_teacher_temp": 0.06,
  "train_momentum": 0.9988,
  "use_gram_loss": true,
  "gram_loss_weight": 0.05
}
```

Within each epoch, the training loop prints a running summary every `log_every` batches and computes extended feature metrics (entropy, collapse coefficient) every `metrics_every` batches.

---

## Reproducibility

Seeding is set at the start of training via `set_seed(seed, deterministic=deterministic)`:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Setting `deterministic: true` enables fully deterministic CUDA operations at the cost of speed. For research requiring exact reproducibility, enable this and set `restore_rng_state: true` for checkpoint resumes.

---

## Memory Estimates

Rough GPU memory usage for training (forward + backward + optimizer states, batch size 64, bf16):

| Config | VRAM (no offload) | VRAM (teacher offload) |
|---|---|---|
| ViT-Tiny/8 (64 px) | ~3 GB | ~2 GB |
| ViT-Small/8 (224 px) | ~14 GB | ~9 GB |
| ViT-Base/16 (224 px) | ~28 GB | ~18 GB |
| ViT-Large/16 (224 px) | ~60 GB | ~38 GB |

*Gram loss adds ~0.5–2 GB depending on N (patch count). Disable it (`use_gram_loss: false`) if tight on memory.*
