# Configuration Reference

All training runs are driven by a YAML config file. The file is divided into six sections: `model`, `dino_head`, `dino`, `optim`, `train`, `checkpoint`, and `data`. Every field has a sensible default; you only need to override what differs from the baseline config (`configs/set_up.yaml`).

---

## Choosing a Config

| File | Use case |
|---|---|
| `configs/set_up.yaml` | First experiment, 4 GB VRAM, TinyImageNet at 64×64 |
| `configs/vit_small.yaml` | Standard training, 12 GB VRAM, 224×224 images |
| `configs/vit_base.yaml` | High-capacity training matching the DINO paper |
| `configs/vit_large.yaml` | Maximum capacity; needs teacher offload or multi-GPU |
| `configs/debug.yaml` | Smoke test / CI: 2 epochs, 10 batches, CPU-compatible |

---

## `model` — ViT Backbone

| Key | Type | Description |
|---|---|---|
| `img_size` | int | Spatial size of global crop input. Must equal `data.global_crop_size`. |
| `patch_size` | int | Side length of each patch. `img_size` must be divisible by `patch_size`. Smaller patches → more tokens → richer spatial features but higher memory. |
| `in_chans` | int | Number of input channels. Always 3 for RGB. |
| `embed_dim` | int | Transformer hidden dimension. Scales model capacity quadratically with depth. |
| `depth` | int | Number of transformer blocks. |
| `num_heads` | int | Number of attention heads. Must divide `embed_dim`. |
| `mlp_ratio` | float | FFN hidden dim = `embed_dim * mlp_ratio`. Standard is 4.0. |
| `qkv_bias` | bool | Add bias terms to the Q/K/V projection. Recommended `true`. |
| `patch_norm` | bool | Apply LayerNorm after PatchEmbedding. Rarely needed; default `false`. |
| `drop_rate` | float | Dropout applied after positional embedding. Usually 0.0 for SSL. |
| `attn_dropout` | float | Dropout on attention weights. Usually 0.0. |
| `proj_dropout` | float | Dropout after attention projection. Usually 0.0. |
| `mlp_dropout` | float | Dropout inside FFN. Usually 0.0. |
| `drop_path_rate` | float | Stochastic depth rate. Values in [0.1, 0.2]. Higher for larger models. |
| `pos_embed_type` | str | `"learned"` (bicubic interpolation for multi-crop), `"rope"` (2D RoPE, recommended for ViT-L), `"none"` (no positional encoding). |
| `use_cls_token` | bool | Add CLS token. Should always be `true` for DINO. |
| `num_register_tokens` | int | Number of register tokens (DINOv2 feature). 0 = disabled. Use 4–8 for ViT-S/B/L. |
| `rope_base` | float | Base frequency for RoPE. `10000.0` is standard. |
| `strict_img_size` | bool | Raise an error if input image size does not match `img_size`. Set `false` to allow multi-crop with variable sizes. |

---

## `dino_head` — Projection Head

| Key | Type | Description |
|---|---|---|
| `in_dim` | int | Input dimension. Must equal `model.embed_dim`. |
| `out_dim` | int | Number of prototype dimensions (vocabulary size). Typical values: 8192 (small runs), 65536 (large runs). |
| `hidden_dim` | int | Hidden dimension of the MLP. Typically 2048 or 4096. |
| `bottleneck_dim` | int | Bottleneck dimension before L2 normalization. Typically 256–512. |
| `num_layers` | int | Total MLP layers including the final bottleneck layer. Minimum 1 (direct linear). |
| `use_bn` | bool | Apply BatchNorm1d after each hidden linear. Disabled by default; can help on very small batches. |
| `norm_last_layer` | bool | If `true`, the `weight_g` (scale) of the weight-normalized last layer is frozen at 1.0. This stabilizes early training. |
| `bias` | bool | Add bias to linear layers in the MLP (but never to the last weight-normalized layer). |
| `cls_key` | str | Key in the backbone output dict to use as head input. Always `"cls"`. |

---

## `dino` — DINO Training Hyper-Parameters

| Key | Type | Description |
|---|---|---|
| `num_global_crops` | int | Number of high-resolution crops per image. Standard is 2. Both student and teacher process global crops. |
| `num_local_crops` | int | Number of low-resolution crops per image. Student processes all crops; teacher processes global only. |
| `student_temp` | float | Temperature for student softmax. Higher = softer predictions. Typical: 0.1. |
| `teacher_momentum_start` | float | EMA momentum at the start of training. Typical: 0.996. |
| `teacher_momentum_end` | float | EMA momentum at the end of training. Standard is 1.0 (teacher stops moving). |
| `warmup_teacher_temp` | float | Initial teacher temperature before warmup. Typical: 0.04. |
| `teacher_temp` | float | Target teacher temperature after warmup. Typical: 0.07. Lower = sharper targets. |
| `warmup_teacher_temp_epochs` | int | Number of epochs to warm teacher temperature from `warmup_teacher_temp` to `teacher_temp`. |
| `use_gram_loss` | bool | Enable patch-level Gram matrix regularization loss. |
| `gram_loss_weight` | float | Weight `λ` applied to the Gram loss: `L = L_DINO + λ * L_gram`. Typical: 0.05. |

---

## `optim` — Optimizer

AdamW is the only supported optimizer. All parameters are in a single parameter group; weight decay is not applied to LayerNorm or bias parameters (this is the standard DINO practice and is handled internally).

| Key | Type | Description |
|---|---|---|
| `learning_rate` | float | Peak learning rate after warmup. Typical: 5e-4 for small models, 1e-4 for large. Scale linearly with effective batch size. |
| `min_learning_rate` | float | Minimum LR at the end of cosine decay. Typical: 1e-6. |
| `weight_decay` | float | Initial weight decay. Typical: 0.04. |
| `weight_decay_end` | float | Final weight decay (cosine-scheduled upward). Typical: 0.4. |
| `betas` | list[float] | Adam `(β1, β2)`. Typical: `[0.9, 0.95]`. |
| `eps` | float | Adam epsilon for numerical stability. Typical: 1e-8. |

**LR scaling rule:** When changing effective batch size, scale the learning rate linearly:
```
lr_new = lr_ref * (effective_batch_new / effective_batch_ref)
```
Reference: `lr=5e-4` at batch 256. So for batch 128: `lr=2.5e-4`.

---

## `train` — Training Loop

| Key | Type | Description |
|---|---|---|
| `epochs` | int | Total training epochs. |
| `warmup_epochs` | int | LR warmup duration in epochs. |
| `grad_clip` | float | Max global gradient norm. `1.0` for small, `3.0` for large models. Set `null` to disable. |
| `grad_accum_steps` | int | Gradient accumulation steps. Effective batch = `batch_size * grad_accum_steps`. |
| `max_batches_per_epoch` | int or null | Cap the number of batches per epoch. Useful for debugging; set `null` for full epoch. |
| `log_every` | int | Print running metrics every N batches. |
| `metrics_every` | int | Compute extended metrics (feature entropy, collapse) every N batches. |
| `teacher_offload` | bool | Move teacher to CPU between forward passes (saves ~50% VRAM). |
| `exclusive_gpu_model` | bool | Reserved for future single-GPU exclusive mode. Keep `false`. |
| `empty_cache_after_offload` | bool | Call `torch.cuda.empty_cache()` after each teacher offload. Useful for very tight VRAM; adds latency. |

---

## `checkpoint` — Saving and Monitoring

| Key | Type | Description |
|---|---|---|
| `ckpt_dir` | str | Directory where checkpoints and metrics JSONL are saved. Created automatically. |
| `run_name` | str | Prefix for checkpoint file names. Use a descriptive name like `dino_vitb16_imagenet`. |
| `save_every` | int | Save a periodic checkpoint every N epochs. Set to 1 to save every epoch; set to a larger value for long runs. |
| `keep_last_n_checkpoints` | int | Maximum number of periodic checkpoints to keep. Older ones are deleted automatically. Best checkpoint is always kept. |
| `monitor_name` | str | Metric to monitor for best-checkpoint selection. `"loss"` is the only standard metric. |
| `monitor_mode` | str | `"min"` (lower is better) or `"max"` (higher is better). Use `"min"` for loss. |

---

## `data` — Dataset and Loading

| Key | Type | Description |
|---|---|---|
| `dataset_name` | str | Hugging Face dataset identifier. |
| `dataset_revision` | str or null | Pin dataset to a specific git revision for reproducibility. |
| `data_dir` | str | Local cache directory for the dataset. |
| `artifacts_dir` | str | Output directory for quality check artifacts. |
| `reports_dir` | str | Output directory for profiling reports. |
| `seed` | int | RNG seed for the train/val split. |
| `val_ratio` | float | Fraction of training data held out for validation. |
| `global_crop_size` | int | Spatial resolution of global crops. Must match `model.img_size`. |
| `local_crop_size` | int | Spatial resolution of local crops. |
| `global_crop_scale` | list[float] | Min/max area fraction for global RandomResizedCrop. |
| `local_crop_scale` | list[float] | Min/max area fraction for local RandomResizedCrop. |
| `batch_size` | int | Batch size per GPU. |
| `num_workers` | int | DataLoader worker processes. Set to 0 for debugging. |
| `pin_memory` | bool | Pin tensors in page-locked memory for faster GPU transfer. |
| `persistent_workers` | bool | Keep DataLoader workers alive between epochs to avoid re-initialization overhead. |
| `shuffle_train` | bool | Shuffle training data each epoch. |
| `drop_last_train` | bool | Drop the last incomplete batch during training (important for consistent batch sizes with AMP). |
| `drop_last_val` | bool | Drop the last incomplete batch during validation. |

### `data.quality` — Quality Check Thresholds

| Key | Type | Description |
|---|---|---|
| `blur_variance_threshold` | float | Images with Laplacian variance below this are flagged as blurry. |
| `dark_mean_threshold` | float | Images with mean pixel value below this are flagged as too dark. |
| `bright_mean_threshold` | float | Images with mean pixel value above this are flagged as too bright. |
| `low_contrast_std_threshold` | float | Images with pixel std below this are flagged as low contrast. |
| `saturated_pixel_fraction_threshold` | float | Images with this fraction of saturated pixels are flagged. |
| `tiny_side_threshold` | int | Images with minimum side below this pixel count are flagged. |
| `flag_top_k_examples` | int | Number of top-flagged examples to export during inspection. |

---

## Config Inheritance / Overrides

To quickly experiment without duplicating entire files, start from the closest preset and override only the fields that change. For example, to try RoPE positional embeddings on the ViT-Small config:

```yaml
# my_vit_small_rope.yaml
# Copy vit_small.yaml and change only:

model:
  pos_embed_type: rope
  num_register_tokens: 4

checkpoint:
  run_name: dino_vits8_rope
  ckpt_dir: checkpoints/dino_vits8_rope
```

The training script will load the base config and merge these overrides on top.
