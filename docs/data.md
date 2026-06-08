# Data

## Dataset

The default dataset is **TinyImageNet** (`zh-plus/tiny-imagenet` on Hugging Face Hub), a scaled-down variant of ImageNet with:

- 200 classes
- 500 training images per class (100,000 total)
- 50 validation images per class (10,000 total)
- Image size: 64×64 pixels

TinyImageNet is used as the default because it is small enough to run meaningful experiments on a single GPU without the infrastructure overhead of full ImageNet, while still being semantically rich enough to produce meaningful self-supervised features.

The dataset is loaded via Hugging Face `datasets` and does not require manual downloading. It is streamed and cached automatically on first use.

---

## Multi-Crop Data Loading

**File:** `data/load_data.py`

The DINO dataloader does not return a single `(image, label)` pair per sample. Instead, it returns a **list of crop tensors** representing different views of the same image:

```python
batch = [
    Tensor[B, 3, global_crop_size, global_crop_size],   # global crop 1
    Tensor[B, 3, global_crop_size, global_crop_size],   # global crop 2
    Tensor[B, 3, local_crop_size,  local_crop_size],    # local crop 1
    Tensor[B, 3, local_crop_size,  local_crop_size],    # local crop 2
    ...                                                  # more local crops
]
```

Labels are discarded; they are not used anywhere in the training pipeline.

### Augmentation per crop type

**Global crops** (high-resolution, large scale):
```
RandomResizedCrop(global_crop_size, scale=global_crop_scale)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)  (p=0.8)
RandomGrayscale(p=0.2)
GaussianBlur(sigma=[0.1, 2.0])  (p=1.0 for first global, p=0.1 for second)
Solarize  (p=0.2 on second global crop only)
ToTensor + Normalize(mean, std)
```

**Local crops** (low-resolution, small scale):
```
RandomResizedCrop(local_crop_size, scale=local_crop_scale)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)  (p=0.8)
RandomGrayscale(p=0.2)
GaussianBlur(sigma=[0.1, 2.0])  (p=0.5)
ToTensor + Normalize(mean, std)
```

Normalization uses ImageNet statistics:
```
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
```

---

## Data Configuration

**File:** `data/data_config.py`

All data parameters have defaults in `DEFAULT_DATA_CONFIG` and can be overridden via the `data:` section of any YAML config. Use `merge_data_config(overrides)` to merge custom settings with the defaults.

| Parameter | Default | Description |
|---|---|---|
| `dataset_name` | `zh-plus/tiny-imagenet` | Hugging Face dataset identifier |
| `dataset_revision` | `null` | Pin to a specific dataset commit/version |
| `data_dir` | `data` | Local cache directory |
| `artifacts_dir` | `data/artifacts` | Output directory for quality artifacts |
| `reports_dir` | `data/reports` | Output directory for profiling reports |
| `seed` | `42` | RNG seed for train/val split |
| `val_ratio` | `0.1` | Fraction of training data used as validation |
| `global_crop_size` | `64` | Pixel size of global crops |
| `local_crop_size` | `32` | Pixel size of local crops |
| `global_crop_scale` | `[0.5, 1.0]` | Area fraction range for global crop |
| `local_crop_scale` | `[0.2, 0.5]` | Area fraction range for local crop |
| `batch_size` | `64` | Images per batch (before multi-crop) |
| `num_workers` | `2` | DataLoader worker processes |
| `pin_memory` | `true` | Pin tensors in page-locked memory for faster GPU transfer |
| `persistent_workers` | `true` | Keep workers alive between epochs |
| `shuffle_train` | `true` | Shuffle training data each epoch |
| `drop_last_train` | `true` | Drop last incomplete batch during training |
| `drop_last_val` | `false` | Keep last incomplete batch during validation |

---

## Data Governance

**File:** `data/dataops.py`

The `dataops` module provides data quality checks, profiling, and data cards for reproducibility and auditing. These are not used during training — they are tools for understanding and validating the dataset before training.

### Quality Assessment

Per-image quality flags are computed on a sample of the dataset:

| Check | Method | Threshold |
|---|---|---|
| **Blur** | Laplacian variance of grayscale image | `blur_variance_threshold: 0.002` |
| **Dark** | Mean pixel value | `dark_mean_threshold: 0.08` |
| **Bright** | Mean pixel value | `bright_mean_threshold: 0.92` |
| **Low contrast** | Standard deviation of pixel values | `low_contrast_std_threshold: 0.05` |
| **Saturated pixels** | Fraction of pixels near 0 or 1 | `saturated_pixel_fraction_threshold: 0.2` |
| **Tiny image** | Minimum side length | `tiny_side_threshold: 32` |

Images failing one or more checks are flagged. The top-K flagged examples (`flag_top_k_examples`) can be exported as a grid image for visual inspection.

### Dataset Profiling

The profiler computes summary statistics across the dataset:

- Class distribution (histogram)
- Image size distribution (H, W, aspect ratio)
- Per-channel mean and standard deviation
- Quality flag rates

Output is saved to `reports_dir` as JSON and CSV.

### Data Cards

A human-readable data card is generated describing the dataset, configuration used, split sizes, and quality metrics. This is useful for documenting experiments and enabling reproducibility.

---

## Inspection Script

```bash
# Profile TinyImageNet with the default config
python scripts/inspect_data.py

# Custom config
python scripts/inspect_data.py --config configs/vit_small.yaml

# Export flagged images to a directory
python scripts/inspect_data.py --export-flagged outputs/flagged/
```

---

## Adding a Custom Dataset

To train on a different dataset:

1. Implement a dataloader that returns `List[Tensor[B, 3, H, W]]` (one tensor per crop).
2. Update `data_config.py` with the new defaults.
3. Point the YAML config's `data.dataset_name` to the new identifier.

The multi-crop transform pipeline is fully reusable — pass it as the `transform` argument to any map-style dataset.
