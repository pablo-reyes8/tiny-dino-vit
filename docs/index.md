# Documentation

This directory contains the technical documentation for the DINO self-supervised learning project.

## Contents

| File | Description |
|---|---|
| [overview.md](overview.md) | What DINO is, the self-distillation idea, key emergent properties, and what this implementation adds |
| [architecture.md](architecture.md) | Full model architecture: PatchEmbedding, positional embeddings (learned / RoPE), attention, transformer blocks, DINOHead |
| [losses.md](losses.md) | DINO loss (cross-entropy self-distillation + EMA centering) and Gram loss (patch Gram matrix regularization) |
| [training.md](training.md) | Training pipeline, multi-crop augmentation, EMA teacher update, schedulers, AMP, checkpointing, metrics |
| [data.md](data.md) | Data loading, multi-crop augmentation pipeline, TinyImageNet, data governance and quality checks |
| [inference.md](inference.md) | Inference pipeline, segmentation strategies, visualization, FastAPI server, REST API reference |
| [configs.md](configs.md) | Full reference for all YAML config parameters, organized by section |

## Suggested reading order

If you are new to the project:

1. Start with [overview.md](overview.md) to understand what DINO is and why it works.
2. Read [architecture.md](architecture.md) to understand the model components.
3. Read [losses.md](losses.md) for the mathematical details of the training objectives.
4. Read [training.md](training.md) to understand how to run and configure training.
5. Refer to [configs.md](configs.md) for a complete parameter reference when customizing a run.
6. See [inference.md](inference.md) when ready to use a trained checkpoint.

For a quick practical start, see the main [README](../README.md).
