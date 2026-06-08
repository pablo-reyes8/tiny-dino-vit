# Project Overview

## What is DINO?

**DINO** stands for **Self-Distillation with No Labels**. It is a self-supervised learning method for Vision Transformers (ViTs) introduced by Caron et al. at Facebook AI Research in 2021. The core idea is that a neural network can teach itself — without any human-annotated labels — by producing consistent feature representations across different views of the same image.

The training signal comes entirely from the data itself. No classification labels, no bounding boxes, no segmentation masks. Despite this, models trained with DINO develop representations that are semantically rich enough to directly segment objects from scenes, perform k-NN classification, and serve as strong feature extractors for downstream tasks.

---

## The Self-Distillation Idea

DINO uses two identical network instances called the **student** and the **teacher**. Both share the same architecture. Their relationship:

- The **student** is trained with standard gradient descent.
- The **teacher** is never updated by backpropagation. Instead, it is updated as an **Exponential Moving Average (EMA)** of the student's weights.
- Both networks see different views (crops) of the same image.
- The student is trained to predict the teacher's output distribution for those views.

Because the teacher is a smoothed, time-averaged version of the student, it provides a more stable training target than using the student itself (which would collapse). This setup is closely related to knowledge distillation, but here the "teacher" emerges from the training process rather than being pre-trained separately — hence *self*-distillation.

```
Image ──→ augment ──→ [view_1, view_2, ..., view_N]
                           │                 │
                        student           teacher (EMA)
                           │                 │
                        DINO Head         DINO Head
                           │                 │
                        logits_s          logits_t
                           │                 │
                      Cross-entropy loss (student predicts teacher)
```

---

## Why Does It Work Without Labels?

DINO avoids the two failure modes that plague naive self-supervised approaches:

1. **Representation collapse** — when the network learns to output the same vector regardless of input. DINO prevents this through *centering* and *sharpening* of the teacher's output distribution.
   - **Centering** subtracts a running mean from the teacher logits before the softmax, preventing any single output dimension from dominating.
   - **Sharpening** uses a low teacher temperature, producing peaked distributions that force the student to make specific predictions.

2. **Trivial solutions** — using multiple augmented views of the same image and requiring the network to produce consistent predictions across views (multi-crop) creates a rich, non-trivial pretext task.

---

## Key Emergent Properties

The original DINO paper demonstrated several surprising properties:

- **Attention maps look like segmentation masks.** The [CLS] token's self-attention in the last transformer block naturally segments the dominant object from the background — without ever seeing a segmentation label.
- **k-NN classification without fine-tuning.** The frozen CLS token features from a DINO-trained ViT-S/8 achieve ~74% top-1 accuracy on ImageNet with a simple k-nearest neighbour classifier.
- **Strong linear probing.** The features transfer well to downstream tasks with minimal adaptation.

---

## This Implementation

This codebase implements the full DINO training pipeline with the following additions:

| Addition | Description |
|---|---|
| **Gram loss** | Optional patch-level Gram matrix regularization, inspired by DINOv3. Encourages the student's patch token co-activation patterns to match the teacher's. |
| **RoPE positional embeddings** | Alternative to learned absolute positional embeddings. 2D Rotary Position Embeddings are applied inside each attention block to patch tokens only. Enables better generalization to variable crop sizes. |
| **Register tokens** | Introduced in DINOv2. Extra learnable tokens that absorb artifact patterns from high-norm patches, producing cleaner attention maps and patch features. |
| **CPU teacher offload** | On low-VRAM GPUs, the teacher can be offloaded to CPU between forward passes, halving the GPU memory footprint at the cost of transfer overhead. |
| **FastAPI inference server** | A production-ready REST API for running inference from checkpoints without writing any Python. |

---

## References

- Caron et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Darcet et al. (2023). *Vision Transformers Need Registers.* ICLR 2024. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
- Dosovitskiy et al. (2020). *An Image is Worth 16×16 Words.* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
