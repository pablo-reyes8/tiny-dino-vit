# Architecture

The model is composed of three stacked components: the **ViT backbone**, the **DINO projection head**, and a student-teacher wrapping layer. The same architecture is shared between student and teacher — the only difference is how their weights are updated.

---

## Overview

```
Input image [B, 3, H, W]
        │
        ▼
┌─────────────────────────┐
│     PatchEmbedding      │   Conv2d(3, D, kernel=P, stride=P)
│     (patch_size=P)      │   [B, N, D]   where N = (H/P) * (W/P)
└─────────────────────────┘
        │
        ▼
Prepend [CLS], [REG_1..R]  →  [B, 1+R+N, D]
        │
        ▼
Add positional embeddings  →  learned (interpolated) or RoPE (in-block)
        │
        ▼
┌─────────────────────────┐
│  TransformerEncoderBlock│  × depth
│  ┌───────────────────┐  │
│  │  LayerNorm        │  │
│  │  MHSA + (RoPE)    │  │  + residual
│  │  LayerNorm        │  │
│  │  FFN (GELU MLP)   │  │  + residual
│  │  DropPath         │  │
│  └───────────────────┘  │
└─────────────────────────┘
        │
        ▼
   LayerNorm (final)
        │
        ├──→ cls  [B, D]         global image representation
        ├──→ registers [B, R, D] (if num_register_tokens > 0)
        └──→ patches [B, N, D]   spatial feature map
                │
                ▼
        ┌──────────────────┐
        │    DINOHead      │   MLP → bottleneck → L2-norm → WN Linear
        └──────────────────┘
                │
                ▼
           logits [B, K]         K = out_dim (e.g. 8192 or 65536)
```

---

## PatchEmbedding

**File:** `src/model/embeddings.py`

Converts an image tensor `[B, C, H, W]` into a sequence of patch tokens `[B, N, D]`.

```
image [B, 3, H, W]
  ──→ Conv2d(in_chans=3, out_chans=embed_dim, kernel_size=patch_size, stride=patch_size)
  ──→ flatten spatial dims
  ──→ transpose to [B, N, D]
  ──→ optional LayerNorm
```

The resulting grid size is `(H / patch_size, W / patch_size)`. For a 64×64 image with `patch_size=8`, this produces an 8×8 = 64-token sequence.

The `grid_size` tuple is returned alongside the tokens and is passed to each attention block so RoPE and positional interpolation know the spatial layout.

---

## Special Tokens

Before entering the transformer, two kinds of learnable tokens are prepended to the patch sequence:

### CLS Token

A single learnable vector `[1, 1, D]` expanded to `[B, 1, D]` and prepended. After the final LayerNorm, the CLS token is extracted as the global image representation and fed to the DINO head. When no CLS token is used (`use_cls_token=False`), the mean of patch tokens is used instead.

### Register Tokens

Introduced in DINOv2 (*Vision Transformers Need Registers*, Darcet et al. 2023). A bank of `num_register_tokens` learnable vectors prepended after the CLS token:

```
token order: [CLS] [REG_1] [REG_2] ... [REG_R] [PATCH_1] ... [PATCH_N]
```

Registers absorb high-norm "artifact" patches — positions where the ViT would otherwise store global information unrelated to local scene content. The effect is that patch tokens become more spatially consistent, improving dense prediction and segmentation quality. Registers are discarded before computing the DINO loss; only CLS and patch tokens are used.

---

## Positional Embeddings

Two strategies are supported, controlled by `pos_embed_type` in the config.

### Learned Absolute Embeddings (`pos_embed_type: learned`)

A parameter `pos_embed` of shape `[1, num_special_tokens + num_patches, D]` is added to the full token sequence after prepending special tokens.

**Multi-crop interpolation.** DINO passes crops of different sizes (e.g. 64×64 global and 32×32 local) through the same backbone. The base positional embedding is trained at the global crop resolution; local crops produce fewer patches. To handle this, the patch portion of `pos_embed` is bicubically interpolated from the base grid size to the current grid size:

```python
# Special token positions are not interpolated — only patch positions.
patch_pos = pos_embed[:, num_special_tokens:]      # [1, N_base, D]
patch_pos = patch_pos.reshape(1, H_base, W_base, D).permute(0, 3, 1, 2)
patch_pos = F.interpolate(patch_pos, size=(H_target, W_target), mode="bicubic")
patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, N_target, D)
pos_embed = torch.cat([special_pos, patch_pos], dim=1)
```

### Rotary Position Embeddings — 2D RoPE (`pos_embed_type: rope`)

**File:** `src/model/rope.py`

RoPE encodes position information directly inside the query and key projections of each attention block, rather than in the token embeddings. This makes it *relative* in nature: the attention between two tokens depends on their relative spatial offset, not their absolute position.

The implementation uses 2D RoPE: the `head_dim`-dimensional space is split in half, with one half encoding the row index and the other encoding the column index of each patch.

Rotation matrices are constructed from sinusoidal frequencies:

```
θ_i = 1 / base^(2i / head_dim)

for row position r:  rotate dims 0..head_dim/4   by r * θ
for col position c:  rotate dims head_dim/4..D/2 by c * θ
```

RoPE is applied only to patch token queries and keys. CLS and register tokens use a zero-rotation (i.e., no positional encoding), which is the standard practice.

**Advantage over learned embeddings:** RoPE does not need interpolation when the crop size changes because position is encoded relative to neighbors, not absolutely. This is particularly valuable for ViT-Large configs where large absolute position embedding matrices become memory-heavy.

---

## MultiHeadSelfAttention

**File:** `src/model/attention.py`

Standard scaled dot-product attention:

```
QKV = Linear(x)                      # [B, T, 3D]
Q, K, V = split(QKV)                 # each [B, H, T, head_dim]

if use_rope:
    Q_patch, K_patch = RoPE(Q_patch, K_patch, grid_size)
    Q = cat([Q_special, Q_patch])
    K = cat([K_special, K_patch])

A = softmax(Q @ K^T / sqrt(head_dim))
out = A @ V
out = proj(out)
```

The attention matrix `A` of shape `[B, H, T, T]` can optionally be returned (set `return_attention=True` in `forward_features`). The last-layer attention maps are used by the inference pipeline to generate segmentation heatmaps.

---

## TransformerEncoderBlock

**File:** `src/model/transformer_block.py`

Pre-LayerNorm residual block following the ViT convention:

```
x = x + DropPath(MHSA(LN(x)))
x = x + DropPath(FFN(LN(x)))
```

**DropPath (Stochastic Depth):** During training, each block's residual contribution is dropped entirely with probability `drop_path_rate`, linearly increasing from 0 at the first block to `drop_path_rate` at the last. This acts as a strong regularizer for deep models.

**FFN:** A two-layer MLP with GELU activation and an optional dropout. The hidden dimension is `embed_dim * mlp_ratio` (typically 4×).

---

## VisionTransformer

**File:** `src/model/vit_model.py`

The `VisionTransformer` class orchestrates all components. Its `forward_features` method returns a dictionary:

```python
{
    "cls":       Tensor[B, D],          # global image feature
    "registers": Tensor[B, R, D],       # register tokens (or None)
    "patches":   Tensor[B, N, D],       # per-patch spatial features
    "grid_size": Tuple[int, int],        # (H_patches, W_patches)
    "x_norm":    Tensor[B, T, D],       # full normalized sequence
    "attentions": List[Tensor[B, H, T, T]]  # optional, per-block
}
```

For DINO training, the `cls` token is passed to the DINO head. The `patches` tensor is passed to the Gram loss.

---

## DINOHead

**File:** `src/model/dino_head.py`

A projection head that maps the `D`-dimensional backbone output to a `K`-dimensional prototype space.

```
input [B, D]
    │
    ▼  Linear(D → hidden_dim) + GELU              ┐
    │  Linear(hidden_dim → hidden_dim) + GELU  ×(L-2)  │  MLP
    │  Linear(hidden_dim → bottleneck_dim)         ┘
    │
    ▼  L2-normalize                                # unit hypersphere
    │
    ▼  WeightNorm Linear(bottleneck_dim → K)        # weight-normalized, no bias
    │
output [B, K]                                      # logits (not probabilities)
```

Key design decisions:

- **L2 normalization before the last layer** constrains features to the unit sphere. This prevents the loss from being driven by feature magnitude and forces the network to learn directional similarity.
- **Weight normalization on the last linear layer** decouples the direction of the weight vectors from their magnitude. The magnitude (`weight_g`) is optionally frozen (`norm_last_layer=True`) so only the directions (`weight_v`) are learned. This stabilizes early training.
- **No bias on the last layer** ensures that the prototype matching is purely geometric (dot product on the sphere).
- **Truncated normal initialization** (`std=0.02`) throughout, following the original ViT paper.

---

## DINOModel (Student/Teacher Wrapper)

**File:** `src/model/dino_full_model.py`

`DINOModel` wraps a `VisionTransformer` backbone and a `DINOHead` together. It handles the routing of backbone outputs to the head and exposes named feature keys for both the loss functions and the inference pipeline.

The `build_dino_bundle()` function constructs a student-teacher pair:

1. Creates one `DINOModel` (student).
2. Deep-copies it to produce an identical teacher.
3. Freezes all teacher parameters.
4. Returns a `bundle` dict with both models, config, and model metadata.

---

## Parameter counts by config

| Config | embed_dim | depth | heads | N patches (global) | Backbone params | Total params |
|---|---|---|---|---|---|---|
| ViT-Tiny/8 | 192 | 6 | 3 | 64 | ~5 M | ~6 M |
| ViT-Small/8 | 384 | 12 | 6 | 784 | ~22 M | ~24 M |
| ViT-Base/16 | 768 | 12 | 12 | 196 | ~86 M | ~88 M |
| ViT-Large/16 | 1024 | 24 | 16 | 196 | ~307 M | ~310 M |

*Patch count assumes 224×224 input. DINO head adds ~5–10 M parameters depending on `out_dim`.*
