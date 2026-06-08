# Loss Functions

Two losses are composed during training: the primary **DINO Loss** (self-distillation cross-entropy) and the optional **Gram Loss** (patch Gram matrix regularization).

---

## DINO Loss

**File:** `src/model/loss/dino_loss.py`  
**Class:** `DINOLoss(out_dim, center_momentum=0.9)`

### Intuition

The student and teacher each receive a different crop of the same image and produce logit vectors. The student is trained to predict the teacher's output distribution. Because the teacher sees a different crop, the student must learn to capture the same semantic content regardless of viewpoint, scale, and augmentation — building view-invariant representations.

### Formulation

Let `s_i` be the student logit vector for crop `i` and `t_j` be the teacher logit vector for crop `j`. The DINO loss averages cross-entropy over all valid cross-view pairs (pairs where `i ≠ j`):

```
L_DINO = (1 / |pairs|) * Σ_{i≠j} H(p_t^j, p_s^i)
```

where the distributions are computed as:

```
p_s^i = softmax(s_i / τ_s)           # student probability
p_t^j = softmax((t_j − c) / τ_t)     # teacher probability, centered
```

and `H(p, q) = −Σ p * log(q)` is the cross-entropy.

- `τ_s` — student temperature (typically 0.1). Higher → softer student predictions.
- `τ_t` — teacher temperature (typically 0.04–0.07). Lower → sharper teacher targets.
- `c` — the centering vector (described below).

The log-softmax / softmax form used in the code is equivalent:

```python
student_log_probs = log_softmax(s / τ_s)
teacher_probs     = softmax((t − c) / τ_t).detach()
loss              = −(teacher_probs * student_log_probs).sum(dim=-1).mean()
```

### Centering

Centering subtracts a running average from the teacher logits before computing the softmax. Without it, one prototype dimension could dominate and cause collapse.

```
c ← momentum_c * c + (1 − momentum_c) * mean(teacher_logits_in_batch)
```

The center is updated *after* computing the loss for the current batch (using the pre-update center). The EMA momentum `center_momentum` is typically 0.9, which smooths the center over ~10 recent batches.

This is implemented as a registered buffer (`register_buffer`) so it is saved with the model state dict and survives checkpoint/resume.

### Sharpening

The teacher temperature `τ_t` is kept lower than the student temperature `τ_s`. This sharpens the teacher's output distribution (makes it more peaked / confident), which gives the student a clearer training signal. If both temperatures were equal, the loss would receive weak gradients from near-uniform distributions.

The teacher temperature is linearly warmed up from `warmup_teacher_temp` to `teacher_temp` over `warmup_teacher_temp_epochs` epochs to prevent early instability.

### Cross-view pairs

For 2 global crops + 4 local crops (the default), the student sees 6 views, and the teacher sees 2 (global only — teacher receives only global crops in the standard DINO setup). The number of valid pairs is:

```
n_pairs = n_student_crops * n_teacher_crops − n_same_view_pairs
        = 6 * 2 − 2 = 10
```

The loss averages uniformly over these 10 pairs.

---

## Gram Loss

**File:** `src/model/loss/dino_gram_loss.py`  
**Class:** `DINOGramLoss(patch_key="patches", normalize_features=True, loss_type="mse")`

### Intuition

The DINO loss trains the CLS token to be view-consistent. It says nothing about the *patch* tokens (the spatial feature map). The Gram loss adds a complementary objective: it asks the student's patch tokens to reproduce the teacher's patch-to-patch co-activation patterns.

The **Gram matrix** of a set of feature vectors captures pairwise inner products — which patches fire together. By matching student and teacher Gram matrices, we encourage the student to develop spatial feature structures that mirror the teacher's, even when the two might differ in embedding dimension (which never happens here, but the loss is designed to handle it).

This is analogous to style matching in neural style transfer (Gatys et al. 2015), but applied to self-supervised features rather than style/content decomposition.

### Formulation

For each crop pair `(student_crop_i, teacher_crop_i)`, given patch token matrices:

```
P_s ∈ ℝ^{B × N × D}     (student patches, optionally L2-normalized per patch)
P_t ∈ ℝ^{B × N × D}     (teacher patches, optionally L2-normalized per patch)
```

The Gram matrices are:

```
G_s = P_s @ P_s^T    ∈ ℝ^{B × N × N}
G_t = P_t @ P_t^T    ∈ ℝ^{B × N × N}
```

Each entry `G[b, i, j]` is the dot product between the features of patch `i` and patch `j` in image `b`. With L2-normalized features this equals the cosine similarity between patches.

The per-pair loss is:

```
L_gram_i = MSE(G_s, sg(G_t))     # sg = stop gradient (teacher is detached)
```

The total Gram loss averages over all crop pairs:

```
L_gram = (1 / n_crops) * Σ_i L_gram_i
```

The final training loss combines both objectives:

```
L_total = L_DINO + λ_gram * L_gram
```

where `λ_gram` = `gram_loss_weight` (default 0.05).

### Configuration options

| Parameter | Default | Effect |
|---|---|---|
| `normalize_features` | `true` | L2-normalize patches before Gram computation; makes entries cosine similarities |
| `loss_type` | `"mse"` | `"mse"`, `"smooth_l1"`, or `"l1"` — controls the regression loss between Gram matrices |
| `patch_key` | `"patches"` | Key in the model output dict where patch tokens are stored |

### Memory considerations

The Gram matrix is `[B, N, N]`. For a ViT-S/8 at 224×224, N = 784, so the Gram matrix is `[B, 784, 784]`. At batch size 128 in float32, this is roughly 784² × 128 × 4 bytes ≈ 314 MB. With `normalize_features=True` and bfloat16 training, this is approximately halved. Reduce `batch_size` or disable the Gram loss if memory is tight.

---

## Combined Training Objective

During training, the two losses are computed and weighted:

```python
loss_dino = dino_loss_fn(
    student_logits=student_cls_logits,   # list of [B, K] tensors, one per crop
    teacher_logits=teacher_cls_logits,   # list of [B, K] tensors, global crops only
    student_temp=student_temp,
    teacher_temp=scheduled_teacher_temp,
)["loss"]

if use_gram_loss:
    loss_gram = gram_loss_fn(
        student_outputs=student_patch_outputs,   # list of dicts with "patches" key
        teacher_outputs=teacher_patch_outputs,
    )["loss"]
    loss = loss_dino + gram_loss_weight * loss_gram
else:
    loss = loss_dino
```

The gradient flows only through the student. The teacher's logits and patch tokens are detached before being used as targets.

---

## Stability Tips

- If loss diverges early, reduce `learning_rate` or increase `warmup_epochs`.
- If the model collapses (all crops produce the same output), check that centering is running and that `teacher_temp < student_temp`.
- If Gram loss produces NaN, check that patch features do not contain NaN (which can happen with very high learning rates or without gradient clipping). `grad_clip: 1.0` is recommended.
- Monitor `train/teacher_temp` and `train/momentum` in metrics JSONL — if momentum never reaches ~0.999+, the EMA is not properly tracking the student.
