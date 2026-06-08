
"""
Advanced pytest suite for src.model.loss.dino_gram_loss.py

Run:
    pytest -q test_model_dino_gram_loss.py

Assumed module path:
    src.model.loss.dino_gram_loss

Covers:
    - DINOGramLoss initialization and validation
    - patch extraction error handling
    - Gram matrix mathematical properties
    - normalized vs unnormalized Gram behavior
    - student/teacher feature dimensions may differ
    - manual equivalence for mse / l1 / smooth_l1
    - teacher detach behavior
    - gradient flow only through student patches
    - pair averaging across multiple views
    - dtype casting behavior
    - device compatibility and CUDA when available
    - numerical stability with zero vectors
"""

import importlib

import pytest
import torch
import torch.nn.functional as F


MODULE_PATH = "src.model.loss.dino_gram_loss"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_module():
    return importlib.import_module(MODULE_PATH)


def make_loss(
    *,
    patch_key="patches",
    normalize_features=True,
    loss_type="mse",
    eps=1e-6,
):
    m = _load_module()
    return m.DINOGramLoss(
        patch_key=patch_key,
        normalize_features=normalize_features,
        loss_type=loss_type,
        eps=eps,
    )


def assert_close(a, b, atol=1e-6, rtol=1e-5):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"Max abs diff: {(a - b).abs().max().item()}"
    )


def manual_gram(patches, normalize_features=True, eps=1e-6):
    x = patches.float()
    if normalize_features:
        x = F.normalize(x, dim=-1, eps=eps)
    return x @ x.transpose(-1, -2)


def manual_pair_loss(student_patches, teacher_patches, loss_type, normalize_features=True, eps=1e-6):
    student_gram = manual_gram(student_patches, normalize_features=normalize_features, eps=eps)
    teacher_gram = manual_gram(teacher_patches, normalize_features=normalize_features, eps=eps)
    target = teacher_gram.detach().to(device=student_gram.device, dtype=student_gram.dtype)

    if loss_type == "mse":
        return F.mse_loss(student_gram, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(student_gram, target)
    if loss_type == "l1":
        return F.l1_loss(student_gram, target)
    raise ValueError(loss_type)


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

def test_initialization_stores_configuration():
    loss = make_loss(
        patch_key="tokens",
        normalize_features=False,
        loss_type="l1",
        eps=1e-4,
    )

    assert loss.patch_key == "tokens"
    assert loss.normalize_features is False
    assert loss.loss_type == "l1"
    assert loss.eps == 1e-4


@pytest.mark.parametrize("loss_type", ["mse", "smooth_l1", "l1"])
def test_valid_loss_types_are_accepted(loss_type):
    loss = make_loss(loss_type=loss_type)
    assert loss.loss_type == loss_type


@pytest.mark.parametrize("bad_loss_type", ["mae", "cosine", "kl", "", None])
def test_invalid_loss_type_raises_value_error(bad_loss_type):
    with pytest.raises(ValueError, match="loss_type must be"):
        make_loss(loss_type=bad_loss_type)


def test_normalize_features_is_cast_to_bool_and_eps_to_float():
    loss = make_loss(normalize_features=1, eps=1)

    assert loss.normalize_features is True
    assert isinstance(loss.eps, float)
    assert loss.eps == 1.0


# ---------------------------------------------------------------------
# _extract_patches validation
# ---------------------------------------------------------------------

def test_extract_patches_success():
    loss = make_loss(patch_key="patches")
    patches = torch.randn(2, 4, 8)

    out = loss._extract_patches({"patches": patches}, index=0, side="student")

    assert out is patches


def test_extract_patches_custom_patch_key():
    loss = make_loss(patch_key="student_patch_tokens")
    patches = torch.randn(2, 4, 8)

    out = loss._extract_patches(
        {"student_patch_tokens": patches},
        index=2,
        side="teacher",
    )

    assert out is patches


def test_extract_patches_missing_key_raises_clear_key_error():
    loss = make_loss(patch_key="patches")

    with pytest.raises(KeyError, match="does not contain key 'patches'"):
        loss._extract_patches({"cls": torch.randn(2, 8)}, index=3, side="student")


def test_extract_patches_non_tensor_raises_type_error():
    loss = make_loss(patch_key="patches")

    with pytest.raises(TypeError, match="must be a Tensor"):
        loss._extract_patches({"patches": [[1, 2, 3]]}, index=0, side="teacher")


@pytest.mark.parametrize(
    "bad_shape",
    [
        (2, 8),
        (2, 4, 8, 1),
        (8,),
        (),
    ],
)
def test_extract_patches_wrong_rank_raises_value_error(bad_shape):
    loss = make_loss(patch_key="patches")
    patches = torch.randn(*bad_shape) if bad_shape else torch.tensor(1.0)

    with pytest.raises(ValueError, match=r"\[B, N, D\]"):
        loss._extract_patches({"patches": patches}, index=0, side="student")


# ---------------------------------------------------------------------
# Gram matrix properties
# ---------------------------------------------------------------------

def test_gram_shape_symmetry_and_finiteness():
    torch.manual_seed(0)

    loss = make_loss(normalize_features=True)
    patches = torch.randn(3, 5, 7)

    gram = loss._gram(patches)

    assert gram.shape == (3, 5, 5)
    assert torch.isfinite(gram).all()
    assert_close(gram, gram.transpose(-1, -2))


def test_gram_with_normalization_has_unit_diagonal_for_nonzero_vectors():
    torch.manual_seed(1)

    loss = make_loss(normalize_features=True)
    patches = torch.randn(2, 6, 9)

    gram = loss._gram(patches)
    diag = torch.diagonal(gram, dim1=-2, dim2=-1)

    assert_close(diag, torch.ones_like(diag), atol=1e-5, rtol=1e-5)


def test_gram_with_normalization_handles_zero_vectors_without_nan():
    loss = make_loss(normalize_features=True, eps=1e-6)
    patches = torch.zeros(2, 4, 8)

    gram = loss._gram(patches)

    assert torch.isfinite(gram).all()
    assert torch.equal(gram, torch.zeros_like(gram))


def test_gram_without_normalization_matches_raw_dot_products():
    torch.manual_seed(2)

    loss = make_loss(normalize_features=False)
    patches = torch.randn(2, 4, 5)

    gram = loss._gram(patches)
    expected = patches.float() @ patches.float().transpose(-1, -2)

    assert_close(gram, expected)


def test_gram_always_casts_to_float32_from_float16():
    loss = make_loss(normalize_features=True)
    patches = torch.randn(2, 4, 8, dtype=torch.float16)

    gram = loss._gram(patches)

    assert gram.dtype == torch.float32


def test_gram_allows_student_and_teacher_different_feature_dims_manual_property():
    """
    The loss compares [B, N, N] Gram matrices, so D_student and D_teacher
    may differ as long as B and N match.
    """
    torch.manual_seed(3)

    loss = make_loss(normalize_features=True)

    student_patches = torch.randn(2, 5, 7)
    teacher_patches = torch.randn(2, 5, 11)

    student_gram = loss._gram(student_patches)
    teacher_gram = loss._gram(teacher_patches)

    assert student_gram.shape == teacher_gram.shape == (2, 5, 5)


# ---------------------------------------------------------------------
# _pair_loss behavior
# ---------------------------------------------------------------------

@pytest.mark.parametrize("loss_type", ["mse", "smooth_l1", "l1"])
def test_pair_loss_matches_torch_function(loss_type):
    torch.manual_seed(4)

    loss = make_loss(loss_type=loss_type)
    student_gram = torch.randn(2, 5, 5)
    teacher_gram = torch.randn(2, 5, 5, requires_grad=True)

    actual = loss._pair_loss(student_gram, teacher_gram)

    target = teacher_gram.detach().to(device=student_gram.device, dtype=student_gram.dtype)
    if loss_type == "mse":
        expected = F.mse_loss(student_gram, target)
    elif loss_type == "smooth_l1":
        expected = F.smooth_l1_loss(student_gram, target)
    else:
        expected = F.l1_loss(student_gram, target)

    assert_close(actual, expected)


def test_pair_loss_detaches_teacher_gram():
    torch.manual_seed(5)

    loss = make_loss(loss_type="mse")

    student_gram = torch.randn(2, 5, 5, requires_grad=True)
    teacher_gram = torch.randn(2, 5, 5, requires_grad=True)

    out = loss._pair_loss(student_gram, teacher_gram)
    out.backward()

    assert student_gram.grad is not None
    assert teacher_gram.grad is None


def test_pair_loss_casts_teacher_to_student_dtype():
    torch.manual_seed(6)

    loss = make_loss(loss_type="mse")

    student_gram = torch.randn(2, 5, 5, dtype=torch.float64)
    teacher_gram = torch.randn(2, 5, 5, dtype=torch.float32)

    out = loss._pair_loss(student_gram, teacher_gram)

    assert out.dtype == torch.float64


# ---------------------------------------------------------------------
# Forward validation
# ---------------------------------------------------------------------

def test_forward_rejects_different_number_of_student_teacher_outputs():
    loss = make_loss()

    student_outputs = [{"patches": torch.randn(2, 4, 8)}]
    teacher_outputs = [
        {"patches": torch.randn(2, 4, 8)},
        {"patches": torch.randn(2, 4, 8)},
    ]

    with pytest.raises(ValueError, match="same number"):
        loss(student_outputs=student_outputs, teacher_outputs=teacher_outputs)


def test_forward_rejects_empty_lists():
    loss = make_loss()

    with pytest.raises(ValueError, match="empty output lists"):
        loss(student_outputs=[], teacher_outputs=[])


@pytest.mark.parametrize(
    "student_shape,teacher_shape",
    [
        ((2, 4, 8), (3, 4, 8)),   # B mismatch
        ((2, 4, 8), (2, 5, 8)),   # N mismatch
        ((2, 4, 8), (2, 5, 11)),  # N mismatch even if D differs
    ],
)
def test_forward_rejects_batch_or_patch_count_mismatch(student_shape, teacher_shape):
    loss = make_loss()

    student_outputs = [{"patches": torch.randn(*student_shape)}]
    teacher_outputs = [{"patches": torch.randn(*teacher_shape)}]

    with pytest.raises(ValueError, match="patch shape mismatch"):
        loss(student_outputs=student_outputs, teacher_outputs=teacher_outputs)


def test_forward_allows_different_feature_dimensions_when_patch_count_matches():
    torch.manual_seed(7)

    loss = make_loss(normalize_features=True, loss_type="mse")

    student_outputs = [{"patches": torch.randn(2, 4, 8)}]
    teacher_outputs = [{"patches": torch.randn(2, 4, 13)}]

    out = loss(student_outputs=student_outputs, teacher_outputs=teacher_outputs)

    assert set(out.keys()) == {"loss", "pair_loss_mean"}
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert_close(out["pair_loss_mean"], out["loss"].detach())


@pytest.mark.parametrize("loss_type", ["mse", "smooth_l1", "l1"])
def test_forward_single_pair_matches_manual_computation(loss_type):
    torch.manual_seed(8)

    loss = make_loss(normalize_features=True, loss_type=loss_type, eps=1e-6)

    student_patches = torch.randn(2, 5, 7)
    teacher_patches = torch.randn(2, 5, 11)

    out = loss(
        student_outputs=[{"patches": student_patches}],
        teacher_outputs=[{"patches": teacher_patches}],
    )

    expected = manual_pair_loss(
        student_patches,
        teacher_patches,
        loss_type=loss_type,
        normalize_features=True,
        eps=1e-6,
    )

    assert_close(out["loss"], expected)
    assert_close(out["pair_loss_mean"], expected.detach())


def test_forward_multiple_pairs_returns_mean_of_pair_losses():
    torch.manual_seed(9)

    loss = make_loss(normalize_features=False, loss_type="mse")

    s1 = torch.randn(2, 4, 6)
    t1 = torch.randn(2, 4, 9)

    s2 = torch.randn(2, 3, 5)
    t2 = torch.randn(2, 3, 7)

    out = loss(
        student_outputs=[{"patches": s1}, {"patches": s2}],
        teacher_outputs=[{"patches": t1}, {"patches": t2}],
    )

    expected_1 = manual_pair_loss(s1, t1, "mse", normalize_features=False)
    expected_2 = manual_pair_loss(s2, t2, "mse", normalize_features=False)
    expected = torch.stack([expected_1, expected_2]).mean()

    assert_close(out["loss"], expected)
    assert_close(out["pair_loss_mean"], expected.detach())


def test_forward_custom_patch_key():
    torch.manual_seed(10)

    loss = make_loss(patch_key="patch_tokens")

    student_outputs = [{"patch_tokens": torch.randn(2, 4, 8)}]
    teacher_outputs = [{"patch_tokens": torch.randn(2, 4, 8)}]

    out = loss(student_outputs=student_outputs, teacher_outputs=teacher_outputs)

    assert torch.isfinite(out["loss"])


# ---------------------------------------------------------------------
# Gradient behavior
# ---------------------------------------------------------------------

def test_forward_backpropagates_to_student_not_teacher():
    torch.manual_seed(11)

    loss_fn = make_loss(normalize_features=True, loss_type="mse")

    student_patches = torch.randn(2, 5, 7, requires_grad=True)
    teacher_patches = torch.randn(2, 5, 11, requires_grad=True)

    out = loss_fn(
        student_outputs=[{"patches": student_patches}],
        teacher_outputs=[{"patches": teacher_patches}],
    )

    out["loss"].backward()

    assert student_patches.grad is not None
    assert torch.isfinite(student_patches.grad).all()
    assert student_patches.grad.abs().sum().item() > 0

    assert teacher_patches.grad is None


def test_pair_loss_mean_is_detached_from_graph():
    torch.manual_seed(12)

    loss_fn = make_loss()

    student_patches = torch.randn(2, 5, 7, requires_grad=True)
    teacher_patches = torch.randn(2, 5, 11, requires_grad=True)

    out = loss_fn(
        student_outputs=[{"patches": student_patches}],
        teacher_outputs=[{"patches": teacher_patches}],
    )

    assert out["loss"].requires_grad is True
    assert out["pair_loss_mean"].requires_grad is False


def test_loss_is_zero_when_student_and_teacher_patches_identical():
    torch.manual_seed(13)

    loss_fn = make_loss(normalize_features=True, loss_type="mse")

    patches = torch.randn(2, 6, 9)

    out = loss_fn(
        student_outputs=[{"patches": patches}],
        teacher_outputs=[{"patches": patches.clone()}],
    )

    assert_close(out["loss"], torch.tensor(0.0))


def test_loss_is_zero_when_student_teacher_related_by_feature_orthogonal_transform():
    """
    Gram matching should be invariant to orthogonal rotations in feature space
    when normalize_features=False, because XQ (XQ)^T = XQQ^T X^T = XX^T.
    """
    torch.manual_seed(14)

    loss_fn = make_loss(normalize_features=False, loss_type="mse")

    patches = torch.randn(2, 5, 8)

    # Build random orthogonal matrix Q.
    q_matrix, _ = torch.linalg.qr(torch.randn(8, 8))
    rotated = patches @ q_matrix

    out = loss_fn(
        student_outputs=[{"patches": patches}],
        teacher_outputs=[{"patches": rotated}],
    )

    assert_close(out["loss"], torch.tensor(0.0), atol=1e-10, rtol=1e-6)


def test_normalized_gram_is_scale_invariant_for_positive_scaling():
    torch.manual_seed(15)

    loss_fn = make_loss(normalize_features=True, loss_type="mse")

    patches = torch.randn(2, 5, 8)
    scaled = patches * 123.0

    out = loss_fn(
        student_outputs=[{"patches": patches}],
        teacher_outputs=[{"patches": scaled}],
    )

    assert_close(out["loss"], torch.tensor(0.0), atol=1e-10, rtol=1e-6)


def test_unnormalized_gram_is_not_scale_invariant():
    torch.manual_seed(16)

    loss_fn = make_loss(normalize_features=False, loss_type="mse")

    patches = torch.randn(2, 5, 8)
    scaled = patches * 2.0

    out = loss_fn(
        student_outputs=[{"patches": patches}],
        teacher_outputs=[{"patches": scaled}],
    )

    assert out["loss"].item() > 0.0


# ---------------------------------------------------------------------
# Dtype, device, numerical stability
# ---------------------------------------------------------------------

def test_forward_casts_float16_inputs_to_float32_loss():
    torch.manual_seed(17)

    loss_fn = make_loss(normalize_features=True, loss_type="mse")

    student_patches = torch.randn(2, 5, 7, dtype=torch.float16)
    teacher_patches = torch.randn(2, 5, 11, dtype=torch.float16)

    out = loss_fn(
        student_outputs=[{"patches": student_patches}],
        teacher_outputs=[{"patches": teacher_patches}],
    )

    assert out["loss"].dtype == torch.float32
    assert out["pair_loss_mean"].dtype == torch.float32


def test_forward_float64_inputs_currently_return_float32_loss_due_to_gram_cast():
    """
    The implementation calls patches.float(), so even float64 inputs produce
    float32 Gram matrices and a float32 loss. This test documents that behavior.
    """
    torch.manual_seed(18)

    loss_fn = make_loss(normalize_features=True, loss_type="mse")

    student_patches = torch.randn(2, 5, 7, dtype=torch.float64)
    teacher_patches = torch.randn(2, 5, 11, dtype=torch.float64)

    out = loss_fn(
        student_outputs=[{"patches": student_patches}],
        teacher_outputs=[{"patches": teacher_patches}],
    )

    assert out["loss"].dtype == torch.float32


def test_forward_with_zero_vectors_is_finite():
    loss_fn = make_loss(normalize_features=True, loss_type="mse", eps=1e-6)

    student_patches = torch.zeros(2, 5, 7, requires_grad=True)
    teacher_patches = torch.zeros(2, 5, 11, requires_grad=True)

    out = loss_fn(
        student_outputs=[{"patches": student_patches}],
        teacher_outputs=[{"patches": teacher_patches}],
    )

    assert torch.isfinite(out["loss"])
    assert_close(out["loss"], torch.tensor(0.0))

    out["loss"].backward()
    assert student_patches.grad is not None
    assert torch.isfinite(student_patches.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_forward_cuda_outputs_remain_on_cuda():
    torch.manual_seed(19)

    loss_fn = make_loss(normalize_features=True, loss_type="mse").cuda()

    student_patches = torch.randn(2, 5, 7, device="cuda", requires_grad=True)
    teacher_patches = torch.randn(2, 5, 11, device="cuda", requires_grad=True)

    out = loss_fn(
        student_outputs=[{"patches": student_patches}],
        teacher_outputs=[{"patches": teacher_patches}],
    )

    assert out["loss"].device.type == "cuda"
    assert out["pair_loss_mean"].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_consistency_for_same_inputs():
    torch.manual_seed(20)

    loss_cpu = make_loss(normalize_features=True, loss_type="smooth_l1").eval()
    loss_cuda = make_loss(normalize_features=True, loss_type="smooth_l1").cuda().eval()

    student_cpu = torch.randn(2, 5, 7)
    teacher_cpu = torch.randn(2, 5, 11)

    student_cuda = student_cpu.cuda()
    teacher_cuda = teacher_cpu.cuda()

    out_cpu = loss_cpu(
        student_outputs=[{"patches": student_cpu}],
        teacher_outputs=[{"patches": teacher_cpu}],
    )
    out_cuda = loss_cuda(
        student_outputs=[{"patches": student_cuda}],
        teacher_outputs=[{"patches": teacher_cuda}],
    )

    assert_close(out_cpu["loss"], out_cuda["loss"].cpu(), atol=1e-6, rtol=1e-6)


def test_large_patch_count_no_nan_or_inf():
    torch.manual_seed(21)

    loss_fn = make_loss(normalize_features=True, loss_type="mse")

    student_patches = torch.randn(2, 128, 64)
    teacher_patches = torch.randn(2, 128, 96)

    out = loss_fn(
        student_outputs=[{"patches": student_patches}],
        teacher_outputs=[{"patches": teacher_patches}],
    )

    assert torch.isfinite(out["loss"])
    assert out["loss"].item() >= 0.0
