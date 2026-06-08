
"""
Advanced pytest suite for src.model.rope.py

Run:
    pytest -q test_model_rope.py

Assumed module path:
    src.model.rope

Covers:
    - rotate_half pairwise algebra
    - RotaryEmbedding2D initialization and buffers
    - exact/manual construction of 2D axial cos/sin
    - identity behavior at grid position (0, 0)
    - norm preservation under RoPE rotation
    - q/k independent transformation
    - broadcasting across batch and heads
    - dtype/device behavior
    - gradient flow
    - invalid head_dim and token/grid mismatch errors
    - CPU/CUDA consistency when CUDA is available
"""

import importlib
import math

import pytest
import torch
import torch.nn as nn


MODULE_PATH = "src.model.rope"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_module():
    return importlib.import_module(MODULE_PATH)


def assert_close(a, b, atol=1e-6, rtol=1e-5):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"Max abs diff: {(a - b).abs().max().item()}"
    )


def manual_rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def manual_2d_cos_sin(head_dim, base, grid_size, device, dtype):
    """
    Mirrors the implementation mathematically, independently from the class.
    """
    if head_dim % 4 != 0:
        raise ValueError("head_dim must be divisible by 4")

    axis_dim = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, axis_dim, 2, device=device).float() / axis_dim)
    )

    gh, gw = grid_size
    y = torch.arange(gh, device=device, dtype=inv_freq.dtype)
    x = torch.arange(gw, device=device, dtype=inv_freq.dtype)

    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    freqs_y = torch.einsum("n,d->nd", yy, inv_freq)
    freqs_x = torch.einsum("n,d->nd", xx, inv_freq)

    emb_y = torch.cat([freqs_y, freqs_y], dim=-1)
    emb_x = torch.cat([freqs_x, freqs_x], dim=-1)
    emb = torch.cat([emb_y, emb_x], dim=-1)

    cos = emb.cos().to(dtype=dtype)[None, None, :, :]
    sin = emb.sin().to(dtype=dtype)[None, None, :, :]

    return cos, sin


# ---------------------------------------------------------------------
# rotate_half tests
# ---------------------------------------------------------------------

def test_rotate_half_basic_pairwise_contract():
    m = _load_module()

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = m.rotate_half(x)

    expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]])
    assert torch.equal(y, expected)


def test_rotate_half_matches_manual_implementation_for_high_rank_tensor():
    m = _load_module()

    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 8)

    y = m.rotate_half(x)
    expected = manual_rotate_half(x)

    assert y.shape == x.shape
    assert_close(y, expected)


def test_rotate_half_is_linear():
    m = _load_module()

    torch.manual_seed(1)
    x = torch.randn(2, 3, 8)
    z = torch.randn(2, 3, 8)
    a = 1.7
    b = -0.3

    lhs = m.rotate_half(a * x + b * z)
    rhs = a * m.rotate_half(x) + b * m.rotate_half(z)

    assert_close(lhs, rhs)


def test_rotate_half_applied_twice_is_negative_identity():
    m = _load_module()

    torch.manual_seed(2)
    x = torch.randn(2, 5, 12)

    y = m.rotate_half(m.rotate_half(x))

    assert_close(y, -x)


def test_rotate_half_preserves_l2_norm_pairwise_and_globally():
    m = _load_module()

    torch.manual_seed(3)
    x = torch.randn(4, 6, 16)

    y = m.rotate_half(x)

    assert_close(y.norm(dim=-1), x.norm(dim=-1))


def test_rotate_half_odd_last_dimension_raises_runtime_error():
    """
    The implementation assumes paired features. An odd final dimension makes
    x_even and x_odd incompatible in torch.stack.
    """
    m = _load_module()

    x = torch.randn(2, 3, 5)

    with pytest.raises(RuntimeError):
        m.rotate_half(x)


def test_rotate_half_preserves_dtype_and_device_cpu():
    m = _load_module()

    x = torch.randn(2, 3, 8, dtype=torch.float64)
    y = m.rotate_half(x)

    assert y.dtype == torch.float64
    assert y.device == x.device


# ---------------------------------------------------------------------
# RotaryEmbedding2D initialization and buffer tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("head_dim", [4, 8, 16, 32])
def test_rotary_embedding_accepts_head_dim_divisible_by_4(head_dim):
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=head_dim, base=10000.0)

    assert isinstance(rope, nn.Module)
    assert rope.head_dim == head_dim
    assert rope.axis_dim == head_dim // 2
    assert rope.inv_freq.shape == (head_dim // 4,)
    assert rope.inv_freq.dtype == torch.float32
    assert rope.inv_freq.requires_grad is False


@pytest.mark.parametrize("bad_head_dim", [1, 2, 3, 5, 6, 10, 14])
def test_rotary_embedding_rejects_head_dim_not_divisible_by_4(bad_head_dim):
    m = _load_module()

    with pytest.raises(ValueError, match="divisible by 4"):
        m.RotaryEmbedding2D(head_dim=bad_head_dim)


def test_inv_freq_is_non_persistent_buffer():
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=16)

    assert "inv_freq" in dict(rope.named_buffers())
    assert "inv_freq" not in rope.state_dict(), (
        "inv_freq is registered with persistent=False and should not be serialized."
    )


def test_inv_freq_matches_formula_exactly():
    m = _load_module()

    head_dim = 16
    base = 1000.0
    rope = m.RotaryEmbedding2D(head_dim=head_dim, base=base)

    axis_dim = head_dim // 2
    expected = 1.0 / (
        base ** (torch.arange(0, axis_dim, 2).float() / axis_dim)
    )

    assert_close(rope.inv_freq, expected)


# ---------------------------------------------------------------------
# _build_2d_cos_sin tests
# ---------------------------------------------------------------------

def test_build_2d_cos_sin_shapes_dtype_device_and_finiteness():
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=8)
    cos, sin = rope._build_2d_cos_sin(
        grid_size=(3, 5),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert cos.shape == (1, 1, 15, 8)
    assert sin.shape == (1, 1, 15, 8)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32
    assert cos.device.type == "cpu"
    assert sin.device.type == "cpu"
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()


def test_build_2d_cos_sin_matches_manual_formula():
    m = _load_module()

    head_dim = 16
    base = 7777.0
    grid_size = (3, 4)
    dtype = torch.float64

    rope = m.RotaryEmbedding2D(head_dim=head_dim, base=base)

    cos, sin = rope._build_2d_cos_sin(
        grid_size=grid_size,
        device=torch.device("cpu"),
        dtype=dtype,
    )

    expected_cos, expected_sin = manual_2d_cos_sin(
        head_dim=head_dim,
        base=base,
        grid_size=grid_size,
        device=torch.device("cpu"),
        dtype=dtype,
    )

    assert_close(cos, expected_cos, atol=1e-12, rtol=1e-12)
    assert_close(sin, expected_sin, atol=1e-12, rtol=1e-12)


def test_build_2d_cos_sin_position_zero_is_identity_rotation():
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=16)
    cos, sin = rope._build_2d_cos_sin(
        grid_size=(2, 3),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Row-major first token corresponds to y=0, x=0.
    assert_close(cos[0, 0, 0], torch.ones(16))
    assert_close(sin[0, 0, 0], torch.zeros(16))


def test_build_2d_cos_sin_row_major_axial_encoding():
    """
    For grid_size=(2, 3), token indices are:
        0 -> y=0, x=0
        1 -> y=0, x=1
        2 -> y=0, x=2
        3 -> y=1, x=0
        4 -> y=1, x=1
        5 -> y=1, x=2

    First half of head_dim encodes y, second half encodes x.
    """
    m = _load_module()

    head_dim = 8
    rope = m.RotaryEmbedding2D(head_dim=head_dim, base=10000.0)
    cos, sin = rope._build_2d_cos_sin(
        grid_size=(2, 3),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Token 1: y=0, x=1 -> y part identity, x part nontrivial.
    assert_close(cos[0, 0, 1, :4], torch.ones(4))
    assert_close(sin[0, 0, 1, :4], torch.zeros(4))
    assert not torch.allclose(cos[0, 0, 1, 4:], torch.ones(4))
    assert not torch.allclose(sin[0, 0, 1, 4:], torch.zeros(4))

    # Token 3: y=1, x=0 -> y part nontrivial, x part identity.
    assert not torch.allclose(cos[0, 0, 3, :4], torch.ones(4))
    assert not torch.allclose(sin[0, 0, 3, :4], torch.zeros(4))
    assert_close(cos[0, 0, 3, 4:], torch.ones(4))
    assert_close(sin[0, 0, 3, 4:], torch.zeros(4))


def test_cos_sin_satisfy_unit_circle_identity():
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=32)
    cos, sin = rope._build_2d_cos_sin(
        grid_size=(5, 7),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert_close(cos.pow(2) + sin.pow(2), torch.ones_like(cos), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------

def test_forward_shapes_and_dtype_are_preserved():
    m = _load_module()

    torch.manual_seed(10)
    rope = m.RotaryEmbedding2D(head_dim=8)

    q = torch.randn(2, 3, 12, 8)
    k = torch.randn(2, 3, 12, 8)

    q_out, k_out = rope(q, k, grid_size=(3, 4))

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert q_out.dtype == q.dtype
    assert k_out.dtype == k.dtype
    assert q_out.device == q.device
    assert k_out.device == k.device
    assert torch.isfinite(q_out).all()
    assert torch.isfinite(k_out).all()


def test_forward_matches_manual_rope_formula():
    m = _load_module()

    torch.manual_seed(11)

    head_dim = 16
    base = 12345.0
    grid_size = (2, 3)

    rope = m.RotaryEmbedding2D(head_dim=head_dim, base=base)

    q = torch.randn(2, 4, 6, head_dim)
    k = torch.randn(2, 4, 6, head_dim)

    q_out, k_out = rope(q, k, grid_size=grid_size)

    cos, sin = manual_2d_cos_sin(
        head_dim=head_dim,
        base=base,
        grid_size=grid_size,
        device=q.device,
        dtype=q.dtype,
    )

    expected_q = (q * cos) + (manual_rotate_half(q) * sin)
    expected_k = (k * cos) + (manual_rotate_half(k) * sin)

    assert_close(q_out, expected_q)
    assert_close(k_out, expected_k)


def test_forward_first_grid_position_is_identity_for_q_and_k():
    m = _load_module()

    torch.manual_seed(12)

    rope = m.RotaryEmbedding2D(head_dim=16)
    q = torch.randn(2, 3, 6, 16)
    k = torch.randn(2, 3, 6, 16)

    q_out, k_out = rope(q, k, grid_size=(2, 3))

    assert_close(q_out[:, :, 0, :], q[:, :, 0, :])
    assert_close(k_out[:, :, 0, :], k[:, :, 0, :])


def test_forward_preserves_l2_norm_per_token_for_q_and_k():
    """
    RoPE is a block-wise orthogonal rotation, so it must preserve the L2 norm
    of each token/head vector.
    """
    m = _load_module()

    torch.manual_seed(13)

    rope = m.RotaryEmbedding2D(head_dim=32)
    q = torch.randn(4, 5, 20, 32)
    k = torch.randn(4, 5, 20, 32)

    q_out, k_out = rope(q, k, grid_size=(4, 5))

    assert_close(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)
    assert_close(k_out.norm(dim=-1), k.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_forward_broadcasts_same_rotation_across_batch_and_heads():
    m = _load_module()

    torch.manual_seed(14)

    rope = m.RotaryEmbedding2D(head_dim=8)

    # Same q/k vector repeated across batch and heads.
    base_q = torch.randn(1, 1, 6, 8)
    base_k = torch.randn(1, 1, 6, 8)

    q = base_q.expand(3, 4, 6, 8).clone()
    k = base_k.expand(3, 4, 6, 8).clone()

    q_out, k_out = rope(q, k, grid_size=(2, 3))

    for b in range(3):
        for h in range(4):
            assert_close(q_out[b, h], q_out[0, 0])
            assert_close(k_out[b, h], k_out[0, 0])


def test_forward_transforms_q_and_k_independently():
    m = _load_module()

    torch.manual_seed(15)

    rope = m.RotaryEmbedding2D(head_dim=16)

    q = torch.randn(2, 3, 6, 16)
    k = torch.zeros_like(q)

    q_out, k_out = rope(q, k, grid_size=(2, 3))

    assert q_out.abs().sum().item() > 0
    assert torch.equal(k_out, torch.zeros_like(k_out))


def test_forward_does_not_modify_inputs_in_place():
    m = _load_module()

    torch.manual_seed(16)

    rope = m.RotaryEmbedding2D(head_dim=16)
    q = torch.randn(2, 3, 6, 16)
    k = torch.randn(2, 3, 6, 16)

    q_before = q.clone()
    k_before = k.clone()

    _ = rope(q, k, grid_size=(2, 3))

    assert torch.equal(q, q_before)
    assert torch.equal(k, k_before)


def test_forward_rejects_q_grid_token_mismatch():
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=16)
    q = torch.randn(2, 3, 5, 16)
    k = torch.randn(2, 3, 6, 16)

    with pytest.raises(ValueError, match="q has 5 patch tokens"):
        rope(q, k, grid_size=(2, 3))


def test_forward_rejects_k_grid_token_mismatch():
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=16)
    q = torch.randn(2, 3, 6, 16)
    k = torch.randn(2, 3, 5, 16)

    with pytest.raises(ValueError, match="k has 5 patch tokens"):
        rope(q, k, grid_size=(2, 3))


def test_forward_wrong_head_dim_fails_loudly():
    """
    The module is initialized for one head_dim. If q/k are passed with a different
    final dim, broadcasting/multiplication should fail instead of silently working.
    """
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=16)
    q = torch.randn(2, 3, 6, 8)
    k = torch.randn(2, 3, 6, 8)

    with pytest.raises(RuntimeError):
        rope(q, k, grid_size=(2, 3))


def test_forward_with_empty_patch_grid_returns_empty_tokens():
    """
    Edge case: grid with zero patches. Not usually meaningful for ViT, but the
    internal tensor logic should remain well-defined.
    """
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=8)
    q = torch.randn(2, 3, 0, 8)
    k = torch.randn(2, 3, 0, 8)

    q_out, k_out = rope(q, k, grid_size=(0, 4))

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


# ---------------------------------------------------------------------
# Gradient behavior
# ---------------------------------------------------------------------

def test_forward_backward_populates_q_and_k_gradients():
    m = _load_module()

    torch.manual_seed(17)

    rope = m.RotaryEmbedding2D(head_dim=16)

    q = torch.randn(2, 3, 6, 16, requires_grad=True)
    k = torch.randn(2, 3, 6, 16, requires_grad=True)

    q_out, k_out = rope(q, k, grid_size=(2, 3))
    loss = q_out.pow(2).mean() + 0.7 * k_out.pow(2).mean()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert q.grad.abs().sum().item() > 0
    assert k.grad.abs().sum().item() > 0


def test_inv_freq_does_not_receive_gradients():
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=16)

    q = torch.randn(2, 3, 6, 16, requires_grad=True)
    k = torch.randn(2, 3, 6, 16, requires_grad=True)

    q_out, k_out = rope(q, k, grid_size=(2, 3))
    loss = q_out.sum() + k_out.sum()
    loss.backward()

    assert rope.inv_freq.requires_grad is False
    assert rope.inv_freq.grad is None


def test_no_grad_forward_returns_non_grad_tensors():
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=16)

    q = torch.randn(2, 3, 6, 16, requires_grad=True)
    k = torch.randn(2, 3, 6, 16, requires_grad=True)

    with torch.no_grad():
        q_out, k_out = rope(q, k, grid_size=(2, 3))

    assert q_out.requires_grad is False
    assert k_out.requires_grad is False


# ---------------------------------------------------------------------
# Dtype and device behavior
# ---------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_preserves_float_dtype(dtype):
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=16)

    q = torch.randn(2, 3, 6, 16, dtype=dtype)
    k = torch.randn(2, 3, 6, 16, dtype=dtype)

    q_out, k_out = rope(q, k, grid_size=(2, 3))

    assert q_out.dtype == dtype
    assert k_out.dtype == dtype


def test_forward_supports_float16_on_cpu_for_output_dtype():
    """
    The implementation casts cos/sin to q.dtype. On CPU, some operations may
    internally compute in float32, but the final output should preserve float16
    for this simple expression.
    """
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=16)

    q = torch.randn(2, 3, 6, 16, dtype=torch.float16)
    k = torch.randn(2, 3, 6, 16, dtype=torch.float16)

    q_out, k_out = rope(q, k, grid_size=(2, 3))

    assert q_out.dtype == torch.float16
    assert k_out.dtype == torch.float16
    assert torch.isfinite(q_out.float()).all()
    assert torch.isfinite(k_out.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_consistency_for_same_inputs():
    m = _load_module()

    torch.manual_seed(18)

    rope_cpu = m.RotaryEmbedding2D(head_dim=16, base=10000.0).eval()
    rope_cuda = m.RotaryEmbedding2D(head_dim=16, base=10000.0).eval().cuda()

    q_cpu = torch.randn(2, 3, 6, 16)
    k_cpu = torch.randn(2, 3, 6, 16)

    q_cuda = q_cpu.cuda()
    k_cuda = k_cpu.cuda()

    with torch.no_grad():
        q_out_cpu, k_out_cpu = rope_cpu(q_cpu, k_cpu, grid_size=(2, 3))
        q_out_cuda, k_out_cuda = rope_cuda(q_cuda, k_cuda, grid_size=(2, 3))

    assert_close(q_out_cpu, q_out_cuda.cpu(), atol=1e-5, rtol=1e-5)
    assert_close(k_out_cpu, k_out_cuda.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_forward_cuda_outputs_remain_on_cuda():
    m = _load_module()

    rope = m.RotaryEmbedding2D(head_dim=16).cuda()

    q = torch.randn(2, 3, 6, 16, device="cuda")
    k = torch.randn(2, 3, 6, 16, device="cuda")

    q_out, k_out = rope(q, k, grid_size=(2, 3))

    assert q_out.device.type == "cuda"
    assert k_out.device.type == "cuda"


# ---------------------------------------------------------------------
# Numerical diagnostics
# ---------------------------------------------------------------------

def test_large_grid_no_nan_or_inf():
    m = _load_module()

    torch.manual_seed(19)

    rope = m.RotaryEmbedding2D(head_dim=32)
    grid_size = (16, 16)
    n = grid_size[0] * grid_size[1]

    q = torch.randn(1, 2, n, 32)
    k = torch.randn(1, 2, n, 32)

    q_out, k_out = rope(q, k, grid_size=grid_size)

    assert torch.isfinite(q_out).all()
    assert torch.isfinite(k_out).all()
    assert_close(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)
    assert_close(k_out.norm(dim=-1), k.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_different_base_changes_nonzero_position_rotation():
    m = _load_module()

    torch.manual_seed(20)

    q = torch.randn(1, 1, 6, 16)
    k = torch.randn(1, 1, 6, 16)

    rope_a = m.RotaryEmbedding2D(head_dim=16, base=10000.0)
    rope_b = m.RotaryEmbedding2D(head_dim=16, base=10.0)

    q_a, k_a = rope_a(q, k, grid_size=(2, 3))
    q_b, k_b = rope_b(q, k, grid_size=(2, 3))

    # Position zero is identity for both bases.
    assert_close(q_a[:, :, 0], q_b[:, :, 0])
    assert_close(k_a[:, :, 0], k_b[:, :, 0])

    # Later positions should differ for at least some channels.
    assert not torch.allclose(q_a[:, :, 5], q_b[:, :, 5])
    assert not torch.allclose(k_a[:, :, 5], k_b[:, :, 5])
