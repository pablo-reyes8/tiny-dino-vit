
"""
Advanced pytest suite for src.model.vit_utils.py

Run:
    pytest -q test_model_vit_utils_attention.py

Assumed module path:
    src.model.vit_utils

This file tests the MultiHeadSelfAttention class as provided in vit_utils.

Covers:
    - initialization contracts
    - exact manual self-attention equivalence without RoPE/dropout
    - attention probability invariants
    - return_attention behavior
    - qkv_bias behavior
    - dropout train/eval behavior
    - RoPE integration over patch tokens only
    - special tokens are not passed to RoPE
    - error handling for invalid dim/heads and missing/mismatched grid_size
    - gradient flow
    - dtype/device behavior
    - CPU/CUDA consistency when CUDA is available
    - permutation equivariance without positional encodings
"""

import importlib
import pytest
import torch
import torch.nn as nn


MODULE_PATH = "src.model.vit_utils"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_module():
    return importlib.import_module(MODULE_PATH)


def assert_close(a, b, atol=1e-6, rtol=1e-5):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"Max abs diff: {(a - b).abs().max().item()}"
    )


def make_attention(
    *,
    dim=16,
    num_heads=4,
    qkv_bias=True,
    attn_dropout=0.0,
    proj_dropout=0.0,
    use_rope=False,
    rope_base=10000.0,
):
    m = _load_module()
    return m.MultiHeadSelfAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        attn_dropout=attn_dropout,
        proj_dropout=proj_dropout,
        use_rope=use_rope,
        rope_base=rope_base,
    )


def manual_attention_no_rope(module, x):
    """
    Independent reference implementation for MultiHeadSelfAttention
    when use_rope=False.
    """
    b, t, d = x.shape

    qkv = module.qkv(x)
    qkv = qkv.reshape(b, t, 3, module.num_heads, module.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv.unbind(dim=0)

    logits = (q @ k.transpose(-2, -1)) * module.scale
    attn = logits.softmax(dim=-1)
    attn = module.attn_drop(attn)

    out = attn @ v
    out = out.transpose(1, 2).reshape(b, t, d)
    out = module.proj(out)
    out = module.proj_drop(out)

    return out, attn


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

def test_attention_initializes_attributes_without_rope():
    attn = make_attention(dim=32, num_heads=4, use_rope=False)

    assert isinstance(attn, nn.Module)
    assert attn.dim == 32
    assert attn.num_heads == 4
    assert attn.head_dim == 8
    assert attn.scale == 8 ** -0.5
    assert attn.use_rope is False

    assert isinstance(attn.qkv, nn.Linear)
    assert attn.qkv.in_features == 32
    assert attn.qkv.out_features == 96

    assert isinstance(attn.proj, nn.Linear)
    assert attn.proj.in_features == 32
    assert attn.proj.out_features == 32

    assert isinstance(attn.attn_drop, nn.Dropout)
    assert isinstance(attn.proj_drop, nn.Dropout)
    assert attn.rope is None


def test_attention_initializes_rope_when_requested():
    attn = make_attention(dim=32, num_heads=4, use_rope=True, rope_base=1234.0)

    assert attn.use_rope is True
    assert attn.rope is not None
    assert attn.rope.head_dim == 8
    assert attn.rope.base == 1234.0


@pytest.mark.parametrize(
    "dim,num_heads",
    [
        (10, 3),
        (31, 4),
        (64, 7),
    ],
)
def test_attention_rejects_dim_not_divisible_by_num_heads(dim, num_heads):
    with pytest.raises(ValueError, match="dim must be divisible"):
        make_attention(dim=dim, num_heads=num_heads)


def test_attention_with_rope_rejects_head_dim_not_divisible_by_4():
    """
    dim % num_heads is valid here, but head_dim=6 is invalid for 2D RoPE.
    """
    with pytest.raises(ValueError, match="divisible by 4"):
        make_attention(dim=24, num_heads=4, use_rope=True)


def test_qkv_bias_flag_is_respected():
    with_bias = make_attention(qkv_bias=True)
    without_bias = make_attention(qkv_bias=False)

    assert with_bias.qkv.bias is not None
    assert without_bias.qkv.bias is None


# ---------------------------------------------------------------------
# Forward without RoPE
# ---------------------------------------------------------------------

def test_forward_without_rope_shape_and_return_attention_contract():
    torch.manual_seed(0)

    attn = make_attention(dim=16, num_heads=4, use_rope=False).eval()
    x = torch.randn(2, 5, 16)

    out = attn(x)
    out2, weights = attn(x, return_attention=True)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 5, 16)
    assert out2.shape == (2, 5, 16)
    assert weights.shape == (2, 4, 5, 5)
    assert_close(out, out2)


def test_forward_without_rope_matches_manual_formula():
    torch.manual_seed(1)

    attn = make_attention(
        dim=16,
        num_heads=4,
        use_rope=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ).eval()

    x = torch.randn(2, 5, 16)

    expected_out, expected_weights = manual_attention_no_rope(attn, x)
    actual_out, actual_weights = attn(x, return_attention=True)

    assert_close(actual_out, expected_out)
    assert_close(actual_weights, expected_weights)


def test_attention_weights_are_probabilities_without_dropout():
    torch.manual_seed(2)

    attn = make_attention(
        dim=16,
        num_heads=4,
        use_rope=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ).eval()

    x = torch.randn(3, 7, 16)
    _, weights = attn(x, return_attention=True)

    assert weights.min().item() >= 0.0
    assert weights.max().item() <= 1.0
    assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_single_token_attention_is_exactly_one():
    torch.manual_seed(3)

    attn = make_attention(dim=16, num_heads=4, use_rope=False).eval()
    x = torch.randn(2, 1, 16)

    _, weights = attn(x, return_attention=True)

    assert weights.shape == (2, 4, 1, 1)
    assert_close(weights, torch.ones_like(weights))


def test_zero_token_sequence_preserves_empty_shapes():
    attn = make_attention(dim=16, num_heads=4, use_rope=False).eval()
    x = torch.randn(2, 0, 16)

    out, weights = attn(x, return_attention=True)

    assert out.shape == (2, 0, 16)
    assert weights.shape == (2, 4, 0, 0)


def test_wrong_embedding_dim_fails_loudly():
    attn = make_attention(dim=16, num_heads=4)
    x = torch.randn(2, 5, 15)

    with pytest.raises(RuntimeError):
        attn(x)


def test_non_3d_input_fails_loudly():
    attn = make_attention(dim=16, num_heads=4)

    with pytest.raises(ValueError):
        attn(torch.randn(2, 16))

    with pytest.raises(ValueError):
        attn(torch.randn(2, 5, 16, 1))


# ---------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------

def test_dropout_is_disabled_in_eval_mode():
    torch.manual_seed(4)

    attn = make_attention(
        dim=16,
        num_heads=4,
        attn_dropout=0.5,
        proj_dropout=0.5,
        use_rope=False,
    ).eval()

    x = torch.randn(2, 5, 16)

    y1, w1 = attn(x, return_attention=True)
    y2, w2 = attn(x, return_attention=True)

    assert_close(y1, y2)
    assert_close(w1, w2)


def test_dropout_train_mode_is_reproducible_with_seed():
    torch.manual_seed(5)

    attn = make_attention(
        dim=16,
        num_heads=4,
        attn_dropout=0.5,
        proj_dropout=0.5,
        use_rope=False,
    ).train()

    x = torch.randn(2, 5, 16)

    torch.manual_seed(999)
    y1, w1 = attn(x, return_attention=True)

    torch.manual_seed(999)
    y2, w2 = attn(x, return_attention=True)

    assert_close(y1, y2)
    assert_close(w1, w2)


def test_attention_dropout_after_softmax_can_break_row_stochasticity():
    torch.manual_seed(6)

    attn = make_attention(
        dim=16,
        num_heads=4,
        attn_dropout=0.9,
        proj_dropout=0.0,
        use_rope=False,
    ).train()

    x = torch.randn(2, 8, 16)

    torch.manual_seed(123)
    _, weights = attn(x, return_attention=True)

    row_sums = weights.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums))


# ---------------------------------------------------------------------
# RoPE integration
# ---------------------------------------------------------------------

def test_rope_forward_requires_grid_size():
    attn = make_attention(dim=32, num_heads=4, use_rope=True)
    x = torch.randn(2, 10, 32)

    with pytest.raises(ValueError, match="grid_size must be provided"):
        attn(x)


def test_rope_forward_rejects_token_grid_mismatch():
    attn = make_attention(dim=32, num_heads=4, use_rope=True)
    x = torch.randn(2, 10, 32)

    # num_special_tokens=1 + grid 2x4 = 9, not 10.
    with pytest.raises(ValueError, match="Token mismatch"):
        attn(x, grid_size=(2, 4), num_special_tokens=1)


def test_rope_forward_valid_shape_and_attention_contract():
    torch.manual_seed(7)

    attn = make_attention(dim=32, num_heads=4, use_rope=True).eval()
    x = torch.randn(2, 7, 32)  # 1 special + 2x3 patches

    out, weights = attn(
        x,
        grid_size=(2, 3),
        num_special_tokens=1,
        return_attention=True,
    )

    assert out.shape == (2, 7, 32)
    assert weights.shape == (2, 4, 7, 7)
    assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_rope_forward_supports_multiple_special_tokens():
    torch.manual_seed(8)

    attn = make_attention(dim=32, num_heads=4, use_rope=True).eval()
    x = torch.randn(2, 9, 32)  # 3 special + 2x3 patches

    out, weights = attn(
        x,
        grid_size=(2, 3),
        num_special_tokens=3,
        return_attention=True,
    )

    assert out.shape == (2, 9, 32)
    assert weights.shape == (2, 4, 9, 9)


def test_rope_receives_only_patch_tokens_with_mock_module():
    torch.manual_seed(9)

    attn = make_attention(dim=32, num_heads=4, use_rope=True).eval()

    calls = {}

    class FakeRope(nn.Module):
        def forward(self, q, k, grid_size):
            calls["q_shape"] = tuple(q.shape)
            calls["k_shape"] = tuple(k.shape)
            calls["grid_size"] = grid_size
            return q + 10.0, k - 10.0

    attn_identity = make_attention(dim=32, num_heads=4, use_rope=True).eval()
    attn_identity.load_state_dict(attn.state_dict())

    class IdentityRope(nn.Module):
        def forward(self, q, k, grid_size):
            return q, k

    attn.rope = FakeRope()
    attn_identity.rope = IdentityRope()

    x = torch.randn(2, 9, 32)  # 3 special + 6 patches

    y_fake = attn(x, grid_size=(2, 3), num_special_tokens=3)
    y_identity = attn_identity(x, grid_size=(2, 3), num_special_tokens=3)

    assert calls["q_shape"] == (2, 4, 6, 8)
    assert calls["k_shape"] == (2, 4, 6, 8)
    assert calls["grid_size"] == (2, 3)
    assert not torch.allclose(y_fake, y_identity)


def test_rope_zero_patch_tokens_edge_case():
    torch.manual_seed(10)

    attn = make_attention(dim=32, num_heads=4, use_rope=True).eval()
    x = torch.randn(2, 3, 32)  # 3 special, 0 patches

    out, weights = attn(
        x,
        grid_size=(0, 4),
        num_special_tokens=3,
        return_attention=True,
    )

    assert out.shape == (2, 3, 32)
    assert weights.shape == (2, 4, 3, 3)


def test_rope_generally_changes_output_relative_to_no_rope():
    torch.manual_seed(11)

    x = torch.randn(2, 7, 32)  # 1 special + 6 patches

    attn_no_rope = make_attention(dim=32, num_heads=4, use_rope=False).eval()
    attn_rope = make_attention(dim=32, num_heads=4, use_rope=True).eval()

    # Copy compatible qkv/proj parameters only.
    no_rope_state = attn_no_rope.state_dict()
    rope_state = attn_rope.state_dict()
    compatible_state = {
        key: value
        for key, value in no_rope_state.items()
        if key in rope_state and rope_state[key].shape == value.shape
    }
    attn_rope.load_state_dict(compatible_state, strict=False)

    y_no_rope = attn_no_rope(x)
    y_rope = attn_rope(x, grid_size=(2, 3), num_special_tokens=1)

    assert y_no_rope.shape == y_rope.shape
    assert not torch.allclose(y_no_rope, y_rope)


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------

def test_backward_without_rope_populates_input_and_parameter_gradients():
    torch.manual_seed(12)

    attn = make_attention(dim=16, num_heads=4, use_rope=False)
    x = torch.randn(2, 5, 16, requires_grad=True)

    out = attn(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0

    for name, param in attn.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_backward_with_rope_populates_input_and_parameter_gradients():
    torch.manual_seed(13)

    attn = make_attention(dim=32, num_heads=4, use_rope=True)
    x = torch.randn(2, 7, 32, requires_grad=True)

    out = attn(x, grid_size=(2, 3), num_special_tokens=1)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0

    for name, param in attn.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_no_grad_forward_returns_non_grad_tensors():
    torch.manual_seed(14)

    attn = make_attention(dim=16, num_heads=4, use_rope=False)
    x = torch.randn(2, 5, 16, requires_grad=True)

    with torch.no_grad():
        out, weights = attn(x, return_attention=True)

    assert out.requires_grad is False
    assert weights.requires_grad is False


# ---------------------------------------------------------------------
# Dtype and device
# ---------------------------------------------------------------------

def test_forward_preserves_float64_when_module_is_double_without_rope():
    torch.manual_seed(15)

    attn = make_attention(dim=16, num_heads=4, use_rope=False).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64)

    out, weights = attn(x, return_attention=True)

    assert out.dtype == torch.float64
    assert weights.dtype == torch.float64


def test_forward_preserves_float64_when_module_is_double_with_rope():
    torch.manual_seed(16)

    attn = make_attention(dim=32, num_heads=4, use_rope=True).double()
    x = torch.randn(2, 7, 32, dtype=torch.float64)

    out, weights = attn(
        x,
        grid_size=(2, 3),
        num_special_tokens=1,
        return_attention=True,
    )

    assert out.dtype == torch.float64
    assert weights.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_consistency_without_rope():
    torch.manual_seed(17)

    attn_cpu = make_attention(dim=16, num_heads=4, use_rope=False).eval()
    attn_cuda = make_attention(dim=16, num_heads=4, use_rope=False).eval().cuda()
    attn_cuda.load_state_dict(attn_cpu.state_dict())

    x_cpu = torch.randn(2, 5, 16)
    x_cuda = x_cpu.cuda()

    with torch.no_grad():
        out_cpu, w_cpu = attn_cpu(x_cpu, return_attention=True)
        out_cuda, w_cuda = attn_cuda(x_cuda, return_attention=True)

    assert_close(out_cpu, out_cuda.cpu(), atol=1e-5, rtol=1e-5)
    assert_close(w_cpu, w_cuda.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_consistency_with_rope():
    torch.manual_seed(18)

    attn_cpu = make_attention(dim=32, num_heads=4, use_rope=True).eval()
    attn_cuda = make_attention(dim=32, num_heads=4, use_rope=True).eval().cuda()
    attn_cuda.load_state_dict(attn_cpu.state_dict())

    x_cpu = torch.randn(2, 7, 32)
    x_cuda = x_cpu.cuda()

    with torch.no_grad():
        out_cpu, w_cpu = attn_cpu(
            x_cpu,
            grid_size=(2, 3),
            num_special_tokens=1,
            return_attention=True,
        )
        out_cuda, w_cuda = attn_cuda(
            x_cuda,
            grid_size=(2, 3),
            num_special_tokens=1,
            return_attention=True,
        )

    assert_close(out_cpu, out_cuda.cpu(), atol=1e-5, rtol=1e-5)
    assert_close(w_cpu, w_cuda.cpu(), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------
# Serialization and numerical sanity
# ---------------------------------------------------------------------

def test_state_dict_roundtrip_preserves_outputs_without_rope():
    torch.manual_seed(19)

    attn_a = make_attention(dim=16, num_heads=4, use_rope=False).eval()
    attn_b = make_attention(dim=16, num_heads=4, use_rope=False).eval()

    x = torch.randn(2, 5, 16)

    attn_b.load_state_dict(attn_a.state_dict())

    y_a, w_a = attn_a(x, return_attention=True)
    y_b, w_b = attn_b(x, return_attention=True)

    assert_close(y_a, y_b)
    assert_close(w_a, w_b)


def test_state_dict_roundtrip_preserves_outputs_with_rope():
    torch.manual_seed(20)

    attn_a = make_attention(dim=32, num_heads=4, use_rope=True).eval()
    attn_b = make_attention(dim=32, num_heads=4, use_rope=True).eval()

    x = torch.randn(2, 7, 32)

    attn_b.load_state_dict(attn_a.state_dict())

    y_a, w_a = attn_a(
        x,
        grid_size=(2, 3),
        num_special_tokens=1,
        return_attention=True,
    )
    y_b, w_b = attn_b(
        x,
        grid_size=(2, 3),
        num_special_tokens=1,
        return_attention=True,
    )

    assert_close(y_a, y_b)
    assert_close(w_a, w_b)


def test_large_token_count_no_nan_or_inf():
    torch.manual_seed(21)

    attn = make_attention(dim=64, num_heads=8, use_rope=True).eval()

    # 1 special + 8x8 patches = 65 tokens.
    x = torch.randn(2, 65, 64)

    out, weights = attn(
        x,
        grid_size=(8, 8),
        num_special_tokens=1,
        return_attention=True,
    )

    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()
    assert_close(
        weights.sum(dim=-1),
        torch.ones_like(weights.sum(dim=-1)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_without_positional_encoding_attention_is_permutation_equivariant():
    """
    Without RoPE or learned positional encodings, self-attention should be
    permutation equivariant: permuting tokens permutes outputs in the same way.
    """
    torch.manual_seed(22)

    attn = make_attention(dim=16, num_heads=4, use_rope=False).eval()
    x = torch.randn(2, 5, 16)

    perm = torch.tensor([2, 0, 4, 1, 3])
    inv_perm = torch.argsort(perm)

    y = attn(x)
    y_perm = attn(x[:, perm, :])

    assert_close(y_perm[:, inv_perm, :], y)
