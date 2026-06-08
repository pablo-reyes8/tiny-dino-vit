
"""
Advanced pytest suite for src.model.attention.py

Run:
    pytest -q test_model_attention.py

Assumed module path:
    src.model.attention

Covers:
    - MultiHeadSelfAttention initialization contracts
    - exact manual attention equivalence without RoPE/dropout
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
"""

import importlib
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


MODULE_PATH = "src.model.attention"


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
    Independent reference implementation for this module when use_rope=False.
    """
    b, t, d = x.shape
    qkv = module.qkv(x)
    qkv = qkv.reshape(b, t, 3, module.num_heads, module.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(dim=0)

    attn_logits = (q @ k.transpose(-2, -1)) * module.scale
    attn = attn_logits.softmax(dim=-1)
    attn_after_drop = module.attn_drop(attn)

    out = attn_after_drop @ v
    out = out.transpose(1, 2).reshape(b, t, d)
    out = module.proj(out)
    out = module.proj_drop(out)

    return out, attn_after_drop


# ---------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------

def test_attention_initializes_core_attributes_without_rope():
    attn = make_attention(dim=32, num_heads=4, use_rope=False)

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


def test_attention_with_rope_requires_head_dim_divisible_by_4():
    """
    dim % num_heads can be valid while head_dim % 4 is invalid.
    RoPE should reject such configurations through RotaryEmbedding2D.
    """
    with pytest.raises(ValueError, match="divisible by 4"):
        make_attention(dim=24, num_heads=4, use_rope=True)  # head_dim=6


def test_qkv_bias_flag_is_respected():
    attn_with_bias = make_attention(qkv_bias=True)
    attn_without_bias = make_attention(qkv_bias=False)

    assert attn_with_bias.qkv.bias is not None
    assert attn_without_bias.qkv.bias is None


# ---------------------------------------------------------------------
# Forward tests without RoPE
# ---------------------------------------------------------------------

def test_forward_without_rope_shape_and_return_attention_contract():
    torch.manual_seed(0)

    attn = make_attention(dim=16, num_heads=4, use_rope=False)
    x = torch.randn(2, 5, 16)

    out = attn(x)
    out_with_attn, weights = attn(x, return_attention=True)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 5, 16)

    assert out_with_attn.shape == (2, 5, 16)
    assert weights.shape == (2, 4, 5, 5)

    assert_close(out, out_with_attn)


def test_forward_without_rope_matches_manual_attention_formula():
    torch.manual_seed(1)

    attn = make_attention(
        dim=16,
        num_heads=4,
        use_rope=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ).eval()

    x = torch.randn(2, 5, 16)

    expected_out, expected_attn = manual_attention_no_rope(attn, x)
    actual_out, actual_attn = attn(x, return_attention=True)

    assert_close(actual_out, expected_out)
    assert_close(actual_attn, expected_attn)


def test_attention_probabilities_are_row_stochastic_without_dropout():
    torch.manual_seed(2)

    attn = make_attention(
        dim=16,
        num_heads=4,
        attn_dropout=0.0,
        proj_dropout=0.0,
        use_rope=False,
    ).eval()

    x = torch.randn(3, 7, 16)
    _, weights = attn(x, return_attention=True)

    assert weights.min().item() >= 0.0
    assert weights.max().item() <= 1.0
    assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_single_token_attention_is_one_without_dropout():
    torch.manual_seed(3)

    attn = make_attention(dim=16, num_heads=4, use_rope=False).eval()
    x = torch.randn(2, 1, 16)

    _, weights = attn(x, return_attention=True)

    assert weights.shape == (2, 4, 1, 1)
    assert_close(weights, torch.ones_like(weights))


def test_forward_accepts_zero_tokens_without_crashing():
    """
    Edge case: not meaningful for real ViT, but shape plumbing should remain valid.
    """
    attn = make_attention(dim=16, num_heads=4, use_rope=False).eval()
    x = torch.randn(2, 0, 16)

    out, weights = attn(x, return_attention=True)

    assert out.shape == (2, 0, 16)
    assert weights.shape == (2, 4, 0, 0)


def test_forward_wrong_embedding_dim_fails_loudly():
    attn = make_attention(dim=16, num_heads=4)
    x = torch.randn(2, 5, 15)

    with pytest.raises(RuntimeError):
        attn(x)


def test_forward_non_3d_input_fails_loudly():
    """
    Current implementation destructures x.shape directly, so non-3D tensors
    raise ValueError from Python unpacking.
    """
    attn = make_attention(dim=16, num_heads=4)

    with pytest.raises(ValueError):
        attn(torch.randn(2, 16))

    with pytest.raises(ValueError):
        attn(torch.randn(2, 5, 16, 1))


# ---------------------------------------------------------------------
# Dropout behavior
# ---------------------------------------------------------------------

def test_dropout_disabled_in_eval_mode_is_deterministic():
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


def test_dropout_train_mode_reproducible_with_same_seed():
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


def test_attention_dropout_breaks_row_stochasticity_in_train_mode_when_active():
    """
    Dropout is applied after softmax. Therefore returned attention weights are
    post-dropout weights and need not sum to 1 in train mode.
    """
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
# RoPE behavior
# ---------------------------------------------------------------------

def test_forward_with_rope_requires_grid_size():
    attn = make_attention(dim=32, num_heads=4, use_rope=True)
    x = torch.randn(2, 10, 32)

    with pytest.raises(ValueError, match="grid_size must be provided"):
        attn(x, grid_size=None)


def test_forward_with_rope_rejects_token_grid_mismatch():
    attn = make_attention(dim=32, num_heads=4, use_rope=True)
    x = torch.randn(2, 10, 32)

    # num_special_tokens=1, grid=(2, 4) implies 9 tokens, not 10.
    with pytest.raises(ValueError, match="Token mismatch"):
        attn(x, grid_size=(2, 4), num_special_tokens=1)


def test_forward_with_rope_valid_shape_and_attention_contract():
    torch.manual_seed(7)

    attn = make_attention(dim=32, num_heads=4, use_rope=True).eval()

    # 1 CLS token + 2x3 patch tokens = 7 tokens.
    x = torch.randn(2, 7, 32)

    out, weights = attn(
        x,
        grid_size=(2, 3),
        num_special_tokens=1,
        return_attention=True,
    )

    assert out.shape == (2, 7, 32)
    assert weights.shape == (2, 4, 7, 7)
    assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_forward_with_rope_supports_multiple_special_tokens():
    torch.manual_seed(8)

    attn = make_attention(dim=32, num_heads=4, use_rope=True).eval()

    # 3 special tokens + 2x3 patch tokens = 9 tokens.
    x = torch.randn(2, 9, 32)

    out, weights = attn(
        x,
        grid_size=(2, 3),
        num_special_tokens=3,
        return_attention=True,
    )

    assert out.shape == (2, 9, 32)
    assert weights.shape == (2, 4, 9, 9)


def test_rope_is_applied_only_to_patch_tokens_by_mocking_rope():
    """
    We replace `attn.rope` with a mock module that asserts the token count it
    receives equals only the patch token count, not special + patch tokens.

    It also adds a constant to q_patch and subtracts a constant from k_patch,
    proving that forward uses the returned q/k tensors.
    """
    torch.manual_seed(9)

    attn = make_attention(dim=32, num_heads=4, use_rope=True).eval()

    calls = {}

    class FakeRope(nn.Module):
        def forward(self, q, k, grid_size):
            calls["q_shape"] = tuple(q.shape)
            calls["k_shape"] = tuple(k.shape)
            calls["grid_size"] = grid_size
            return q + 10.0, k - 10.0

    x = torch.randn(2, 9, 32)  # 3 special + 6 patches

    # Compare against a version where fake rope returns identity.
    attn_identity = make_attention(dim=32, num_heads=4, use_rope=True).eval()
    attn_identity.load_state_dict(attn.state_dict())

    class IdentityRope(nn.Module):
        def forward(self, q, k, grid_size):
            return q, k

    attn.rope = FakeRope()
    attn_identity.rope = IdentityRope()

    y_fake = attn(x, grid_size=(2, 3), num_special_tokens=3)
    y_identity = attn_identity(x, grid_size=(2, 3), num_special_tokens=3)

    assert calls["q_shape"] == (2, 4, 6, 8)
    assert calls["k_shape"] == (2, 4, 6, 8)
    assert calls["grid_size"] == (2, 3)

    assert not torch.allclose(y_fake, y_identity), (
        "Changing q_patch/k_patch through RoPE should affect the final output."
    )


def test_special_tokens_are_not_rotated_when_only_special_tokens_exist():
    """
    Edge case: grid_size=(0, N) implies zero patch tokens. The module should
    pass an empty q_patch/k_patch to RoPE and preserve shape.
    """
    torch.manual_seed(10)

    attn = make_attention(dim=32, num_heads=4, use_rope=True).eval()
    x = torch.randn(2, 3, 32)  # three special tokens, zero patches

    out, weights = attn(
        x,
        grid_size=(0, 4),
        num_special_tokens=3,
        return_attention=True,
    )

    assert out.shape == (2, 3, 32)
    assert weights.shape == (2, 4, 3, 3)


def test_rope_changes_output_for_nonzero_patch_positions_but_not_due_to_shape_error():
    torch.manual_seed(11)

    x = torch.randn(2, 7, 32)  # 1 special + 6 patches

    attn_no_rope = make_attention(dim=32, num_heads=4, use_rope=False).eval()
    attn_rope = make_attention(dim=32, num_heads=4, use_rope=True).eval()

    # qkv/proj parameters are shared; only RoPE differs.
    state = attn_no_rope.state_dict()
    compatible_state = {
        k: v for k, v in state.items()
        if k in attn_rope.state_dict() and attn_rope.state_dict()[k].shape == v.shape
    }
    attn_rope.load_state_dict(compatible_state, strict=False)

    y_no_rope = attn_no_rope(x)
    y_rope = attn_rope(x, grid_size=(2, 3), num_special_tokens=1)

    assert y_no_rope.shape == y_rope.shape
    assert not torch.allclose(y_no_rope, y_rope), (
        "With nonzero spatial positions, RoPE should generally alter attention output."
    )


# ---------------------------------------------------------------------
# Gradient tests
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

    trainable_params = dict(attn.named_parameters())
    assert "qkv.weight" in trainable_params
    assert "proj.weight" in trainable_params

    for name, param in trainable_params.items():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_no_grad_forward_returns_non_grad_output_and_attention():
    torch.manual_seed(14)

    attn = make_attention(dim=16, num_heads=4, use_rope=False)
    x = torch.randn(2, 5, 16, requires_grad=True)

    with torch.no_grad():
        out, weights = attn(x, return_attention=True)

    assert out.requires_grad is False
    assert weights.requires_grad is False


# ---------------------------------------------------------------------
# Dtype/device tests
# ---------------------------------------------------------------------

def test_forward_preserves_float64_when_module_is_double():
    torch.manual_seed(15)

    attn = make_attention(dim=16, num_heads=4, use_rope=False).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64)

    out, weights = attn(x, return_attention=True)

    assert out.dtype == torch.float64
    assert weights.dtype == torch.float64


def test_forward_with_rope_preserves_float64_when_module_is_double():
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
    assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6, rtol=1e-6)


def test_attention_output_changes_when_input_token_order_changes():
    """
    Self-attention is permutation equivariant without positional encodings.
    With no RoPE and no other positional signal, permuting tokens should permute
    outputs accordingly.
    """
    torch.manual_seed(22)

    attn = make_attention(dim=16, num_heads=4, use_rope=False).eval()
    x = torch.randn(2, 5, 16)

    perm = torch.tensor([2, 0, 4, 1, 3])
    inv_perm = torch.argsort(perm)

    y = attn(x)
    y_perm = attn(x[:, perm, :])

    assert_close(y_perm[:, inv_perm, :], y)
