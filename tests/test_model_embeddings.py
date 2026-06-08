
"""
Advanced pytest suite for src.model.embeddings.py

Run:
    pytest -q test_model_embeddings.py

Assumed module path:
    src.model.embeddings

Covers:
    - to_2tuple
    - PatchEmbeddingOutput dataclass contract
    - PatchEmbedding forward contracts
    - exact Conv2d equivalence
    - flatten=True / flatten=False behavior
    - strict and dynamic image sizes
    - norm_layer behavior
    - gradient flow
    - invalid input handling
    - CPU/CUDA consistency when CUDA is available
"""

import importlib
from dataclasses import is_dataclass

import pytest
import torch
import torch.nn as nn


MODULE_PATH = "src.model.embeddings"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_module():
    return importlib.import_module(MODULE_PATH)


def assert_close(a, b, atol=1e-6, rtol=1e-5):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"Max abs diff: {(a - b).abs().max().item()}"
    )


def make_patch_embedding(
    *,
    img_size=8,
    patch_size=4,
    in_chans=3,
    embed_dim=5,
    norm_layer=None,
    flatten=True,
    strict_img_size=False,
    bias=True,
):
    m = _load_module()
    return m.PatchEmbedding(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
        flatten=flatten,
        strict_img_size=strict_img_size,
        bias=bias,
    )


# ---------------------------------------------------------------------
# to_2tuple tests
# ---------------------------------------------------------------------

def test_to_2tuple_accepts_int_and_tuple():
    m = _load_module()

    assert m.to_2tuple(8) == (8, 8)
    assert m.to_2tuple((8, 4)) == (8, 4)
    assert m.to_2tuple((1, 999)) == (1, 999)


@pytest.mark.parametrize(
    "bad_value",
    [
        [8, 4],
        (8,),
        (8, 4, 2),
        "8",
        8.0,
        None,
        {"h": 8, "w": 4},
    ],
)
def test_to_2tuple_rejects_invalid_inputs(bad_value):
    m = _load_module()

    with pytest.raises(ValueError, match="Expected int or tuple"):
        m.to_2tuple(bad_value)


# ---------------------------------------------------------------------
# Dataclass contract
# ---------------------------------------------------------------------

def test_patch_embedding_output_is_dataclass_with_expected_fields():
    m = _load_module()

    assert is_dataclass(m.PatchEmbeddingOutput)

    tokens = torch.randn(2, 4, 5)
    out = m.PatchEmbeddingOutput(
        tokens=tokens,
        grid_size=(2, 2),
        num_patches=4,
        patch_size=(4, 4),
    )

    assert out.tokens is tokens
    assert out.grid_size == (2, 2)
    assert out.num_patches == 4
    assert out.patch_size == (4, 4)


# ---------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------

def test_patch_embedding_initializes_nominal_grid_and_num_patches():
    pe = make_patch_embedding(
        img_size=(16, 12),
        patch_size=(4, 3),
        in_chans=3,
        embed_dim=7,
    )

    assert pe.img_size == (16, 12)
    assert pe.patch_size == (4, 3)
    assert pe.grid_size == (4, 4)
    assert pe.num_patches == 16
    assert pe.embed_dim == 7
    assert pe.in_chans == 3

    assert isinstance(pe.proj, nn.Conv2d)
    assert pe.proj.kernel_size == (4, 3)
    assert pe.proj.stride == (4, 3)
    assert pe.proj.in_channels == 3
    assert pe.proj.out_channels == 7


def test_patch_embedding_rejects_non_divisible_reference_img_size():
    with pytest.raises(ValueError, match="must be divisible"):
        make_patch_embedding(img_size=(15, 16), patch_size=(4, 4))


def test_patch_embedding_bias_flag_is_respected():
    pe_with_bias = make_patch_embedding(bias=True)
    pe_without_bias = make_patch_embedding(bias=False)

    assert pe_with_bias.proj.bias is not None
    assert pe_without_bias.proj.bias is None


def test_patch_embedding_norm_identity_when_norm_layer_none():
    pe = make_patch_embedding(norm_layer=None)
    assert isinstance(pe.norm, nn.Identity)


def test_patch_embedding_norm_layer_is_instantiated_with_embed_dim():
    pe = make_patch_embedding(embed_dim=11, norm_layer=nn.LayerNorm)
    assert isinstance(pe.norm, nn.LayerNorm)
    assert pe.norm.normalized_shape == (11,)


# ---------------------------------------------------------------------
# Forward tests: flatten=True
# ---------------------------------------------------------------------

def test_forward_flatten_true_returns_metadata_and_expected_shape():
    pe = make_patch_embedding(
        img_size=(8, 12),
        patch_size=(4, 3),
        in_chans=3,
        embed_dim=5,
        flatten=True,
        strict_img_size=True,
    )

    x = torch.randn(2, 3, 8, 12)
    out = pe(x, return_metadata=True)

    assert out.tokens.shape == (2, 8, 5)  # Gh=2, Gw=4, N=8
    assert out.grid_size == (2, 4)
    assert out.num_patches == 8
    assert out.patch_size == (4, 3)
    assert out.tokens.dtype == x.dtype
    assert torch.isfinite(out.tokens).all()


def test_forward_return_metadata_false_returns_raw_tensor_only():
    pe = make_patch_embedding(
        img_size=8,
        patch_size=4,
        embed_dim=5,
        flatten=True,
    )

    x = torch.randn(2, 3, 8, 8)
    y = pe(x, return_metadata=False)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 4, 5)


def test_forward_flatten_true_matches_manual_conv_flatten_transpose_without_norm():
    """
    Critical equivalence test:
        PatchEmbedding(flatten=True, norm=None)
        must equal:
        Conv2d -> flatten spatial -> transpose to [B, N, D]
    """
    torch.manual_seed(123)

    pe = make_patch_embedding(
        img_size=(8, 12),
        patch_size=(4, 3),
        in_chans=3,
        embed_dim=5,
        flatten=True,
        norm_layer=None,
        bias=True,
    )

    x = torch.randn(2, 3, 8, 12)

    out = pe(x, return_metadata=True).tokens
    expected = pe.proj(x).flatten(2).transpose(1, 2)

    assert_close(out, expected)


def test_forward_flatten_true_patch_order_is_row_major():
    """
    Uses a 1-channel image with patch_size=2 and a conv kernel of ones.
    Each token becomes the sum of its patch. This allows us to verify that
    flattening preserves row-major spatial order:
        top-left, top-right, bottom-left, bottom-right.
    """
    pe = make_patch_embedding(
        img_size=(4, 4),
        patch_size=(2, 2),
        in_chans=1,
        embed_dim=1,
        flatten=True,
        norm_layer=None,
        bias=False,
    )

    with torch.no_grad():
        pe.proj.weight.fill_(1.0)

    x = torch.tensor(
        [[
            [
                [1.0, 2.0, 10.0, 20.0],
                [3.0, 4.0, 30.0, 40.0],
                [100.0, 200.0, 1000.0, 2000.0],
                [300.0, 400.0, 3000.0, 4000.0],
            ]
        ]]
    )

    tokens = pe(x, return_metadata=False).squeeze(0).squeeze(-1)

    expected = torch.tensor([
        1.0 + 2.0 + 3.0 + 4.0,
        10.0 + 20.0 + 30.0 + 40.0,
        100.0 + 200.0 + 300.0 + 400.0,
        1000.0 + 2000.0 + 3000.0 + 4000.0,
    ])

    assert_close(tokens, expected)


def test_forward_flatten_true_layernorm_normalizes_last_dimension_per_token():
    torch.manual_seed(0)

    pe = make_patch_embedding(
        img_size=8,
        patch_size=4,
        in_chans=3,
        embed_dim=6,
        flatten=True,
        norm_layer=nn.LayerNorm,
        bias=True,
    )

    x = torch.randn(4, 3, 8, 8)
    tokens = pe(x, return_metadata=True).tokens

    # LayerNorm should produce approximately zero mean and unit variance
    # over the embedding dimension for each token.
    per_token_mean = tokens.mean(dim=-1)
    per_token_var = tokens.var(dim=-1, unbiased=False)

    assert torch.allclose(per_token_mean, torch.zeros_like(per_token_mean), atol=1e-5)
    assert torch.allclose(per_token_var, torch.ones_like(per_token_var), atol=2e-4)


# ---------------------------------------------------------------------
# Forward tests: flatten=False
# ---------------------------------------------------------------------

def test_forward_flatten_false_returns_spatial_feature_map_and_metadata():
    pe = make_patch_embedding(
        img_size=(8, 12),
        patch_size=(4, 3),
        in_chans=3,
        embed_dim=5,
        flatten=False,
    )

    x = torch.randn(2, 3, 8, 12)
    out = pe(x, return_metadata=True)

    assert out.tokens.shape == (2, 5, 2, 4)
    assert out.grid_size == (2, 4)
    assert out.num_patches == 8
    assert out.patch_size == (4, 3)


def test_forward_flatten_false_matches_raw_conv2d_output():
    torch.manual_seed(555)

    pe = make_patch_embedding(
        img_size=8,
        patch_size=4,
        in_chans=3,
        embed_dim=5,
        flatten=False,
        norm_layer=None,
        bias=True,
    )

    x = torch.randn(2, 3, 8, 8)

    out = pe(x, return_metadata=False)
    expected = pe.proj(x)

    assert out.shape == (2, 5, 2, 2)
    assert_close(out, expected)


def test_forward_flatten_false_does_not_apply_norm_layer():
    """
    Important behavior of the current implementation:
    norm is only applied inside `if self.flatten`.
    This test documents/protects that behavior.
    """
    torch.manual_seed(11)

    pe = make_patch_embedding(
        img_size=8,
        patch_size=4,
        in_chans=3,
        embed_dim=5,
        flatten=False,
        norm_layer=nn.LayerNorm,
        bias=True,
    )

    x = torch.randn(2, 3, 8, 8)
    y = pe(x, return_metadata=False)

    expected = pe.proj(x)
    assert_close(y, expected)


# ---------------------------------------------------------------------
# Dynamic multi-crop behavior
# ---------------------------------------------------------------------

def test_forward_allows_dynamic_divisible_sizes_when_not_strict():
    pe = make_patch_embedding(
        img_size=64,
        patch_size=8,
        in_chans=3,
        embed_dim=16,
        flatten=True,
        strict_img_size=False,
    )

    x_64 = torch.randn(2, 3, 64, 64)
    x_32 = torch.randn(2, 3, 32, 32)
    x_rect = torch.randn(2, 3, 32, 48)

    out_64 = pe(x_64)
    out_32 = pe(x_32)
    out_rect = pe(x_rect)

    assert out_64.grid_size == (8, 8)
    assert out_64.num_patches == 64
    assert out_64.tokens.shape == (2, 64, 16)

    assert out_32.grid_size == (4, 4)
    assert out_32.num_patches == 16
    assert out_32.tokens.shape == (2, 16, 16)

    assert out_rect.grid_size == (4, 6)
    assert out_rect.num_patches == 24
    assert out_rect.tokens.shape == (2, 24, 16)


def test_forward_strict_img_size_rejects_dynamic_crop_size():
    pe = make_patch_embedding(
        img_size=64,
        patch_size=8,
        strict_img_size=True,
    )

    x = torch.randn(2, 3, 32, 32)

    with pytest.raises(ValueError, match="does not match"):
        pe(x)


def test_forward_rejects_non_divisible_runtime_size_even_when_not_strict():
    pe = make_patch_embedding(
        img_size=64,
        patch_size=8,
        strict_img_size=False,
    )

    x = torch.randn(2, 3, 62, 64)

    with pytest.raises(ValueError, match="must be divisible"):
        pe(x)


# ---------------------------------------------------------------------
# Invalid input handling
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_shape",
    [
        (3, 64, 64),        # missing batch
        (2, 3, 64),         # missing width
        (2, 3, 64, 64, 1),  # extra dim
        (64,),              # completely wrong
    ],
)
def test_forward_rejects_non_4d_inputs(bad_shape):
    pe = make_patch_embedding()

    x = torch.randn(*bad_shape)

    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        pe(x)


def test_forward_rejects_wrong_number_of_channels():
    pe = make_patch_embedding(in_chans=3)

    x = torch.randn(2, 1, 64, 64)

    with pytest.raises(ValueError, match="input channels"):
        pe(x)


def test_forward_accepts_batch_size_one_and_zero_batch():
    """
    Batch size zero can appear in edge cases during distributed/eval plumbing.
    The module should preserve a valid empty batch instead of crashing.
    """
    pe = make_patch_embedding(
        img_size=8,
        patch_size=4,
        in_chans=3,
        embed_dim=5,
        flatten=True,
    )

    x_one = torch.randn(1, 3, 8, 8)
    x_zero = torch.randn(0, 3, 8, 8)

    y_one = pe(x_one, return_metadata=False)
    y_zero = pe(x_zero, return_metadata=False)

    assert y_one.shape == (1, 4, 5)
    assert y_zero.shape == (0, 4, 5)


# ---------------------------------------------------------------------
# Gradient and training behavior
# ---------------------------------------------------------------------

def test_forward_backward_populates_gradients_for_input_weight_and_bias():
    torch.manual_seed(9)

    pe = make_patch_embedding(
        img_size=8,
        patch_size=4,
        in_chans=3,
        embed_dim=5,
        flatten=True,
        bias=True,
    )

    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    tokens = pe(x, return_metadata=False)

    loss = tokens.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert pe.proj.weight.grad is not None
    assert pe.proj.bias.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(pe.proj.weight.grad).all()
    assert torch.isfinite(pe.proj.bias.grad).all()

    assert pe.proj.weight.grad.abs().sum().item() > 0


def test_no_grad_forward_returns_non_grad_tensor():
    pe = make_patch_embedding(
        img_size=8,
        patch_size=4,
        in_chans=3,
        embed_dim=5,
        flatten=True,
    )

    x = torch.randn(2, 3, 8, 8, requires_grad=True)

    with torch.no_grad():
        y = pe(x, return_metadata=False)

    assert y.requires_grad is False


def test_eval_and_train_mode_have_same_output_without_stochastic_layers():
    """
    PatchEmbedding has Conv2d + optional deterministic norm only.
    For LayerNorm, train/eval should be identical.
    """
    torch.manual_seed(33)

    pe = make_patch_embedding(
        img_size=8,
        patch_size=4,
        embed_dim=5,
        norm_layer=nn.LayerNorm,
        flatten=True,
    )

    x = torch.randn(2, 3, 8, 8)

    pe.train()
    y_train = pe(x, return_metadata=False)

    pe.eval()
    y_eval = pe(x, return_metadata=False)

    assert_close(y_train, y_eval)


# ---------------------------------------------------------------------
# Dtype and device behavior
# ---------------------------------------------------------------------

def test_forward_preserves_float64_when_module_is_double():
    pe = make_patch_embedding(
        img_size=8,
        patch_size=4,
        embed_dim=5,
        flatten=True,
    ).double()

    x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    y = pe(x, return_metadata=False)

    assert y.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_consistency_for_same_weights():
    torch.manual_seed(101)

    pe_cpu = make_patch_embedding(
        img_size=8,
        patch_size=4,
        embed_dim=5,
        flatten=True,
        norm_layer=nn.LayerNorm,
    ).eval()

    pe_cuda = make_patch_embedding(
        img_size=8,
        patch_size=4,
        embed_dim=5,
        flatten=True,
        norm_layer=nn.LayerNorm,
    ).eval().cuda()

    pe_cuda.load_state_dict(pe_cpu.state_dict())

    x_cpu = torch.randn(2, 3, 8, 8)
    x_cuda = x_cpu.cuda()

    with torch.no_grad():
        y_cpu = pe_cpu(x_cpu, return_metadata=False)
        y_cuda = pe_cuda(x_cuda, return_metadata=False).cpu()

    assert_close(y_cpu, y_cuda, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------
# State dict and serialization sanity
# ---------------------------------------------------------------------

def test_state_dict_roundtrip_preserves_outputs():
    torch.manual_seed(202)

    pe_a = make_patch_embedding(
        img_size=(8, 12),
        patch_size=(4, 3),
        embed_dim=5,
        flatten=True,
        norm_layer=nn.LayerNorm,
    )

    pe_b = make_patch_embedding(
        img_size=(8, 12),
        patch_size=(4, 3),
        embed_dim=5,
        flatten=True,
        norm_layer=nn.LayerNorm,
    )

    x = torch.randn(2, 3, 8, 12)

    pe_b.load_state_dict(pe_a.state_dict())

    y_a = pe_a(x, return_metadata=False)
    y_b = pe_b(x, return_metadata=False)

    assert_close(y_a, y_b)


def test_metadata_matches_runtime_output_not_nominal_img_size():
    """
    This is crucial for DINO multi-crop:
    metadata should describe the actual crop passed to forward,
    not only the nominal img_size configured in __init__.
    """
    pe = make_patch_embedding(
        img_size=64,
        patch_size=8,
        embed_dim=5,
        strict_img_size=False,
        flatten=True,
    )

    x = torch.randn(2, 3, 40, 56)
    out = pe(x, return_metadata=True)

    assert pe.grid_size == (8, 8)      # nominal
    assert pe.num_patches == 64        # nominal

    assert out.grid_size == (5, 7)     # runtime
    assert out.num_patches == 35       # runtime
    assert out.tokens.shape == (2, 35, 5)
