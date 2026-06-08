
"""
Advanced pytest suite for src.model.transformer_block.py

Run:
    pytest -q test_model_transformer_block.py

Assumed module path:
    src.model.transformer_block

Covers:
    - TransformerEncoderBlock initialization contract
    - PreNorm ordering with hooks
    - residual connections
    - return_attention behavior
    - RoPE argument propagation through attention
    - exact identity cases when attention/MLP branches are zeroed
    - DropPath train/eval behavior
    - gradient flow
    - dtype/device behavior
    - state_dict roundtrip
    - numerical sanity
"""

import importlib
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


MODULE_PATH = "src.model.transformer_block"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_module():
    return importlib.import_module(MODULE_PATH)


def assert_close(a, b, atol=1e-6, rtol=1e-5):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"Max abs diff: {(a - b).abs().max().item()}"
    )


def make_block(
    *,
    dim=32,
    num_heads=4,
    mlp_ratio=4.0,
    qkv_bias=True,
    attn_dropout=0.0,
    proj_dropout=0.0,
    mlp_dropout=0.0,
    drop_path=0.0,
    use_rope=False,
    rope_base=10000.0,
    norm_layer=nn.LayerNorm,
):
    m = _load_module()
    return m.TransformerEncoderBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        attn_dropout=attn_dropout,
        proj_dropout=proj_dropout,
        mlp_dropout=mlp_dropout,
        drop_path=drop_path,
        use_rope=use_rope,
        rope_base=rope_base,
        norm_layer=norm_layer,
    )


def zero_module_parameters(module):
    for p in module.parameters():
        with torch.no_grad():
            p.zero_()


def set_attention_to_zero_output(block):
    """
    Make attention branch exactly zero after projection.
    """
    zero_module_parameters(block.attn)


def set_mlp_to_zero_output(block):
    """
    Make MLP branch exactly zero.
    """
    zero_module_parameters(block.mlp)


def get_linear_layers(module):
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

def test_block_initializes_expected_submodules():
    block = make_block(
        dim=32,
        num_heads=4,
        mlp_ratio=3.0,
        qkv_bias=True,
        use_rope=False,
        drop_path=0.1,
    )

    assert isinstance(block, nn.Module)
    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)
    assert block.norm1.normalized_shape == (32,)
    assert block.norm2.normalized_shape == (32,)

    assert hasattr(block, "attn")
    assert block.attn.dim == 32
    assert block.attn.num_heads == 4
    assert block.attn.use_rope is False

    assert hasattr(block, "mlp")
    assert hasattr(block, "drop_path1")
    assert hasattr(block, "drop_path2")


def test_block_initializes_attention_with_rope():
    block = make_block(
        dim=32,
        num_heads=4,
        use_rope=True,
        rope_base=1234.0,
    )

    assert block.attn.use_rope is True
    assert block.attn.rope is not None
    assert block.attn.rope.base == 1234.0


def test_block_rejects_invalid_attention_dim_heads():
    with pytest.raises(ValueError, match="dim must be divisible"):
        make_block(dim=30, num_heads=8)


def test_custom_norm_layer_is_used_for_both_norms():
    block = make_block(dim=32, num_heads=4, norm_layer=nn.BatchNorm1d)

    assert isinstance(block.norm1, nn.BatchNorm1d)
    assert isinstance(block.norm2, nn.BatchNorm1d)


@pytest.mark.parametrize("mlp_ratio", [0.5, 1.0, 2.0, 4.0])
def test_mlp_ratio_controls_hidden_dim_when_feedforward_uses_linears(mlp_ratio):
    block = make_block(dim=32, num_heads=4, mlp_ratio=mlp_ratio)

    linears = get_linear_layers(block.mlp)
    assert len(linears) >= 2

    expected_hidden_dim = int(32 * mlp_ratio)
    assert linears[0].in_features == 32
    assert linears[0].out_features == expected_hidden_dim
    assert linears[-1].out_features == 32


# ---------------------------------------------------------------------
# Forward shape and attention contract
# ---------------------------------------------------------------------

def test_forward_shape_without_return_attention():
    torch.manual_seed(0)

    block = make_block(dim=32, num_heads=4).eval()
    x = torch.randn(2, 7, 32)

    y = block(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_forward_return_attention_contract_without_rope():
    torch.manual_seed(1)

    block = make_block(dim=32, num_heads=4).eval()
    x = torch.randn(2, 7, 32)

    y, attn = block(x, return_attention=True)

    assert y.shape == x.shape
    assert attn.shape == (2, 4, 7, 7)
    assert torch.isfinite(y).all()
    assert torch.isfinite(attn).all()
    assert_close(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))


def test_forward_return_attention_false_matches_first_element_when_true_in_eval():
    torch.manual_seed(2)

    block = make_block(dim=32, num_heads=4).eval()
    x = torch.randn(2, 7, 32)

    y = block(x, return_attention=False)
    y_with_attn, _ = block(x, return_attention=True)

    assert_close(y, y_with_attn)


def test_forward_with_rope_requires_valid_grid_size_from_attention():
    block = make_block(dim=32, num_heads=4, use_rope=True)
    x = torch.randn(2, 7, 32)

    with pytest.raises(ValueError, match="grid_size must be provided"):
        block(x, grid_size=None, num_special_tokens=1)


def test_forward_with_rope_valid_grid_size():
    torch.manual_seed(3)

    block = make_block(dim=32, num_heads=4, use_rope=True).eval()
    x = torch.randn(2, 7, 32)  # 1 special + 2x3 patches

    y, attn = block(
        x,
        grid_size=(2, 3),
        num_special_tokens=1,
        return_attention=True,
    )

    assert y.shape == x.shape
    assert attn.shape == (2, 4, 7, 7)


def test_forward_with_rope_rejects_token_grid_mismatch():
    block = make_block(dim=32, num_heads=4, use_rope=True)
    x = torch.randn(2, 8, 32)

    with pytest.raises(ValueError, match="Token mismatch"):
        block(x, grid_size=(2, 3), num_special_tokens=1)


def test_forward_supports_multiple_special_tokens_with_rope():
    torch.manual_seed(4)

    block = make_block(dim=32, num_heads=4, use_rope=True).eval()
    x = torch.randn(2, 9, 32)  # 3 special + 2x3 patches

    y, attn = block(
        x,
        grid_size=(2, 3),
        num_special_tokens=3,
        return_attention=True,
    )

    assert y.shape == x.shape
    assert attn.shape == (2, 4, 9, 9)


# ---------------------------------------------------------------------
# Residual and PreNorm behavior
# ---------------------------------------------------------------------

def test_zero_attention_and_zero_mlp_make_block_exact_identity():
    torch.manual_seed(5)

    block = make_block(dim=32, num_heads=4, drop_path=0.0).eval()
    set_attention_to_zero_output(block)
    set_mlp_to_zero_output(block)

    x = torch.randn(2, 7, 32)
    y = block(x)

    assert_close(y, x)


def test_zero_mlp_branch_leaves_attention_residual_only():
    torch.manual_seed(6)

    block = make_block(dim=32, num_heads=4, drop_path=0.0).eval()
    set_mlp_to_zero_output(block)

    x = torch.randn(2, 7, 32)

    with torch.no_grad():
        attn_out = block.attn(block.norm1(x), return_attention=False)
        expected = x + attn_out
        actual = block(x)

    assert_close(actual, expected)


def test_zero_attention_branch_leaves_mlp_residual_only():
    torch.manual_seed(7)

    block = make_block(dim=32, num_heads=4, drop_path=0.0).eval()
    set_attention_to_zero_output(block)

    x = torch.randn(2, 7, 32)

    with torch.no_grad():
        after_attn = x
        expected = after_attn + block.mlp(block.norm2(after_attn))
        actual = block(x)

    assert_close(actual, expected)


def test_forward_matches_manual_two_residual_prenorm_computation():
    torch.manual_seed(8)

    block = make_block(dim=32, num_heads=4, drop_path=0.0).eval()
    x = torch.randn(2, 7, 32)

    with torch.no_grad():
        attn_out = block.attn(block.norm1(x), return_attention=False)
        h = x + block.drop_path1(attn_out)
        mlp_out = block.mlp(block.norm2(h))
        expected = h + block.drop_path2(mlp_out)
        actual = block(x)

    assert_close(actual, expected)


def test_prenorm_attention_receives_norm1_output_with_hook():
    torch.manual_seed(9)

    block = make_block(dim=32, num_heads=4, drop_path=0.0).eval()
    x = torch.randn(2, 7, 32)

    captured = {}

    def attn_pre_hook(module, inputs):
        captured["attn_input"] = inputs[0].detach().clone()

    handle = block.attn.register_forward_pre_hook(attn_pre_hook)

    _ = block(x)

    handle.remove()

    expected = block.norm1(x)
    assert_close(captured["attn_input"], expected)


def test_prenorm_mlp_receives_norm2_of_post_attention_state_with_hooks():
    torch.manual_seed(10)

    block = make_block(dim=32, num_heads=4, drop_path=0.0).eval()
    x = torch.randn(2, 7, 32)

    captured = {}

    def mlp_pre_hook(module, inputs):
        captured["mlp_input"] = inputs[0].detach().clone()

    handle = block.mlp.register_forward_pre_hook(mlp_pre_hook)

    with torch.no_grad():
        attn_out = block.attn(block.norm1(x), return_attention=False)
        h = x + attn_out
        expected = block.norm2(h)

    _ = block(x)

    handle.remove()

    assert_close(captured["mlp_input"], expected)


# ---------------------------------------------------------------------
# Argument propagation to attention
# ---------------------------------------------------------------------

def test_grid_size_and_num_special_tokens_are_forwarded_to_attention_with_mock():
    torch.manual_seed(11)

    block = make_block(dim=32, num_heads=4, use_rope=True).eval()

    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(
            self,
            x,
            grid_size=None,
            num_special_tokens=1,
            return_attention=False,
        ):
            self.calls.append(
                {
                    "shape": tuple(x.shape),
                    "grid_size": grid_size,
                    "num_special_tokens": num_special_tokens,
                    "return_attention": return_attention,
                }
            )
            out = torch.zeros_like(x)
            if return_attention:
                b, t, _ = x.shape
                fake_attn = torch.zeros(b, 4, t, t, device=x.device, dtype=x.dtype)
                return out, fake_attn
            return out

    fake_attn = FakeAttention()
    block.attn = fake_attn
    set_mlp_to_zero_output(block)

    x = torch.randn(2, 9, 32)

    y, attn = block(
        x,
        grid_size=(2, 3),
        num_special_tokens=3,
        return_attention=True,
    )

    assert fake_attn.calls == [
        {
            "shape": (2, 9, 32),
            "grid_size": (2, 3),
            "num_special_tokens": 3,
            "return_attention": True,
        }
    ]
    assert y.shape == x.shape
    assert attn.shape == (2, 4, 9, 9)


# ---------------------------------------------------------------------
# DropPath behavior
# ---------------------------------------------------------------------

def test_drop_path_zero_eval_and_train_are_deterministic_for_same_input():
    torch.manual_seed(12)

    block = make_block(dim=32, num_heads=4, drop_path=0.0)
    x = torch.randn(2, 7, 32)

    block.train()
    y_train_1 = block(x)
    y_train_2 = block(x)

    block.eval()
    y_eval_1 = block(x)
    y_eval_2 = block(x)

    assert_close(y_train_1, y_train_2)
    assert_close(y_eval_1, y_eval_2)


def test_drop_path_train_mode_reproducible_with_same_seed():
    torch.manual_seed(13)

    block = make_block(dim=32, num_heads=4, drop_path=0.5).train()
    x = torch.randn(8, 7, 32)

    torch.manual_seed(999)
    y1 = block(x)

    torch.manual_seed(999)
    y2 = block(x)

    assert_close(y1, y2)


def test_drop_path_eval_mode_disabled_even_when_probability_positive():
    torch.manual_seed(14)

    block = make_block(dim=32, num_heads=4, drop_path=0.9).eval()
    x = torch.randn(8, 7, 32)

    y1 = block(x)
    y2 = block(x)

    assert_close(y1, y2)


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------

def test_backward_populates_input_and_parameter_gradients():
    torch.manual_seed(15)

    block = make_block(dim=32, num_heads=4, drop_path=0.0)
    x = torch.randn(2, 7, 32, requires_grad=True)

    y = block(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0

    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_backward_with_return_attention_populates_gradients():
    torch.manual_seed(16)

    block = make_block(dim=32, num_heads=4, drop_path=0.0)
    x = torch.randn(2, 7, 32, requires_grad=True)

    y, attn = block(x, return_attention=True)
    loss = y.pow(2).mean() + 0.01 * attn.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0


def test_no_grad_forward_returns_non_grad_tensors():
    torch.manual_seed(17)

    block = make_block(dim=32, num_heads=4).eval()
    x = torch.randn(2, 7, 32, requires_grad=True)

    with torch.no_grad():
        y, attn = block(x, return_attention=True)

    assert y.requires_grad is False
    assert attn.requires_grad is False


# ---------------------------------------------------------------------
# Dtype/device
# ---------------------------------------------------------------------

def test_forward_preserves_float64_when_module_is_double():
    torch.manual_seed(18)

    block = make_block(dim=32, num_heads=4).double().eval()
    x = torch.randn(2, 7, 32, dtype=torch.float64)

    y, attn = block(x, return_attention=True)

    assert y.dtype == torch.float64
    assert attn.dtype == torch.float64
    assert torch.isfinite(y).all()
    assert torch.isfinite(attn).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward_outputs_remain_on_cuda():
    torch.manual_seed(19)

    block = make_block(dim=32, num_heads=4, use_rope=True).cuda().eval()
    x = torch.randn(2, 7, 32, device="cuda")

    y, attn = block(
        x,
        grid_size=(2, 3),
        num_special_tokens=1,
        return_attention=True,
    )

    assert y.device.type == "cuda"
    assert attn.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_consistency_without_rope():
    torch.manual_seed(20)

    block_cpu = make_block(dim=32, num_heads=4, use_rope=False).eval()
    block_cuda = make_block(dim=32, num_heads=4, use_rope=False).eval().cuda()
    block_cuda.load_state_dict(block_cpu.state_dict())

    x_cpu = torch.randn(2, 7, 32)
    x_cuda = x_cpu.cuda()

    with torch.no_grad():
        y_cpu, a_cpu = block_cpu(x_cpu, return_attention=True)
        y_cuda, a_cuda = block_cuda(x_cuda, return_attention=True)

    assert_close(y_cpu, y_cuda.cpu(), atol=1e-5, rtol=1e-5)
    assert_close(a_cpu, a_cuda.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_consistency_with_rope():
    torch.manual_seed(21)

    block_cpu = make_block(dim=32, num_heads=4, use_rope=True).eval()
    block_cuda = make_block(dim=32, num_heads=4, use_rope=True).eval().cuda()
    block_cuda.load_state_dict(block_cpu.state_dict())

    x_cpu = torch.randn(2, 7, 32)
    x_cuda = x_cpu.cuda()

    with torch.no_grad():
        y_cpu, a_cpu = block_cpu(
            x_cpu,
            grid_size=(2, 3),
            num_special_tokens=1,
            return_attention=True,
        )
        y_cuda, a_cuda = block_cuda(
            x_cuda,
            grid_size=(2, 3),
            num_special_tokens=1,
            return_attention=True,
        )

    assert_close(y_cpu, y_cuda.cpu(), atol=1e-5, rtol=1e-5)
    assert_close(a_cpu, a_cuda.cpu(), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------
# Serialization and numerical sanity
# ---------------------------------------------------------------------

def test_state_dict_roundtrip_preserves_outputs():
    torch.manual_seed(22)

    block_a = make_block(dim=32, num_heads=4, use_rope=True).eval()
    block_b = make_block(dim=32, num_heads=4, use_rope=True).eval()

    x = torch.randn(2, 7, 32)

    block_b.load_state_dict(block_a.state_dict())

    y_a, attn_a = block_a(
        x,
        grid_size=(2, 3),
        num_special_tokens=1,
        return_attention=True,
    )
    y_b, attn_b = block_b(
        x,
        grid_size=(2, 3),
        num_special_tokens=1,
        return_attention=True,
    )

    assert_close(y_a, y_b)
    assert_close(attn_a, attn_b)


def test_large_token_count_no_nan_or_inf():
    torch.manual_seed(23)

    block = make_block(dim=64, num_heads=8, use_rope=True).eval()
    x = torch.randn(2, 65, 64)  # 1 special + 8x8 patches

    y, attn = block(
        x,
        grid_size=(8, 8),
        num_special_tokens=1,
        return_attention=True,
    )

    assert y.shape == x.shape
    assert attn.shape == (2, 8, 65, 65)
    assert torch.isfinite(y).all()
    assert torch.isfinite(attn).all()
    assert_close(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-6, rtol=1e-6)


def test_block_output_is_not_trivially_equal_to_input_with_active_branches():
    torch.manual_seed(24)

    block = make_block(dim=32, num_heads=4, drop_path=0.0).eval()
    x = torch.randn(2, 7, 32)

    y = block(x)

    assert y.shape == x.shape
    assert not torch.allclose(y, x)


def test_zero_batch_preserves_shape():
    block = make_block(dim=32, num_heads=4).eval()
    x = torch.randn(0, 7, 32)

    y, attn = block(x, return_attention=True)

    assert y.shape == (0, 7, 32)
    assert attn.shape == (0, 4, 7, 7)
