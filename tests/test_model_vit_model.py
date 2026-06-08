
"""
Advanced pytest suite for src.model.vit_model.py

Run:
    pytest -q test_model_vit_model.py

Assumed module path:
    src.model.vit_model

Designed to run comfortably on CPU with small tensors.

Covers:
    - VisionTransformer initialization contract
    - learned / rope / none positional modes
    - CLS token and register token logic
    - small-image patch embedding behavior
    - learned positional interpolation for variable crop sizes
    - _prepare_tokens contract
    - forward_features dictionary contract
    - return_features vs logits forward behavior
    - return_all_tokens behavior
    - return_attention behavior
    - no-CLS mean pooling behavior
    - classification head vs Identity head
    - depth=0 edge case
    - strict_img_size behavior
    - gradient flow
    - dtype/device behavior
    - state_dict roundtrip
"""

import importlib

import pytest
import torch
import torch.nn as nn


MODULE_PATH = "src.model.vit_model"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_module():
    return importlib.import_module(MODULE_PATH)


def assert_close(a, b, atol=1e-6, rtol=1e-5):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"Max abs diff: {(a - b).abs().max().item()}"
    )


def make_vit(
    *,
    img_size=16,
    patch_size=4,
    in_chans=3,
    num_classes=0,
    embed_dim=16,
    depth=2,
    num_heads=4,
    mlp_ratio=2.0,
    qkv_bias=True,
    patch_norm=False,
    drop_rate=0.0,
    attn_dropout=0.0,
    proj_dropout=0.0,
    mlp_dropout=0.0,
    drop_path_rate=0.0,
    pos_embed_type="learned",
    use_cls_token=True,
    num_register_tokens=0,
    rope_base=10000.0,
    norm_layer=nn.LayerNorm,
    strict_img_size=False,
):
    m = _load_module()
    return m.VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        patch_norm=patch_norm,
        drop_rate=drop_rate,
        attn_dropout=attn_dropout,
        proj_dropout=proj_dropout,
        mlp_dropout=mlp_dropout,
        drop_path_rate=drop_path_rate,
        pos_embed_type=pos_embed_type,
        use_cls_token=use_cls_token,
        num_register_tokens=num_register_tokens,
        rope_base=rope_base,
        norm_layer=norm_layer,
        strict_img_size=strict_img_size,
    )


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

def test_initialization_learned_pos_with_cls_and_registers_small_cpu():
    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=2,
        num_heads=4,
        pos_embed_type="learned",
        use_cls_token=True,
        num_register_tokens=2,
    )

    assert isinstance(model, nn.Module)
    assert model.img_size == 16
    assert model.patch_size == 4
    assert model.in_chans == 3
    assert model.embed_dim == 16
    assert model.num_features == 16
    assert model.depth == 2
    assert model.num_heads == 4
    assert model.pos_embed_type == "learned"
    assert model.use_cls_token is True
    assert model.num_register_tokens == 2
    assert model.num_special_tokens == 3

    assert model.num_patches == 16
    assert model.base_grid_size == (4, 4)

    assert model.cls_token.shape == (1, 1, 16)
    assert model.register_tokens.shape == (1, 2, 16)
    assert model.pos_embed.shape == (1, 19, 16)

    assert len(model.blocks) == 2
    assert isinstance(model.norm, nn.LayerNorm)
    assert isinstance(model.head, nn.Identity)


@pytest.mark.parametrize("pos_embed_type", ["rope", "none"])
def test_initialization_without_learned_pos_has_no_pos_embed(pos_embed_type):
    model = make_vit(
        pos_embed_type=pos_embed_type,
        use_cls_token=True,
        num_register_tokens=1,
        depth=1,
    )

    assert model.pos_embed_type == pos_embed_type
    assert model.pos_embed is None

    for block in model.blocks:
        assert block.attn.use_rope is (pos_embed_type == "rope")


def test_invalid_pos_embed_type_raises_value_error():
    with pytest.raises(ValueError, match="pos_embed_type"):
        make_vit(pos_embed_type="sinusoidal")


def test_no_cls_no_registers_has_zero_special_tokens():
    model = make_vit(
        pos_embed_type="learned",
        use_cls_token=False,
        num_register_tokens=0,
        depth=1,
    )

    assert model.cls_token is None
    assert model.register_tokens is None
    assert model.num_special_tokens == 0
    assert model.pos_embed.shape == (1, model.num_patches, model.embed_dim)


def test_num_classes_controls_head_type():
    backbone = make_vit(num_classes=0)
    classifier = make_vit(num_classes=5)

    assert isinstance(backbone.head, nn.Identity)
    assert isinstance(classifier.head, nn.Linear)
    assert classifier.head.in_features == classifier.embed_dim
    assert classifier.head.out_features == 5


def test_depth_zero_creates_no_blocks_but_forward_still_works():
    model = make_vit(
        depth=0,
        pos_embed_type="learned",
        use_cls_token=True,
        num_register_tokens=1,
        num_classes=0,
    ).eval()

    assert len(model.blocks) == 0

    x = torch.randn(2, 3, 16, 16)
    features = model(x, return_features=True)

    assert features["cls"].shape == (2, 16)
    assert features["registers"].shape == (2, 1, 16)
    assert features["patches"].shape == (2, 16, 16)
    assert features["x_norm"].shape == (2, 18, 16)


def test_patch_norm_adds_norm_inside_patch_embedding():
    model_no_norm = make_vit(patch_norm=False)
    model_with_norm = make_vit(patch_norm=True)

    assert isinstance(model_no_norm.patch_embed.norm, nn.Identity)
    assert isinstance(model_with_norm.patch_embed.norm, nn.LayerNorm)


# ---------------------------------------------------------------------
# Positional embedding interpolation
# ---------------------------------------------------------------------

def test_interpolate_pos_encoding_base_grid_returns_pos_embed_casted():
    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        pos_embed_type="learned",
        use_cls_token=True,
        num_register_tokens=1,
    )

    pos = model._interpolate_pos_encoding(
        grid_size=(4, 4),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert pos.shape == (1, 18, 16)
    assert pos.dtype == torch.float64
    assert_close(pos.float(), model.pos_embed.detach())


def test_interpolate_pos_encoding_smaller_grid_keeps_special_tokens_exact():
    torch.manual_seed(0)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        pos_embed_type="learned",
        use_cls_token=True,
        num_register_tokens=2,
    )

    pos = model._interpolate_pos_encoding(
        grid_size=(2, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert pos.shape == (1, 3 + 4, 16)

    expected_special = model.pos_embed[:, :3].detach()
    actual_special = pos[:, :3].detach()

    assert_close(actual_special, expected_special)

    # Patch positional part is interpolated to 2x2 => 4 tokens.
    assert pos[:, 3:].shape == (1, 4, 16)


def test_interpolate_pos_encoding_raises_when_pos_embed_is_none():
    model = make_vit(pos_embed_type="rope")

    with pytest.raises(RuntimeError, match="pos_embed is None"):
        model._interpolate_pos_encoding(
            grid_size=(4, 4),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------
# Token preparation
# ---------------------------------------------------------------------

def test_prepare_tokens_learned_with_cls_and_registers():
    torch.manual_seed(1)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        pos_embed_type="learned",
        use_cls_token=True,
        num_register_tokens=2,
    ).eval()

    x = torch.randn(2, 3, 16, 16)
    tokens, grid_size = model._prepare_tokens(x)

    assert grid_size == (4, 4)
    assert tokens.shape == (2, 19, 16)
    assert torch.isfinite(tokens).all()


def test_prepare_tokens_local_crop_interpolates_learned_pos():
    torch.manual_seed(2)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        pos_embed_type="learned",
        use_cls_token=True,
        num_register_tokens=1,
        strict_img_size=False,
    ).eval()

    x = torch.randn(2, 3, 8, 8)
    tokens, grid_size = model._prepare_tokens(x)

    assert grid_size == (2, 2)
    assert tokens.shape == (2, 1 + 1 + 4, 16)


def test_prepare_tokens_rope_adds_no_absolute_pos_embed():
    torch.manual_seed(3)

    model_rope = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        pos_embed_type="rope",
        use_cls_token=True,
        num_register_tokens=1,
    ).eval()

    model_none = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        pos_embed_type="none",
        use_cls_token=True,
        num_register_tokens=1,
    ).eval()

    # Copy components that affect token preparation.
    model_none.patch_embed.load_state_dict(model_rope.patch_embed.state_dict())
    with torch.no_grad():
        model_none.cls_token.copy_(model_rope.cls_token)
        model_none.register_tokens.copy_(model_rope.register_tokens)

    x = torch.randn(2, 3, 16, 16)

    tokens_rope, grid_rope = model_rope._prepare_tokens(x)
    tokens_none, grid_none = model_none._prepare_tokens(x)

    assert grid_rope == grid_none == (4, 4)
    assert_close(tokens_rope, tokens_none)


def test_prepare_tokens_without_cls_uses_only_registers_and_patches():
    torch.manual_seed(4)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        pos_embed_type="learned",
        use_cls_token=False,
        num_register_tokens=2,
    ).eval()

    x = torch.randn(2, 3, 16, 16)
    tokens, grid_size = model._prepare_tokens(x)

    assert grid_size == (4, 4)
    assert tokens.shape == (2, 2 + 16, 16)


def test_strict_img_size_rejects_local_crop():
    model = make_vit(
        img_size=16,
        patch_size=4,
        strict_img_size=True,
        pos_embed_type="learned",
    )

    x = torch.randn(2, 3, 8, 8)

    with pytest.raises(ValueError, match="does not match"):
        model._prepare_tokens(x)


# ---------------------------------------------------------------------
# forward_features contract
# ---------------------------------------------------------------------

@pytest.mark.parametrize("pos_embed_type", ["learned", "rope", "none"])
def test_forward_features_contract_small_model(pos_embed_type):
    torch.manual_seed(5)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        pos_embed_type=pos_embed_type,
        use_cls_token=True,
        num_register_tokens=2,
        drop_path_rate=0.0,
    ).eval()

    x = torch.randn(2, 3, 16, 16)
    out = model.forward_features(x, return_all_tokens=True, return_attention=False)

    assert set(out.keys()) == {"cls", "registers", "patches", "grid_size", "x_norm"}

    assert out["cls"].shape == (2, 16)
    assert out["registers"].shape == (2, 2, 16)
    assert out["patches"].shape == (2, 16, 16)
    assert out["grid_size"] == (4, 4)
    assert out["x_norm"].shape == (2, 19, 16)

    # x_norm should concatenate [CLS], registers, patches in that exact order.
    assert_close(out["cls"], out["x_norm"][:, 0])
    assert_close(out["registers"], out["x_norm"][:, 1:3])
    assert_close(out["patches"], out["x_norm"][:, 3:])


def test_forward_features_return_all_tokens_false_excludes_x_norm():
    torch.manual_seed(6)

    model = make_vit(depth=1).eval()
    x = torch.randn(2, 3, 16, 16)

    out = model.forward_features(x, return_all_tokens=False)

    assert "x_norm" not in out
    assert set(out.keys()) == {"cls", "registers", "patches", "grid_size"}


def test_forward_features_return_attention_collects_one_map_per_block():
    torch.manual_seed(7)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=2,
        num_heads=4,
        pos_embed_type="learned",
        use_cls_token=True,
        num_register_tokens=1,
        drop_path_rate=0.0,
    ).eval()

    x = torch.randn(2, 3, 16, 16)

    out = model.forward_features(
        x,
        return_all_tokens=True,
        return_attention=True,
    )

    assert "attentions" in out
    assert isinstance(out["attentions"], list)
    assert len(out["attentions"]) == 2

    for attn in out["attentions"]:
        assert attn.shape == (2, 4, 18, 18)
        assert torch.isfinite(attn).all()
        assert_close(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))


def test_forward_features_no_cls_uses_mean_pooling_over_patches():
    torch.manual_seed(8)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=0,
        pos_embed_type="learned",
        use_cls_token=False,
        num_register_tokens=2,
    ).eval()

    x = torch.randn(2, 3, 16, 16)
    out = model.forward_features(x, return_all_tokens=True)

    # With no CLS token, cls is mean over patch tokens only.
    expected_cls = out["x_norm"][:, model.num_special_tokens :].mean(dim=1)

    assert out["cls"].shape == (2, 16)
    assert_close(out["cls"], expected_cls)

    assert out["registers"].shape == (2, 2, 16)
    assert out["patches"].shape == (2, 16, 16)


def test_forward_features_no_cls_no_registers_mean_pools_all_tokens():
    torch.manual_seed(9)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=0,
        pos_embed_type="learned",
        use_cls_token=False,
        num_register_tokens=0,
    ).eval()

    x = torch.randn(2, 3, 16, 16)
    out = model.forward_features(x, return_all_tokens=True)

    assert out["registers"] is None
    assert out["patches"].shape == (2, 16, 16)
    assert_close(out["cls"], out["x_norm"].mean(dim=1))


def test_forward_features_local_crop_with_rope_small_cpu():
    torch.manual_seed(10)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        pos_embed_type="rope",
        use_cls_token=True,
        num_register_tokens=1,
        strict_img_size=False,
    ).eval()

    x = torch.randn(2, 3, 8, 8)
    out = model.forward_features(
        x,
        return_all_tokens=True,
        return_attention=True,
    )

    assert out["grid_size"] == (2, 2)
    assert out["patches"].shape == (2, 4, 16)
    assert out["x_norm"].shape == (2, 6, 16)
    assert out["attentions"][0].shape == (2, 4, 6, 6)


# ---------------------------------------------------------------------
# Standard forward
# ---------------------------------------------------------------------

def test_forward_return_features_true_returns_feature_dict():
    torch.manual_seed(11)

    model = make_vit(num_classes=5, depth=1).eval()
    x = torch.randn(2, 3, 16, 16)

    out = model(x, return_features=True)

    assert isinstance(out, dict)
    assert "cls" in out
    assert "patches" in out


def test_forward_num_classes_zero_returns_cls_features_via_identity_head():
    torch.manual_seed(12)

    model = make_vit(num_classes=0, depth=1).eval()
    x = torch.randn(2, 3, 16, 16)

    logits = model(x, return_features=False)
    features = model(x, return_features=True)

    assert logits.shape == (2, 16)
    assert_close(logits, features["cls"])


def test_forward_with_classification_head_returns_logits():
    torch.manual_seed(13)

    model = make_vit(num_classes=7, depth=1).eval()
    x = torch.randn(2, 3, 16, 16)

    logits = model(x)

    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()


def test_forward_return_attention_with_return_features_true():
    torch.manual_seed(14)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=2,
        num_heads=4,
        pos_embed_type="rope",
        use_cls_token=True,
        num_register_tokens=1,
    ).eval()

    x = torch.randn(2, 3, 16, 16)

    features = model(
        x,
        return_features=True,
        return_all_tokens=True,
        return_attention=True,
    )

    assert "attentions" in features
    assert len(features["attentions"]) == 2
    assert features["attentions"][0].shape == (2, 4, 18, 18)


# ---------------------------------------------------------------------
# Initialization sanity
# ---------------------------------------------------------------------

def test_special_and_pos_tokens_are_initialized_finite_and_nonzero():
    torch.manual_seed(15)

    model = make_vit(
        pos_embed_type="learned",
        use_cls_token=True,
        num_register_tokens=2,
    )

    assert torch.isfinite(model.cls_token).all()
    assert torch.isfinite(model.register_tokens).all()
    assert torch.isfinite(model.pos_embed).all()

    assert model.cls_token.abs().sum().item() > 0
    assert model.register_tokens.abs().sum().item() > 0
    assert model.pos_embed.abs().sum().item() > 0


def test_linear_conv_layernorm_initialization_sanity():
    torch.manual_seed(16)

    model = make_vit(depth=1, num_classes=3)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert torch.isfinite(module.weight).all()
            assert module.weight.abs().sum().item() > 0
            if module.bias is not None:
                assert_close(module.bias.detach(), torch.zeros_like(module.bias))
        elif isinstance(module, nn.Conv2d):
            assert torch.isfinite(module.weight).all()
            assert module.weight.abs().sum().item() > 0
            if module.bias is not None:
                assert_close(module.bias.detach(), torch.zeros_like(module.bias))
        elif isinstance(module, nn.LayerNorm):
            assert_close(module.weight.detach(), torch.ones_like(module.weight))
            assert_close(module.bias.detach(), torch.zeros_like(module.bias))


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------

def test_backward_through_features_populates_gradients_small_cpu():
    torch.manual_seed(17)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        num_classes=0,
        pos_embed_type="learned",
        use_cls_token=True,
        num_register_tokens=1,
        drop_path_rate=0.0,
    )

    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0

    grad_names = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }

    assert "cls_token" in grad_names
    assert "register_tokens" in grad_names
    assert "pos_embed" in grad_names
    assert any(name.startswith("patch_embed.proj") for name in grad_names)
    assert any(name.startswith("blocks.0") for name in grad_names)


def test_no_grad_forward_features_outputs_non_grad_tensors():
    torch.manual_seed(18)

    model = make_vit(depth=1).eval()
    x = torch.randn(2, 3, 16, 16, requires_grad=True)

    with torch.no_grad():
        out = model(x, return_features=True, return_attention=True)

    assert out["cls"].requires_grad is False
    assert out["patches"].requires_grad is False
    assert out["x_norm"].requires_grad is False
    assert out["attentions"][0].requires_grad is False


# ---------------------------------------------------------------------
# Dtype/device and serialization
# ---------------------------------------------------------------------

def test_forward_preserves_float64_when_module_is_double():
    torch.manual_seed(19)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        pos_embed_type="learned",
    ).double().eval()

    x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    out = model(x, return_features=True)

    assert out["cls"].dtype == torch.float64
    assert out["patches"].dtype == torch.float64
    assert out["x_norm"].dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward_outputs_remain_on_cuda_small_model():
    torch.manual_seed(20)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        pos_embed_type="rope",
    ).cuda().eval()

    x = torch.randn(2, 3, 16, 16, device="cuda")
    out = model(x, return_features=True, return_attention=True)

    assert out["cls"].device.type == "cuda"
    assert out["patches"].device.type == "cuda"
    assert out["attentions"][0].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_consistency_small_model():
    torch.manual_seed(21)

    model_cpu = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        pos_embed_type="learned",
        drop_path_rate=0.0,
    ).eval()

    model_cuda = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        pos_embed_type="learned",
        drop_path_rate=0.0,
    ).eval().cuda()

    model_cuda.load_state_dict(model_cpu.state_dict())

    x_cpu = torch.randn(2, 3, 16, 16)
    x_cuda = x_cpu.cuda()

    with torch.no_grad():
        out_cpu = model_cpu(x_cpu, return_features=True, return_attention=True)
        out_cuda = model_cuda(x_cuda, return_features=True, return_attention=True)

    assert_close(out_cpu["cls"], out_cuda["cls"].cpu(), atol=1e-5, rtol=1e-5)
    assert_close(out_cpu["patches"], out_cuda["patches"].cpu(), atol=1e-5, rtol=1e-5)
    assert_close(out_cpu["attentions"][0], out_cuda["attentions"][0].cpu(), atol=1e-5, rtol=1e-5)


def test_state_dict_roundtrip_preserves_outputs_small_cpu():
    torch.manual_seed(22)

    model_a = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        pos_embed_type="learned",
        num_classes=5,
        drop_path_rate=0.0,
    ).eval()

    model_b = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        pos_embed_type="learned",
        num_classes=5,
        drop_path_rate=0.0,
    ).eval()

    model_b.load_state_dict(model_a.state_dict())

    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        y_a = model_a(x)
        y_b = model_b(x)

    assert_close(y_a, y_b)


def test_same_seed_same_initialization_small_model():
    torch.manual_seed(123)
    model_a = make_vit(depth=1)

    torch.manual_seed(123)
    model_b = make_vit(depth=1)

    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(),
        model_b.named_parameters(),
    ):
        assert name_a == name_b
        assert_close(param_a.detach(), param_b.detach())


# ---------------------------------------------------------------------
# Numerical and edge-case sanity
# ---------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("crop_size,expected_grid,expected_patches", [(16, (4, 4), 16), (8, (2, 2), 4)])
def test_variable_crop_sizes_are_small_and_valid(batch_size, crop_size, expected_grid, expected_patches):
    torch.manual_seed(23)

    model = make_vit(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        pos_embed_type="learned",
        use_cls_token=True,
        num_register_tokens=1,
        strict_img_size=False,
    ).eval()

    x = torch.randn(batch_size, 3, crop_size, crop_size)
    out = model(x, return_features=True, return_attention=True)

    assert out["grid_size"] == expected_grid
    assert out["patches"].shape == (batch_size, expected_patches, 16)
    assert out["x_norm"].shape == (batch_size, 2 + expected_patches, 16)
    assert out["attentions"][0].shape == (
        batch_size,
        4,
        2 + expected_patches,
        2 + expected_patches,
    )

    assert torch.isfinite(out["cls"]).all()
    assert torch.isfinite(out["patches"]).all()
    assert torch.isfinite(out["attentions"][0]).all()


def test_invalid_runtime_image_size_not_divisible_by_patch_raises():
    model = make_vit(
        img_size=16,
        patch_size=4,
        strict_img_size=False,
    )

    x = torch.randn(2, 3, 10, 10)

    with pytest.raises(ValueError, match="must be divisible"):
        model(x, return_features=True)


def test_wrong_number_of_input_channels_raises():
    model = make_vit(in_chans=3)
    x = torch.randn(2, 1, 16, 16)

    with pytest.raises(ValueError, match="input channels"):
        model(x, return_features=True)


def test_small_model_outputs_no_nan_or_inf_all_pos_modes():
    torch.manual_seed(24)

    for pos_embed_type in ["learned", "rope", "none"]:
        model = make_vit(
            img_size=16,
            patch_size=4,
            embed_dim=16,
            depth=2,
            num_heads=4,
            pos_embed_type=pos_embed_type,
            use_cls_token=True,
            num_register_tokens=1,
            drop_path_rate=0.0,
        ).eval()

        x = torch.randn(2, 3, 16, 16)
        out = model(x, return_features=True, return_attention=True)

        assert torch.isfinite(out["cls"]).all()
        assert torch.isfinite(out["patches"]).all()
        assert torch.isfinite(out["x_norm"]).all()
        for attn in out["attentions"]:
            assert torch.isfinite(attn).all()
            assert_close(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-6, rtol=1e-6)
