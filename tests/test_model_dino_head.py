
"""
Advanced pytest suite for src.model.dino_head.py

Run:
    pytest -q test_model_dino_head.py

Assumed module path:
    src.model.dino_head

Covers:
    - DINOHead initialization contract
    - num_layers=1 vs num_layers>1 architecture
    - BatchNorm insertion when use_bn=True
    - invalid num_layers
    - weight normalization parameters
    - norm_last_layer freezing behavior
    - initialization behavior for Linear and BatchNorm1d
    - forward shape and manual equivalence
    - L2 normalization before last layer
    - gradient flow and frozen weight_g behavior
    - train/eval behavior with BatchNorm
    - dtype/device behavior
    - state_dict roundtrip
"""

import importlib
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_PATH = "src.model.dino_head"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_module():
    return importlib.import_module(MODULE_PATH)


def make_head(
    *,
    in_dim=32,
    out_dim=128,
    hidden_dim=64,
    bottleneck_dim=16,
    num_layers=3,
    use_bn=False,
    norm_last_layer=True,
    bias=True,
):
    m = _load_module()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return m.DINOHead(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_layers=num_layers,
            use_bn=use_bn,
            norm_last_layer=norm_last_layer,
            bias=bias,
        )


def assert_close(a, b, atol=1e-6, rtol=1e-5):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"Max abs diff: {(a - b).abs().max().item()}"
    )


def get_linear_layers(module):
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def get_bn_layers(module):
    return [m for m in module.modules() if isinstance(m, nn.BatchNorm1d)]


# ---------------------------------------------------------------------
# Initialization and architecture
# ---------------------------------------------------------------------

def test_initialization_stores_configuration():
    head = make_head(
        in_dim=32,
        out_dim=100,
        hidden_dim=64,
        bottleneck_dim=17,
        num_layers=3,
        use_bn=True,
        norm_last_layer=False,
        bias=False,
    )

    assert head.in_dim == 32
    assert head.out_dim == 100
    assert head.hidden_dim == 64
    assert head.bottleneck_dim == 17
    assert head.num_layers == 3
    assert head.use_bn is True
    assert head.norm_last_layer is False


@pytest.mark.parametrize("bad_num_layers", [0, -1, -5])
def test_invalid_num_layers_raises_value_error(bad_num_layers):
    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        make_head(num_layers=bad_num_layers)


def test_num_layers_one_uses_single_linear_mlp():
    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=1,
        use_bn=True,   # should be ignored for num_layers=1
        bias=True,
    )

    assert isinstance(head.mlp, nn.Linear)
    assert head.mlp.in_features == 32
    assert head.mlp.out_features == 16
    assert head.mlp.bias is not None

    # one MLP linear + parametrized last_layer linear
    linear_layers = get_linear_layers(head)
    assert len(linear_layers) == 2

    bn_layers = get_bn_layers(head)
    assert len(bn_layers) == 0


def test_num_layers_three_without_bn_architecture():
    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=False,
        bias=True,
    )

    assert isinstance(head.mlp, nn.Sequential)

    mlp_layers = list(head.mlp)
    assert len(mlp_layers) == 5

    assert isinstance(mlp_layers[0], nn.Linear)
    assert mlp_layers[0].in_features == 32
    assert mlp_layers[0].out_features == 64

    assert isinstance(mlp_layers[1], nn.GELU)

    assert isinstance(mlp_layers[2], nn.Linear)
    assert mlp_layers[2].in_features == 64
    assert mlp_layers[2].out_features == 64

    assert isinstance(mlp_layers[3], nn.GELU)

    assert isinstance(mlp_layers[4], nn.Linear)
    assert mlp_layers[4].in_features == 64
    assert mlp_layers[4].out_features == 16


def test_num_layers_four_with_bn_architecture():
    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=4,
        use_bn=True,
        bias=True,
    )

    assert isinstance(head.mlp, nn.Sequential)

    mlp_layers = list(head.mlp)

    # Linear+BN+GELU, Linear+BN+GELU, Linear+BN+GELU, final Linear
    assert len(mlp_layers) == 10

    assert isinstance(mlp_layers[0], nn.Linear)
    assert isinstance(mlp_layers[1], nn.BatchNorm1d)
    assert isinstance(mlp_layers[2], nn.GELU)

    assert isinstance(mlp_layers[3], nn.Linear)
    assert isinstance(mlp_layers[4], nn.BatchNorm1d)
    assert isinstance(mlp_layers[5], nn.GELU)

    assert isinstance(mlp_layers[6], nn.Linear)
    assert isinstance(mlp_layers[7], nn.BatchNorm1d)
    assert isinstance(mlp_layers[8], nn.GELU)

    assert isinstance(mlp_layers[9], nn.Linear)
    assert mlp_layers[9].out_features == 16

    assert len(get_bn_layers(head)) == 3


def test_bias_flag_applies_to_mlp_linears_but_not_last_layer():
    head_bias = make_head(num_layers=3, bias=True)
    head_no_bias = make_head(num_layers=3, bias=False)

    mlp_linears_bias = [m for m in head_bias.mlp if isinstance(m, nn.Linear)]
    mlp_linears_no_bias = [m for m in head_no_bias.mlp if isinstance(m, nn.Linear)]

    assert all(layer.bias is not None for layer in mlp_linears_bias)
    assert all(layer.bias is None for layer in mlp_linears_no_bias)

    # Last layer is explicitly bias=False regardless of bias flag.
    assert head_bias.last_layer.bias is None
    assert head_no_bias.last_layer.bias is None


# ---------------------------------------------------------------------
# Weight norm and last-layer behavior
# ---------------------------------------------------------------------

def test_last_layer_is_weight_normalized_with_expected_parameters():
    head = make_head(
        in_dim=32,
        out_dim=128,
        bottleneck_dim=16,
        norm_last_layer=True,
    )

    # Old torch.nn.utils.weight_norm exposes weight_g and weight_v.
    assert hasattr(head.last_layer, "weight_g")
    assert hasattr(head.last_layer, "weight_v")

    assert head.last_layer.weight_g.shape == (128, 1)
    assert head.last_layer.weight_v.shape == (128, 16)

    assert_close(head.last_layer.weight_g.detach(), torch.ones_like(head.last_layer.weight_g))


def test_norm_last_layer_true_freezes_weight_g():
    head = make_head(norm_last_layer=True)
    assert head.last_layer.weight_g.requires_grad is False
    assert head.last_layer.weight_v.requires_grad is True


def test_norm_last_layer_false_allows_weight_g_gradients():
    head = make_head(norm_last_layer=False)
    assert head.last_layer.weight_g.requires_grad is True
    assert head.last_layer.weight_v.requires_grad is True


def test_last_layer_has_no_bias():
    head = make_head()
    assert head.last_layer.bias is None


# ---------------------------------------------------------------------
# Initialization behavior
# ---------------------------------------------------------------------

def test_linear_biases_initialized_to_zero():
    head = make_head(num_layers=4, use_bn=True, bias=True)

    for layer in get_linear_layers(head):
        if layer.bias is not None:
            assert_close(layer.bias.detach(), torch.zeros_like(layer.bias))


def test_batchnorm_initialized_to_identity_affine():
    head = make_head(num_layers=4, use_bn=True)

    for bn in get_bn_layers(head):
        assert_close(bn.weight.detach(), torch.ones_like(bn.weight))
        assert_close(bn.bias.detach(), torch.zeros_like(bn.bias))


def test_linear_weights_are_finite_and_nonzero_after_initialization():
    torch.manual_seed(0)
    head = make_head(num_layers=3, use_bn=False)

    for layer in get_linear_layers(head):
        assert torch.isfinite(layer.weight).all()
        assert layer.weight.abs().sum().item() > 0


# ---------------------------------------------------------------------
# Forward behavior
# ---------------------------------------------------------------------

@pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
@pytest.mark.parametrize("use_bn", [False, True])
def test_forward_output_shape_for_various_architectures(num_layers, use_bn):
    torch.manual_seed(1)

    # BatchNorm1d in train mode needs batch size > 1.
    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=num_layers,
        use_bn=use_bn,
    ).eval()

    x = torch.randn(5, 32)
    y = head(x)

    assert y.shape == (5, 128)
    assert torch.isfinite(y).all()


def test_forward_matches_manual_mlp_normalize_last_layer():
    torch.manual_seed(2)

    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=False,
        norm_last_layer=True,
    ).eval()

    x = torch.randn(4, 32)

    with torch.no_grad():
        mlp_out = head.mlp(x)
        normalized = F.normalize(mlp_out, dim=-1, p=2)
        expected = head.last_layer(normalized)
        actual = head(x)

    assert_close(actual, expected)


def test_bottleneck_features_are_l2_normalized_before_last_layer_using_hook():
    torch.manual_seed(3)

    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=False,
    ).eval()

    captured = {}

    def pre_hook(module, inputs):
        captured["last_layer_input"] = inputs[0].detach().clone()

    handle = head.last_layer.register_forward_pre_hook(pre_hook)

    x = torch.randn(6, 32)
    _ = head(x)

    handle.remove()

    z = captured["last_layer_input"]
    norms = z.norm(dim=-1)

    assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6)


def test_forward_with_zero_input_is_finite():
    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=False,
    ).eval()

    x = torch.zeros(4, 32)
    y = head(x)

    assert y.shape == (4, 128)
    assert torch.isfinite(y).all()


def test_wrong_input_dim_fails_loudly():
    head = make_head(in_dim=32)

    x = torch.randn(4, 31)

    with pytest.raises(RuntimeError):
        head(x)


def test_forward_accepts_higher_rank_input_supported_by_linear():
    """
    nn.Linear supports arbitrary leading dimensions. The DINOHead docstring
    says [B, in_dim], but the implementation also works for e.g. [B, T, in_dim]
    when no BatchNorm is used.
    """
    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=False,
    ).eval()

    x = torch.randn(2, 3, 32)
    y = head(x)

    assert y.shape == (2, 3, 128)


# ---------------------------------------------------------------------
# BatchNorm behavior
# ---------------------------------------------------------------------

def test_batchnorm_train_mode_updates_running_statistics():
    torch.manual_seed(4)

    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=True,
    ).train()

    bn_layers = get_bn_layers(head)
    assert len(bn_layers) > 0

    before = [bn.running_mean.detach().clone() for bn in bn_layers]

    x = torch.randn(16, 32) + 10.0
    _ = head(x)

    after = [bn.running_mean.detach().clone() for bn in bn_layers]

    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_batchnorm_eval_mode_is_deterministic():
    torch.manual_seed(5)

    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=True,
    ).eval()

    x = torch.randn(8, 32)

    y1 = head(x)
    y2 = head(x)

    assert_close(y1, y2)


def test_batchnorm_with_batch_size_one_in_train_mode_fails_loudly():
    """
    BatchNorm1d cannot estimate batch statistics with a single sample in train mode.
    This documents the expected PyTorch behavior.
    """
    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=True,
    ).train()

    x = torch.randn(1, 32)

    with pytest.raises(ValueError):
        head(x)


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------

def test_backward_populates_gradients_for_input_and_trainable_parameters():
    torch.manual_seed(6)

    head = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=False,
        norm_last_layer=True,
    )

    x = torch.randn(4, 32, requires_grad=True)
    y = head(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0

    for name, param in head.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    assert head.last_layer.weight_g.requires_grad is False
    assert head.last_layer.weight_g.grad is None


def test_norm_last_layer_false_weight_g_receives_gradient():
    torch.manual_seed(7)

    head = make_head(norm_last_layer=False)

    x = torch.randn(4, 32, requires_grad=True)
    y = head(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert head.last_layer.weight_g.requires_grad is True
    assert head.last_layer.weight_g.grad is not None
    assert torch.isfinite(head.last_layer.weight_g.grad).all()


def test_no_grad_forward_returns_non_grad_tensor():
    torch.manual_seed(8)

    head = make_head()
    x = torch.randn(4, 32, requires_grad=True)

    with torch.no_grad():
        y = head(x)

    assert y.requires_grad is False


# ---------------------------------------------------------------------
# Dtype and device
# ---------------------------------------------------------------------

def test_forward_preserves_float64_when_module_is_double():
    torch.manual_seed(9)

    head = make_head().double().eval()
    x = torch.randn(4, 32, dtype=torch.float64)

    y = head(x)

    assert y.dtype == torch.float64
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward_outputs_remain_on_cuda():
    torch.manual_seed(10)

    head = make_head().cuda().eval()
    x = torch.randn(4, 32, device="cuda")

    y = head(x)

    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_cuda_consistency_for_same_weights():
    torch.manual_seed(11)

    head_cpu = make_head().eval()
    head_cuda = make_head().eval().cuda()
    head_cuda.load_state_dict(head_cpu.state_dict())

    x_cpu = torch.randn(4, 32)
    x_cuda = x_cpu.cuda()

    with torch.no_grad():
        y_cpu = head_cpu(x_cpu)
        y_cuda = head_cuda(x_cuda).cpu()

    assert_close(y_cpu, y_cuda, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------
# Serialization and reproducibility
# ---------------------------------------------------------------------

def test_state_dict_roundtrip_preserves_outputs():
    torch.manual_seed(12)

    head_a = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=False,
    ).eval()

    head_b = make_head(
        in_dim=32,
        out_dim=128,
        hidden_dim=64,
        bottleneck_dim=16,
        num_layers=3,
        use_bn=False,
    ).eval()

    x = torch.randn(4, 32)

    head_b.load_state_dict(head_a.state_dict())

    y_a = head_a(x)
    y_b = head_b(x)

    assert_close(y_a, y_b)


def test_same_seed_same_initialization():
    torch.manual_seed(123)
    head_a = make_head()

    torch.manual_seed(123)
    head_b = make_head()

    for (name_a, param_a), (name_b, param_b) in zip(
        head_a.named_parameters(),
        head_b.named_parameters(),
    ):
        assert name_a == name_b
        assert_close(param_a.detach(), param_b.detach())


def test_different_seed_generally_changes_initialization():
    torch.manual_seed(123)
    head_a = make_head()

    torch.manual_seed(124)
    head_b = make_head()

    diffs = []
    for (name_a, param_a), (name_b, param_b) in zip(
        head_a.named_parameters(),
        head_b.named_parameters(),
    ):
        assert name_a == name_b
        if param_a.shape == param_b.shape and param_a.numel() > 1:
            diffs.append(not torch.allclose(param_a.detach(), param_b.detach()))

    assert any(diffs)


def test_large_output_dimension_no_nan_or_inf():
    torch.manual_seed(13)

    head = make_head(
        in_dim=64,
        out_dim=8192,
        hidden_dim=128,
        bottleneck_dim=32,
        num_layers=3,
        use_bn=False,
    ).eval()

    x = torch.randn(3, 64)
    y = head(x)

    assert y.shape == (3, 8192)
    assert torch.isfinite(y).all()
