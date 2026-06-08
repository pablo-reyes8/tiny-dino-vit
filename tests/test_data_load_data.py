
"""
Advanced pytest suite for data.load_data.py

Run:
    pytest -q test_data_load_data.py

Assumed module path:
    data.load_data

These tests intentionally mock Hugging Face `load_dataset` so they do not
download TinyImageNet or require internet access.
"""

import importlib
import math

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader


MODULE_PATH = "data.load_data"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_module():
    return importlib.import_module(MODULE_PATH)


def make_rgb_image(width=80, height=72, color=(120, 80, 40)):
    return Image.new("RGB", (width, height), color=color)


def make_grayscale_image(width=80, height=72, value=127):
    return Image.new("L", (width, height), color=value)


def make_rgba_image(width=80, height=72, color=(120, 80, 40, 180)):
    return Image.new("RGBA", (width, height), color=color)


class FakeHFDataset:
    """
    Minimal Hugging Face-like dataset.

    Supports:
        len(dataset)
        dataset[idx] -> {"image": PIL.Image, "label": int}
    """

    def __init__(self, n=8, image_factory=None, label_offset=0):
        self.n = n
        self.image_factory = image_factory or (lambda i: make_rgb_image())
        self.label_offset = label_offset

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if not 0 <= idx < self.n:
            raise IndexError(idx)

        return {
            "image": self.image_factory(idx),
            "label": int(idx + self.label_offset),
        }


def base_config(**overrides):
    cfg = {
        "seed": 123,
        "dataset_name": "fake/tiny-imagenet",
        "global_crop_size": 64,
        "local_crop_size": 32,
        "num_global_crops": 2,
        "num_local_crops": 3,
        "global_crop_scale": (0.5, 1.0),
        "local_crop_scale": (0.2, 0.5),
        "val_ratio": 0.25,
        "batch_size": 4,
        "shuffle_train": False,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last_train": False,
        "drop_last_val": False,
        "persistent_workers": True,
    }
    cfg.update(overrides)
    return cfg


def assert_normalized_image_tensor(x, expected_size):
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, expected_size, expected_size)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all(), "Crop contains NaN or Inf."

    # Normalized ImageNet tensors should not usually lie in [0, 1].
    # This catches missing Normalize or missing ToTensor in many cases.
    assert x.min().item() < 0.0 or x.max().item() > 1.0


# ---------------------------------------------------------------------
# DINOTransform tests
# ---------------------------------------------------------------------

def test_dino_transform_returns_ordered_multicrop_contract():
    m = _load_module()

    transform = m.DINOTransform(
        global_crop_size=64,
        local_crop_size=32,
        num_global_crops=2,
        num_local_crops=4,
        global_crop_scale=(0.6, 1.0),
        local_crop_scale=(0.2, 0.5),
    )

    image = make_rgb_image(96, 80)
    crops = transform(image)

    assert isinstance(crops, list)
    assert len(crops) == 6

    for crop in crops[:2]:
        assert_normalized_image_tensor(crop, expected_size=64)

    for crop in crops[2:]:
        assert_normalized_image_tensor(crop, expected_size=32)


def test_dino_transform_respects_zero_crop_counts():
    m = _load_module()

    transform = m.DINOTransform(
        global_crop_size=64,
        local_crop_size=32,
        num_global_crops=0,
        num_local_crops=0,
    )

    crops = transform(make_rgb_image())
    assert crops == []


def test_dino_transform_is_reproducible_under_same_torch_seed():
    """
    torchvision random transforms draw from torch RNG in current versions.
    This test catches accidental use of uncontrolled randomness in the pipeline.
    """
    m = _load_module()

    transform = m.DINOTransform(
        global_crop_size=64,
        local_crop_size=32,
        num_global_crops=2,
        num_local_crops=2,
    )

    image = make_rgb_image(96, 96, color=(40, 130, 220))

    torch.manual_seed(999)
    crops_a = transform(image)

    torch.manual_seed(999)
    crops_b = transform(image)

    assert len(crops_a) == len(crops_b)
    for a, b in zip(crops_a, crops_b):
        assert torch.equal(a, b), "Same seed should reproduce identical crops."


def test_dino_transform_produces_non_aliasing_tensors():
    """
    The crop list should contain independent tensors. This protects against
    accidentally appending the same tensor reference repeatedly.
    """
    m = _load_module()

    transform = m.DINOTransform(
        global_crop_size=64,
        local_crop_size=32,
        num_global_crops=2,
        num_local_crops=2,
    )

    torch.manual_seed(77)
    crops = transform(make_rgb_image(96, 96))

    data_ptrs = [crop.data_ptr() for crop in crops]
    assert len(set(data_ptrs)) == len(data_ptrs)

    original_second_crop = crops[1].clone()
    crops[0].zero_()
    assert torch.equal(crops[1], original_second_crop), (
        "Mutating one crop should not mutate another crop."
    )


def test_dino_transform_accepts_grayscale_and_rgba_after_dataset_rgb_conversion():
    """
    DINOTransform itself expects PIL-compatible RGB-like input. The RGB conversion
    is the responsibility of TinyImageNetDataset, tested here through the wrapper.
    """
    m = _load_module()

    transform = m.DINOTransform(
        global_crop_size=64,
        local_crop_size=32,
        num_global_crops=1,
        num_local_crops=1,
    )

    gray_hf = FakeHFDataset(n=1, image_factory=lambda i: make_grayscale_image())
    rgba_hf = FakeHFDataset(n=1, image_factory=lambda i: make_rgba_image())

    gray_ds = m.TinyImageNetDataset(gray_hf, transform=transform, return_label=False)
    rgba_ds = m.TinyImageNetDataset(rgba_hf, transform=transform, return_label=False)

    gray_crops = gray_ds[0]
    rgba_crops = rgba_ds[0]

    assert len(gray_crops) == 2
    assert len(rgba_crops) == 2
    assert gray_crops[0].shape[0] == 3
    assert rgba_crops[0].shape[0] == 3


# ---------------------------------------------------------------------
# build_eval_transform tests
# ---------------------------------------------------------------------

def test_eval_transform_is_deterministic_and_matches_manual_normalization():
    m = _load_module()

    transform = m.build_eval_transform(image_size=64)
    image = Image.new("RGB", (32, 48), color=(255, 0, 0))

    out_a = transform(image)
    out_b = transform(image)

    assert torch.equal(out_a, out_b)
    assert out_a.shape == (3, 64, 64)
    assert out_a.dtype == torch.float32
    assert torch.isfinite(out_a).all()

    # Since the input is a constant red image and Resize preserves constants,
    # every spatial location should equal ImageNet normalization of RGB=(1,0,0).
    expected = torch.tensor([
        (1.0 - 0.485) / 0.229,
        (0.0 - 0.456) / 0.224,
        (0.0 - 0.406) / 0.225,
    ], dtype=torch.float32)

    observed = out_a[:, 0, 0]
    assert torch.allclose(observed, expected, atol=1e-5)

    # Each channel should be spatially constant for a constant image.
    channel_std = out_a.flatten(1).std(dim=1)
    assert torch.all(channel_std < 1e-6)


# ---------------------------------------------------------------------
# TinyImageNetDataset tests
# ---------------------------------------------------------------------

def test_tinyimagenet_dataset_len_and_label_contract_without_transform():
    m = _load_module()

    hf = FakeHFDataset(n=5)
    ds_without_label = m.TinyImageNetDataset(hf, transform=None, return_label=False)
    ds_with_label = m.TinyImageNetDataset(hf, transform=None, return_label=True)

    assert len(ds_without_label) == 5
    assert len(ds_with_label) == 5

    image_only = ds_without_label[2]
    image_with_label, label = ds_with_label[2]

    assert isinstance(image_only, Image.Image)
    assert image_only.mode == "RGB"
    assert isinstance(image_with_label, Image.Image)
    assert image_with_label.mode == "RGB"
    assert label == 2


def test_tinyimagenet_dataset_applies_transform_after_rgb_conversion():
    m = _load_module()

    seen_modes = []

    def mode_recording_transform(image):
        seen_modes.append(image.mode)
        return torch.ones(3, 10, 10)

    hf = FakeHFDataset(n=2, image_factory=lambda i: make_grayscale_image())
    ds = m.TinyImageNetDataset(
        hf_dataset=hf,
        transform=mode_recording_transform,
        return_label=True,
    )

    x, y = ds[0]

    assert seen_modes == ["RGB"]
    assert torch.equal(x, torch.ones(3, 10, 10))
    assert y == 0


def test_tinyimagenet_dataset_does_not_swallow_hf_index_errors():
    m = _load_module()

    hf = FakeHFDataset(n=1)
    ds = m.TinyImageNetDataset(hf, transform=None, return_label=False)

    with pytest.raises(IndexError):
        _ = ds[999]


# ---------------------------------------------------------------------
# seed_worker tests
# ---------------------------------------------------------------------

def test_seed_worker_uses_global_data_config_and_worker_id(monkeypatch):
    m = _load_module()

    monkeypatch.setattr(m, "DATA_CONFIG", {"seed": 10}, raising=False)

    m.seed_worker(worker_id=7)
    sample_a = torch.rand(5)

    torch.manual_seed(17)
    sample_b = torch.rand(5)

    assert torch.equal(sample_a, sample_b), (
        "seed_worker should set torch seed to DATA_CONFIG['seed'] + worker_id."
    )


# ---------------------------------------------------------------------
# build_tinyimagenet_dataloaders tests
# ---------------------------------------------------------------------

def test_build_dataloaders_uses_existing_validation_split(monkeypatch, capsys):
    m = _load_module()

    fake_hf = {
        "train": FakeHFDataset(n=8),
        "validation": FakeHFDataset(n=4, label_offset=100),
    }

    monkeypatch.setattr(m, "load_dataset", lambda name: fake_hf)

    cfg = base_config(
        batch_size=4,
        num_workers=0,
        num_global_crops=2,
        num_local_crops=2,
        global_crop_size=64,
        local_crop_size=32,
    )

    train_dataset, val_dataset, train_loader, val_loader = (
        m.build_tinyimagenet_dataloaders(config=cfg)
    )

    captured = capsys.readouterr().out
    assert "Using 'validation' as validation split." in captured

    assert len(train_dataset) == 8
    assert len(val_dataset) == 4
    assert train_loader.persistent_workers is False
    assert val_loader.persistent_workers is False

    train_batch = next(iter(train_loader))
    assert isinstance(train_batch, list)
    assert len(train_batch) == 4

    assert train_batch[0].shape == (4, 3, 64, 64)
    assert train_batch[1].shape == (4, 3, 64, 64)
    assert train_batch[2].shape == (4, 3, 32, 32)
    assert train_batch[3].shape == (4, 3, 32, 32)

    for crop_batch in train_batch:
        assert crop_batch.dtype == torch.float32
        assert torch.isfinite(crop_batch).all()

    val_images, val_labels = next(iter(val_loader))
    assert val_images.shape == (4, 3, 64, 64)
    assert val_images.dtype == torch.float32
    assert torch.equal(val_labels, torch.tensor([100, 101, 102, 103]))


def test_build_dataloaders_prefers_validation_over_test(monkeypatch, capsys):
    m = _load_module()

    fake_hf = {
        "train": FakeHFDataset(n=6),
        "validation": FakeHFDataset(n=2, label_offset=100),
        "test": FakeHFDataset(n=3, label_offset=200),
    }

    monkeypatch.setattr(m, "load_dataset", lambda name: fake_hf)

    cfg = base_config(batch_size=2, num_workers=0)
    _, val_dataset, _, val_loader = m.build_tinyimagenet_dataloaders(config=cfg)

    captured = capsys.readouterr().out
    assert "Using 'validation' as validation split." in captured
    assert len(val_dataset) == 2

    _, labels = next(iter(val_loader))
    assert torch.equal(labels, torch.tensor([100, 101]))


def test_build_dataloaders_falls_back_to_manual_split_reproducibly(monkeypatch):
    m = _load_module()

    fake_hf = {
        "train": FakeHFDataset(n=20),
    }

    monkeypatch.setattr(m, "load_dataset", lambda name: fake_hf)

    cfg = base_config(
        seed=2025,
        batch_size=5,
        val_ratio=0.2,
        num_workers=0,
        shuffle_train=False,
    )

    train_ds_a, val_ds_a, train_loader_a, val_loader_a = (
        m.build_tinyimagenet_dataloaders(config=cfg)
    )

    train_ds_b, val_ds_b, train_loader_b, val_loader_b = (
        m.build_tinyimagenet_dataloaders(config=cfg)
    )

    assert len(train_ds_a) == 16
    assert len(val_ds_a) == 4

    train_indices_a = list(train_ds_a.indices)
    val_indices_a = list(val_ds_a.indices)
    train_indices_b = list(train_ds_b.indices)
    val_indices_b = list(val_ds_b.indices)

    assert train_indices_a == train_indices_b
    assert val_indices_a == val_indices_b
    assert set(train_indices_a).isdisjoint(set(val_indices_a))
    assert sorted(train_indices_a + val_indices_a) == list(range(20))

    # Validate train multicrop batch contract after Subset wrapping.
    train_batch = next(iter(train_loader_a))
    assert isinstance(train_batch, list)
    assert len(train_batch) == cfg["num_global_crops"] + cfg["num_local_crops"]

    # Validate val label order follows Subset indices exactly.
    _, val_labels = next(iter(val_loader_a))
    assert torch.equal(val_labels, torch.tensor(val_indices_a[: len(val_labels)]))


def test_build_dataloaders_raises_clear_error_without_train_split(monkeypatch):
    m = _load_module()

    fake_hf = {
        "validation": FakeHFDataset(n=4),
    }

    monkeypatch.setattr(m, "load_dataset", lambda name: fake_hf)

    with pytest.raises(ValueError, match="split 'train'"):
        m.build_tinyimagenet_dataloaders(config=base_config())


def test_build_dataloaders_drop_last_train_and_val_are_respected(monkeypatch):
    m = _load_module()

    fake_hf = {
        "train": FakeHFDataset(n=10),
        "validation": FakeHFDataset(n=7, label_offset=50),
    }

    monkeypatch.setattr(m, "load_dataset", lambda name: fake_hf)

    cfg = base_config(
        batch_size=4,
        drop_last_train=True,
        drop_last_val=True,
        num_workers=0,
    )

    _, _, train_loader, val_loader = m.build_tinyimagenet_dataloaders(config=cfg)

    assert len(train_loader) == 2  # floor(10 / 4)
    assert len(val_loader) == 1    # floor(7 / 4)

    train_batches = list(iter(train_loader))
    val_batches = list(iter(val_loader))

    assert all(batch[0].shape[0] == 4 for batch in train_batches)
    assert all(images.shape[0] == 4 for images, labels in val_batches)


def test_build_dataloaders_is_not_silently_deterministic_across_epochs_without_generator(monkeypatch):
    """
    This is a diagnostic test rather than a failure expectation.

    With num_workers=0 and shuffle_train=True, DataLoader uses the global torch RNG
    unless a generator is explicitly passed. The builder seeds torch.manual_seed once.
    This test documents that two loaders built with the same config initially produce
    the same first shuffled batch, which is useful for reproducible experiments.
    """
    m = _load_module()

    fake_hf = {
        "train": FakeHFDataset(n=12),
        "validation": FakeHFDataset(n=4, label_offset=100),
    }

    monkeypatch.setattr(m, "load_dataset", lambda name: fake_hf)

    cfg = base_config(
        seed=42,
        batch_size=4,
        shuffle_train=True,
        num_workers=0,
        num_global_crops=1,
        num_local_crops=0,
    )

    _, _, loader_a, _ = m.build_tinyimagenet_dataloaders(config=cfg)
    first_a = next(iter(loader_a))[0]

    _, _, loader_b, _ = m.build_tinyimagenet_dataloaders(config=cfg)
    first_b = next(iter(loader_b))[0]

    assert torch.equal(first_a, first_b)


# ---------------------------------------------------------------------
# Configuration edge-case tests
# ---------------------------------------------------------------------

def test_invalid_crop_scale_fails_fast():
    """
    torchvision.RandomResizedCrop should reject invalid scale intervals.
    This protects against silent misconfiguration.
    """
    m = _load_module()

    with pytest.raises(Exception):
        m.DINOTransform(
            global_crop_size=64,
            local_crop_size=32,
            global_crop_scale=(1.0, 0.5),
        )


def test_dataloader_multicrop_default_collate_preserves_crop_order(monkeypatch):
    """
    Validates an important DINO invariant: after DataLoader collation, the batch
    is still a list ordered as all global crops first, then all local crops.
    """

    m = _load_module()

    class DeterministicCropTransform:
        def __call__(self, image):
            return [
                torch.full((3, 64, 64), 1.0),
                torch.full((3, 64, 64), 2.0),
                torch.full((3, 32, 32), 3.0),
            ]

    ds = m.TinyImageNetDataset(
        hf_dataset=FakeHFDataset(n=4),
        transform=DeterministicCropTransform(),
        return_label=False,
    )

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    assert isinstance(batch, list)
    assert len(batch) == 3

    assert batch[0].shape == (4, 3, 64, 64)
    assert batch[1].shape == (4, 3, 64, 64)
    assert batch[2].shape == (4, 3, 32, 32)

    assert torch.all(batch[0] == 1.0)
    assert torch.all(batch[1] == 2.0)
    assert torch.all(batch[2] == 3.0)
