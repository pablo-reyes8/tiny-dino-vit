from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


def _move_tensor_tree(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, tuple):
        return tuple(_move_tensor_tree(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_tensor_tree(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_tensor_tree(v, device) for k, v in obj.items()}
    return obj


def _pre_hook_to_device(device: torch.device):
    def hook(_module, inputs):
        return _move_tensor_tree(inputs, device)

    return hook


def shard_vit_blocks(
    model: nn.Module,
    devices: Iterable[str | torch.device],
) -> nn.Module:
    """
    Experimental model-parallel placement for DINO ViT blocks across GPUs.

    This is intended for inference/prototyping or very large ViTs when DDP is not enough.
    For production training, prefer DDP through `scripts/train_dino_multi_gpu.py`.
    """
    devices = [torch.device(device) for device in devices]
    if len(devices) == 0:
        raise ValueError("devices must contain at least one device.")

    backbone = getattr(model, "backbone", model)
    if not hasattr(backbone, "blocks"):
        raise AttributeError("Expected model or model.backbone to expose a 'blocks' ModuleList.")

    first_device = devices[0]
    last_device = devices[-1]

    if hasattr(backbone, "patch_embed"):
        backbone.patch_embed.to(first_device)
        backbone.patch_embed.register_forward_pre_hook(_pre_hook_to_device(first_device))
    if hasattr(backbone, "cls_token") and torch.is_tensor(backbone.cls_token):
        backbone.cls_token.data = backbone.cls_token.data.to(first_device)
    if hasattr(backbone, "pos_embed") and torch.is_tensor(backbone.pos_embed):
        backbone.pos_embed.data = backbone.pos_embed.data.to(first_device)

    blocks: List[nn.Module] = list(backbone.blocks)
    for idx, block in enumerate(blocks):
        device = devices[idx % len(devices)]
        block.to(device)
        block.register_forward_pre_hook(_pre_hook_to_device(device))

    if hasattr(backbone, "norm"):
        backbone.norm.to(last_device)
        backbone.norm.register_forward_pre_hook(_pre_hook_to_device(last_device))
    if hasattr(model, "head"):
        model.head.to(last_device)
        model.head.register_forward_pre_hook(_pre_hook_to_device(last_device))

    return model
