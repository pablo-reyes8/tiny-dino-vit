"""Define the Vision Transformer backbone used by the DINO models."""

from typing import Optional, Tuple, Literal, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.embeddings import * 
from src.model.transformer_block import * 

class VisionTransformer(nn.Module):
    """
    Plain Vision Transformer backbone adapted for DINO-style self-supervised learning.

    Supports:
        - learned absolute positional embeddings
        - 2D RoPE inside attention blocks
        - CLS token
        - optional register tokens
        - variable input crop sizes, e.g. 64x64 global crops and 32x32 local crops
        - returning CLS and patch tokens separately for DINO loss and Gram loss

    Expected dependencies:
        - PatchEmbedding
        - TransformerEncoderBlock

    The TransformerEncoderBlock should support:
        block(
            x,
            grid_size=grid_size,
            num_special_tokens=num_special_tokens,
            return_attention=False,
        )
    if using RoPE. If your block ignores these arguments, learned pos embeddings will still work.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 64,
        patch_size: int | tuple[int, int] = 8,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        patch_norm: bool = False,
        drop_rate: float = 0.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        pos_embed_type: Literal["learned", "rope", "none"] = "learned",
        use_cls_token: bool = True,
        num_register_tokens: int = 0,
        rope_base: float = 10000.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        strict_img_size: bool = False,
    ):
        super().__init__()

        if pos_embed_type not in {"learned", "rope", "none"}:
            raise ValueError(
                f"pos_embed_type must be one of 'learned', 'rope', or 'none'. "
                f"Got {pos_embed_type}."
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.pos_embed_type = pos_embed_type
        self.use_cls_token = use_cls_token
        self.num_register_tokens = num_register_tokens

        self.num_special_tokens = int(use_cls_token) + num_register_tokens

        patch_norm_layer = norm_layer if patch_norm else None

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=patch_norm_layer,
            strict_img_size=strict_img_size,
        )

        self.num_patches = self.patch_embed.num_patches
        self.base_grid_size = self.patch_embed.grid_size

        # ----------------------------------------------------
        # Special tokens: CLS + registers
        # ----------------------------------------------------

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_register_tokens, embed_dim)
            )
        else:
            self.register_tokens = None

        # ----------------------------------------------------
        # Positional embeddings
        # ----------------------------------------------------
        # Learned absolute embeddings are added only when pos_embed_type="learned".
        # RoPE is applied inside attention, so no absolute pos_embed is added here.

        if pos_embed_type == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    self.num_special_tokens + self.num_patches,
                    embed_dim,))

        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(drop_rate)

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        drop_path_values = (
            torch.linspace(0, drop_path_rate, depth).tolist()
            if depth > 0
            else [])

        use_rope = pos_embed_type == "rope"

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_dropout=attn_dropout,
                    proj_dropout=proj_dropout,
                    mlp_dropout=mlp_dropout,
                    drop_path=drop_path_values[idx],
                    use_rope=use_rope,
                    rope_base=rope_base,
                    norm_layer=norm_layer,
                )
                for idx in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Optional classification head. For DINO pretraining, usually num_classes=0.
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        self._init_weights()

    # ========================================================
    # Initialization
    # ========================================================

    def _init_weights(self):
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0.0)

    # ========================================================
    # Positional embedding interpolation
    # ========================================================

    def _interpolate_pos_encoding(
        self,
        grid_size: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,) -> torch.Tensor:
        """
        Interpolates learned absolute positional embeddings to a target patch grid.

        This is needed because DINO uses multi-crop:
            global crop: 64x64 -> 8x8 patches
            local crop: 32x32  -> 4x4 patches

        Special tokens are not interpolated.
        Patch tokens are reshaped as a 2D grid and bicubically interpolated.
        """

        if self.pos_embed is None:
            raise RuntimeError(
                "_interpolate_pos_encoding was called but self.pos_embed is None.")

        target_h, target_w = grid_size
        base_h, base_w = self.base_grid_size

        num_target_patches = target_h * target_w

        special_pos = self.pos_embed[:, : self.num_special_tokens]
        patch_pos = self.pos_embed[:, self.num_special_tokens :]

        if (target_h, target_w) == (base_h, base_w):
            return self.pos_embed.to(device=device, dtype=dtype)

        patch_pos = patch_pos.reshape(
            1,
            base_h,
            base_w,
            self.embed_dim).permute(0, 3, 1, 2)


        patch_pos = F.interpolate(
            patch_pos,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False)

        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
            1,
            num_target_patches,
            self.embed_dim)

        pos_embed = torch.cat([special_pos, patch_pos], dim=1)

        return pos_embed.to(device=device, dtype=dtype)

    # ========================================================
    # Token preparation
    # ========================================================

    def _prepare_tokens(
        self,
        x: torch.Tensor) -> tuple[torch.Tensor, Tuple[int, int]]:
        """
        Converts image tensor into full ViT token sequence:

            image -> patch tokens
            prepend CLS token if enabled
            prepend register tokens if enabled
            add learned positional embeddings if enabled

        Token order:
            [CLS] [REG_1] ... [REG_R] [PATCH_1] ... [PATCH_N]

        This order is important because RoPE is applied only to patch tokens.
        """

        patch_out = self.patch_embed(x)

        # Compatible with either:
        #   old PatchEmbedding returning (tokens, grid_size)
        #   new PatchEmbedding returning PatchEmbeddingOutput
        if isinstance(patch_out, tuple):
            patch_tokens, grid_size = patch_out
        else:
            patch_tokens = patch_out.tokens
            grid_size = patch_out.grid_size

        batch_size = patch_tokens.shape[0]

        tokens = []

        if self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens.append(cls)

        if self.register_tokens is not None:
            registers = self.register_tokens.expand(batch_size, -1, -1)
            tokens.append(registers)

        tokens.append(patch_tokens)

        x = torch.cat(tokens, dim=1)

        if self.pos_embed_type == "learned":
            pos_embed = self._interpolate_pos_encoding(
                grid_size=grid_size,
                dtype=x.dtype,
                device=x.device)

            x = x + pos_embed

        x = self.pos_drop(x)

        return x, grid_size

    # ========================================================
    # Forward features
    # ========================================================

    def forward_features(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = True,
        return_attention: bool = False) -> Dict[str, Any]:
        """
        Returns a dictionary useful for DINO:

            {
                "x_norm": full normalized sequence,
                "cls": CLS token or pooled token,
                "registers": register tokens or None,
                "patches": patch tokens,
                "grid_size": patch grid,
                "attentions": optional list of attention maps
            }
        """

        x, grid_size = self._prepare_tokens(x)

        attentions = [] if return_attention else None

        for block in self.blocks:
            if return_attention:
                x, attn = block(
                    x,
                    grid_size=grid_size,
                    num_special_tokens=self.num_special_tokens,
                    return_attention=True)

                attentions.append(attn)
            else:
                x = block(
                    x,
                    grid_size=grid_size,
                    num_special_tokens=self.num_special_tokens,
                    return_attention=False)

        x = self.norm(x)

        idx = 0

        if self.use_cls_token:
            cls = x[:, idx]
            idx += 1
        else:
            # If no CLS token exists, use mean pooling over patch tokens.
            cls = x[:, self.num_special_tokens :].mean(dim=1)

        if self.num_register_tokens > 0:
            registers = x[:, idx : idx + self.num_register_tokens]
            idx += self.num_register_tokens
        else:
            registers = None

        patch_tokens = x[:, idx:]

        out = {
            "cls": cls,                        # [B, D]
            "registers": registers,            # [B, R, D] or None
            "patches": patch_tokens,           # [B, N, D]
            "grid_size": grid_size,             # (Gh, Gw)
        }

        if return_all_tokens:
            out["x_norm"] = x                  # [B, T, D]

        if return_attention:
            out["attentions"] = attentions

        return out

    # ========================================================
    # Standard forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_all_tokens: bool = True,
        return_attention: bool = False):

        features = self.forward_features(
            x,
            return_all_tokens=return_all_tokens,
            return_attention=return_attention)

        if return_features:
            return features

        logits = self.head(features["cls"])

        return logits
