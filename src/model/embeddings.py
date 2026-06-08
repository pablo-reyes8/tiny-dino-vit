from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

Size2D = Union[int, Tuple[int, int]]


def to_2tuple(x: Size2D) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, tuple) and len(x) == 2:
        return x
    raise ValueError(f"Expected int or tuple[int, int], got {x}")

@dataclass
class PatchEmbeddingOutput:
    tokens: torch.Tensor              # [B, N, D]
    grid_size: Tuple[int, int]         # (Gh, Gw)
    num_patches: int                   # Gh * Gw
    patch_size: Tuple[int, int]        # (Ph, Pw)


class PatchEmbedding(nn.Module):
    """
    Image to patch-token embedding.

    This module only performs:
        image -> Conv2d patch projection -> flattened patch tokens

    It does NOT add:
        - CLS token
        - register tokens
        - learned positional embeddings
        - RoPE

    Reason:
        CLS/register/positional logic belongs to the ViT backbone, not to the
        patch embedding layer. For RoPE, we return grid_size so the attention
        module can construct/apply 2D rotary positions correctly.

    Parameters
    ----------
    img_size:
        Reference image size. Used only to compute the nominal grid_size and
        num_patches. Forward can still accept other divisible sizes if
        strict_img_size=False.

    patch_size:
        Patch size used by the Conv2d projection.

    in_chans:
        Number of image channels.

    embed_dim:
        Token embedding dimension.

    norm_layer:
        Optional normalization over patch tokens.

    flatten:
        If True, returns [B, N, D].
        If False, returns [B, D, Gh, Gw].

    strict_img_size:
        If True, forward requires input size == img_size.
        If False, any input divisible by patch_size is allowed.
        For DINO multi-crop, this should usually be False.
    """

    def __init__(
        self,
        img_size: Size2D = 64,
        patch_size: Size2D = 8,
        in_chans: int = 3,
        embed_dim: int = 192,
        norm_layer: Optional[type[nn.Module]] = None,
        flatten: bool = True,
        strict_img_size: bool = False,
        bias: bool = True,):

        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans

        img_h, img_w = self.img_size
        patch_h, patch_w = self.patch_size

        if img_h % patch_h != 0 or img_w % patch_w != 0:
            raise ValueError(
                f"img_size={self.img_size} must be divisible by "
                f"patch_size={self.patch_size}."
            )

        self.grid_size = (
            img_h // patch_h,
            img_w // patch_w,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )

        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        return_metadata: bool = True,
    ) -> PatchEmbeddingOutput | torch.Tensor:
        """
        Parameters
        ----------
        x:
            Image tensor of shape [B, C, H, W].

        return_metadata:
            If True, returns PatchEmbeddingOutput.
            If False, returns only patch tokens.

        Returns
        -------
        PatchEmbeddingOutput or Tensor
            tokens: [B, N, D] if flatten=True.
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected input of shape [B, C, H, W], got {tuple(x.shape)}."
            )

        batch_size, channels, height, width = x.shape

        if channels != self.in_chans:
            raise ValueError(
                f"Expected {self.in_chans} input channels, got {channels}."
            )

        if self.strict_img_size and (height, width) != self.img_size:
            raise ValueError(
                f"Input image size {(height, width)} does not match "
                f"configured img_size={self.img_size}. "
                f"Set strict_img_size=False to allow multi-crop inputs.")

        patch_h, patch_w = self.patch_size

        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Input image size {(height, width)} must be divisible by "
                f"patch_size={self.patch_size}.")

        x = self.proj(x)  # [B, D, Gh, Gw]

        grid_size = (x.shape[-2], x.shape[-1])
        num_patches = grid_size[0] * grid_size[1]

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, N, D]
            x = self.norm(x)

        if not return_metadata:
            return x

        return PatchEmbeddingOutput(
            tokens=x,
            grid_size=grid_size,
            num_patches=num_patches,
            patch_size=self.patch_size,
        )