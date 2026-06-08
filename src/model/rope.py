from typing import Optional, Callable, Tuple
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates pairs of features:
        [x1, x2] -> [-x2, x1]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rotated = torch.stack((-x_odd, x_even), dim=-1)
    x_rotated = x_rotated.flatten(-2)

    return x_rotated


class RotaryEmbedding2D(nn.Module):
    """
    2D axial RoPE for ViT patch tokens.

    This module builds rotary frequencies for a patch grid (Gh, Gw) and applies
    them to q and k patch tokens.

    Expected q/k shape:
        [B, num_heads, N_patch, head_dim]

    RoPE is applied only to patch tokens, not to CLS/register tokens.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim % 4 != 0:
            raise ValueError(
                f"For 2D RoPE, head_dim must be divisible by 4. Got head_dim={head_dim}."
            )

        self.head_dim = head_dim
        self.base = base

        # Half of head_dim goes to height, half to width.
        self.axis_dim = head_dim // 2

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, self.axis_dim, 2).float() / self.axis_dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_2d_cos_sin(
        self,
        grid_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gh, gw = grid_size

        y = torch.arange(gh, device=device, dtype=self.inv_freq.dtype)
        x = torch.arange(gw, device=device, dtype=self.inv_freq.dtype)

        yy, xx = torch.meshgrid(y, x, indexing="ij")

        yy = yy.reshape(-1)  # [N]
        xx = xx.reshape(-1)  # [N]

        freqs_y = torch.einsum("n,d->nd", yy, self.inv_freq)  # [N, axis_dim/2]
        freqs_x = torch.einsum("n,d->nd", xx, self.inv_freq)  # [N, axis_dim/2]

        emb_y = torch.cat([freqs_y, freqs_y], dim=-1)  # [N, axis_dim]
        emb_x = torch.cat([freqs_x, freqs_x], dim=-1)  # [N, axis_dim]

        emb = torch.cat([emb_y, emb_x], dim=-1)  # [N, head_dim]

        cos = emb.cos().to(dtype=dtype)[None, None, :, :]  # [1, 1, N, head_dim]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]  # [1, 1, N, head_dim]

        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k:
            [B, num_heads, N_patch, head_dim]
        """

        expected_num_patches = grid_size[0] * grid_size[1]

        if q.shape[-2] != expected_num_patches:
            raise ValueError(
                f"q has {q.shape[-2]} patch tokens, but grid_size={grid_size} "
                f"implies {expected_num_patches} patches."
            )

        if k.shape[-2] != expected_num_patches:
            raise ValueError(
                f"k has {k.shape[-2]} patch tokens, but grid_size={grid_size} "
                f"implies {expected_num_patches} patches."
            )

        cos, sin = self._build_2d_cos_sin(
            grid_size=grid_size,
            device=q.device,
            dtype=q.dtype,
        )

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        return q, k

