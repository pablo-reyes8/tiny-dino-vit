"""Implement multi-head self-attention with optional 2D RoPE support."""

import torch
import torch.nn as nn

from src.model.rope import *

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional 2D RoPE over patch tokens.

    Input:
        x: [B, T, D]

    If use_rope=True:
        - grid_size must be provided.
        - num_special_tokens tells how many leading tokens are not spatial:
            CLS + register tokens.
        - RoPE is applied only to patch tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim must be divisible by num_heads. Got dim={dim}, num_heads={num_heads}."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_dropout)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        if use_rope:
            self.rope = RotaryEmbedding2D(
                head_dim=self.head_dim,
                base=rope_base,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
        num_special_tokens: int = 1,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(dim=0)
        # q, k, v: [B, H, T, head_dim]

        if self.use_rope:
            if grid_size is None:
                raise ValueError("grid_size must be provided when use_rope=True.")

            num_patch_tokens = grid_size[0] * grid_size[1]
            expected_tokens = num_special_tokens + num_patch_tokens

            if num_tokens != expected_tokens:
                raise ValueError(
                    f"Token mismatch: got num_tokens={num_tokens}, but "
                    f"num_special_tokens={num_special_tokens} and grid_size={grid_size} "
                    f"imply {expected_tokens} tokens."
                )

            q_special = q[:, :, :num_special_tokens, :]
            q_patch = q[:, :, num_special_tokens:, :]

            k_special = k[:, :, :num_special_tokens, :]
            k_patch = k[:, :, num_special_tokens:, :]

            q_patch, k_patch = self.rope(
                q=q_patch,
                k=k_patch,
                grid_size=grid_size,
            )

            q = torch.cat([q_special, q_patch], dim=2)
            k = torch.cat([k_special, k_patch], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, dim)

        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attention:
            return out, attn

        return out
