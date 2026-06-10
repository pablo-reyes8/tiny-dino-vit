"""Compose ViT attention and feed-forward layers into a Transformer block."""

from src.model.vit_utils import * 
from src.model.attention import * 

class TransformerEncoderBlock(nn.Module):
    """
    PreNorm Transformer encoder block used in ViT.

    Supports optional RoPE through the attention module.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path: float = 0.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            use_rope=use_rope,
            rope_base=rope_base,
        )

        self.drop_path1 = DropPath(drop_path)

        self.norm2 = norm_layer(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=mlp_dropout,
        )

        self.drop_path2 = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
        num_special_tokens: int = 1,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attention:
            attn_out, attn = self.attn(
                self.norm1(x),
                grid_size=grid_size,
                num_special_tokens=num_special_tokens,
                return_attention=True,
            )

            x = x + self.drop_path1(attn_out)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))

            return x, attn

        x = x + self.drop_path1(
            self.attn(
                self.norm1(x),
                grid_size=grid_size,
                num_special_tokens=num_special_tokens,
                return_attention=False,
            )
        )

        x = x + self.drop_path2(
            self.mlp(
                self.norm2(x)
            )
        )

        return x
