"""Provide reusable Vision Transformer utility layers and compatibility exports."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from src.model.attention import MultiHeadSelfAttention

class DropPath(nn.Module):
    """
    Stochastic depth per sample.

    This is commonly used in ViT/DeiT/DINO-style models.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob

        if keep_prob <= 0.0:
            return torch.zeros_like(x)

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor = random_tensor.div(keep_prob)

        return x * random_tensor


class FeedForward(nn.Module):
    """
    Standard ViT MLP block.

    Structure:
        Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        bias: bool = True,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x
