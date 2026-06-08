from src.model.vit_model import * 

class DINOHead(nn.Module):
    """
    Projection head used for DINO-style self-distillation.

    Input:
        x: [B, in_dim]

    Output:
        logits: [B, out_dim]

    Canonical structure:
        MLP -> bottleneck -> L2 normalize -> weight-normalized linear layer
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_layers: int = 3,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_layers = int(num_layers)
        self.use_bn = bool(use_bn)
        self.norm_last_layer = bool(norm_last_layer)

        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim, bias=bias)

        else:
            layers = []

            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))

            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.GELU())

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))

                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))

                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))

            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

        self.last_layer.weight_g.data.fill_(1.0)

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)

        return x