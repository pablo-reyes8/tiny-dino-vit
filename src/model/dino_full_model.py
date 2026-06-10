"""Assemble student-teacher DINO models and projection heads."""

from src.model.dino_head import * 
import copy


class DINOModel(nn.Module):
    """
    Full DINO model:

        image
        -> VisionTransformer backbone
        -> CLS token
        -> DINOHead
        -> logits

    Also returns patch tokens for Gram loss.

    Expected backbone output when return_features=True:
        {
            "cls": Tensor[B, D],
            "patches": Tensor[B, N, D],
            "registers": Optional[Tensor[B, R, D]],
            "grid_size": tuple[int, int],
            ...
        }
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        cls_key: str = "cls",
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.cls_key = cls_key

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True,
        return_all_tokens: bool = True,
        return_attention: bool = False,
    ) -> Dict[str, Any] | torch.Tensor:
        features = self.backbone(
            x,
            return_features=True,
            return_all_tokens=return_all_tokens,
            return_attention=return_attention,
        )

        if not isinstance(features, dict):
            raise TypeError(
                f"Backbone must return a dict when return_features=True. Got {type(features)}."
            )

        if self.cls_key not in features:
            raise KeyError(
                f"Backbone output does not contain cls_key='{self.cls_key}'. "
                f"Available keys: {list(features.keys())}"
            )

        cls = features[self.cls_key]
        logits = self.head(cls)

        features["logits"] = logits

        if return_features:
            return features

        return logits
    
######################################################
################## WRAPPER ###########################
######################################################


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Cuenta parámetros del modelo.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Congela todos los parámetros de un modelo.
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_dino_bundle(CFG: dict) -> dict:
    """
    Build a complete DINO bundle.

    Returns
    -------
    bundle_dino:
        {
            "student": DINOModel,
            "teacher": DINOModel,
            "config": CFG,
            "model_info": metadata,
            "device": device,
        }

    Notes
    -----
    - student contains backbone + DINOHead.
    - teacher is initialized as a deepcopy of student.
    - teacher is frozen.
    - teacher is updated with EMA during training.
    """

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = CFG.get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    device = torch.device(device)

    # --------------------------------------------------------
    # Configs
    # --------------------------------------------------------

    model_cfg = CFG["model"]
    head_cfg = CFG.get("dino_head", {})

    embed_dim = int(model_cfg.get("embed_dim", 192))

    # --------------------------------------------------------
    # Student backbone
    # --------------------------------------------------------

    student_backbone = VisionTransformer(
        img_size=model_cfg.get("img_size", 64),
        patch_size=model_cfg.get("patch_size", 8),
        in_chans=model_cfg.get("in_chans", 3),

        # For DINO pretraining, classification head is not used.
        num_classes=0,

        embed_dim=embed_dim,
        depth=model_cfg.get("depth", 6),
        num_heads=model_cfg.get("num_heads", 3),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        qkv_bias=model_cfg.get("qkv_bias", True),
        patch_norm=model_cfg.get("patch_norm", False),
        drop_rate=model_cfg.get("drop_rate", 0.0),
        attn_dropout=model_cfg.get("attn_dropout", 0.0),
        proj_dropout=model_cfg.get("proj_dropout", 0.0),
        mlp_dropout=model_cfg.get("mlp_dropout", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
        pos_embed_type=model_cfg.get("pos_embed_type", "learned"),
        use_cls_token=model_cfg.get("use_cls_token", True),
        num_register_tokens=model_cfg.get("num_register_tokens", 0),
        rope_base=model_cfg.get("rope_base", 10000.0),
        strict_img_size=model_cfg.get("strict_img_size", False),
    )

    # --------------------------------------------------------
    # Student DINO head
    # --------------------------------------------------------

    student_head = DINOHead(
        in_dim=head_cfg.get("in_dim", embed_dim),
        out_dim=head_cfg.get("out_dim", 8192),
        hidden_dim=head_cfg.get("hidden_dim", 2048),
        bottleneck_dim=head_cfg.get("bottleneck_dim", 256),
        num_layers=head_cfg.get("num_layers", 3),
        use_bn=head_cfg.get("use_bn", False),
        norm_last_layer=head_cfg.get("norm_last_layer", True),
        bias=head_cfg.get("bias", True),
    )

    # --------------------------------------------------------
    # Full student model
    # --------------------------------------------------------

    student = DINOModel(
        backbone=student_backbone,
        head=student_head,
        cls_key=head_cfg.get("cls_key", "cls"),
    )

    # --------------------------------------------------------
    # Teacher = exact copy of full student
    # --------------------------------------------------------

    teacher = copy.deepcopy(student)
    freeze_model(teacher)

    student = student.to(device)
    teacher = teacher.to(device)

    student.train()
    teacher.eval()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    model_info = {
        "architecture": "DINOModel(VisionTransformer + DINOHead)",

        "backbone": {
            "type": "Plain Vision Transformer",
            "img_size": model_cfg.get("img_size", 64),
            "patch_size": model_cfg.get("patch_size", 8),
            "embed_dim": embed_dim,
            "depth": model_cfg.get("depth", 6),
            "num_heads": model_cfg.get("num_heads", 3),
            "mlp_ratio": model_cfg.get("mlp_ratio", 4.0),
            "pos_embed_type": model_cfg.get("pos_embed_type", "learned"),
            "use_cls_token": model_cfg.get("use_cls_token", True),
            "num_register_tokens": model_cfg.get("num_register_tokens", 0),
        },

        "head": {
            "type": "DINOHead",
            "in_dim": head_cfg.get("in_dim", embed_dim),
            "out_dim": head_cfg.get("out_dim", 8192),
            "hidden_dim": head_cfg.get("hidden_dim", 2048),
            "bottleneck_dim": head_cfg.get("bottleneck_dim", 256),
            "num_layers": head_cfg.get("num_layers", 3),
            "use_bn": head_cfg.get("use_bn", False),
            "norm_last_layer": head_cfg.get("norm_last_layer", True),
        },

        "student_total_params": count_parameters(student, trainable_only=False),
        "student_trainable_params": count_parameters(student, trainable_only=True),

        "teacher_total_params": count_parameters(teacher, trainable_only=False),
        "teacher_trainable_params": count_parameters(teacher, trainable_only=True),
    }

    # --------------------------------------------------------
    # Bundle final
    # --------------------------------------------------------

    bundle_dino = {
        "student": student,
        "teacher": teacher,
        "config": CFG,
        "model_info": model_info,
        "device": device,
    }

    return bundle_dino
