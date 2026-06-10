"""Compute Gram-matrix consistency loss over student and teacher patch tokens."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOGramLoss(nn.Module):
    """
    Patch-token Gram matching loss for DINO-style training.

    Expected inputs are lists of model output dicts containing:
        output[patch_key] -> Tensor [B, N, D]

    The loss compares per-image patch similarity matrices, so student and
    teacher feature dimensions may differ as long as the patch count matches.
    """

    def __init__(
        self,
        patch_key: str = "patches",
        normalize_features: bool = True,
        loss_type: str = "mse",
        eps: float = 1e-6,
    ):
        super().__init__()

        if loss_type not in {"mse", "smooth_l1", "l1"}:
            raise ValueError(
                f"loss_type must be 'mse', 'smooth_l1' or 'l1', got {loss_type}."
            )

        self.patch_key = patch_key
        self.normalize_features = bool(normalize_features)
        self.loss_type = loss_type
        self.eps = float(eps)

    def _extract_patches(self, output: Dict[str, Any], index: int, side: str) -> torch.Tensor:
        if self.patch_key not in output:
            raise KeyError(
                f"{side}_outputs[{index}] does not contain key '{self.patch_key}'. "
                f"Available keys: {list(output.keys())}"
            )

        patches = output[self.patch_key]
        if not torch.is_tensor(patches):
            raise TypeError(
                f"{side}_outputs[{index}]['{self.patch_key}'] must be a Tensor, "
                f"got {type(patches)}."
            )

        if patches.ndim != 3:
            raise ValueError(
                f"{side}_outputs[{index}]['{self.patch_key}'] must have shape [B, N, D], "
                f"got {tuple(patches.shape)}."
            )

        return patches

    def _gram(self, patches: torch.Tensor) -> torch.Tensor:
        x = patches.float()

        if self.normalize_features:
            x = F.normalize(x, dim=-1, eps=self.eps)

        return x @ x.transpose(-1, -2)

    def _pair_loss(self, student_gram: torch.Tensor, teacher_gram: torch.Tensor) -> torch.Tensor:
        target = teacher_gram.detach().to(
            device=student_gram.device,
            dtype=student_gram.dtype,
        )

        if self.loss_type == "mse":
            return F.mse_loss(student_gram, target)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(student_gram, target)
        return F.l1_loss(student_gram, target)

    def forward(
        self,
        *,
        student_outputs: List[Dict[str, Any]],
        teacher_outputs: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        if len(student_outputs) != len(teacher_outputs):
            raise ValueError(
                f"DINOGramLoss expects same number of student/teacher outputs. "
                f"Got {len(student_outputs)} student and {len(teacher_outputs)} teacher."
            )

        if len(student_outputs) == 0:
            raise ValueError("DINOGramLoss received empty output lists.")

        pair_losses = []

        for index, (student_output, teacher_output) in enumerate(
            zip(student_outputs, teacher_outputs)
        ):
            student_patches = self._extract_patches(student_output, index, "student")
            teacher_patches = self._extract_patches(teacher_output, index, "teacher")

            if student_patches.shape[:2] != teacher_patches.shape[:2]:
                raise ValueError(
                    f"DINOGramLoss patch shape mismatch at pair {index}: "
                    f"student={tuple(student_patches.shape)}, "
                    f"teacher={tuple(teacher_patches.shape)}."
                )

            student_gram = self._gram(student_patches)
            teacher_gram = self._gram(teacher_patches)
            pair_losses.append(self._pair_loss(student_gram, teacher_gram))

        loss = torch.stack(pair_losses).mean()

        return {
            "loss": loss,
            "pair_loss_mean": loss.detach(),
        }
