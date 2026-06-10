"""Compute the standard DINO cross-view teacher-student loss."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    DINO self-distillation loss with EMA teacher centering.
    """

    def __init__(
        self,
        out_dim: int,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, self.out_dim))

    @torch.no_grad()
    def update_center(self, teacher_logits: List[torch.Tensor]) -> None:
        teacher_batch = torch.cat([x.detach() for x in teacher_logits], dim=0)
        batch_center = teacher_batch.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)

    def forward(
        self,
        *,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
    ) -> Dict[str, torch.Tensor]:
        if len(student_logits) == 0 or len(teacher_logits) == 0:
            raise ValueError("DINOLoss expects non-empty student and teacher logits.")

        student_log_probs = [F.log_softmax(s.float() / float(student_temp), dim=-1) for s in student_logits]
        teacher_probs = [
            F.softmax((t.float() - self.center.to(t.device, t.dtype)) / float(teacher_temp), dim=-1).detach()
            for t in teacher_logits
        ]

        total_loss = student_log_probs[0].new_tensor(0.0)
        n_terms = 0

        for teacher_idx, teacher_prob in enumerate(teacher_probs):
            for student_idx, student_log_prob in enumerate(student_log_probs):
                if student_idx == teacher_idx:
                    continue
                total_loss = total_loss + -(teacher_prob * student_log_prob).sum(dim=-1).mean()
                n_terms += 1

        if n_terms == 0:
            raise ValueError("DINOLoss found no valid cross-view pairs.")

        loss = total_loss / n_terms
        self.update_center(teacher_logits)

        return {"loss": loss, "num_terms": loss.new_tensor(float(n_terms))}
