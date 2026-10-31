from __future__ import annotations

import torch
import torch.nn as nn


class BaseCognitiveDiagnosisModel(nn.Module):
    """Base interface for cognitive diagnosis models."""

    def forward(
        self,
        student_ids: torch.LongTensor,
        exercise_ids: torch.LongTensor,
        q_vectors: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy(predictions, labels)

    def predict(
        self,
        student_ids: torch.LongTensor,
        exercise_ids: torch.LongTensor,
        q_vectors: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(student_ids, exercise_ids, q_vectors)
