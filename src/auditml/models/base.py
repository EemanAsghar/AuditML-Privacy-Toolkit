"""Abstract base class for all AuditML models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Base class that all AuditML target/shadow models must extend.

    Subclasses must implement ``forward`` and ``get_features``.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass returning logits of shape ``(batch, num_classes)``."""
        ...

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate-layer feature vector.

        Used by membership inference attacks to analyse the model's internal
        representations. Shape: ``(batch, feature_dim)``.
        """
        ...


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Returns
    -------
    dict
        Keys: ``"total"``, ``"trainable"``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
