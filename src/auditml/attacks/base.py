"""Abstract base class for all AuditML privacy attacks.

Every concrete attack (threshold MIA, shadow MIA, model inversion,
attribute inference) inherits from ``BaseAttack``.  This guarantees a
consistent API so the CLI and report generator can treat all attacks
uniformly.

The class also provides **shared utility methods** that multiple attacks
need: extracting softmax probabilities from the model, computing
per-sample loss values, and calculating standard evaluation metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from auditml.attacks.results import AttackResult
from auditml.config.schema import AuditMLConfig


class BaseAttack(ABC):
    """Abstract base for all privacy attacks.

    Parameters
    ----------
    target_model:
        The trained model being attacked. Must be in ``eval()`` mode.
    config:
        Full AuditML configuration (the attack reads its own section).
    device:
        Torch device the model lives on.
    """

    attack_name: str = "base"  # overridden by each subclass

    def __init__(
        self,
        target_model: nn.Module,
        config: AuditMLConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.target_model = target_model
        self.target_model.eval()  # always eval mode for attacks
        self.config = config
        self.device = torch.device(device)
        self.result: AttackResult | None = None

    # ------------------------------------------------------------------
    # Abstract methods — each concrete attack MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        member_loader: DataLoader,
        nonmember_loader: DataLoader,
    ) -> AttackResult:
        """Execute the attack.

        Parameters
        ----------
        member_loader:
            DataLoader over samples the target model WAS trained on.
        nonmember_loader:
            DataLoader over samples the target model was NOT trained on.

        Returns
        -------
        AttackResult
            Predictions, ground truth, and confidence scores.
        """
        ...

    # ------------------------------------------------------------------
    # Evaluation — shared across all attacks
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, float]:
        """Compute standard metrics from the most recent ``run()``.

        Returns
        -------
        dict
            Keys: accuracy, precision, recall, f1, auc_roc, auc_pr,
            tpr_at_1fpr, tpr_at_01fpr.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called yet.
        """
        if self.result is None:
            raise RuntimeError("Call run() before evaluate().")
        return self._compute_metrics(
            self.result.predictions,
            self.result.ground_truth,
            self.result.confidence_scores,
        )

    # ------------------------------------------------------------------
    # Shared utility methods — used by multiple attacks
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_model_outputs(
        self, loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the target model on every sample in *loader*.

        Returns
        -------
        (probabilities, logits, labels)
            - probabilities: ``(N, num_classes)`` softmax output
            - logits: ``(N, num_classes)`` raw model output
            - labels: ``(N,)`` true class labels from the dataset
        """
        all_probs: list[np.ndarray] = []
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            logits = self.target_model(inputs)
            probs = F.softmax(logits, dim=1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(targets.numpy())

        return (
            np.concatenate(all_probs),
            np.concatenate(all_logits),
            np.concatenate(all_labels),
        )

    @torch.no_grad()
    def get_loss_values(self, loader: DataLoader) -> np.ndarray:
        """Compute **per-sample** cross-entropy loss for every sample.

        This is critical for threshold-based MIA: training samples
        typically have lower loss because the model has seen them before.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — one loss value per sample.
        """
        criterion = nn.CrossEntropyLoss(reduction="none")  # per-sample
        all_losses: list[np.ndarray] = []

        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            logits = self.target_model(inputs)
            losses = criterion(logits, targets)
            all_losses.append(losses.cpu().numpy())

        return np.concatenate(all_losses)

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        confidence_scores: np.ndarray,
    ) -> dict[str, float]:
        """Compute a comprehensive set of binary classification metrics.

        Parameters
        ----------
        predictions:
            Binary array (0/1) — the attack's prediction.
        ground_truth:
            Binary array (0/1) — the true membership label.
        confidence_scores:
            Continuous score — higher means "more likely member".

        Returns
        -------
        dict with keys:
            accuracy, precision, recall, f1, auc_roc, auc_pr,
            tpr_at_1fpr, tpr_at_01fpr
        """
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(ground_truth, predictions)),
            "precision": float(precision_score(ground_truth, predictions, zero_division=0)),
            "recall": float(recall_score(ground_truth, predictions, zero_division=0)),
            "f1": float(f1_score(ground_truth, predictions, zero_division=0)),
        }

        # ROC-based metrics (need continuous scores)
        if len(np.unique(ground_truth)) == 2:
            fpr, tpr, _ = roc_curve(ground_truth, confidence_scores)
            metrics["auc_roc"] = float(roc_auc_score(ground_truth, confidence_scores))

            # TPR at specific FPR thresholds — realistic adversary constraints
            metrics["tpr_at_1fpr"] = float(np.interp(0.01, fpr, tpr))
            metrics["tpr_at_01fpr"] = float(np.interp(0.001, fpr, tpr))

            # Precision-Recall AUC
            prec_arr, rec_arr, _ = precision_recall_curve(ground_truth, confidence_scores)
            metrics["auc_pr"] = float(auc(rec_arr, prec_arr))
        else:
            # Edge case: if all samples have the same label, AUC is undefined
            metrics["auc_roc"] = 0.0
            metrics["tpr_at_1fpr"] = 0.0
            metrics["tpr_at_01fpr"] = 0.0
            metrics["auc_pr"] = 0.0

        return metrics
