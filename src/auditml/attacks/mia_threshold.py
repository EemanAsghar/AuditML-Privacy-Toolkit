"""Threshold-based Membership Inference Attack.

The simplest MIA approach: exploit the fact that a model behaves
differently on data it was trained on (members) versus data it has
never seen (non-members).  Specifically, for members the model tends to
produce **lower loss**, **higher confidence**, and **lower prediction
entropy**.

The attack computes one of these metrics for every sample, picks an
optimal threshold, and classifies samples on one side as members and
the other side as non-members.

References
----------
Yeom et al., "Privacy Risk in Machine Learning: Analyzing the Connection
to Overfitting", IEEE CSF 2018.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from auditml.attacks.base import BaseAttack
from auditml.attacks.results import AttackResult
from auditml.config.schema import AuditMLConfig


class ThresholdMIA(BaseAttack):
    """Threshold-based Membership Inference Attack.

    Supports three signal metrics:

    - ``"loss"`` — per-sample cross-entropy loss (lower → more likely member)
    - ``"confidence"`` — max softmax probability (higher → more likely member)
    - ``"entropy"`` — prediction entropy (lower → more likely member)

    Parameters
    ----------
    target_model:
        The trained model to attack.
    config:
        AuditML config. Reads ``config.attack_params.mia_threshold`` for
        ``metric`` and ``percentile`` settings.
    device:
        Torch device.
    """

    attack_name = "mia_threshold"

    def __init__(self, target_model, config, device="cpu"):
        super().__init__(target_model, config, device)
        params = config.attack_params.mia_threshold
        self.metric = params.metric          # "loss", "confidence", or "entropy"
        self.percentile = params.percentile  # for fixed-percentile threshold

        # Intermediate values stored after run() for analysis/plotting
        self.member_scores: np.ndarray | None = None
        self.nonmember_scores: np.ndarray | None = None
        self.threshold: float | None = None
        # Class labels for per-class evaluation (stored during run())
        self.member_labels: np.ndarray | None = None
        self.nonmember_labels: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Main attack logic
    # ------------------------------------------------------------------

    def run(
        self,
        member_loader: DataLoader,
        nonmember_loader: DataLoader,
    ) -> AttackResult:
        """Execute the threshold MIA.

        Steps:
        1. Compute the chosen metric (loss/confidence/entropy) for every
           member and non-member sample.
        2. Find the optimal threshold that maximises accuracy.
        3. Classify each sample as member or non-member using that
           threshold.

        Parameters
        ----------
        member_loader:
            DataLoader for samples the model WAS trained on.
        nonmember_loader:
            DataLoader for samples the model was NOT trained on.

        Returns
        -------
        AttackResult
        """
        # Step 1: Compute signal metric for both groups
        self.member_scores = self._compute_signal(member_loader)
        self.nonmember_scores = self._compute_signal(nonmember_loader)

        # Store class labels for per-class evaluation
        self.member_labels = self._extract_labels(member_loader)
        self.nonmember_labels = self._extract_labels(nonmember_loader)

        # Combine into single arrays
        all_scores = np.concatenate([self.member_scores, self.nonmember_scores])
        ground_truth = np.concatenate([
            np.ones(len(self.member_scores)),    # 1 = member
            np.zeros(len(self.nonmember_scores)),  # 0 = non-member
        ])

        # Step 2: Find optimal threshold
        self.threshold = self._find_optimal_threshold(all_scores, ground_truth)

        # Step 3: Classify using threshold
        predictions = self._apply_threshold(all_scores, self.threshold)

        # Confidence scores: how "member-like" each sample is
        # We normalise so that higher = more likely member, regardless of metric
        confidence_scores = self._scores_to_confidence(all_scores)

        self.result = AttackResult(
            predictions=predictions,
            ground_truth=ground_truth,
            confidence_scores=confidence_scores,
            attack_name=self.attack_name,
            metadata={
                "metric": self.metric,
                "threshold": self.threshold,
                "member_mean": float(np.mean(self.member_scores)),
                "nonmember_mean": float(np.mean(self.nonmember_scores)),
                "member_std": float(np.std(self.member_scores)),
                "nonmember_std": float(np.std(self.nonmember_scores)),
            },
        )
        return self.result

    # ------------------------------------------------------------------
    # Signal computation — the core of the attack
    # ------------------------------------------------------------------

    def _compute_signal(self, loader: DataLoader) -> np.ndarray:
        """Compute the chosen metric for every sample in *loader*.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — one score per sample.
        """
        if self.metric == "loss":
            return self._compute_loss_signal(loader)
        elif self.metric == "confidence":
            return self._compute_confidence_signal(loader)
        elif self.metric == "entropy":
            return self._compute_entropy_signal(loader)
        else:
            raise ValueError(
                f"Unknown metric {self.metric!r}. "
                f"Choose from: 'loss', 'confidence', 'entropy'"
            )

    def _compute_loss_signal(self, loader: DataLoader) -> np.ndarray:
        """Per-sample cross-entropy loss.

        Members typically have LOWER loss because the model was trained
        to minimise loss on them.
        """
        return self.get_loss_values(loader)

    def _compute_confidence_signal(self, loader: DataLoader) -> np.ndarray:
        """Max softmax probability (prediction confidence).

        Members typically have HIGHER confidence because the model has
        seen them before and is more certain about them.
        """
        probs, _, _ = self.get_model_outputs(loader)
        return np.max(probs, axis=1)

    def _compute_entropy_signal(self, loader: DataLoader) -> np.ndarray:
        """Prediction entropy: -sum(p * log(p)).

        Members typically have LOWER entropy (less uncertainty) because
        the model is more confident about them.
        """
        probs, _, _ = self.get_model_outputs(loader)
        # Clip to avoid log(0)
        probs_clipped = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
        return entropy

    # ------------------------------------------------------------------
    # Threshold selection
    # ------------------------------------------------------------------

    def _find_optimal_threshold(
        self,
        scores: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """Find the threshold that maximises attack accuracy.

        Tries every unique score value as a potential threshold and picks
        the one that correctly classifies the most samples.

        Parameters
        ----------
        scores:
            Signal values for all samples (members + non-members).
        ground_truth:
            Binary labels (1 = member, 0 = non-member).

        Returns
        -------
        float
            The optimal threshold value.
        """
        # Get sorted unique thresholds — we test each one
        sorted_scores = np.sort(np.unique(scores))

        # For efficiency, subsample if there are too many unique values
        if len(sorted_scores) > 1000:
            indices = np.linspace(0, len(sorted_scores) - 1, 1000, dtype=int)
            candidates = sorted_scores[indices]
        else:
            candidates = sorted_scores

        best_acc = 0.0
        best_threshold = float(np.median(scores))

        for t in candidates:
            preds = self._apply_threshold(scores, t)
            acc = float(np.mean(preds == ground_truth))
            if acc > best_acc:
                best_acc = acc
                best_threshold = float(t)

        return best_threshold

    def _apply_threshold(
        self, scores: np.ndarray, threshold: float,
    ) -> np.ndarray:
        """Classify samples as member/non-member using *threshold*.

        The direction depends on the metric:
        - loss: score < threshold → member (lower loss = member)
        - confidence: score > threshold → member (higher conf = member)
        - entropy: score < threshold → member (lower entropy = member)

        Returns
        -------
        np.ndarray
            Binary predictions (1 = member, 0 = non-member).
        """
        if self.metric == "confidence":
            # Higher confidence → more likely member
            return (scores >= threshold).astype(int)
        else:
            # Lower loss / lower entropy → more likely member
            return (scores <= threshold).astype(int)

    def _scores_to_confidence(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw scores to confidence values where higher = more
        likely member.

        This normalisation is needed so that ROC curves and AUC work
        correctly regardless of which metric was used.
        """
        if self.metric == "confidence":
            # Already in the right direction: higher = more member-like
            return scores
        else:
            # Loss and entropy: lower = more member-like, so negate
            return -scores

    # ------------------------------------------------------------------
    # Label extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_labels(loader: DataLoader) -> np.ndarray:
        """Extract the class labels from a DataLoader.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — one integer class label per sample.
        """
        all_labels: list[np.ndarray] = []
        for _, targets in loader:
            all_labels.append(targets.numpy())
        return np.concatenate(all_labels)

    # ------------------------------------------------------------------
    # Per-class evaluation
    # ------------------------------------------------------------------

    def evaluate_per_class(self) -> dict[int, dict[str, float]]:
        """Compute evaluation metrics **separately for each class**.

        This reveals whether the attack works better on certain classes.
        For example, rare classes might be easier to identify as members
        because the model memorises them more.

        Returns
        -------
        dict[int, dict[str, float]]
            Mapping from class label → metric dictionary. Each inner dict
            has the same keys as ``evaluate()`` (accuracy, precision, …).

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called yet.
        """
        if self.result is None:
            raise RuntimeError("Call run() before evaluate_per_class().")

        all_labels = np.concatenate([self.member_labels, self.nonmember_labels])
        unique_classes = np.unique(all_labels)

        per_class: dict[int, dict[str, float]] = {}
        for cls in unique_classes:
            mask = all_labels == cls
            # Need at least 2 samples AND both member/non-member in this class
            preds_cls = self.result.predictions[mask]
            gt_cls = self.result.ground_truth[mask]
            scores_cls = self.result.confidence_scores[mask]

            if len(gt_cls) < 2 or len(np.unique(gt_cls)) < 2:
                # Not enough data for meaningful per-class metrics
                per_class[int(cls)] = {
                    "accuracy": float(np.mean(preds_cls == gt_cls)) if len(gt_cls) > 0 else 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "auc_roc": 0.0,
                    "auc_pr": 0.0,
                    "tpr_at_1fpr": 0.0,
                    "tpr_at_01fpr": 0.0,
                    "n_samples": int(mask.sum()),
                }
                continue

            metrics = self._compute_metrics(preds_cls, gt_cls, scores_cls)
            metrics["n_samples"] = int(mask.sum())
            per_class[int(cls)] = metrics

        return per_class

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_dir: str | Path) -> Path:
        """Generate a complete evaluation report with metrics and plots.

        Creates the following files in *output_dir*:

        - ``metrics.json`` — overall evaluation metrics
        - ``per_class_metrics.json`` — per-class breakdown
        - ``roc_curve.png`` — ROC curve plot
        - ``score_distributions.png`` — histogram of member vs non-member scores
        - ``per_class_accuracy.png`` — bar chart of per-class attack accuracy
        - ``summary.txt`` — human-readable text summary

        Parameters
        ----------
        output_dir:
            Directory where all report files are saved. Created if it
            doesn't exist.

        Returns
        -------
        Path
            The output directory.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called yet.
        """
        if self.result is None:
            raise RuntimeError("Call run() before generate_report().")

        # Lazy import to avoid matplotlib overhead when not needed
        from auditml.attacks.visualization import (
            plot_per_class_metrics,
            plot_roc_curve,
            plot_score_distributions,
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Overall metrics
        metrics = self.evaluate()
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 2. Per-class metrics
        per_class = self.evaluate_per_class()
        # JSON keys must be strings
        per_class_str = {str(k): v for k, v in per_class.items()}
        with open(out / "per_class_metrics.json", "w") as f:
            json.dump(per_class_str, f, indent=2)

        # 3. ROC curve
        plot_roc_curve(
            ground_truth=self.result.ground_truth,
            confidence_scores=self.result.confidence_scores,
            save_path=out / "roc_curve.png",
        )

        # 4. Score distributions histogram
        plot_score_distributions(
            member_scores=self.member_scores,
            nonmember_scores=self.nonmember_scores,
            metric_name=self.metric,
            threshold=self.threshold,
            save_path=out / "score_distributions.png",
        )

        # 5. Per-class accuracy bar chart
        plot_per_class_metrics(
            per_class_metrics=per_class,
            save_path=out / "per_class_accuracy.png",
        )

        # 6. Human-readable summary
        self._write_summary(out / "summary.txt", metrics, per_class)

        return out

    def _write_summary(
        self,
        path: Path,
        metrics: dict[str, float],
        per_class: dict[int, dict[str, float]],
    ) -> None:
        """Write a human-readable text summary of the attack results."""
        lines = [
            "=" * 60,
            "AuditML — Threshold MIA Report",
            "=" * 60,
            "",
            f"Metric used:     {self.metric}",
            f"Threshold:       {self.threshold:.6f}",
            f"Total samples:   {len(self.result.predictions)}",
            f"  Members:       {int(self.result.ground_truth.sum())}",
            f"  Non-members:   {int((1 - self.result.ground_truth).sum())}",
            "",
            "--- Overall Metrics ---",
        ]
        for key, val in metrics.items():
            lines.append(f"  {key:<20s}: {val:.4f}")

        lines.append("")
        lines.append("--- Per-Class Breakdown ---")
        for cls in sorted(per_class.keys()):
            m = per_class[cls]
            lines.append(
                f"  Class {cls:>3d}:  acc={m['accuracy']:.3f}  "
                f"auc={m['auc_roc']:.3f}  n={m['n_samples']}"
            )

        lines.append("")
        lines.append("--- Metadata ---")
        for key, val in self.result.metadata.items():
            lines.append(f"  {key}: {val}")

        lines.append("")
        path.write_text("\n".join(lines))
