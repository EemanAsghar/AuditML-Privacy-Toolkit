"""DP vs Non-DP comparison module.

Compares attack results obtained from a standard (non-private) model
against a differentially-private (DP) model.  The comparison answers
the core question: **does DP training reduce privacy leakage?**

Expected workflow:

1. Train a standard model and run attacks → ``baseline_results``.
2. Train a DP model (same architecture, same data) and run the same
   attacks → ``dp_results``.
3. Pass both to ``DPComparison`` to compute deltas, generate plots,
   and produce a combined report.

The module is attack-agnostic — it works with any ``AttackResult``
from any of the four attack types.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from auditml.attacks.base import BaseAttack
from auditml.attacks.results import AttackResult

logger = logging.getLogger(__name__)


class DPComparison:
    """Compare attack effectiveness between standard and DP models.

    Parameters
    ----------
    baseline_result:
        Attack result from the standard (non-DP) model.
    dp_result:
        Attack result from the DP-trained model.
    baseline_metrics:
        Pre-computed metrics for the baseline. If ``None``, computed
        from the result arrays.
    dp_metrics:
        Pre-computed metrics for the DP model. If ``None``, computed
        from the result arrays.
    epsilon:
        The privacy budget (epsilon) used during DP training.
    model_accuracy:
        Optional dict with ``{"baseline": float, "dp": float}``
        holding the utility (test accuracy) of each model.
    """

    def __init__(
        self,
        baseline_result: AttackResult,
        dp_result: AttackResult,
        baseline_metrics: dict[str, float] | None = None,
        dp_metrics: dict[str, float] | None = None,
        epsilon: float | None = None,
        model_accuracy: dict[str, float] | None = None,
    ) -> None:
        self.baseline_result = baseline_result
        self.dp_result = dp_result
        self.epsilon = epsilon
        self.model_accuracy = model_accuracy or {}

        # Compute metrics if not provided
        self.baseline_metrics = baseline_metrics or BaseAttack._compute_metrics(
            baseline_result.predictions,
            baseline_result.ground_truth,
            baseline_result.confidence_scores,
        )
        self.dp_metrics = dp_metrics or BaseAttack._compute_metrics(
            dp_result.predictions,
            dp_result.ground_truth,
            dp_result.confidence_scores,
        )

    # ------------------------------------------------------------------
    # Core comparison
    # ------------------------------------------------------------------

    def compute_deltas(self) -> dict[str, float]:
        """Compute the change in each metric: ``dp_value - baseline_value``.

        Negative deltas mean the DP model is more private (attacks are
        less effective).  The most important metrics:

        - ``accuracy_delta``: negative = DP makes the attack less accurate
        - ``auc_roc_delta``: negative = DP makes the attack less
          discriminative
        - ``tpr_at_1fpr_delta``: negative = DP reduces true positive rate
          at realistic operating points

        Returns
        -------
        dict[str, float]
            Keys are ``"<metric>_delta"`` for each shared metric.
        """
        deltas: dict[str, float] = {}
        for key in self.baseline_metrics:
            if key in self.dp_metrics:
                deltas[f"{key}_delta"] = (
                    self.dp_metrics[key] - self.baseline_metrics[key]
                )
        return deltas

    def compute_privacy_gain(self) -> dict[str, float]:
        """Summarise the privacy improvement from DP training.

        Returns
        -------
        dict with keys:
            - ``attack_accuracy_reduction``: how much attack accuracy dropped
            - ``auc_roc_reduction``: how much AUC-ROC dropped
            - ``baseline_attack_accuracy``: baseline attack accuracy
            - ``dp_attack_accuracy``: DP attack accuracy
            - ``baseline_auc_roc``: baseline AUC-ROC
            - ``dp_auc_roc``: DP AUC-ROC
            - ``epsilon``: privacy budget (if known)
            - ``utility_cost``: drop in model accuracy (if known)
        """
        gain: dict[str, float] = {
            "attack_accuracy_reduction": (
                self.baseline_metrics.get("accuracy", 0.0)
                - self.dp_metrics.get("accuracy", 0.0)
            ),
            "auc_roc_reduction": (
                self.baseline_metrics.get("auc_roc", 0.0)
                - self.dp_metrics.get("auc_roc", 0.0)
            ),
            "baseline_attack_accuracy": self.baseline_metrics.get("accuracy", 0.0),
            "dp_attack_accuracy": self.dp_metrics.get("accuracy", 0.0),
            "baseline_auc_roc": self.baseline_metrics.get("auc_roc", 0.0),
            "dp_auc_roc": self.dp_metrics.get("auc_roc", 0.0),
        }

        if self.epsilon is not None:
            gain["epsilon"] = self.epsilon

        if "baseline" in self.model_accuracy and "dp" in self.model_accuracy:
            gain["utility_cost"] = (
                self.model_accuracy["baseline"] - self.model_accuracy["dp"]
            )
            gain["baseline_model_accuracy"] = self.model_accuracy["baseline"]
            gain["dp_model_accuracy"] = self.model_accuracy["dp"]

        return gain

    def summary_dict(self) -> dict[str, Any]:
        """Return a comprehensive summary combining all comparisons.

        Returns
        -------
        dict
            Contains ``baseline_metrics``, ``dp_metrics``, ``deltas``,
            ``privacy_gain``, and metadata.
        """
        return {
            "baseline_metrics": self.baseline_metrics,
            "dp_metrics": self.dp_metrics,
            "deltas": self.compute_deltas(),
            "privacy_gain": self.compute_privacy_gain(),
            "attack_name": self.baseline_result.attack_name,
            "epsilon": self.epsilon,
            "model_accuracy": self.model_accuracy,
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_dir: str | Path) -> Path:
        """Generate a comparison report with metrics, plots, and summary.

        Creates:

        - ``comparison.json`` — full comparison data
        - ``comparison_bar_chart.png`` — side-by-side metric comparison
        - ``roc_comparison.png`` — overlaid ROC curves
        - ``score_comparison.png`` — confidence score distributions
        - ``summary.txt`` — human-readable summary

        Parameters
        ----------
        output_dir:
            Directory to write report files.

        Returns
        -------
        Path
            The output directory.
        """
        from auditml.reporting.visualization import (
            plot_metric_comparison,
            plot_roc_comparison,
            plot_score_comparison,
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. JSON data
        summary = self.summary_dict()
        with open(out / "comparison.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # 2. Bar chart comparing metrics
        plot_metric_comparison(
            baseline_metrics=self.baseline_metrics,
            dp_metrics=self.dp_metrics,
            save_path=out / "comparison_bar_chart.png",
        )

        # 3. ROC curves
        plot_roc_comparison(
            baseline_gt=self.baseline_result.ground_truth,
            baseline_scores=self.baseline_result.confidence_scores,
            dp_gt=self.dp_result.ground_truth,
            dp_scores=self.dp_result.confidence_scores,
            save_path=out / "roc_comparison.png",
        )

        # 4. Score distributions
        plot_score_comparison(
            baseline_scores=self.baseline_result.confidence_scores,
            baseline_gt=self.baseline_result.ground_truth,
            dp_scores=self.dp_result.confidence_scores,
            dp_gt=self.dp_result.ground_truth,
            save_path=out / "score_comparison.png",
        )

        # 5. Summary text
        self._write_summary(out / "summary.txt")

        return out

    def _write_summary(self, path: Path) -> None:
        """Write a human-readable comparison summary."""
        gain = self.compute_privacy_gain()
        deltas = self.compute_deltas()

        lines = [
            "=" * 60,
            "AuditML — DP vs Non-DP Comparison Report",
            "=" * 60,
            "",
            f"Attack: {self.baseline_result.attack_name}",
        ]

        if self.epsilon is not None:
            lines.append(f"DP epsilon: {self.epsilon:.2f}")

        lines.append("")
        lines.append("--- Model Utility ---")
        if self.model_accuracy:
            for k, v in self.model_accuracy.items():
                lines.append(f"  {k}: {v:.4f}")
            if "utility_cost" in gain:
                lines.append(f"  Utility cost (accuracy drop): {gain['utility_cost']:.4f}")
        else:
            lines.append("  (not provided)")

        lines.append("")
        lines.append("--- Attack Metrics ---")
        lines.append(f"  {'Metric':<20s}  {'Baseline':>10s}  {'DP':>10s}  {'Delta':>10s}")
        lines.append(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
        for key in sorted(self.baseline_metrics.keys()):
            bv = self.baseline_metrics[key]
            dv = self.dp_metrics.get(key, 0.0)
            delta = deltas.get(f"{key}_delta", 0.0)
            lines.append(f"  {key:<20s}  {bv:>10.4f}  {dv:>10.4f}  {delta:>+10.4f}")

        lines.append("")
        lines.append("--- Privacy Gain Summary ---")
        lines.append(f"  Attack accuracy reduction: {gain['attack_accuracy_reduction']:.4f}")
        lines.append(f"  AUC-ROC reduction:         {gain['auc_roc_reduction']:.4f}")

        direction = "improved" if gain["auc_roc_reduction"] > 0 else "worsened"
        lines.append(f"  Privacy {direction} with DP training.")

        lines.append("")
        path.write_text("\n".join(lines))
