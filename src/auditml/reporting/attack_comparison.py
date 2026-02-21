"""Cross-attack comparison module.

Compares results from **multiple attack types** run against the same
model, answering questions like:

- Which attack is the most effective at inferring membership?
- Which classes are most vulnerable, and does this vary by attack?
- How do confidence-score distributions differ across attacks?

Expected workflow:

1. Run two or more attacks (threshold MIA, shadow MIA, model inversion,
   attribute inference) against the same target model.
2. Pass all ``AttackResult`` objects to ``AttackComparison``.
3. Call ``rank_attacks()``, ``generate_report()``, etc.

This module is complementary to ``DPComparison`` (Task 2.11), which
compares the *same* attack across DP vs non-DP models.
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


class AttackComparison:
    """Compare effectiveness across multiple attack types.

    Parameters
    ----------
    results:
        Mapping from attack name to its ``AttackResult``. At least two
        entries are required for a meaningful comparison.  Example::

            {
                "mia_threshold": threshold_result,
                "mia_shadow": shadow_result,
                "model_inversion": inversion_result,
            }
    """

    def __init__(self, results: dict[str, AttackResult]) -> None:
        if len(results) < 1:
            raise ValueError("At least one AttackResult is required.")
        self.results = results
        self._metrics: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_all_metrics(self) -> dict[str, dict[str, float]]:
        """Compute standard metrics for every attack.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping from attack name to its metric dictionary.
        """
        if not self._metrics:
            for name, result in self.results.items():
                self._metrics[name] = BaseAttack._compute_metrics(
                    result.predictions,
                    result.ground_truth,
                    result.confidence_scores,
                )
        return self._metrics

    def rank_attacks(self, metric: str = "auc_roc") -> list[tuple[str, float]]:
        """Rank attacks by a chosen metric (descending).

        Parameters
        ----------
        metric:
            The metric key to rank by (default ``"auc_roc"``).

        Returns
        -------
        list of (attack_name, metric_value), sorted descending.
        """
        all_metrics = self.compute_all_metrics()
        ranked = [
            (name, m.get(metric, 0.0))
            for name, m in all_metrics.items()
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def best_attack(self, metric: str = "auc_roc") -> str:
        """Return the name of the most effective attack.

        Parameters
        ----------
        metric:
            The metric to compare by.

        Returns
        -------
        str
            Name of the top-ranked attack.
        """
        return self.rank_attacks(metric)[0][0]

    def summary_table(self) -> dict[str, dict[str, float]]:
        """Build a table of all attacks and all metrics.

        Returns
        -------
        dict[str, dict[str, float]]
            Outer key = attack name, inner dict = metric values.
        """
        return self.compute_all_metrics()

    def summary_dict(self) -> dict[str, Any]:
        """Return a comprehensive summary for serialisation.

        Returns
        -------
        dict containing ``metrics``, ``ranking``, ``best_attack``, and
        ``attack_names``.
        """
        return {
            "attack_names": list(self.results.keys()),
            "metrics": self.compute_all_metrics(),
            "ranking_by_auc_roc": self.rank_attacks("auc_roc"),
            "ranking_by_accuracy": self.rank_attacks("accuracy"),
            "best_attack_auc_roc": self.best_attack("auc_roc"),
            "best_attack_accuracy": self.best_attack("accuracy"),
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_dir: str | Path) -> Path:
        """Generate a cross-attack comparison report.

        Creates:

        - ``attack_comparison.json`` — full metrics and rankings
        - ``attack_comparison_bar.png`` — grouped bar chart of metrics
        - ``attack_roc_overlay.png`` — overlaid ROC curves
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
        from auditml.reporting.attack_visualization import (
            plot_attack_comparison_bar,
            plot_attack_roc_overlay,
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. JSON data
        summary = self.summary_dict()
        with open(out / "attack_comparison.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # 2. Bar chart
        plot_attack_comparison_bar(
            metrics=self.compute_all_metrics(),
            save_path=out / "attack_comparison_bar.png",
        )

        # 3. ROC overlay
        plot_attack_roc_overlay(
            results=self.results,
            save_path=out / "attack_roc_overlay.png",
        )

        # 4. Summary text
        self._write_summary(out / "summary.txt")

        return out

    def _write_summary(self, path: Path) -> None:
        """Write a human-readable comparison summary."""
        all_metrics = self.compute_all_metrics()
        ranking = self.rank_attacks("auc_roc")

        lines = [
            "=" * 60,
            "AuditML — Cross-Attack Comparison Report",
            "=" * 60,
            "",
            f"Attacks compared: {len(self.results)}",
            f"Attack names: {', '.join(self.results.keys())}",
            "",
            "--- Ranking by AUC-ROC ---",
        ]
        for i, (name, val) in enumerate(ranking, 1):
            lines.append(f"  {i}. {name:<25s}  AUC-ROC = {val:.4f}")

        lines.append("")
        lines.append("--- Full Metrics Table ---")

        # Header
        metric_keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]
        header = f"  {'Attack':<25s}"
        for mk in metric_keys:
            header += f"  {mk:>10s}"
        lines.append(header)
        lines.append(f"  {'-'*25}" + f"  {'-'*10}" * len(metric_keys))

        for name in self.results:
            m = all_metrics[name]
            row = f"  {name:<25s}"
            for mk in metric_keys:
                row += f"  {m.get(mk, 0.0):>10.4f}"
            lines.append(row)

        lines.append("")
        lines.append(f"Best attack (AUC-ROC): {self.best_attack('auc_roc')}")
        lines.append(f"Best attack (Accuracy): {self.best_attack('accuracy')}")

        lines.append("")
        path.write_text("\n".join(lines))
