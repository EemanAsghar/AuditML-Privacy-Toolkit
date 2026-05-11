"""Unified report generator for a complete AuditML audit.

Orchestrates individual attack reports, DP comparison, and cross-attack
comparison into a **single output directory** with a master summary.

Expected workflow:

1. Run one or more attacks against a target model.
2. Optionally run the same attacks against a DP-trained model.
3. Pass all results to ``ReportGenerator``.
4. Call ``generate()`` to produce the full report.

Output structure::

    <output_dir>/
    ├── summary.txt              # Master text summary
    ├── audit_summary.json       # Machine-readable summary
    ├── attacks/
    │   ├── mia_threshold/       # Per-attack report dirs
    │   ├── mia_shadow/
    │   ├── model_inversion/
    │   └── attribute_inference/
    ├── attack_comparison/       # Cross-attack comparison (if 2+ attacks)
    │   ├── attack_comparison.json
    │   ├── attack_comparison_bar.png
    │   └── attack_roc_overlay.png
    └── dp_comparison/           # DP vs non-DP (if DP results provided)
        ├── comparison.json
        ├── roc_comparison.png
        └── ...
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from auditml.attacks.base import BaseAttack
from auditml.attacks.results import AttackResult
from auditml.reporting.attack_comparison import AttackComparison
from auditml.reporting.comparison import DPComparison

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Unified report generator for a full privacy audit.

    Parameters
    ----------
    experiment_name:
        Human-readable name for this audit (used in titles and filenames).
    attack_results:
        Mapping from attack name to ``(attack_instance, result)`` tuples.
        The attack instances are used to call per-attack ``generate_report()``.
    dp_attack_results:
        Same structure but for attacks run against the DP model.
        If provided, a DP comparison section is added.
    epsilon:
        The privacy budget used for DP training. Required if
        ``dp_attack_results`` is provided.
    model_accuracy:
        Optional dict ``{"baseline": float, "dp": float}`` with model
        test accuracy for the utility vs privacy analysis.
    """

    def __init__(
        self,
        experiment_name: str = "audit",
        attack_results: dict[str, tuple[BaseAttack, AttackResult]] | None = None,
        dp_attack_results: dict[str, tuple[BaseAttack, AttackResult]] | None = None,
        epsilon: float | None = None,
        model_accuracy: dict[str, float] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.attack_results = attack_results or {}
        self.dp_attack_results = dp_attack_results or {}
        self.epsilon = epsilon
        self.model_accuracy = model_accuracy or {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, output_dir: str | Path) -> Path:
        """Generate the complete audit report.

        Parameters
        ----------
        output_dir:
            Root directory for the report. Created if it doesn't exist.

        Returns
        -------
        Path
            The output directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary_data: dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "num_attacks": len(self.attack_results),
            "attack_names": list(self.attack_results.keys()),
            "dp_enabled": len(self.dp_attack_results) > 0,
        }

        # 1. Per-attack reports
        attack_metrics = self._generate_attack_reports(out)
        summary_data["attack_metrics"] = attack_metrics

        # 2. Cross-attack comparison (if 2+ attacks)
        if len(self.attack_results) >= 2:
            self._generate_attack_comparison(out)
            summary_data["attack_comparison"] = True
        else:
            summary_data["attack_comparison"] = False

        # 3. DP comparison (if DP results provided)
        dp_summaries = {}
        if self.dp_attack_results:
            dp_summaries = self._generate_dp_comparisons(out)
            summary_data["dp_comparison"] = dp_summaries
            summary_data["epsilon"] = self.epsilon
        else:
            summary_data["dp_comparison"] = None

        # 4. Model accuracy
        if self.model_accuracy:
            summary_data["model_accuracy"] = self.model_accuracy

        # 5. Write master JSON
        with open(out / "audit_summary.json", "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        # 6. Write master text summary
        self._write_master_summary(out / "summary.txt", summary_data, dp_summaries)

        # 7. Generate self-contained HTML report
        try:
            from auditml.reporting.html_report import generate_html_report
            dp_metrics = {
                name: attack.evaluate()
                for name, (attack, _) in self.dp_attack_results.items()
            } if self.dp_attack_results else None
            html_path = generate_html_report(
                output_dir=out,
                experiment_name=self.experiment_name,
                attack_results=self.attack_results,
                dp_attack_results=self.dp_attack_results or None,
                attack_metrics=attack_metrics,
                dp_attack_metrics=dp_metrics,
                model_accuracy=self.model_accuracy or None,
                epsilon=self.epsilon,
            )
            summary_data["html_report"] = str(html_path)
            logger.info("HTML report generated at %s", html_path)
        except Exception as exc:
            logger.warning("HTML report generation failed: %s", exc, exc_info=True)

        logger.info("Report generated at %s", out)
        return out

    # ------------------------------------------------------------------
    # Per-attack reports
    # ------------------------------------------------------------------

    def _generate_attack_reports(
        self, output_dir: Path,
    ) -> dict[str, dict[str, float]]:
        """Generate individual report for each attack.

        Returns mapping from attack name to its metrics dict.
        """
        attacks_dir = output_dir / "attacks"
        attacks_dir.mkdir(exist_ok=True)

        metrics_all: dict[str, dict[str, float]] = {}

        for name, (attack, result) in self.attack_results.items():
            attack_dir = attacks_dir / name
            logger.info("Generating report for %s ...", name)

            if hasattr(attack, "generate_report"):
                attack.generate_report(attack_dir)
            else:
                # Fallback: just save metrics
                attack_dir.mkdir(parents=True, exist_ok=True)
                metrics = attack.evaluate()
                with open(attack_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

            metrics_all[name] = attack.evaluate()

        return metrics_all

    # ------------------------------------------------------------------
    # Cross-attack comparison
    # ------------------------------------------------------------------

    def _generate_attack_comparison(self, output_dir: Path) -> None:
        """Generate cross-attack comparison report."""
        results_only = {
            name: result
            for name, (_, result) in self.attack_results.items()
        }
        comp = AttackComparison(results_only)
        comp.generate_report(output_dir / "attack_comparison")

    # ------------------------------------------------------------------
    # DP comparison
    # ------------------------------------------------------------------

    def _generate_dp_comparisons(
        self, output_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        """Generate DP vs non-DP comparison for each shared attack.

        Returns mapping from attack name to privacy gain summary.
        """
        dp_dir = output_dir / "dp_comparison"
        dp_dir.mkdir(exist_ok=True)

        dp_summaries: dict[str, dict[str, Any]] = {}

        for name in self.attack_results:
            if name not in self.dp_attack_results:
                continue

            _, baseline_result = self.attack_results[name]
            _, dp_result = self.dp_attack_results[name]

            comp = DPComparison(
                baseline_result=baseline_result,
                dp_result=dp_result,
                epsilon=self.epsilon,
                model_accuracy=self.model_accuracy,
            )
            comp.generate_report(dp_dir / name)
            dp_summaries[name] = comp.compute_privacy_gain()

        return dp_summaries

    # ------------------------------------------------------------------
    # Master summary
    # ------------------------------------------------------------------

    def _write_master_summary(
        self,
        path: Path,
        summary_data: dict[str, Any],
        dp_summaries: dict[str, dict[str, Any]],
    ) -> None:
        """Write the top-level human-readable summary."""
        lines = [
            "=" * 70,
            f"AuditML Privacy Audit Report — {self.experiment_name}",
            "=" * 70,
            "",
            f"Generated: {summary_data['timestamp']}",
            f"Attacks run: {summary_data['num_attacks']}",
            f"Attack types: {', '.join(summary_data['attack_names'])}",
            f"DP comparison: {'Yes' if summary_data['dp_enabled'] else 'No'}",
        ]

        if self.model_accuracy:
            lines.append("")
            lines.append("--- Model Utility ---")
            for k, v in self.model_accuracy.items():
                lines.append(f"  {k}: {v:.4f}")

        # Per-attack metrics
        lines.append("")
        lines.append("--- Attack Results ---")
        attack_metrics = summary_data.get("attack_metrics", {})

        if attack_metrics:
            header = f"  {'Attack':<25s}  {'Accuracy':>10s}  {'AUC-ROC':>10s}  {'F1':>10s}"
            lines.append(header)
            lines.append(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}")
            for name, m in attack_metrics.items():
                lines.append(
                    f"  {name:<25s}  {m.get('accuracy', 0.0):>10.4f}  "
                    f"{m.get('auc_roc', 0.0):>10.4f}  {m.get('f1', 0.0):>10.4f}"
                )

        # DP comparison
        if dp_summaries:
            lines.append("")
            lines.append("--- DP Privacy Gain ---")
            if self.epsilon is not None:
                lines.append(f"  Epsilon: {self.epsilon:.2f}")
            for name, gain in dp_summaries.items():
                lines.append(
                    f"  {name}: accuracy reduction = "
                    f"{gain.get('attack_accuracy_reduction', 0.0):.4f}, "
                    f"AUC-ROC reduction = "
                    f"{gain.get('auc_roc_reduction', 0.0):.4f}"
                )

        # Report structure
        lines.append("")
        lines.append("--- Report Files ---")
        lines.append("  attacks/<attack_name>/     Per-attack detailed reports")
        if summary_data.get("attack_comparison"):
            lines.append("  attack_comparison/         Cross-attack comparison")
        if summary_data.get("dp_comparison"):
            lines.append("  dp_comparison/<attack>/    DP vs non-DP comparison")
        lines.append("  audit_summary.json         Machine-readable summary")

        lines.append("")
        path.write_text("\n".join(lines))
