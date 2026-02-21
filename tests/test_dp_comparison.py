"""Tests for Task 2.11 — DP vs Non-DP Comparison.

Tests the DPComparison class, delta computation, privacy gain summary,
report generation, and comparison visualization functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from auditml.attacks.results import AttackResult
from auditml.reporting.comparison import DPComparison
from auditml.reporting.visualization import (
    plot_metric_comparison,
    plot_roc_comparison,
    plot_score_comparison,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_result(
    n: int = 100,
    attack_accuracy: float = 0.7,
    attack_name: str = "mia_threshold",
    seed: int = 0,
) -> AttackResult:
    """Create a synthetic AttackResult with controlled accuracy."""
    rng = np.random.RandomState(seed)
    gt = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])

    # Generate scores so that attack_accuracy fraction are correct
    scores = rng.random(n)
    # Bias member scores upward and non-member scores downward
    scores[:n // 2] += (attack_accuracy - 0.5) * 2
    scores[n // 2:] -= (attack_accuracy - 0.5) * 2
    scores = np.clip(scores, 0.01, 0.99)

    preds = (scores >= 0.5).astype(np.int32)

    return AttackResult(
        predictions=preds,
        ground_truth=gt,
        confidence_scores=scores,
        attack_name=attack_name,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def baseline_result() -> AttackResult:
    """Stronger attack (higher accuracy) — baseline model."""
    return _make_result(n=200, attack_accuracy=0.75, seed=0)


@pytest.fixture()
def dp_result() -> AttackResult:
    """Weaker attack (lower accuracy) — DP model."""
    return _make_result(n=200, attack_accuracy=0.55, seed=1)


@pytest.fixture()
def comparison(baseline_result, dp_result) -> DPComparison:
    return DPComparison(
        baseline_result=baseline_result,
        dp_result=dp_result,
        epsilon=5.0,
        model_accuracy={"baseline": 0.95, "dp": 0.85},
    )


# ── DPComparison construction ───────────────────────────────────────────


class TestDPComparisonInit:
    def test_creates_comparison(self, comparison) -> None:
        assert isinstance(comparison, DPComparison)
        assert comparison.epsilon == 5.0

    def test_metrics_computed(self, comparison) -> None:
        assert "accuracy" in comparison.baseline_metrics
        assert "accuracy" in comparison.dp_metrics
        assert "auc_roc" in comparison.baseline_metrics

    def test_custom_metrics(self, baseline_result, dp_result) -> None:
        custom_base = {"accuracy": 0.8, "auc_roc": 0.85}
        custom_dp = {"accuracy": 0.6, "auc_roc": 0.55}
        comp = DPComparison(
            baseline_result, dp_result,
            baseline_metrics=custom_base,
            dp_metrics=custom_dp,
        )
        assert comp.baseline_metrics["accuracy"] == 0.8
        assert comp.dp_metrics["accuracy"] == 0.6

    def test_model_accuracy_stored(self, comparison) -> None:
        assert comparison.model_accuracy["baseline"] == 0.95
        assert comparison.model_accuracy["dp"] == 0.85


# ── compute_deltas ──────────────────────────────────────────────────────


class TestComputeDeltas:
    def test_returns_dict(self, comparison) -> None:
        deltas = comparison.compute_deltas()
        assert isinstance(deltas, dict)

    def test_has_delta_keys(self, comparison) -> None:
        deltas = comparison.compute_deltas()
        assert "accuracy_delta" in deltas
        assert "auc_roc_delta" in deltas

    def test_dp_weaker_means_negative_delta(self, comparison) -> None:
        """DP model should have lower attack accuracy → negative delta."""
        deltas = comparison.compute_deltas()
        assert deltas["accuracy_delta"] < 0

    def test_delta_math_correct(self) -> None:
        base = _make_result(n=100, attack_accuracy=0.8, seed=10)
        dp = _make_result(n=100, attack_accuracy=0.6, seed=11)
        comp = DPComparison(
            base, dp,
            baseline_metrics={"accuracy": 0.8, "auc_roc": 0.9},
            dp_metrics={"accuracy": 0.6, "auc_roc": 0.55},
        )
        deltas = comp.compute_deltas()
        assert abs(deltas["accuracy_delta"] - (-0.2)) < 1e-10
        assert abs(deltas["auc_roc_delta"] - (-0.35)) < 1e-10


# ── compute_privacy_gain ────────────────────────────────────────────────


class TestComputePrivacyGain:
    def test_returns_dict(self, comparison) -> None:
        gain = comparison.compute_privacy_gain()
        assert isinstance(gain, dict)

    def test_has_expected_keys(self, comparison) -> None:
        gain = comparison.compute_privacy_gain()
        assert "attack_accuracy_reduction" in gain
        assert "auc_roc_reduction" in gain
        assert "epsilon" in gain

    def test_positive_reduction_means_better_privacy(self, comparison) -> None:
        gain = comparison.compute_privacy_gain()
        # Baseline is stronger → reduction should be positive
        assert gain["attack_accuracy_reduction"] > 0

    def test_utility_cost(self, comparison) -> None:
        gain = comparison.compute_privacy_gain()
        assert "utility_cost" in gain
        assert abs(gain["utility_cost"] - 0.10) < 1e-10

    def test_epsilon_included(self, comparison) -> None:
        gain = comparison.compute_privacy_gain()
        assert gain["epsilon"] == 5.0

    def test_no_epsilon(self, baseline_result, dp_result) -> None:
        comp = DPComparison(baseline_result, dp_result)
        gain = comp.compute_privacy_gain()
        assert "epsilon" not in gain


# ── summary_dict ────────────────────────────────────────────────────────


class TestSummaryDict:
    def test_has_all_sections(self, comparison) -> None:
        summary = comparison.summary_dict()
        assert "baseline_metrics" in summary
        assert "dp_metrics" in summary
        assert "deltas" in summary
        assert "privacy_gain" in summary
        assert "attack_name" in summary

    def test_json_serializable(self, comparison) -> None:
        summary = comparison.summary_dict()
        # Should not raise
        json.dumps(summary, default=str)


# ── Visualization functions ─────────────────────────────────────────────


class TestPlotMetricComparison:
    def test_returns_figure(self) -> None:
        base = {"accuracy": 0.8, "precision": 0.7, "auc_roc": 0.85}
        dp = {"accuracy": 0.6, "precision": 0.5, "auc_roc": 0.55}
        fig = plot_metric_comparison(base, dp)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path) -> None:
        base = {"accuracy": 0.8, "auc_roc": 0.9}
        dp = {"accuracy": 0.6, "auc_roc": 0.55}
        path = tmp_path / "comparison.png"
        plot_metric_comparison(base, dp, save_path=path)
        assert path.exists()


class TestPlotROCComparison:
    def test_returns_figure(self, baseline_result, dp_result) -> None:
        fig = plot_roc_comparison(
            baseline_result.ground_truth,
            baseline_result.confidence_scores,
            dp_result.ground_truth,
            dp_result.confidence_scores,
        )
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, baseline_result, dp_result, tmp_path) -> None:
        path = tmp_path / "roc.png"
        plot_roc_comparison(
            baseline_result.ground_truth,
            baseline_result.confidence_scores,
            dp_result.ground_truth,
            dp_result.confidence_scores,
            save_path=path,
        )
        assert path.exists()


class TestPlotScoreComparison:
    def test_returns_figure(self, baseline_result, dp_result) -> None:
        fig = plot_score_comparison(
            baseline_result.confidence_scores,
            baseline_result.ground_truth,
            dp_result.confidence_scores,
            dp_result.ground_truth,
        )
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, baseline_result, dp_result, tmp_path) -> None:
        path = tmp_path / "scores.png"
        plot_score_comparison(
            baseline_result.confidence_scores,
            baseline_result.ground_truth,
            dp_result.confidence_scores,
            dp_result.ground_truth,
            save_path=path,
        )
        assert path.exists()


# ── Report generation ────────────────────────────────────────────────────


class TestGenerateReport:
    def test_creates_output_dir(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        assert out.is_dir()

    def test_creates_comparison_json(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        f = out / "comparison.json"
        assert f.exists()
        data = json.loads(f.read_text())
        assert "baseline_metrics" in data
        assert "dp_metrics" in data
        assert "deltas" in data

    def test_creates_bar_chart(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        assert (out / "comparison_bar_chart.png").exists()

    def test_creates_roc_comparison(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        assert (out / "roc_comparison.png").exists()

    def test_creates_score_comparison(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        assert (out / "score_comparison.png").exists()

    def test_creates_summary_txt(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        summary = out / "summary.txt"
        assert summary.exists()
        text = summary.read_text()
        assert "DP vs Non-DP" in text
        assert "Baseline" in text
        assert "epsilon" in text.lower()

    def test_returns_output_dir(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        result = comparison.generate_report(out)
        assert result == out


# ── Imports ──────────────────────────────────────────────────────────────


class TestImports:
    def test_import_from_reporting(self) -> None:
        from auditml.reporting import DPComparison
        assert DPComparison is not None
