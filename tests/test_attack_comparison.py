"""Tests for Task 2.12 — Attack Comparison Module.

Tests the AttackComparison class, ranking, report generation, and
cross-attack visualization functions.
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
from auditml.reporting.attack_comparison import AttackComparison
from auditml.reporting.attack_visualization import (
    plot_attack_comparison_bar,
    plot_attack_roc_overlay,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_result(
    n: int = 100,
    attack_accuracy: float = 0.7,
    attack_name: str = "test",
    seed: int = 0,
) -> AttackResult:
    """Create a synthetic AttackResult with controlled accuracy."""
    rng = np.random.RandomState(seed)
    gt = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    scores = rng.random(n)
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
def three_results() -> dict[str, AttackResult]:
    """Three attacks with different effectiveness levels."""
    return {
        "mia_threshold": _make_result(200, 0.65, "mia_threshold", seed=0),
        "mia_shadow": _make_result(200, 0.80, "mia_shadow", seed=1),
        "attribute_inference": _make_result(200, 0.55, "attribute_inference", seed=2),
    }


@pytest.fixture()
def comparison(three_results) -> AttackComparison:
    return AttackComparison(three_results)


# ── Construction ─────────────────────────────────────────────────────────


class TestAttackComparisonInit:
    def test_creates_comparison(self, comparison) -> None:
        assert isinstance(comparison, AttackComparison)
        assert len(comparison.results) == 3

    def test_single_result_allowed(self) -> None:
        r = _make_result(100, 0.7, "test", seed=0)
        comp = AttackComparison({"test": r})
        assert len(comp.results) == 1

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            AttackComparison({})


# ── Metrics ──────────────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_returns_dict_per_attack(self, comparison) -> None:
        metrics = comparison.compute_all_metrics()
        assert len(metrics) == 3
        for name in ("mia_threshold", "mia_shadow", "attribute_inference"):
            assert name in metrics

    def test_each_attack_has_standard_keys(self, comparison) -> None:
        metrics = comparison.compute_all_metrics()
        for name, m in metrics.items():
            assert "accuracy" in m
            assert "auc_roc" in m
            assert "f1" in m

    def test_values_in_range(self, comparison) -> None:
        metrics = comparison.compute_all_metrics()
        for name, m in metrics.items():
            assert 0.0 <= m["accuracy"] <= 1.0
            assert 0.0 <= m["auc_roc"] <= 1.0

    def test_caching(self, comparison) -> None:
        m1 = comparison.compute_all_metrics()
        m2 = comparison.compute_all_metrics()
        assert m1 is m2  # should be same object (cached)


# ── Ranking ──────────────────────────────────────────────────────────────


class TestRanking:
    def test_rank_returns_list(self, comparison) -> None:
        ranking = comparison.rank_attacks("auc_roc")
        assert isinstance(ranking, list)
        assert len(ranking) == 3

    def test_rank_descending(self, comparison) -> None:
        ranking = comparison.rank_attacks("auc_roc")
        values = [v for _, v in ranking]
        assert values == sorted(values, reverse=True)

    def test_rank_by_accuracy(self, comparison) -> None:
        ranking = comparison.rank_attacks("accuracy")
        values = [v for _, v in ranking]
        assert values == sorted(values, reverse=True)

    def test_best_attack(self, comparison) -> None:
        best = comparison.best_attack("auc_roc")
        ranking = comparison.rank_attacks("auc_roc")
        assert best == ranking[0][0]

    def test_shadow_mia_strongest(self, comparison) -> None:
        """Shadow MIA was set to 0.80 accuracy — should rank highest."""
        best = comparison.best_attack("accuracy")
        assert best == "mia_shadow"


# ── Summary ──────────────────────────────────────────────────────────────


class TestSummaryDict:
    def test_has_all_sections(self, comparison) -> None:
        summary = comparison.summary_dict()
        assert "attack_names" in summary
        assert "metrics" in summary
        assert "ranking_by_auc_roc" in summary
        assert "best_attack_auc_roc" in summary

    def test_json_serializable(self, comparison) -> None:
        summary = comparison.summary_dict()
        json.dumps(summary, default=str)  # should not raise

    def test_summary_table(self, comparison) -> None:
        table = comparison.summary_table()
        assert len(table) == 3
        for name, m in table.items():
            assert "accuracy" in m


# ── Visualization ────────────────────────────────────────────────────────


class TestPlotAttackComparisonBar:
    def test_returns_figure(self) -> None:
        metrics = {
            "attack_a": {"accuracy": 0.8, "auc_roc": 0.85},
            "attack_b": {"accuracy": 0.6, "auc_roc": 0.55},
        }
        fig = plot_attack_comparison_bar(metrics)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path) -> None:
        metrics = {
            "a": {"accuracy": 0.8, "auc_roc": 0.9},
            "b": {"accuracy": 0.6, "auc_roc": 0.5},
        }
        path = tmp_path / "bar.png"
        plot_attack_comparison_bar(metrics, save_path=path)
        assert path.exists()

    def test_custom_title(self) -> None:
        metrics = {"a": {"accuracy": 0.7}}
        fig = plot_attack_comparison_bar(metrics, title="Custom")
        assert isinstance(fig, plt.Figure)


class TestPlotAttackROCOverlay:
    def test_returns_figure(self, three_results) -> None:
        fig = plot_attack_roc_overlay(three_results)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, three_results, tmp_path) -> None:
        path = tmp_path / "roc.png"
        plot_attack_roc_overlay(three_results, save_path=path)
        assert path.exists()


# ── Report generation ────────────────────────────────────────────────────


class TestGenerateReport:
    def test_creates_output_dir(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        assert out.is_dir()

    def test_creates_json(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        f = out / "attack_comparison.json"
        assert f.exists()
        data = json.loads(f.read_text())
        assert "metrics" in data
        assert "ranking_by_auc_roc" in data

    def test_creates_bar_chart(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        assert (out / "attack_comparison_bar.png").exists()

    def test_creates_roc_overlay(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        assert (out / "attack_roc_overlay.png").exists()

    def test_creates_summary_txt(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        comparison.generate_report(out)
        summary = out / "summary.txt"
        assert summary.exists()
        text = summary.read_text()
        assert "Cross-Attack Comparison" in text
        assert "Ranking" in text
        assert "mia_shadow" in text

    def test_returns_output_dir(self, comparison, tmp_path) -> None:
        out = tmp_path / "report"
        result = comparison.generate_report(out)
        assert result == out


# ── Imports ──────────────────────────────────────────────────────────────


class TestImports:
    def test_import_from_reporting(self) -> None:
        from auditml.reporting import AttackComparison
        assert AttackComparison is not None
