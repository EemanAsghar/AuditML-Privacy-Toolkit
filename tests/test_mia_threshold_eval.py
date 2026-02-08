"""Tests for Task 2.3 — MIA Threshold Evaluation & Visualization.

Tests per-class evaluation, visualization functions, and report generation.
Uses a SimpleCNN on synthetic data (same approach as test_mia_threshold.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from auditml.attacks.mia_threshold import ThresholdMIA
from auditml.attacks.visualization import (
    plot_per_class_metrics,
    plot_roc_curve,
    plot_score_distributions,
)
from auditml.config import default_config
from auditml.models import SimpleCNN


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def model() -> SimpleCNN:
    torch.manual_seed(0)
    m = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
    m.eval()
    return m


@pytest.fixture()
def config():
    return default_config()


@pytest.fixture()
def member_loader() -> DataLoader:
    torch.manual_seed(1)
    ds = TensorDataset(torch.randn(80, 1, 28, 28), torch.randint(0, 10, (80,)))
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def nonmember_loader() -> DataLoader:
    torch.manual_seed(2)
    ds = TensorDataset(torch.randn(80, 1, 28, 28), torch.randint(0, 10, (80,)))
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def run_attack(model, config, member_loader, nonmember_loader) -> ThresholdMIA:
    """Return a ThresholdMIA that has already been run."""
    attack = ThresholdMIA(model, config, device="cpu")
    attack.run(member_loader, nonmember_loader)
    return attack


# ── Per-class evaluation tests ───────────────────────────────────────────

class TestPerClassEvaluation:
    def test_returns_dict(self, run_attack) -> None:
        per_class = run_attack.evaluate_per_class()
        assert isinstance(per_class, dict)
        assert len(per_class) > 0

    def test_keys_are_integers(self, run_attack) -> None:
        per_class = run_attack.evaluate_per_class()
        for key in per_class:
            assert isinstance(key, int)

    def test_each_class_has_standard_metrics(self, run_attack) -> None:
        per_class = run_attack.evaluate_per_class()
        expected_keys = [
            "accuracy", "precision", "recall", "f1",
            "auc_roc", "auc_pr", "tpr_at_1fpr", "tpr_at_01fpr",
            "n_samples",
        ]
        for cls, metrics in per_class.items():
            for key in expected_keys:
                assert key in metrics, f"Class {cls} missing metric: {key}"

    def test_accuracy_in_valid_range(self, run_attack) -> None:
        per_class = run_attack.evaluate_per_class()
        for cls, metrics in per_class.items():
            assert 0.0 <= metrics["accuracy"] <= 1.0, (
                f"Class {cls} accuracy out of range: {metrics['accuracy']}"
            )

    def test_n_samples_positive(self, run_attack) -> None:
        per_class = run_attack.evaluate_per_class()
        for cls, metrics in per_class.items():
            assert metrics["n_samples"] > 0

    def test_n_samples_sum_to_total(self, run_attack) -> None:
        per_class = run_attack.evaluate_per_class()
        total = sum(m["n_samples"] for m in per_class.values())
        assert total == 160  # 80 members + 80 non-members

    def test_raises_before_run(self, model, config) -> None:
        attack = ThresholdMIA(model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.evaluate_per_class()

    def test_labels_stored_after_run(self, run_attack) -> None:
        assert run_attack.member_labels is not None
        assert run_attack.nonmember_labels is not None
        assert len(run_attack.member_labels) == 80
        assert len(run_attack.nonmember_labels) == 80


# ── Visualization tests ──────────────────────────────────────────────────

class TestPlotROCCurve:
    def test_returns_figure(self) -> None:
        gt = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        fig = plot_roc_curve(gt, scores)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path) -> None:
        gt = np.array([1, 1, 0, 0, 1, 0])
        scores = np.array([0.9, 0.7, 0.3, 0.1, 0.6, 0.4])
        path = tmp_path / "roc.png"
        plot_roc_curve(gt, scores, save_path=path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_custom_title(self) -> None:
        gt = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        fig = plot_roc_curve(gt, scores, title="Custom Title")
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Title"


class TestPlotScoreDistributions:
    def test_returns_figure(self) -> None:
        member = np.random.randn(50)
        nonmember = np.random.randn(50) + 1
        fig = plot_score_distributions(member, nonmember)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path) -> None:
        member = np.random.randn(50)
        nonmember = np.random.randn(50) + 1
        path = tmp_path / "hist.png"
        plot_score_distributions(member, nonmember, save_path=path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_with_threshold_line(self) -> None:
        member = np.random.randn(50)
        nonmember = np.random.randn(50) + 1
        fig = plot_score_distributions(member, nonmember, threshold=0.5)
        assert isinstance(fig, plt.Figure)

    def test_custom_metric_name(self) -> None:
        member = np.random.randn(50)
        nonmember = np.random.randn(50)
        fig = plot_score_distributions(member, nonmember, metric_name="entropy")
        assert isinstance(fig, plt.Figure)


class TestPlotPerClassMetrics:
    def test_returns_figure(self) -> None:
        per_class = {
            0: {"accuracy": 0.6, "n_samples": 20},
            1: {"accuracy": 0.7, "n_samples": 25},
            2: {"accuracy": 0.5, "n_samples": 15},
        }
        fig = plot_per_class_metrics(per_class)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path) -> None:
        per_class = {
            0: {"accuracy": 0.6, "n_samples": 20},
            1: {"accuracy": 0.7, "n_samples": 25},
        }
        path = tmp_path / "per_class.png"
        plot_per_class_metrics(per_class, save_path=path)
        assert path.exists()

    def test_different_metric_key(self) -> None:
        per_class = {
            0: {"accuracy": 0.6, "auc_roc": 0.55, "n_samples": 20},
            1: {"accuracy": 0.7, "auc_roc": 0.65, "n_samples": 25},
        }
        fig = plot_per_class_metrics(per_class, metric_key="auc_roc")
        assert isinstance(fig, plt.Figure)


# ── Report generation tests ──────────────────────────────────────────────

class TestGenerateReport:
    def test_creates_output_directory(self, run_attack, tmp_path) -> None:
        out = tmp_path / "report"
        run_attack.generate_report(out)
        assert out.is_dir()

    def test_creates_all_files(self, run_attack, tmp_path) -> None:
        out = tmp_path / "report"
        run_attack.generate_report(out)
        expected_files = [
            "metrics.json",
            "per_class_metrics.json",
            "roc_curve.png",
            "score_distributions.png",
            "per_class_accuracy.png",
            "summary.txt",
        ]
        for fname in expected_files:
            assert (out / fname).exists(), f"Missing: {fname}"

    def test_metrics_json_valid(self, run_attack, tmp_path) -> None:
        out = tmp_path / "report"
        run_attack.generate_report(out)
        with open(out / "metrics.json") as f:
            data = json.load(f)
        assert "accuracy" in data
        assert "auc_roc" in data
        assert isinstance(data["accuracy"], float)

    def test_per_class_json_valid(self, run_attack, tmp_path) -> None:
        out = tmp_path / "report"
        run_attack.generate_report(out)
        with open(out / "per_class_metrics.json") as f:
            data = json.load(f)
        assert len(data) > 0
        # Keys should be string representations of class labels
        for key in data:
            assert key.isdigit()

    def test_summary_txt_content(self, run_attack, tmp_path) -> None:
        out = tmp_path / "report"
        run_attack.generate_report(out)
        text = (out / "summary.txt").read_text()
        assert "Threshold MIA Report" in text
        assert "accuracy" in text
        assert run_attack.metric in text

    def test_plots_are_non_empty(self, run_attack, tmp_path) -> None:
        out = tmp_path / "report"
        run_attack.generate_report(out)
        for fname in ["roc_curve.png", "score_distributions.png", "per_class_accuracy.png"]:
            assert (out / fname).stat().st_size > 1000, f"{fname} seems too small"

    def test_raises_before_run(self, model, config, tmp_path) -> None:
        attack = ThresholdMIA(model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.generate_report(tmp_path / "report")

    def test_returns_path(self, run_attack, tmp_path) -> None:
        out = tmp_path / "report"
        result = run_attack.generate_report(out)
        assert isinstance(result, Path)
        assert result == out
