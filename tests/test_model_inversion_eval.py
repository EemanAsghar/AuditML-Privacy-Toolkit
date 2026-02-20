"""Tests for Task 2.7 — Model Inversion Evaluation & Visualization.

Tests reconstruction visualization, confidence bar charts, similarity
distributions, and full report generation.
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

from auditml.attacks.model_inversion import ModelInversion
from auditml.attacks.visualization import (
    plot_reconstruction_confidence,
    plot_reconstructions,
)
from auditml.config import default_config
from auditml.config.schema import DatasetName
from auditml.models import SimpleCNN


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def config():
    cfg = default_config()
    cfg.model.num_classes = 10
    cfg.data.dataset = DatasetName.MNIST
    cfg.attack_params.model_inversion.num_iterations = 10
    cfg.attack_params.model_inversion.target_class = 3  # single class for speed
    return cfg


@pytest.fixture()
def model() -> SimpleCNN:
    torch.manual_seed(0)
    m = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
    m.eval()
    return m


@pytest.fixture()
def member_loader() -> DataLoader:
    torch.manual_seed(1)
    ds = TensorDataset(torch.randn(40, 1, 28, 28), torch.randint(0, 10, (40,)))
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def nonmember_loader() -> DataLoader:
    torch.manual_seed(2)
    ds = TensorDataset(torch.randn(40, 1, 28, 28), torch.randint(0, 10, (40,)))
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def run_attack(model, config, member_loader, nonmember_loader) -> ModelInversion:
    """ModelInversion that has been run."""
    attack = ModelInversion(model, config, device="cpu")
    attack.run(member_loader, nonmember_loader)
    return attack


# ── Visualization tests ──────────────────────────────────────────────────

class TestPlotReconstructions:
    def test_returns_figure_grayscale(self) -> None:
        recons = {0: np.random.randn(1, 1, 28, 28), 1: np.random.randn(1, 1, 28, 28)}
        fig = plot_reconstructions(recons)
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_rgb(self) -> None:
        recons = {0: np.random.randn(1, 3, 32, 32), 1: np.random.randn(1, 3, 32, 32)}
        fig = plot_reconstructions(recons)
        assert isinstance(fig, plt.Figure)

    def test_with_confidences(self) -> None:
        recons = {0: np.random.randn(1, 1, 28, 28)}
        confs = {0: 0.95}
        fig = plot_reconstructions(recons, confidences=confs)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path) -> None:
        recons = {i: np.random.randn(1, 1, 28, 28) for i in range(5)}
        path = tmp_path / "recons.png"
        plot_reconstructions(recons, save_path=path)
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_single_class(self) -> None:
        recons = {3: np.random.randn(1, 1, 28, 28)}
        fig = plot_reconstructions(recons)
        assert isinstance(fig, plt.Figure)

    def test_many_classes(self) -> None:
        recons = {i: np.random.randn(1, 1, 28, 28) for i in range(10)}
        confs = {i: float(np.random.rand()) for i in range(10)}
        fig = plot_reconstructions(recons, confidences=confs)
        assert isinstance(fig, plt.Figure)

    def test_3d_input_no_batch(self) -> None:
        """Handle (C, H, W) without batch dimension."""
        recons = {0: np.random.randn(1, 28, 28)}
        fig = plot_reconstructions(recons)
        assert isinstance(fig, plt.Figure)


class TestPlotReconstructionConfidence:
    def test_returns_figure(self) -> None:
        confs = {0: 0.9, 1: 0.7, 2: 0.5}
        fig = plot_reconstruction_confidence(confs)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path) -> None:
        confs = {0: 0.9, 1: 0.7}
        path = tmp_path / "conf.png"
        plot_reconstruction_confidence(confs, save_path=path)
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_single_class(self) -> None:
        confs = {5: 0.85}
        fig = plot_reconstruction_confidence(confs)
        assert isinstance(fig, plt.Figure)


# ── Score storage tests ──────────────────────────────────────────────────

class TestScoreStorage:
    def test_member_scores_stored(self, run_attack) -> None:
        assert run_attack.member_scores is not None
        assert len(run_attack.member_scores) == 40

    def test_nonmember_scores_stored(self, run_attack) -> None:
        assert run_attack.nonmember_scores is not None
        assert len(run_attack.nonmember_scores) == 40


# ── Report generation tests ──────────────────────────────────────────────

class TestGenerateReport:
    def test_creates_output_directory(self, run_attack, tmp_path) -> None:
        out = tmp_path / "mi_report"
        run_attack.generate_report(out)
        assert out.is_dir()

    def test_creates_all_files(self, run_attack, tmp_path) -> None:
        out = tmp_path / "mi_report"
        run_attack.generate_report(out)
        expected_files = [
            "metrics.json",
            "reconstructions.png",
            "reconstruction_confidence.png",
            "similarity_distributions.png",
            "roc_curve.png",
            "summary.txt",
        ]
        for fname in expected_files:
            assert (out / fname).exists(), f"Missing: {fname}"

    def test_metrics_json_valid(self, run_attack, tmp_path) -> None:
        out = tmp_path / "mi_report"
        run_attack.generate_report(out)
        with open(out / "metrics.json") as f:
            data = json.load(f)
        assert "accuracy" in data
        assert "auc_roc" in data

    def test_summary_content(self, run_attack, tmp_path) -> None:
        out = tmp_path / "mi_report"
        run_attack.generate_report(out)
        text = (out / "summary.txt").read_text()
        assert "Model Inversion Report" in text
        assert "Lambda TV" in text
        assert "Reconstruction Confidences" in text

    def test_plots_non_empty(self, run_attack, tmp_path) -> None:
        out = tmp_path / "mi_report"
        run_attack.generate_report(out)
        for fname in ["reconstructions.png", "reconstruction_confidence.png",
                       "similarity_distributions.png", "roc_curve.png"]:
            assert (out / fname).stat().st_size > 1000, f"{fname} too small"

    def test_raises_before_run(self, model, config, tmp_path) -> None:
        attack = ModelInversion(model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.generate_report(tmp_path / "report")

    def test_returns_path(self, run_attack, tmp_path) -> None:
        out = tmp_path / "mi_report"
        result = run_attack.generate_report(out)
        assert isinstance(result, Path)
        assert result == out
