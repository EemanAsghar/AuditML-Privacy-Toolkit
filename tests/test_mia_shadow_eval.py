"""Tests for Task 2.5 — Shadow Model MIA Evaluation & Visualization.

Tests per-class evaluation, report generation, and reuse of the shared
visualization functions for shadow MIA output.
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

from auditml.attacks.mia_shadow import AttackMLP, ShadowMIA
from auditml.config import default_config
from auditml.config.schema import DatasetName
from auditml.models import SimpleCNN


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def config():
    cfg = default_config()
    cfg.attack_params.mia_shadow.num_shadow_models = 2
    cfg.attack_params.mia_shadow.shadow_epochs = 2
    cfg.model.num_classes = 10
    cfg.model.arch = "cnn"
    cfg.data.dataset = DatasetName.MNIST
    return cfg


@pytest.fixture()
def target_model() -> SimpleCNN:
    torch.manual_seed(0)
    m = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
    m.eval()
    return m


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
def pre_trained_shadows():
    """Create pre-trained shadow models for fast tests."""
    shadows = []
    for i in range(2):
        torch.manual_seed(10 + i)
        shadow = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
        shadow.eval()

        torch.manual_seed(20 + i)
        mem_ds = TensorDataset(torch.randn(50, 1, 28, 28), torch.randint(0, 10, (50,)))
        nonmem_ds = TensorDataset(torch.randn(50, 1, 28, 28), torch.randint(0, 10, (50,)))

        shadows.append((
            shadow,
            DataLoader(mem_ds, batch_size=16),
            DataLoader(nonmem_ds, batch_size=16),
        ))
    return shadows


@pytest.fixture()
def run_attack(target_model, config, member_loader, nonmember_loader, pre_trained_shadows):
    """ShadowMIA that has been run with pre-trained shadows."""
    attack = ShadowMIA(
        target_model, config, device="cpu",
        shadow_models=pre_trained_shadows,
    )
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
            assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_n_samples_sum_to_total(self, run_attack) -> None:
        per_class = run_attack.evaluate_per_class()
        total = sum(m["n_samples"] for m in per_class.values())
        assert total == 160  # 80 members + 80 non-members

    def test_raises_before_run(self, target_model, config) -> None:
        attack = ShadowMIA(target_model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.evaluate_per_class()

    def test_labels_stored_after_run(self, run_attack) -> None:
        assert run_attack.member_labels is not None
        assert run_attack.nonmember_labels is not None
        assert len(run_attack.member_labels) == 80
        assert len(run_attack.nonmember_labels) == 80

    def test_confidence_stored_after_run(self, run_attack) -> None:
        assert run_attack.member_confidence is not None
        assert run_attack.nonmember_confidence is not None
        assert len(run_attack.member_confidence) == 80
        assert len(run_attack.nonmember_confidence) == 80
        # Confidences should be in [0, 1]
        assert (run_attack.member_confidence >= 0).all()
        assert (run_attack.member_confidence <= 1).all()


# ── Report generation tests ──────────────────────────────────────────────

class TestGenerateReport:
    def test_creates_output_directory(self, run_attack, tmp_path) -> None:
        out = tmp_path / "shadow_report"
        run_attack.generate_report(out)
        assert out.is_dir()

    def test_creates_all_files(self, run_attack, tmp_path) -> None:
        out = tmp_path / "shadow_report"
        run_attack.generate_report(out)
        expected_files = [
            "metrics.json",
            "per_class_metrics.json",
            "roc_curve.png",
            "confidence_distributions.png",
            "per_class_accuracy.png",
            "summary.txt",
        ]
        for fname in expected_files:
            assert (out / fname).exists(), f"Missing: {fname}"

    def test_metrics_json_valid(self, run_attack, tmp_path) -> None:
        out = tmp_path / "shadow_report"
        run_attack.generate_report(out)
        with open(out / "metrics.json") as f:
            data = json.load(f)
        assert "accuracy" in data
        assert "auc_roc" in data
        assert isinstance(data["accuracy"], float)

    def test_per_class_json_valid(self, run_attack, tmp_path) -> None:
        out = tmp_path / "shadow_report"
        run_attack.generate_report(out)
        with open(out / "per_class_metrics.json") as f:
            data = json.load(f)
        assert len(data) > 0
        for key in data:
            assert key.isdigit()

    def test_summary_txt_content(self, run_attack, tmp_path) -> None:
        out = tmp_path / "shadow_report"
        run_attack.generate_report(out)
        text = (out / "summary.txt").read_text()
        assert "Shadow Model MIA Report" in text
        assert "accuracy" in text
        assert "Shadow models" in text

    def test_plots_are_non_empty(self, run_attack, tmp_path) -> None:
        out = tmp_path / "shadow_report"
        run_attack.generate_report(out)
        for fname in ["roc_curve.png", "confidence_distributions.png", "per_class_accuracy.png"]:
            assert (out / fname).stat().st_size > 1000, f"{fname} seems too small"

    def test_raises_before_run(self, target_model, config, tmp_path) -> None:
        attack = ShadowMIA(target_model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.generate_report(tmp_path / "report")

    def test_returns_path(self, run_attack, tmp_path) -> None:
        out = tmp_path / "shadow_report"
        result = run_attack.generate_report(out)
        assert isinstance(result, Path)
        assert result == out

    def test_summary_includes_shadow_count(self, run_attack, tmp_path) -> None:
        out = tmp_path / "shadow_report"
        run_attack.generate_report(out)
        text = (out / "summary.txt").read_text()
        assert "2" in text  # num_shadow_models = 2
