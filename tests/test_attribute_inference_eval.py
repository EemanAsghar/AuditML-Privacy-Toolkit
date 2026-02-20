"""Tests for Task 2.9 — Attribute Inference Evaluation & Visualization.

Tests per-class evaluation, per-group attribute accuracy, report
generation, and the new ``plot_attribute_accuracy`` visualization.
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

from auditml.attacks.attribute_inference import AttributeAttackMLP, AttributeInference
from auditml.attacks.visualization import plot_attribute_accuracy
from auditml.config import default_config
from auditml.config.schema import DatasetName
from auditml.models import SimpleCNN


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def config():
    cfg = default_config()
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
    ds = TensorDataset(
        torch.randn(80, 1, 28, 28),
        torch.randint(0, 10, (80,)),
    )
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def nonmember_loader() -> DataLoader:
    torch.manual_seed(2)
    ds = TensorDataset(
        torch.randn(80, 1, 28, 28),
        torch.randint(0, 10, (80,)),
    )
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def attack_with_result(target_model, config, member_loader, nonmember_loader):
    """Run the attack once and return the attack object."""
    attack = AttributeInference(target_model, config, device="cpu")
    attack.run(member_loader, nonmember_loader)
    return attack


# ── Per-class evaluation ─────────────────────────────────────────────────


class TestEvaluatePerClass:
    def test_returns_dict(self, attack_with_result) -> None:
        per_class = attack_with_result.evaluate_per_class()
        assert isinstance(per_class, dict)
        assert len(per_class) > 0

    def test_keys_are_class_labels(self, attack_with_result) -> None:
        per_class = attack_with_result.evaluate_per_class()
        for k in per_class:
            assert isinstance(k, int)
            assert 0 <= k <= 9

    def test_each_class_has_metrics(self, attack_with_result) -> None:
        per_class = attack_with_result.evaluate_per_class()
        for cls, metrics in per_class.items():
            assert "accuracy" in metrics
            assert "n_samples" in metrics
            assert metrics["n_samples"] > 0

    def test_accuracy_in_range(self, attack_with_result) -> None:
        per_class = attack_with_result.evaluate_per_class()
        for cls, metrics in per_class.items():
            assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_before_run_raises(self, target_model, config) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.evaluate_per_class()


# ── Per-group evaluation ─────────────────────────────────────────────────


class TestEvaluatePerGroup:
    def test_returns_member_and_nonmember(self, attack_with_result) -> None:
        per_group = attack_with_result.evaluate_per_group()
        assert "member" in per_group
        assert "nonmember" in per_group

    def test_group_ids_valid(self, attack_with_result) -> None:
        per_group = attack_with_result.evaluate_per_group()
        for split in ("member", "nonmember"):
            for g in per_group[split]:
                assert isinstance(g, int)
                assert 0 <= g < attack_with_result.num_groups

    def test_accuracy_in_range(self, attack_with_result) -> None:
        per_group = attack_with_result.evaluate_per_group()
        for split in ("member", "nonmember"):
            for g, acc in per_group[split].items():
                assert 0.0 <= acc <= 1.0

    def test_before_run_raises(self, target_model, config) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.evaluate_per_group()


# ── plot_attribute_accuracy ──────────────────────────────────────────────


class TestPlotAttributeAccuracy:
    def test_returns_figure(self) -> None:
        mem = {0: 0.8, 1: 0.7, 2: 0.6}
        nonmem = {0: 0.5, 1: 0.4, 2: 0.3}
        fig = plot_attribute_accuracy(mem, nonmem)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path) -> None:
        mem = {0: 0.8, 1: 0.7}
        nonmem = {0: 0.5, 1: 0.4}
        save = tmp_path / "attr_acc.png"
        plot_attribute_accuracy(mem, nonmem, save_path=save)
        assert save.exists()
        assert save.stat().st_size > 0

    def test_custom_title(self) -> None:
        mem = {0: 0.9}
        nonmem = {0: 0.4}
        fig = plot_attribute_accuracy(mem, nonmem, title="Custom Title")
        assert isinstance(fig, plt.Figure)

    def test_mismatched_groups_handled(self) -> None:
        mem = {0: 0.8, 1: 0.7}
        nonmem = {0: 0.5, 2: 0.3}  # group 2 only in nonmember
        fig = plot_attribute_accuracy(mem, nonmem)
        assert isinstance(fig, plt.Figure)


# ── Report generation ────────────────────────────────────────────────────


class TestGenerateReport:
    def test_creates_output_dir(self, attack_with_result, tmp_path) -> None:
        out = tmp_path / "report"
        attack_with_result.generate_report(out)
        assert out.is_dir()

    def test_creates_metrics_json(self, attack_with_result, tmp_path) -> None:
        out = tmp_path / "report"
        attack_with_result.generate_report(out)
        metrics_file = out / "metrics.json"
        assert metrics_file.exists()
        data = json.loads(metrics_file.read_text())
        assert "accuracy" in data
        assert "auc_roc" in data

    def test_creates_per_class_json(self, attack_with_result, tmp_path) -> None:
        out = tmp_path / "report"
        attack_with_result.generate_report(out)
        pc_file = out / "per_class_metrics.json"
        assert pc_file.exists()
        data = json.loads(pc_file.read_text())
        assert len(data) > 0

    def test_creates_per_group_json(self, attack_with_result, tmp_path) -> None:
        out = tmp_path / "report"
        attack_with_result.generate_report(out)
        pg_file = out / "per_group_accuracy.json"
        assert pg_file.exists()
        data = json.loads(pg_file.read_text())
        assert "member" in data
        assert "nonmember" in data

    def test_creates_roc_curve_png(self, attack_with_result, tmp_path) -> None:
        out = tmp_path / "report"
        attack_with_result.generate_report(out)
        assert (out / "roc_curve.png").exists()

    def test_creates_confidence_distributions_png(
        self, attack_with_result, tmp_path,
    ) -> None:
        out = tmp_path / "report"
        attack_with_result.generate_report(out)
        assert (out / "confidence_distributions.png").exists()

    def test_creates_per_class_accuracy_png(
        self, attack_with_result, tmp_path,
    ) -> None:
        out = tmp_path / "report"
        attack_with_result.generate_report(out)
        assert (out / "per_class_accuracy.png").exists()

    def test_creates_attribute_accuracy_png(
        self, attack_with_result, tmp_path,
    ) -> None:
        out = tmp_path / "report"
        attack_with_result.generate_report(out)
        assert (out / "attribute_accuracy.png").exists()

    def test_creates_summary_txt(self, attack_with_result, tmp_path) -> None:
        out = tmp_path / "report"
        attack_with_result.generate_report(out)
        summary = out / "summary.txt"
        assert summary.exists()
        text = summary.read_text()
        assert "Attribute Inference" in text
        assert "Overall Metrics" in text
        assert "Per-Group" in text

    def test_before_run_raises(self, target_model, config, tmp_path) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.generate_report(tmp_path / "report")

    def test_returns_output_dir(self, attack_with_result, tmp_path) -> None:
        out = tmp_path / "report"
        result = attack_with_result.generate_report(out)
        assert result == out
