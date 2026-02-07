"""Tests for the Threshold-based Membership Inference Attack.

Uses a SimpleCNN on synthetic data. Because the model is untrained and
the data is random, we do NOT expect strong attack performance — the
tests verify correctness of the pipeline, not attack effectiveness.
Real effectiveness is validated when you run on trained baseline models.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from auditml.attacks import get_attack
from auditml.attacks.mia_threshold import ThresholdMIA
from auditml.attacks.results import AttackResult
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


# ── Core run/evaluate tests ──────────────────────────────────────────────

class TestThresholdMIARun:
    def test_run_returns_attack_result(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = ThresholdMIA(model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        assert isinstance(result, AttackResult)

    def test_result_lengths_match(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = ThresholdMIA(model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        total = 80 + 80  # member + nonmember samples
        assert len(result.predictions) == total
        assert len(result.ground_truth) == total
        assert len(result.confidence_scores) == total

    def test_ground_truth_correct(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = ThresholdMIA(model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        # First 80 should be members (1), last 80 should be non-members (0)
        assert np.all(result.ground_truth[:80] == 1)
        assert np.all(result.ground_truth[80:] == 0)

    def test_predictions_are_binary(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = ThresholdMIA(model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        unique_vals = set(np.unique(result.predictions))
        assert unique_vals.issubset({0, 1})

    def test_metadata_stored(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = ThresholdMIA(model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        assert "metric" in result.metadata
        assert "threshold" in result.metadata
        assert "member_mean" in result.metadata
        assert "nonmember_mean" in result.metadata

    def test_intermediate_scores_stored(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = ThresholdMIA(model, config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        assert attack.member_scores is not None
        assert attack.nonmember_scores is not None
        assert attack.threshold is not None
        assert len(attack.member_scores) == 80
        assert len(attack.nonmember_scores) == 80

    def test_evaluate_returns_all_metrics(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = ThresholdMIA(model, config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        metrics = attack.evaluate()
        expected_keys = [
            "accuracy", "precision", "recall", "f1",
            "auc_roc", "auc_pr", "tpr_at_1fpr", "tpr_at_01fpr",
        ]
        for key in expected_keys:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0


# ── All three metric types ───────────────────────────────────────────────

class TestMetricTypes:
    """Verify that all three signal metrics run without errors."""

    def _run_with_metric(self, model, config, member_loader, nonmember_loader, metric):
        config.attack_params.mia_threshold.metric = metric
        attack = ThresholdMIA(model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        metrics = attack.evaluate()
        return result, metrics

    def test_loss_metric(self, model, config, member_loader, nonmember_loader) -> None:
        result, metrics = self._run_with_metric(
            model, config, member_loader, nonmember_loader, "loss",
        )
        assert result.metadata["metric"] == "loss"
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_confidence_metric(self, model, config, member_loader, nonmember_loader) -> None:
        result, metrics = self._run_with_metric(
            model, config, member_loader, nonmember_loader, "confidence",
        )
        assert result.metadata["metric"] == "confidence"
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_entropy_metric(self, model, config, member_loader, nonmember_loader) -> None:
        result, metrics = self._run_with_metric(
            model, config, member_loader, nonmember_loader, "entropy",
        )
        assert result.metadata["metric"] == "entropy"
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_invalid_metric_raises(self, model, config, member_loader, nonmember_loader) -> None:
        config.attack_params.mia_threshold.metric = "invalid_metric"
        attack = ThresholdMIA(model, config, device="cpu")
        with pytest.raises(ValueError, match="Unknown metric"):
            attack.run(member_loader, nonmember_loader)


# ── Threshold logic ──────────────────────────────────────────────────────

class TestThresholdLogic:
    def test_threshold_is_finite(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = ThresholdMIA(model, config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        assert np.isfinite(attack.threshold)

    def test_loss_direction(self, model, config, member_loader, nonmember_loader) -> None:
        """For loss metric: score <= threshold → predicted member."""
        config.attack_params.mia_threshold.metric = "loss"
        attack = ThresholdMIA(model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        all_scores = np.concatenate([attack.member_scores, attack.nonmember_scores])
        # Verify direction: samples with loss <= threshold are predicted as members
        expected = (all_scores <= attack.threshold).astype(int)
        np.testing.assert_array_equal(result.predictions, expected)

    def test_confidence_direction(self, model, config, member_loader, nonmember_loader) -> None:
        """For confidence metric: score >= threshold → predicted member."""
        config.attack_params.mia_threshold.metric = "confidence"
        attack = ThresholdMIA(model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        all_scores = np.concatenate([attack.member_scores, attack.nonmember_scores])
        expected = (all_scores >= attack.threshold).astype(int)
        np.testing.assert_array_equal(result.predictions, expected)


# ── Factory integration ──────────────────────────────────────────────────

class TestFactoryIntegration:
    def test_get_attack_returns_threshold_mia(self, model, config) -> None:
        attack = get_attack("mia_threshold", model, config, device="cpu")
        assert isinstance(attack, ThresholdMIA)

    def test_factory_created_attack_runs(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = get_attack("mia_threshold", model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        assert isinstance(result, AttackResult)
        assert len(result.predictions) == 160
