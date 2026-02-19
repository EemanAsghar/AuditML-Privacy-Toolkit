"""Tests for the attack base class and result container.

Uses a concrete dummy attack (inheriting BaseAttack) to test the shared
utilities, since BaseAttack itself is abstract and cannot be instantiated.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from auditml.attacks.base import BaseAttack
from auditml.attacks.results import AttackResult
from auditml.attacks import get_attack
from auditml.config import AuditMLConfig, default_config
from auditml.models import SimpleCNN


# ── Concrete dummy attack for testing ────────────────────────────────────

class _DummyAttack(BaseAttack):
    """Minimal concrete attack that predicts randomly."""

    attack_name = "dummy"

    def run(self, member_loader, nonmember_loader):
        # Get loss values for members and non-members
        member_losses = self.get_loss_values(member_loader)
        nonmember_losses = self.get_loss_values(nonmember_loader)

        # Ground truth: 1 for members, 0 for non-members
        ground_truth = np.concatenate([
            np.ones(len(member_losses)),
            np.zeros(len(nonmember_losses)),
        ])

        # Use negative loss as confidence (lower loss = more likely member)
        all_losses = np.concatenate([member_losses, nonmember_losses])
        confidence_scores = -all_losses  # negate so higher = more likely member

        # Simple threshold: predict member if loss < median
        threshold = np.median(all_losses)
        predictions = (all_losses < threshold).astype(int)

        self.result = AttackResult(
            predictions=predictions,
            ground_truth=ground_truth,
            confidence_scores=confidence_scores,
            attack_name=self.attack_name,
        )
        return self.result


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def model() -> SimpleCNN:
    m = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
    m.eval()
    return m


@pytest.fixture()
def config() -> AuditMLConfig:
    return default_config()


@pytest.fixture()
def member_loader() -> DataLoader:
    ds = TensorDataset(torch.randn(50, 1, 28, 28), torch.randint(0, 10, (50,)))
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def nonmember_loader() -> DataLoader:
    ds = TensorDataset(torch.randn(50, 1, 28, 28), torch.randint(0, 10, (50,)))
    return DataLoader(ds, batch_size=16)


# ── AttackResult tests ───────────────────────────────────────────────────

class TestAttackResult:
    def test_valid_result(self) -> None:
        r = AttackResult(
            predictions=np.array([1, 0, 1]),
            ground_truth=np.array([1, 1, 0]),
            confidence_scores=np.array([0.9, 0.3, 0.7]),
            attack_name="test",
        )
        assert r.attack_name == "test"
        assert len(r.predictions) == 3

    def test_mismatched_predictions_raises(self) -> None:
        with pytest.raises(ValueError, match="predictions length"):
            AttackResult(
                predictions=np.array([1, 0]),  # 2 elements
                ground_truth=np.array([1, 1, 0]),  # 3 elements
                confidence_scores=np.array([0.9, 0.3, 0.7]),
            )

    def test_mismatched_scores_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_scores length"):
            AttackResult(
                predictions=np.array([1, 0, 1]),
                ground_truth=np.array([1, 1, 0]),
                confidence_scores=np.array([0.9]),  # 1 element
            )

    def test_metadata_default_empty(self) -> None:
        r = AttackResult(
            predictions=np.array([1]),
            ground_truth=np.array([0]),
            confidence_scores=np.array([0.5]),
        )
        assert r.metadata == {}


# ── BaseAttack cannot be instantiated ────────────────────────────────────

class TestBaseAttackAbstract:
    def test_cannot_instantiate(self, model, config) -> None:
        with pytest.raises(TypeError):
            BaseAttack(model, config)  # type: ignore[abstract]


# ── Shared utility methods ───────────────────────────────────────────────

class TestBaseAttackUtilities:
    def test_model_set_to_eval(self, model, config) -> None:
        model.train()  # deliberately set to train mode
        attack = _DummyAttack(model, config, device="cpu")
        assert not attack.target_model.training  # should be eval

    def test_get_model_outputs_shapes(self, model, config, member_loader) -> None:
        attack = _DummyAttack(model, config, device="cpu")
        probs, logits, labels = attack.get_model_outputs(member_loader)

        assert probs.shape == (50, 10)
        assert logits.shape == (50, 10)
        assert labels.shape == (50,)

    def test_get_model_outputs_probabilities_sum_to_one(
        self, model, config, member_loader,
    ) -> None:
        attack = _DummyAttack(model, config, device="cpu")
        probs, _, _ = attack.get_model_outputs(member_loader)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_get_loss_values_shape(self, model, config, member_loader) -> None:
        attack = _DummyAttack(model, config, device="cpu")
        losses = attack.get_loss_values(member_loader)
        assert losses.shape == (50,)

    def test_get_loss_values_positive(self, model, config, member_loader) -> None:
        attack = _DummyAttack(model, config, device="cpu")
        losses = attack.get_loss_values(member_loader)
        assert (losses >= 0).all()


# ── Full run + evaluate flow ─────────────────────────────────────────────

class TestDummyAttackFlow:
    def test_run_returns_result(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = _DummyAttack(model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        assert isinstance(result, AttackResult)
        assert len(result.predictions) == 100  # 50 members + 50 non-members
        assert len(result.ground_truth) == 100

    def test_evaluate_before_run_raises(self, model, config) -> None:
        attack = _DummyAttack(model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.evaluate()

    def test_evaluate_returns_metrics(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = _DummyAttack(model, config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        metrics = attack.evaluate()

        expected_keys = [
            "accuracy", "precision", "recall", "f1",
            "auc_roc", "auc_pr", "tpr_at_1fpr", "tpr_at_01fpr",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float)

    def test_accuracy_in_valid_range(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = _DummyAttack(model, config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        metrics = attack.evaluate()
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc_roc"] <= 1.0


# ── Metrics computation directly ─────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        gt = np.array([1, 1, 0, 0])
        preds = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        metrics = BaseAttack._compute_metrics(preds, gt, scores)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["auc_roc"] == 1.0

    def test_random_predictions(self) -> None:
        rng = np.random.RandomState(42)
        gt = np.concatenate([np.ones(500), np.zeros(500)])
        scores = rng.random(1000)
        preds = (scores > 0.5).astype(int)
        metrics = BaseAttack._compute_metrics(preds, gt, scores)
        # Random should be close to 0.5
        assert 0.3 <= metrics["accuracy"] <= 0.7
        assert 0.3 <= metrics["auc_roc"] <= 0.7

    def test_all_same_label_handled(self) -> None:
        gt = np.ones(10)  # all members — edge case
        preds = np.ones(10)
        scores = np.ones(10) * 0.9
        metrics = BaseAttack._compute_metrics(preds, gt, scores)
        # AUC undefined with single class — should return 0.0, not crash
        assert metrics["auc_roc"] == 0.0


# ── Factory function ─────────────────────────────────────────────────────

class TestGetAttack:
    def test_unimplemented_attack_raises(self, model, config) -> None:
        with pytest.raises(ValueError, match="not yet implemented"):
            get_attack("model_inversion", model, config)

    def test_invalid_attack_name_raises(self, model, config) -> None:
        with pytest.raises(ValueError):
            get_attack("not_a_real_attack", model, config)
