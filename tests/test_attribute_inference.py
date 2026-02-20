"""Tests for Task 2.8 — Attribute Inference Core.

Tests the AttributeInference attack and AttributeAttackMLP using
synthetic data.  Keeps things fast by using small models and datasets.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from auditml.attacks.attribute_inference import AttributeAttackMLP, AttributeInference
from auditml.attacks.results import AttackResult
from auditml.config import default_config
from auditml.config.schema import DatasetName


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def config():
    cfg = default_config()
    cfg.model.num_classes = 10
    cfg.model.arch = "cnn"
    cfg.data.dataset = DatasetName.MNIST  # 1-channel, matches synthetic data
    return cfg


@pytest.fixture()
def target_model():
    """Small CNN that accepts 1×28×28 input, outputs 10 classes."""
    from auditml.models import SimpleCNN

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


# ── AttributeAttackMLP tests ─────────────────────────────────────────────


class TestAttributeAttackMLP:
    def test_forward_shape(self) -> None:
        model = AttributeAttackMLP(input_dim=10, num_groups=5)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 5)

    def test_predict_proba_shape(self) -> None:
        model = AttributeAttackMLP(input_dim=10, num_groups=5)
        x = torch.randn(8, 10)
        probs = model.predict_proba(x)
        assert probs.shape == (8, 5)

    def test_predict_proba_sums_to_one(self) -> None:
        model = AttributeAttackMLP(input_dim=10, num_groups=5)
        x = torch.randn(16, 10)
        probs = model.predict_proba(x)
        np.testing.assert_allclose(
            probs.sum(dim=1).numpy(), 1.0, atol=1e-5,
        )

    def test_custom_hidden_dim(self) -> None:
        model = AttributeAttackMLP(input_dim=10, num_groups=3, hidden_dim=32)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_gradient_flows(self) -> None:
        model = AttributeAttackMLP(input_dim=10, num_groups=5)
        x = torch.randn(4, 10)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ── AttributeInference class-to-group mapping ────────────────────────────


class TestGroupMapping:
    def test_default_mnist_groups(self, target_model, config) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        assert attack.num_groups == 5
        # 0,1 → group 0; 2,3 → group 1; etc.
        assert attack.class_to_group[0] == 0
        assert attack.class_to_group[1] == 0
        assert attack.class_to_group[8] == 4
        assert attack.class_to_group[9] == 4

    def test_default_cifar10_groups(self, target_model, config) -> None:
        config.data.dataset = DatasetName.CIFAR10
        attack = AttributeInference(target_model, config, device="cpu")
        assert attack.num_groups == 5

    def test_cifar100_auto_groups(self, target_model, config) -> None:
        config.data.dataset = DatasetName.CIFAR100
        config.model.num_classes = 100
        attack = AttributeInference(target_model, config, device="cpu")
        # 100 classes → 20 groups (100 // 5)
        assert attack.num_groups == 20

    def test_custom_num_groups(self, target_model, config) -> None:
        attack = AttributeInference(
            target_model, config, device="cpu", num_groups=3,
        )
        assert attack.num_groups == 3
        # All 10 classes mapped to 3 groups via modulo
        for c in range(10):
            assert attack.class_to_group[c] == c % 3

    def test_custom_class_to_group(self, target_model, config) -> None:
        mapping = {i: 0 if i < 5 else 1 for i in range(10)}
        attack = AttributeInference(
            target_model, config, device="cpu", class_to_group=mapping,
        )
        assert attack.num_groups == 2
        assert attack.class_to_group == mapping

    def test_labels_to_groups(self, target_model, config) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        labels = np.array([0, 1, 4, 5, 8, 9])
        groups = attack._labels_to_groups(labels)
        np.testing.assert_array_equal(groups, [0, 0, 2, 2, 4, 4])


# ── Full attack flow ─────────────────────────────────────────────────────


class TestAttributeInferenceRun:
    def test_run_returns_attack_result(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        assert isinstance(result, AttackResult)
        assert result.attack_name == "attribute_inference"

    def test_result_shapes(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        total = 80 + 80  # member + nonmember
        assert len(result.predictions) == total
        assert len(result.ground_truth) == total
        assert len(result.confidence_scores) == total

    def test_ground_truth_labels(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        # First 80 are members (1), last 80 are non-members (0)
        assert result.ground_truth[:80].sum() == 80
        assert result.ground_truth[80:].sum() == 0

    def test_predictions_binary(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        unique_vals = set(np.unique(result.predictions))
        assert unique_vals.issubset({0, 1})

    def test_confidence_scores_bounded(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        # Confidence = softmax probability → [0, 1]
        assert (result.confidence_scores >= 0.0).all()
        assert (result.confidence_scores <= 1.0).all()

    def test_metadata_populated(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        assert "num_groups" in result.metadata
        assert "sensitive_attribute" in result.metadata
        assert "mean_member_attr_confidence" in result.metadata
        assert "mean_nonmember_attr_confidence" in result.metadata

    def test_member_labels_stored(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        assert attack.member_labels is not None
        assert attack.nonmember_labels is not None
        assert len(attack.member_labels) == 80
        assert len(attack.nonmember_labels) == 80

    def test_confidence_stored(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        assert attack.member_confidence is not None
        assert attack.nonmember_confidence is not None
        assert len(attack.member_confidence) == 80
        assert len(attack.nonmember_confidence) == 80

    def test_attack_model_trained(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        assert attack.attack_model is not None
        assert isinstance(attack.attack_model, AttributeAttackMLP)

    def test_evaluate_after_run(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        metrics = attack.evaluate()
        assert "accuracy" in metrics
        assert "auc_roc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_before_run_raises(
        self, target_model, config,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.evaluate()


# ── Predict attributes ────────────────────────────────────────────────────


class TestPredictAttributes:
    def test_predict_attributes_shape(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        probs = np.random.rand(20, 10).astype(np.float32)
        preds = attack.predict_attributes(probs)
        assert preds.shape == (20,)
        # Predictions should be valid group IDs
        assert (preds >= 0).all()
        assert (preds < attack.num_groups).all()

    def test_predict_attributes_before_run_raises(
        self, target_model, config,
    ) -> None:
        attack = AttributeInference(target_model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.predict_attributes(np.random.rand(5, 10).astype(np.float32))


# ── Factory integration ──────────────────────────────────────────────────


class TestFactoryIntegration:
    def test_get_attack_attribute_inference(self, target_model, config) -> None:
        from auditml.attacks import get_attack

        attack = get_attack("attribute_inference", target_model, config)
        assert isinstance(attack, AttributeInference)
        assert attack.attack_name == "attribute_inference"
