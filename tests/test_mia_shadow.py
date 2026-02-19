"""Tests for Task 2.4 — Shadow Model MIA Core.

Uses pre-trained shadow models with synthetic data to keep tests fast.
Shadow training from scratch is tested with a minimal configuration.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from auditml.attacks.mia_shadow import AttackMLP, ShadowMIA
from auditml.attacks.results import AttackResult
from auditml.config import default_config
from auditml.config.schema import DatasetName
from auditml.models import SimpleCNN


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def config():
    cfg = default_config()
    # Use small values for fast tests
    cfg.attack_params.mia_shadow.num_shadow_models = 2
    cfg.attack_params.mia_shadow.shadow_epochs = 2
    cfg.model.num_classes = 10
    cfg.model.arch = "cnn"
    cfg.data.dataset = DatasetName.MNIST  # match synthetic 1-channel data
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
    ds = TensorDataset(torch.randn(60, 1, 28, 28), torch.randint(0, 10, (60,)))
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def nonmember_loader() -> DataLoader:
    torch.manual_seed(2)
    ds = TensorDataset(torch.randn(60, 1, 28, 28), torch.randint(0, 10, (60,)))
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def shadow_dataset() -> TensorDataset:
    """Synthetic dataset for shadow model training."""
    torch.manual_seed(3)
    return TensorDataset(torch.randn(200, 1, 28, 28), torch.randint(0, 10, (200,)))


@pytest.fixture()
def pre_trained_shadows(config) -> list[tuple[nn.Module, DataLoader, DataLoader]]:
    """Create pre-trained shadow models to skip the slow training step."""
    shadows = []
    for i in range(2):
        torch.manual_seed(10 + i)
        shadow = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
        shadow.eval()

        torch.manual_seed(20 + i)
        mem_ds = TensorDataset(torch.randn(40, 1, 28, 28), torch.randint(0, 10, (40,)))
        nonmem_ds = TensorDataset(torch.randn(40, 1, 28, 28), torch.randint(0, 10, (40,)))

        mem_loader = DataLoader(mem_ds, batch_size=16)
        nonmem_loader = DataLoader(nonmem_ds, batch_size=16)
        shadows.append((shadow, mem_loader, nonmem_loader))

    return shadows


@pytest.fixture()
def run_attack_pretrained(
    target_model, config, member_loader, nonmember_loader, pre_trained_shadows,
) -> ShadowMIA:
    """ShadowMIA that has been run with pre-trained shadows."""
    attack = ShadowMIA(
        target_model, config, device="cpu",
        shadow_models=pre_trained_shadows,
    )
    attack.run(member_loader, nonmember_loader)
    return attack


# ── AttackMLP tests ──────────────────────────────────────────────────────

class TestAttackMLP:
    def test_output_shape(self) -> None:
        model = AttackMLP(input_dim=10)
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5, 1)

    def test_predict_proba_range(self) -> None:
        model = AttackMLP(input_dim=10)
        model.eval()
        x = torch.randn(20, 10)
        probs = model.predict_proba(x)
        assert probs.shape == (20,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_different_input_dims(self) -> None:
        for dim in [2, 10, 100]:
            model = AttackMLP(input_dim=dim)
            x = torch.randn(3, dim)
            out = model(x)
            assert out.shape == (3, 1)


# ── ShadowMIA with pre-trained shadows ───────────────────────────────────

class TestShadowMIAPreTrained:
    def test_run_returns_attack_result(self, run_attack_pretrained) -> None:
        assert isinstance(run_attack_pretrained.result, AttackResult)

    def test_result_lengths_match(self, run_attack_pretrained) -> None:
        r = run_attack_pretrained.result
        n = len(r.ground_truth)
        assert n == 120  # 60 members + 60 non-members
        assert len(r.predictions) == n
        assert len(r.confidence_scores) == n

    def test_ground_truth_correct(self, run_attack_pretrained) -> None:
        gt = run_attack_pretrained.result.ground_truth
        assert gt[:60].sum() == 60   # first 60 are members
        assert gt[60:].sum() == 0    # last 60 are non-members

    def test_predictions_are_binary(self, run_attack_pretrained) -> None:
        preds = run_attack_pretrained.result.predictions
        assert set(np.unique(preds)).issubset({0, 1})

    def test_confidence_scores_in_range(self, run_attack_pretrained) -> None:
        scores = run_attack_pretrained.result.confidence_scores
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_metadata_stored(self, run_attack_pretrained) -> None:
        meta = run_attack_pretrained.result.metadata
        assert "num_shadow_models" in meta
        assert "attack_train_samples" in meta
        assert "member_mean_confidence" in meta
        assert "nonmember_mean_confidence" in meta
        assert meta["num_shadow_models"] == 2

    def test_attack_model_trained(self, run_attack_pretrained) -> None:
        assert run_attack_pretrained.attack_model is not None
        assert isinstance(run_attack_pretrained.attack_model, AttackMLP)

    def test_evaluate_returns_all_metrics(self, run_attack_pretrained) -> None:
        metrics = run_attack_pretrained.evaluate()
        expected = [
            "accuracy", "precision", "recall", "f1",
            "auc_roc", "auc_pr", "tpr_at_1fpr", "tpr_at_01fpr",
        ]
        for key in expected:
            assert key in metrics, f"Missing metric: {key}"
            assert 0.0 <= metrics[key] <= 1.0

    def test_attack_name(self, run_attack_pretrained) -> None:
        assert run_attack_pretrained.result.attack_name == "mia_shadow"


# ── Shadow training from dataset ─────────────────────────────────────────

class TestShadowMIAFromDataset:
    def test_train_shadow_models(
        self, target_model, config, member_loader, nonmember_loader, shadow_dataset,
    ) -> None:
        """Test full pipeline including shadow model training."""
        # Use minimal config for speed
        config.attack_params.mia_shadow.num_shadow_models = 1
        config.attack_params.mia_shadow.shadow_epochs = 1
        config.data.dataset = config.data.dataset

        attack = ShadowMIA(
            target_model, config, device="cpu",
            shadow_dataset=shadow_dataset,
        )
        result = attack.run(member_loader, nonmember_loader)

        assert isinstance(result, AttackResult)
        assert len(result.predictions) == 120
        assert len(attack.trained_shadows) == 1

    def test_raises_without_dataset_or_models(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = ShadowMIA(target_model, config, device="cpu")
        with pytest.raises(ValueError, match="shadow_dataset or shadow_models"):
            attack.run(member_loader, nonmember_loader)


# ── Factory integration ──────────────────────────────────────────────────

class TestFactoryIntegration:
    def test_get_attack_returns_shadow_mia(self, target_model, config) -> None:
        from auditml.attacks import get_attack
        attack = get_attack("mia_shadow", target_model, config)
        assert isinstance(attack, ShadowMIA)

    def test_factory_created_attack_needs_data(self, target_model, config, member_loader, nonmember_loader) -> None:
        from auditml.attacks import get_attack
        attack = get_attack("mia_shadow", target_model, config)
        # Factory doesn't set shadow_dataset, so should raise
        with pytest.raises(ValueError, match="shadow_dataset or shadow_models"):
            attack.run(member_loader, nonmember_loader)


# ── Edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_evaluate_before_run_raises(self, target_model, config) -> None:
        attack = ShadowMIA(target_model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.evaluate()

    def test_single_shadow_model(
        self, target_model, config, member_loader, nonmember_loader,
    ) -> None:
        """Works even with just 1 shadow model."""
        torch.manual_seed(99)
        shadow = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
        shadow.eval()
        mem_ds = TensorDataset(torch.randn(30, 1, 28, 28), torch.randint(0, 10, (30,)))
        nonmem_ds = TensorDataset(torch.randn(30, 1, 28, 28), torch.randint(0, 10, (30,)))

        shadows = [(shadow, DataLoader(mem_ds, batch_size=16), DataLoader(nonmem_ds, batch_size=16))]

        config.attack_params.mia_shadow.num_shadow_models = 1
        attack = ShadowMIA(
            target_model, config, device="cpu",
            shadow_models=shadows,
        )
        result = attack.run(member_loader, nonmember_loader)
        assert len(result.predictions) == 120
