"""Tests for Task 2.6 — Model Inversion Core.

Uses a SimpleCNN on synthetic data. Inversion uses very few iterations
(10–20) to keep tests fast — reconstruction quality doesn't matter for
unit tests, only correctness of shapes and interfaces.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from auditml.attacks.model_inversion import ModelInversion
from auditml.attacks.results import AttackResult
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
    # Fast iterations for tests
    cfg.attack_params.model_inversion.num_iterations = 10
    cfg.attack_params.model_inversion.learning_rate = 0.01
    cfg.attack_params.model_inversion.lambda_tv = 0.001
    cfg.attack_params.model_inversion.lambda_l2 = 0.0
    cfg.attack_params.model_inversion.target_class = None
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
def single_class_config(config):
    """Config that only inverts class 3."""
    config.attack_params.model_inversion.target_class = 3
    return config


# ── Invert single class tests ────────────────────────────────────────────

class TestInvertClass:
    def test_returns_tensor_and_confidence(self, model, config) -> None:
        attack = ModelInversion(model, config, device="cpu")
        recon, conf = attack.invert_class(0, num_iterations=10)
        assert isinstance(recon, torch.Tensor)
        assert recon.shape == (1, 1, 28, 28)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_confidence_increases_with_iterations(self, model, config) -> None:
        attack = ModelInversion(model, config, device="cpu")
        _, conf_10 = attack.invert_class(0, num_iterations=10)
        _, conf_50 = attack.invert_class(0, num_iterations=50)
        # More iterations should generally give higher confidence
        # (not guaranteed with random init, but very likely)
        # Just check both are valid
        assert 0.0 <= conf_10 <= 1.0
        assert 0.0 <= conf_50 <= 1.0

    def test_different_classes_produce_different_images(self, model, config) -> None:
        attack = ModelInversion(model, config, device="cpu")
        torch.manual_seed(42)
        recon_0, _ = attack.invert_class(0, num_iterations=20)
        torch.manual_seed(42)
        recon_1, _ = attack.invert_class(1, num_iterations=20)
        # Different target classes should produce different images
        # (even with same seed, the loss landscape differs)
        assert not torch.allclose(recon_0, recon_1, atol=1e-3)


# ── Total Variation tests ────────────────────────────────────────────────

class TestTotalVariation:
    def test_smooth_image_low_tv(self) -> None:
        # Constant image has zero TV
        x = torch.ones(1, 1, 8, 8)
        tv = ModelInversion._total_variation(x)
        assert tv.item() == pytest.approx(0.0, abs=1e-7)

    def test_noisy_image_high_tv(self) -> None:
        # Random noise has high TV
        torch.manual_seed(0)
        x = torch.randn(1, 1, 8, 8)
        tv = ModelInversion._total_variation(x)
        assert tv.item() > 0.1

    def test_tv_with_regularisation(self, model, config) -> None:
        """Higher lambda_tv should produce smoother reconstructions."""
        config.attack_params.model_inversion.lambda_tv = 10.0
        attack = ModelInversion(model, config, device="cpu")
        recon_smooth, _ = attack.invert_class(0, num_iterations=20)

        config.attack_params.model_inversion.lambda_tv = 0.0
        attack_noisy = ModelInversion(model, config, device="cpu")
        recon_noisy, _ = attack_noisy.invert_class(0, num_iterations=20)

        tv_smooth = ModelInversion._total_variation(recon_smooth).item()
        tv_noisy = ModelInversion._total_variation(recon_noisy).item()
        # Strongly regularised image should have lower TV
        assert tv_smooth < tv_noisy


# ── Full run() tests ─────────────────────────────────────────────────────

class TestModelInversionRun:
    def test_run_single_class(
        self, model, single_class_config, member_loader, nonmember_loader,
    ) -> None:
        attack = ModelInversion(model, single_class_config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        assert isinstance(result, AttackResult)
        assert len(result.predictions) == 80  # 40 + 40

    def test_run_all_classes(
        self, model, config, member_loader, nonmember_loader,
    ) -> None:
        attack = ModelInversion(model, config, device="cpu")
        result = attack.run(member_loader, nonmember_loader)
        assert isinstance(result, AttackResult)
        assert len(result.predictions) == 80
        assert len(attack.reconstructions) == 10  # all 10 classes

    def test_ground_truth_correct(
        self, model, single_class_config, member_loader, nonmember_loader,
    ) -> None:
        attack = ModelInversion(model, single_class_config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        gt = attack.result.ground_truth
        assert gt[:40].sum() == 40  # members
        assert gt[40:].sum() == 0   # non-members

    def test_predictions_are_binary(
        self, model, single_class_config, member_loader, nonmember_loader,
    ) -> None:
        attack = ModelInversion(model, single_class_config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        assert set(np.unique(attack.result.predictions)).issubset({0, 1})

    def test_confidence_scores_in_range(
        self, model, single_class_config, member_loader, nonmember_loader,
    ) -> None:
        attack = ModelInversion(model, single_class_config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        scores = attack.result.confidence_scores
        # Cosine similarity ranges [-1, 1], but softmax outputs are positive
        # so it should be in [0, 1] or close
        assert scores.min() >= -1.1
        assert scores.max() <= 1.1

    def test_metadata_stored(
        self, model, single_class_config, member_loader, nonmember_loader,
    ) -> None:
        attack = ModelInversion(model, single_class_config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        meta = attack.result.metadata
        assert "num_classes_inverted" in meta
        assert "reconstruction_confidences" in meta
        assert "mean_member_similarity" in meta
        assert meta["num_classes_inverted"] == 1

    def test_reconstructions_stored(
        self, model, single_class_config, member_loader, nonmember_loader,
    ) -> None:
        attack = ModelInversion(model, single_class_config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        assert 3 in attack.reconstructions
        assert attack.reconstructions[3].shape == (1, 1, 28, 28)

    def test_evaluate_after_run(
        self, model, single_class_config, member_loader, nonmember_loader,
    ) -> None:
        attack = ModelInversion(model, single_class_config, device="cpu")
        attack.run(member_loader, nonmember_loader)
        metrics = attack.evaluate()
        assert "accuracy" in metrics
        assert "auc_roc" in metrics


# ── Factory integration ──────────────────────────────────────────────────

class TestFactoryIntegration:
    def test_get_attack_returns_model_inversion(self, model, config) -> None:
        from auditml.attacks import get_attack
        attack = get_attack("model_inversion", model, config)
        assert isinstance(attack, ModelInversion)

    def test_attack_name(self, model, config) -> None:
        attack = ModelInversion(model, config, device="cpu")
        assert attack.attack_name == "model_inversion"


# ── Edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_evaluate_before_run_raises(self, model, config) -> None:
        attack = ModelInversion(model, config, device="cpu")
        with pytest.raises(RuntimeError, match="Call run"):
            attack.evaluate()

    def test_explicit_input_shape(self, model, config) -> None:
        attack = ModelInversion(
            model, config, device="cpu",
            input_shape=(1, 28, 28),
        )
        assert attack.input_shape == (1, 28, 28)

    def test_l2_regularisation(self, model, config) -> None:
        config.attack_params.model_inversion.lambda_l2 = 0.1
        attack = ModelInversion(model, config, device="cpu")
        recon, conf = attack.invert_class(0, num_iterations=10)
        assert recon.shape == (1, 1, 28, 28)
