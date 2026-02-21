"""Tests for Task 2.10 — Differential Privacy Defense Integration.

Tests the DPTrainer, validate_and_fix_model, is_dp_compatible, and
the end-to-end DP training flow using Opacus.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from auditml.config.schema import DPConfig
from auditml.models import SimpleCNN, SmallResNet
from auditml.training.dp_trainer import (
    DPTrainer,
    is_dp_compatible,
    validate_and_fix_model,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def dp_config() -> DPConfig:
    return DPConfig(
        enabled=True,
        epsilon=10.0,
        delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=1.0,
    )


@pytest.fixture()
def model() -> SimpleCNN:
    torch.manual_seed(0)
    m = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
    return validate_and_fix_model(m)


@pytest.fixture()
def train_loader() -> DataLoader:
    torch.manual_seed(1)
    ds = TensorDataset(
        torch.randn(64, 1, 28, 28),
        torch.randint(0, 10, (64,)),
    )
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def val_loader() -> DataLoader:
    torch.manual_seed(2)
    ds = TensorDataset(
        torch.randn(32, 1, 28, 28),
        torch.randint(0, 10, (32,)),
    )
    return DataLoader(ds, batch_size=16)


@pytest.fixture()
def optimizer(model) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture()
def dp_trainer(model, train_loader, val_loader, optimizer, dp_config) -> DPTrainer:
    return DPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        dp_config=dp_config,
    )


# ── Model compatibility helpers ──────────────────────────────────────────


class TestValidateAndFixModel:
    def test_simple_cnn_compatible(self) -> None:
        model = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
        fixed = validate_and_fix_model(model)
        assert is_dp_compatible(fixed)

    def test_resnet_compatible(self) -> None:
        model = SmallResNet(input_channels=3, num_classes=10)
        fixed = validate_and_fix_model(model)
        assert is_dp_compatible(fixed)

    def test_model_with_batchnorm_gets_fixed(self) -> None:
        """A model with BatchNorm should be automatically fixed."""
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 28 * 28, 10),
        )
        assert not is_dp_compatible(model)
        fixed = validate_and_fix_model(model)
        assert is_dp_compatible(fixed)

    def test_returns_nn_module(self) -> None:
        model = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
        fixed = validate_and_fix_model(model)
        assert isinstance(fixed, nn.Module)


class TestIsDPCompatible:
    def test_compatible_model(self) -> None:
        model = SmallResNet(input_channels=3, num_classes=10)
        assert is_dp_compatible(model)

    def test_incompatible_model(self) -> None:
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
        )
        assert not is_dp_compatible(model)


# ── DPTrainer construction ───────────────────────────────────────────────


class TestDPTrainerInit:
    def test_creates_trainer(self, dp_trainer) -> None:
        assert isinstance(dp_trainer, DPTrainer)
        assert dp_trainer.privacy_engine is None
        assert not dp_trainer._is_private

    def test_inherits_from_trainer(self, dp_trainer) -> None:
        from auditml.training.trainer import Trainer
        assert isinstance(dp_trainer, Trainer)

    def test_dp_config_stored(self, dp_trainer, dp_config) -> None:
        assert dp_trainer.dp_config is dp_config
        assert dp_trainer.dp_config.epsilon == 10.0

    def test_epsilon_history_empty(self, dp_trainer) -> None:
        assert dp_trainer.epsilon_history == []


# ── make_private ─────────────────────────────────────────────────────────


class TestMakePrivate:
    def test_makes_model_private(self, dp_trainer) -> None:
        dp_trainer.make_private()
        assert dp_trainer._is_private
        assert dp_trainer.privacy_engine is not None

    def test_epsilon_zero_before_training(self, dp_trainer) -> None:
        dp_trainer.make_private()
        eps = dp_trainer.get_epsilon()
        assert eps == 0.0


# ── DP training flow ─────────────────────────────────────────────────────


class TestDPTraining:
    def test_train_one_epoch(self, dp_trainer) -> None:
        history = dp_trainer.train(epochs=1, patience=0)
        assert "epsilon" in history
        assert len(history["epsilon"]) == 1
        assert history["epsilon"][0] > 0.0

    def test_epsilon_increases_with_epochs(self, dp_trainer) -> None:
        dp_trainer.train(epochs=2, patience=0)
        assert len(dp_trainer.epsilon_history) == 2
        assert dp_trainer.epsilon_history[1] >= dp_trainer.epsilon_history[0]

    def test_get_epsilon_after_training(self, dp_trainer) -> None:
        dp_trainer.train(epochs=1, patience=0)
        eps = dp_trainer.get_epsilon()
        assert eps > 0.0
        assert eps == dp_trainer.epsilon_history[-1]

    def test_history_has_standard_keys(self, dp_trainer) -> None:
        history = dp_trainer.train(epochs=1, patience=0)
        for key in ("train_loss", "train_acc", "val_loss", "val_acc", "epsilon"):
            assert key in history

    def test_val_metrics_computed(self, dp_trainer) -> None:
        history = dp_trainer.train(epochs=1, patience=0)
        assert 0.0 <= history["val_acc"][0] <= 1.0
        assert history["val_loss"][0] > 0.0

    def test_auto_makes_private(self, dp_trainer) -> None:
        """train() should auto-call make_private() if not done yet."""
        assert not dp_trainer._is_private
        dp_trainer.train(epochs=1, patience=0)
        assert dp_trainer._is_private


# ── Checkpoint with DP metadata ──────────────────────────────────────────


class TestDPCheckpoint:
    def test_save_checkpoint_includes_epsilon(self, dp_trainer, tmp_path) -> None:
        dp_trainer.train(epochs=1, patience=0)
        ckpt_path = dp_trainer.save_checkpoint(tmp_path, epoch=1)
        assert ckpt_path.exists()

        # Check metrics.json has epsilon
        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists()
        data = json.loads(metrics_path.read_text())
        assert "epsilon" in data
        assert data["epsilon"] > 0.0


# ── Noise calibration ───────────────────────────────────────────────────


class TestNoiseCalibration:
    def test_auto_calibration(
        self, model, train_loader, val_loader, optimizer,
    ) -> None:
        """When noise_multiplier is None, it should be auto-calibrated."""
        dp_cfg = DPConfig(
            enabled=True,
            epsilon=10.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=None,
        )
        trainer = DPTrainer(model, train_loader, val_loader, optimizer, dp_config=dp_cfg)
        trainer.make_private()
        assert trainer._is_private

    def test_explicit_noise_multiplier(self, dp_trainer) -> None:
        """When noise_multiplier is set, it should be used directly."""
        dp_trainer.make_private()
        assert dp_trainer._is_private


# ── Evaluate works after DP training ─────────────────────────────────────


class TestEvaluateAfterDP:
    def test_evaluate_returns_metrics(self, dp_trainer, val_loader) -> None:
        dp_trainer.train(epochs=1, patience=0)
        metrics = dp_trainer.evaluate(val_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ── Imports from training package ────────────────────────────────────────


class TestImports:
    def test_dp_trainer_importable(self) -> None:
        from auditml.training import DPTrainer
        assert DPTrainer is not None

    def test_validate_and_fix_importable(self) -> None:
        from auditml.training import validate_and_fix_model
        assert validate_and_fix_model is not None

    def test_is_dp_compatible_importable(self) -> None:
        from auditml.training import is_dp_compatible
        assert is_dp_compatible is not None
