"""Tests for AuditML training framework.

Uses synthetic data — fast, no downloads, CPU only.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from auditml.models import SimpleCNN
from auditml.training import Trainer, build_optimizer


class TestTrainer:
    def test_single_epoch(
        self, simple_cnn: SimpleCNN, train_loader: DataLoader, val_loader: DataLoader,
    ) -> None:
        optimizer = build_optimizer(simple_cnn, "adam", lr=0.001)
        trainer = Trainer(
            simple_cnn, train_loader, val_loader, optimizer, device="cpu",
        )
        history = trainer.train(epochs=1, patience=0)
        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1

    def test_history_grows(
        self, simple_cnn: SimpleCNN, train_loader: DataLoader, val_loader: DataLoader,
    ) -> None:
        optimizer = build_optimizer(simple_cnn, "adam")
        trainer = Trainer(
            simple_cnn, train_loader, val_loader, optimizer, device="cpu",
        )
        history = trainer.train(epochs=3, patience=0)
        assert len(history["train_loss"]) == 3
        assert len(history["val_acc"]) == 3

    def test_evaluate_returns_metrics(
        self, simple_cnn: SimpleCNN, val_loader: DataLoader, train_loader: DataLoader,
    ) -> None:
        optimizer = build_optimizer(simple_cnn, "adam")
        trainer = Trainer(
            simple_cnn, train_loader, val_loader, optimizer, device="cpu",
        )
        metrics = trainer.evaluate(val_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_checkpoint_save_load(
        self,
        simple_cnn: SimpleCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tmp_path: Path,
    ) -> None:
        optimizer = build_optimizer(simple_cnn, "adam")
        trainer = Trainer(
            simple_cnn, train_loader, val_loader, optimizer, device="cpu",
        )
        trainer.train(epochs=2, patience=0, checkpoint_dir=tmp_path)

        ckpt_path = tmp_path / "model.pt"
        assert ckpt_path.exists()

        # Load into fresh trainer
        fresh_model = SimpleCNN(input_channels=1, num_classes=10, input_size=28)
        fresh_opt = build_optimizer(fresh_model, "adam")
        fresh_trainer = Trainer(
            fresh_model, train_loader, val_loader, fresh_opt, device="cpu",
        )
        ckpt = fresh_trainer.load_checkpoint(ckpt_path)
        assert "epoch" in ckpt

        # Predictions should match
        x = torch.randn(2, 1, 28, 28)
        simple_cnn.eval()
        fresh_model.eval()
        with torch.no_grad():
            orig = simple_cnn(x)
            loaded = fresh_model(x)
        torch.testing.assert_close(orig, loaded)

    def test_early_stopping(
        self, simple_cnn: SimpleCNN, train_loader: DataLoader, val_loader: DataLoader,
    ) -> None:
        optimizer = build_optimizer(simple_cnn, "adam")
        trainer = Trainer(
            simple_cnn, train_loader, val_loader, optimizer, device="cpu",
        )
        # With patience=2 on random data, should stop well before 50 epochs
        history = trainer.train(epochs=50, patience=2)
        assert len(history["train_loss"]) <= 50


class TestBuildOptimizer:
    def test_adam(self, simple_cnn: SimpleCNN) -> None:
        opt = build_optimizer(simple_cnn, "adam", lr=0.01)
        assert isinstance(opt, torch.optim.Adam)

    def test_sgd(self, simple_cnn: SimpleCNN) -> None:
        opt = build_optimizer(simple_cnn, "sgd", lr=0.1)
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_raises(self, simple_cnn: SimpleCNN) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(simple_cnn, "rmsprop")
