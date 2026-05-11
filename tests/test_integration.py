"""Integration & end-to-end tests for AuditML (Task 4.3).

These tests wire together real components — no mocks — using tiny synthetic
data so the full pipeline runs in seconds on CPU. They verify that:

1. Full MIA workflow   : train → run ThresholdMIA → metrics produced
2. Full shadow MIA     : train → run ShadowMIA    → metrics produced
3. DP training workflow: train with DP → epsilon tracked → model functional
4. Full audit CLI      : main() with a real temp config → outputs on disk
5. Report generation   : attacks → ReportGenerator → files on disk
6. Rust acceleration   : rust_accel functions give same results as NumPy
7. run_all_attacks helpers: build_config + train_or_load_model work correctly
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

# ── shared constants ──────────────────────────────────────────────────────

N = 120          # total samples
BATCH = 32
EPOCHS = 2       # keep tests fast
CLASSES = 10
IMG_C, IMG_H = 1, 28


# ── helpers ───────────────────────────────────────────────────────────────

def _synthetic_dataset(n: int = N) -> TensorDataset:
    torch.manual_seed(0)
    return TensorDataset(
        torch.randn(n, IMG_C, IMG_H, IMG_H),
        torch.randint(0, CLASSES, (n,)),
    )


def _split_loaders(ds: TensorDataset):
    """Return (train_loader, val_loader, member_loader, nonmember_loader)."""
    half = len(ds) // 2
    member_set = Subset(ds, list(range(half)))
    nonmember_set = Subset(ds, list(range(half, len(ds))))
    kw = dict(batch_size=BATCH, shuffle=False)
    return (
        DataLoader(member_set, batch_size=BATCH, shuffle=True),
        DataLoader(nonmember_set, **kw),
        DataLoader(member_set, **kw),
        DataLoader(nonmember_set, **kw),
    )


def _trained_model(train_loader, val_loader):
    """Train a SimpleCNN for EPOCHS and return the trained model."""
    from auditml.models import SimpleCNN
    from auditml.training import Trainer, build_optimizer

    model = SimpleCNN(input_channels=IMG_C, num_classes=CLASSES, input_size=IMG_H)
    optimizer = build_optimizer(model, "adam", lr=0.01)
    trainer = Trainer(model, train_loader, val_loader, optimizer, device="cpu")
    trainer.train(epochs=EPOCHS, patience=0)
    model.eval()
    return model


def _default_cfg():
    from auditml.config import default_config
    return default_config()


# ── 1. Full MIA Threshold workflow ────────────────────────────────────────

class TestThresholdMIAIntegration:
    def test_full_pipeline_produces_metrics(self):
        ds = _synthetic_dataset()
        train_l, val_l, member_l, nonmember_l = _split_loaders(ds)
        model = _trained_model(train_l, val_l)

        from auditml.attacks.mia_threshold import ThresholdMIA
        attack = ThresholdMIA(model, _default_cfg(), device="cpu")
        result = attack.run(member_l, nonmember_l)
        metrics = attack.evaluate()

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc_roc"] <= 1.0
        assert "tpr_at_1fpr" in metrics
        assert len(result.predictions) == N
        assert len(result.ground_truth) == N

    def test_threshold_is_set_after_run(self):
        ds = _synthetic_dataset()
        train_l, val_l, member_l, nonmember_l = _split_loaders(ds)
        model = _trained_model(train_l, val_l)

        from auditml.attacks.mia_threshold import ThresholdMIA
        attack = ThresholdMIA(model, _default_cfg(), device="cpu")
        attack.run(member_l, nonmember_l)

        assert attack.threshold is not None
        assert isinstance(attack.threshold, float)

    def test_metrics_without_run_raises(self):
        from auditml.attacks.mia_threshold import ThresholdMIA
        from auditml.models import SimpleCNN
        model = SimpleCNN(input_channels=IMG_C, num_classes=CLASSES, input_size=IMG_H)
        attack = ThresholdMIA(model, _default_cfg(), device="cpu")
        with pytest.raises(RuntimeError):
            attack.evaluate()


# ── 2. Full Shadow MIA workflow ───────────────────────────────────────────

class TestShadowMIAIntegration:
    def test_full_pipeline_produces_metrics(self):
        ds = _synthetic_dataset()
        train_l, val_l, member_l, nonmember_l = _split_loaders(ds)
        model = _trained_model(train_l, val_l)

        from auditml.config.schema import DatasetName, MIAShadowConfig
        cfg = _default_cfg()
        cfg.data.dataset = DatasetName.MNIST       # match synthetic 1-channel data
        cfg.model.num_classes = CLASSES
        cfg.attack_params.mia_shadow = MIAShadowConfig(
            num_shadow_models=1,
            shadow_epochs=1,
        )

        from auditml.attacks.mia_shadow import ShadowMIA
        attack = ShadowMIA(model, cfg, device="cpu", shadow_dataset=ds)
        result = attack.run(member_l, nonmember_l)
        metrics = attack.evaluate()

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc_roc"] <= 1.0
        assert len(result.predictions) == N

    def test_attack_model_trained_after_run(self):
        ds = _synthetic_dataset()
        train_l, val_l, member_l, nonmember_l = _split_loaders(ds)
        model = _trained_model(train_l, val_l)

        from auditml.config.schema import DatasetName, MIAShadowConfig
        cfg = _default_cfg()
        cfg.data.dataset = DatasetName.MNIST
        cfg.model.num_classes = CLASSES
        cfg.attack_params.mia_shadow = MIAShadowConfig(num_shadow_models=1, shadow_epochs=1)

        from auditml.attacks.mia_shadow import ShadowMIA
        attack = ShadowMIA(model, cfg, device="cpu", shadow_dataset=ds)
        attack.run(member_l, nonmember_l)

        assert attack.attack_model is not None
        assert len(attack.trained_shadows) == 1


# ── 3. DP training workflow ───────────────────────────────────────────────

class TestDPTrainingIntegration:
    def test_dp_epsilon_tracked(self):
        ds = _synthetic_dataset()
        train_l, val_l, member_l, nonmember_l = _split_loaders(ds)

        from auditml.config.schema import DPConfig
        from auditml.models import SimpleCNN
        from auditml.training import build_optimizer
        from auditml.training.dp_trainer import DPTrainer, validate_and_fix_model

        model = SimpleCNN(input_channels=IMG_C, num_classes=CLASSES, input_size=IMG_H)
        model = validate_and_fix_model(model)
        optimizer = build_optimizer(model, "adam", lr=0.01)
        dp_cfg = DPConfig(enabled=True, epsilon=10.0, delta=1e-5, max_grad_norm=1.0)

        trainer = DPTrainer(model, train_l, val_l, optimizer, dp_cfg, device="cpu")
        trainer.train(epochs=EPOCHS, patience=0)

        eps = trainer.get_epsilon()
        assert eps > 0.0, f"Expected positive epsilon, got {eps}"

    def test_dp_model_produces_predictions(self):
        ds = _synthetic_dataset()
        train_l, val_l, _, _ = _split_loaders(ds)

        from auditml.config.schema import DPConfig
        from auditml.models import SimpleCNN
        from auditml.training import build_optimizer
        from auditml.training.dp_trainer import DPTrainer, validate_and_fix_model

        model = SimpleCNN(input_channels=IMG_C, num_classes=CLASSES, input_size=IMG_H)
        model = validate_and_fix_model(model)
        optimizer = build_optimizer(model, "adam", lr=0.01)
        dp_cfg = DPConfig(enabled=True, epsilon=10.0, delta=1e-5, max_grad_norm=1.0)

        trainer = DPTrainer(model, train_l, val_l, optimizer, dp_cfg, device="cpu")
        trainer.train(epochs=EPOCHS, patience=0)
        trainer.model.eval()

        sample = torch.randn(1, IMG_C, IMG_H, IMG_H)
        with torch.no_grad():
            out = trainer.model(sample)
        assert out.shape == (1, CLASSES)

    def test_dp_val_accuracy_computed(self):
        ds = _synthetic_dataset()
        train_l, val_l, _, _ = _split_loaders(ds)

        from auditml.config.schema import DPConfig
        from auditml.models import SimpleCNN
        from auditml.training import build_optimizer
        from auditml.training.dp_trainer import DPTrainer, validate_and_fix_model

        model = SimpleCNN(input_channels=IMG_C, num_classes=CLASSES, input_size=IMG_H)
        model = validate_and_fix_model(model)
        optimizer = build_optimizer(model, "adam", lr=0.01)
        dp_cfg = DPConfig(enabled=True, epsilon=10.0, delta=1e-5, max_grad_norm=1.0)

        trainer = DPTrainer(model, train_l, val_l, optimizer, dp_cfg, device="cpu")
        trainer.train(epochs=EPOCHS, patience=0)
        metrics = trainer.evaluate(val_l)

        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ── 4. Full audit CLI end-to-end ──────────────────────────────────────────

class TestCLIIntegration:
    def test_show_config_returns_json(self):
        from auditml.cli import main
        # Should print JSON and return 0
        rc = main(["show-config"])
        assert rc == 0

    def test_train_command_saves_checkpoint(self, tmp_path):
        import yaml

        from auditml.cli import main

        config = {
            "experiment_name": "integration_train",
            "attacks": ["mia_threshold"],
            "data": {"dataset": "mnist", "data_dir": str(tmp_path / "data"),
                     "train_ratio": 0.5, "num_workers": 0, "download": True},
            "model": {"arch": "cnn", "num_classes": 10},
            "training": {"epochs": EPOCHS, "batch_size": BATCH,
                         "learning_rate": 0.01, "optimizer": "adam",
                         "weight_decay": 0.0001, "seed": 42, "device": "cpu"},
            "dp": {"enabled": False, "epsilon": 1.0, "delta": 1e-5, "max_grad_norm": 1.0},
            "reporting": {"output_dir": str(tmp_path / "results"),
                          "save_plots": False, "save_json": True},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config))

        rc = main(["train", "-c", str(cfg_path)])
        assert rc == 0

        ckpt = tmp_path / "results" / "integration_train" / "checkpoints" / "model.pt"
        assert ckpt.exists(), f"Checkpoint not found at {ckpt}"

    def test_invalid_config_path_returns_error(self):
        from auditml.cli import main
        rc = main(["train", "-c", "/nonexistent/path.yaml"])
        assert rc == 1

    def test_no_command_returns_error(self):
        from auditml.cli import main
        rc = main([])
        assert rc == 1


# ── 5. Report generation ──────────────────────────────────────────────────

class TestReportGenerationIntegration:
    def test_report_creates_expected_files(self, tmp_path):
        ds = _synthetic_dataset()
        train_l, val_l, member_l, nonmember_l = _split_loaders(ds)
        model = _trained_model(train_l, val_l)

        from auditml.attacks.mia_threshold import ThresholdMIA
        from auditml.reporting.report_generator import ReportGenerator

        cfg = _default_cfg()
        attack = ThresholdMIA(model, cfg, device="cpu")
        result = attack.run(member_l, nonmember_l)

        generator = ReportGenerator(
            experiment_name="test_report",
            attack_results={"mia_threshold": (attack, result)},
        )
        generator.generate(tmp_path)

        assert (tmp_path / "summary.txt").exists()
        assert (tmp_path / "audit_summary.json").exists()
        assert (tmp_path / "report.html").exists(), "HTML report must be generated by ReportGenerator"  # noqa: E501

    def test_report_html_auto_open(self, tmp_path):
        """CLI audit path: report.html is created and webbrowser.open would be called."""
        ds = _synthetic_dataset()
        train_l, val_l, member_l, nonmember_l = _split_loaders(ds)
        model = _trained_model(train_l, val_l)

        from auditml.attacks.mia_threshold import ThresholdMIA
        from auditml.reporting.report_generator import ReportGenerator

        cfg = _default_cfg()
        attack = ThresholdMIA(model, cfg, device="cpu")
        result = attack.run(member_l, nonmember_l)

        generator = ReportGenerator(
            experiment_name="test_autoopen",
            attack_results={"mia_threshold": (attack, result)},
        )
        generator.generate(tmp_path)

        html_report = tmp_path / "report.html"
        assert html_report.exists(), "report.html must exist for browser auto-open"
        content = html_report.read_text()
        assert "AuditML" in content
        assert "mia_threshold" in content.lower() or "Threshold MIA" in content

    def test_audit_summary_json_is_valid(self, tmp_path):
        ds = _synthetic_dataset()
        train_l, val_l, member_l, nonmember_l = _split_loaders(ds)
        model = _trained_model(train_l, val_l)

        from auditml.attacks.mia_threshold import ThresholdMIA
        from auditml.reporting.report_generator import ReportGenerator

        cfg = _default_cfg()
        attack = ThresholdMIA(model, cfg, device="cpu")
        result = attack.run(member_l, nonmember_l)

        generator = ReportGenerator(
            experiment_name="test_json",
            attack_results={"mia_threshold": (attack, result)},
        )
        generator.generate(tmp_path)

        summary = json.loads((tmp_path / "audit_summary.json").read_text())
        assert "experiment_name" in summary
        assert summary["experiment_name"] == "test_json"


# ── 6. Rust acceleration integration ─────────────────────────────────────

class TestRustAccelIntegration:
    def test_rust_available(self):
        from auditml.utils.rust_accel import RUST_AVAILABLE, rust_status
        # Rust extension is optional — just verify the flag and status string are consistent
        status = rust_status()
        if RUST_AVAILABLE:
            assert "ENABLED" in status
        else:
            assert "DISABLED" in status

    def test_find_threshold_matches_numpy(self):
        from auditml.utils.rust_accel import (
            _find_best_threshold_numpy,
            find_best_threshold,
        )
        rng = np.random.default_rng(42)
        scores = rng.random(500).astype(np.float64)
        labels = rng.integers(0, 2, 500).astype(np.int32)

        _, acc_rust = find_best_threshold(scores, labels)
        _, acc_numpy = _find_best_threshold_numpy(scores, labels)

        assert abs(acc_rust - acc_numpy) < 1e-6

    def test_ssim_perfect_match(self):
        from auditml.utils.rust_accel import compute_ssim
        img = np.random.default_rng(0).random(784)
        ssim = compute_ssim(img, img)
        assert abs(ssim - 1.0) < 1e-5

    def test_batch_ssim_length(self):
        from auditml.utils.rust_accel import batch_ssim
        rng = np.random.default_rng(1)
        imgs_a = [rng.random(784) for _ in range(10)]
        imgs_b = [rng.random(784) for _ in range(10)]
        results = batch_ssim(imgs_a, imgs_b)
        assert len(results) == 10
        assert all(-1.0 <= s <= 1.0 for s in results)

    def test_threshold_mia_uses_rust(self):
        """ThresholdMIA.run() should complete without error when Rust is available."""
        ds = _synthetic_dataset()
        train_l, val_l, member_l, nonmember_l = _split_loaders(ds)
        model = _trained_model(train_l, val_l)

        from auditml.attacks.mia_threshold import ThresholdMIA
        attack = ThresholdMIA(model, _default_cfg(), device="cpu")
        attack.run(member_l, nonmember_l)
        assert attack.threshold is not None


# ── 7. Checkpoint save/load round-trip ────────────────────────────────────

class TestCheckpointIntegration:
    def test_save_and_reload_model(self, tmp_path):
        from auditml.models import SimpleCNN
        from auditml.training import Trainer, build_optimizer

        ds = _synthetic_dataset()
        train_l, val_l, _, _ = _split_loaders(ds)

        model = SimpleCNN(input_channels=IMG_C, num_classes=CLASSES, input_size=IMG_H)
        optimizer = build_optimizer(model, "adam", lr=0.01)
        trainer = Trainer(model, train_l, val_l, optimizer, device="cpu")
        trainer.train(epochs=EPOCHS, patience=0)

        ckpt_path = trainer.save_checkpoint(tmp_path, epoch=EPOCHS)
        assert ckpt_path.exists()

        # Reload into a fresh model
        model2 = SimpleCNN(input_channels=IMG_C, num_classes=CLASSES, input_size=IMG_H)
        optimizer2 = build_optimizer(model2, "adam", lr=0.01)
        trainer2 = Trainer(model2, train_l, val_l, optimizer2, device="cpu")
        trainer2.load_checkpoint(ckpt_path)

        # Both models should produce identical outputs (eval mode disables dropout)
        trainer.model.eval()
        trainer2.model.eval()
        sample = torch.randn(1, IMG_C, IMG_H, IMG_H)
        with torch.no_grad():
            out1 = trainer.model(sample)
            out2 = trainer2.model(sample)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_checkpoint_contains_metrics(self, tmp_path):
        from auditml.models import SimpleCNN
        from auditml.training import Trainer, build_optimizer

        ds = _synthetic_dataset()
        train_l, val_l, _, _ = _split_loaders(ds)

        model = SimpleCNN(input_channels=IMG_C, num_classes=CLASSES, input_size=IMG_H)
        optimizer = build_optimizer(model, "adam", lr=0.01)
        trainer = Trainer(model, train_l, val_l, optimizer, device="cpu")
        trainer.train(epochs=EPOCHS, patience=0)

        trainer.save_checkpoint(tmp_path, epoch=EPOCHS, metrics={"val_acc": 0.42})

        ckpt = torch.load(tmp_path / "model.pt", weights_only=False)
        assert ckpt["metrics"]["val_acc"] == pytest.approx(0.42)
