"""Tests for the AuditML CLI (Task 3.2 — wired implementation).

Tests the CLI entry point, argument parsing, and the train/audit/show-config
subcommands. Heavy operations (data loading, model creation, training, attacks)
are mocked so tests run instantly without network access.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from auditml.cli import build_parser, main


# ---------------------------------------------------------------------------
# Patch targets — the actual module paths used by local imports in cli.py
# ---------------------------------------------------------------------------
_P_DATALOADERS = "auditml.data.datasets.get_dataloaders"
_P_GET_MODEL = "auditml.models.get_model"
_P_TRAINER = "auditml.training.Trainer"
_P_BUILD_OPT = "auditml.training.build_optimizer"
_P_DP_TRAINER = "auditml.training.DPTrainer"
_P_VALIDATE = "auditml.training.validate_and_fix_model"
_P_GET_ATTACK = "auditml.attacks.get_attack"
_P_REPORT_GEN = "auditml.reporting.ReportGenerator"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def _fake_dataloaders():
    """Return a dict that looks like get_dataloaders() output."""
    loader = MagicMock()
    return {
        "train_loader": loader,
        "test_loader": loader,
        "member_loader": loader,
        "nonmember_loader": loader,
        "member_indices": np.arange(100),
        "nonmember_indices": np.arange(100, 200),
    }


def _fake_model():
    """Return a mock model with .eval() and .parameters()."""
    model = MagicMock()
    model.eval.return_value = model
    model.parameters.return_value = [MagicMock()]
    return model


def _fake_trainer():
    """Return a mock Trainer with train/evaluate/load_checkpoint."""
    trainer = MagicMock()
    trainer.train.return_value = {"train_loss": [0.5], "val_loss": [0.4]}
    trainer.evaluate.return_value = {"accuracy": 0.85, "loss": 0.35}
    return trainer


def _fake_dp_trainer():
    """Return a mock DPTrainer with get_epsilon."""
    trainer = _fake_trainer()
    trainer.get_epsilon.return_value = 3.14
    return trainer


def _fake_attack():
    """Return a mock attack with run/evaluate/generate_report."""
    attack = MagicMock()
    result = MagicMock()
    attack.run.return_value = result
    attack.evaluate.return_value = {
        "accuracy": 0.72,
        "auc_roc": 0.78,
        "precision": 0.70,
        "recall": 0.69,
        "f1": 0.695,
    }
    attack.generate_report.side_effect = lambda d: Path(d).mkdir(parents=True, exist_ok=True)
    return attack


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit, match="0"):
            parser.parse_args(["--version"])
        assert "auditml" in capsys.readouterr().out

    def test_audit_requires_config(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["audit"])

    def test_train_requires_config(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train"])

    def test_show_config_config_optional(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["show-config"])
        assert args.command == "show-config"
        assert args.config is None

    def test_subcommands_present(self) -> None:
        parser = build_parser()
        for cmd in ["train", "audit"]:
            args = parser.parse_args([cmd, "-c", "x.yaml"])
            assert args.command == cmd
        args = parser.parse_args(["show-config"])
        assert args.command == "show-config"


# ---------------------------------------------------------------------------
# main() — general behaviour
# ---------------------------------------------------------------------------

class TestMain:
    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main([])
        assert rc == 1
        assert "auditml" in capsys.readouterr().out

    def test_bad_config_path(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["audit", "--config", "/does/not/exist.yaml"])
        assert rc == 1
        assert "Error" in capsys.readouterr().err

    def test_invalid_config_content(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = _write_yaml(tmp_path, "bogus_key: 42\n")
        rc = main(["audit", "--config", str(p)])
        assert rc == 1
        assert "Unknown config keys" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# show-config subcommand
# ---------------------------------------------------------------------------

class TestShowConfig:
    def test_show_config_defaults(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["show-config"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["experiment_name"] == "audit"
        assert data["data"]["dataset"] == "cifar10"

    def test_show_config_with_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = _write_yaml(tmp_path, "experiment_name: custom\n")
        rc = main(["show-config", "--config", str(p)])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["experiment_name"] == "custom"


# ---------------------------------------------------------------------------
# train subcommand
# ---------------------------------------------------------------------------

class TestTrainCommand:
    """Tests for ``auditml train`` with mocked heavy operations."""

    @patch(_P_TRAINER)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_train_basic(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_trainer_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        mock_get_model.return_value = _fake_model()
        mock_build_opt.return_value = MagicMock()
        trainer = _fake_trainer()
        mock_trainer_cls.return_value = trainer

        p = _write_yaml(tmp_path, """\
            data:
              dataset: mnist
            training:
              epochs: 5
        """)
        rc = main(["train", "--config", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "mnist" in out
        assert "5" in out
        trainer.train.assert_called_once()
        trainer.evaluate.assert_called_once()

    @patch(_P_TRAINER)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_train_prints_accuracy(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_trainer_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        mock_get_model.return_value = _fake_model()
        mock_build_opt.return_value = MagicMock()
        trainer = _fake_trainer()
        trainer.evaluate.return_value = {"accuracy": 0.9100, "loss": 0.25}
        mock_trainer_cls.return_value = trainer

        p = _write_yaml(tmp_path, "training:\n  epochs: 1\n")
        rc = main(["train", "--config", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "0.9100" in out

    @patch(_P_DP_TRAINER)
    @patch(_P_VALIDATE)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_train_with_dp(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_validate,
        mock_dp_trainer_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        model = _fake_model()
        mock_get_model.return_value = model
        mock_validate.return_value = model
        mock_build_opt.return_value = MagicMock()
        dp_trainer = _fake_dp_trainer()
        mock_dp_trainer_cls.return_value = dp_trainer

        p = _write_yaml(tmp_path, """\
            dp:
              enabled: true
              epsilon: 0.1
        """)
        rc = main(["train", "--config", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "DP enabled : True" in out
        assert "0.1" in out
        mock_validate.assert_called_once()
        dp_trainer.train.assert_called_once()

    @patch(_P_DP_TRAINER)
    @patch(_P_VALIDATE)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_train_dp_prints_epsilon(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_validate,
        mock_dp_trainer_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        model = _fake_model()
        mock_get_model.return_value = model
        mock_validate.return_value = model
        mock_build_opt.return_value = MagicMock()
        dp_trainer = _fake_dp_trainer()
        dp_trainer.get_epsilon.return_value = 2.50
        mock_dp_trainer_cls.return_value = dp_trainer

        p = _write_yaml(tmp_path, """\
            dp:
              enabled: true
              epsilon: 1.0
        """)
        rc = main(["train", "--config", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2.50" in out

    @patch(_P_TRAINER)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_train_checkpoint_path_in_output(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_trainer_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        mock_get_model.return_value = _fake_model()
        mock_build_opt.return_value = MagicMock()
        mock_trainer_cls.return_value = _fake_trainer()

        p = _write_yaml(tmp_path, "training:\n  epochs: 1\n")
        rc = main(["train", "--config", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Checkpoint saved to" in out


# ---------------------------------------------------------------------------
# audit subcommand
# ---------------------------------------------------------------------------

class TestAuditCommand:
    """Tests for ``auditml audit`` with mocked heavy operations."""

    @patch(_P_REPORT_GEN)
    @patch(_P_GET_ATTACK)
    @patch(_P_TRAINER)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_audit_basic(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_trainer_cls,
        mock_get_attack,
        mock_report_gen_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        mock_get_model.return_value = _fake_model()
        mock_build_opt.return_value = MagicMock()
        mock_trainer_cls.return_value = _fake_trainer()
        mock_get_attack.return_value = _fake_attack()
        mock_report_gen_cls.return_value = MagicMock()

        p = _write_yaml(tmp_path, """\
            experiment_name: cli_test
            attacks:
              - mia_threshold
        """)
        rc = main(["audit", "--config", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "cli_test" in out
        assert "mia_threshold" in out

    @patch(_P_REPORT_GEN)
    @patch(_P_GET_ATTACK)
    @patch(_P_TRAINER)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_audit_trains_baseline_when_no_checkpoint(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_trainer_cls,
        mock_get_attack,
        mock_report_gen_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        mock_get_model.return_value = _fake_model()
        mock_build_opt.return_value = MagicMock()
        trainer = _fake_trainer()
        mock_trainer_cls.return_value = trainer
        mock_get_attack.return_value = _fake_attack()
        mock_report_gen_cls.return_value = MagicMock()

        p = _write_yaml(tmp_path, """\
            experiment_name: no_ckpt_test
            attacks:
              - mia_threshold
        """)
        rc = main(["audit", "--config", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No checkpoint found" in out
        trainer.train.assert_called_once()

    @patch(_P_REPORT_GEN)
    @patch(_P_GET_ATTACK)
    @patch(_P_TRAINER)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_audit_runs_all_attacks(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_trainer_cls,
        mock_get_attack,
        mock_report_gen_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        mock_get_model.return_value = _fake_model()
        mock_build_opt.return_value = MagicMock()
        mock_trainer_cls.return_value = _fake_trainer()
        mock_get_attack.return_value = _fake_attack()
        mock_report_gen_cls.return_value = MagicMock()

        p = _write_yaml(tmp_path, """\
            attacks:
              - mia_threshold
              - mia_shadow
        """)
        rc = main(["audit", "--config", str(p)])
        assert rc == 0
        assert mock_get_attack.call_count == 2

    @patch(_P_REPORT_GEN)
    @patch(_P_GET_ATTACK)
    @patch(_P_TRAINER)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_audit_generates_report(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_trainer_cls,
        mock_get_attack,
        mock_report_gen_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        mock_get_model.return_value = _fake_model()
        mock_build_opt.return_value = MagicMock()
        mock_trainer_cls.return_value = _fake_trainer()
        mock_get_attack.return_value = _fake_attack()

        report_gen = MagicMock()
        mock_report_gen_cls.return_value = report_gen

        p = _write_yaml(tmp_path, """\
            attacks:
              - mia_threshold
        """)
        rc = main(["audit", "--config", str(p)])
        assert rc == 0
        report_gen.generate.assert_called_once()

    @patch(_P_REPORT_GEN)
    @patch(_P_GET_ATTACK)
    @patch(_P_DP_TRAINER)
    @patch(_P_VALIDATE)
    @patch(_P_TRAINER)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_audit_with_dp(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_trainer_cls,
        mock_validate,
        mock_dp_trainer_cls,
        mock_get_attack,
        mock_report_gen_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        model = _fake_model()
        mock_get_model.return_value = model
        mock_validate.return_value = model
        mock_build_opt.return_value = MagicMock()
        mock_trainer_cls.return_value = _fake_trainer()

        dp_trainer = _fake_dp_trainer()
        mock_dp_trainer_cls.return_value = dp_trainer

        mock_get_attack.return_value = _fake_attack()

        report_gen = MagicMock()
        mock_report_gen_cls.return_value = report_gen

        p = _write_yaml(tmp_path, """\
            attacks:
              - mia_threshold
            dp:
              enabled: true
              epsilon: 5.0
        """)
        rc = main(["audit", "--config", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "DP" in out
        dp_trainer.train.assert_called_once()
        # Attacks run twice: once on baseline, once on DP model
        assert mock_get_attack.call_count == 2
        # Report generator receives dp_attack_results
        call_kwargs = mock_report_gen_cls.call_args[1]
        assert call_kwargs.get("dp_attack_results") is not None

    @patch(_P_REPORT_GEN)
    @patch(_P_GET_ATTACK)
    @patch(_P_TRAINER)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_audit_prints_attack_metrics(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_trainer_cls,
        mock_get_attack,
        mock_report_gen_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        mock_get_model.return_value = _fake_model()
        mock_build_opt.return_value = MagicMock()
        mock_trainer_cls.return_value = _fake_trainer()
        mock_report_gen_cls.return_value = MagicMock()

        attack = _fake_attack()
        attack.evaluate.return_value = {"accuracy": 0.7500, "auc_roc": 0.8200}
        mock_get_attack.return_value = attack

        p = _write_yaml(tmp_path, """\
            attacks:
              - mia_threshold
        """)
        rc = main(["audit", "--config", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "0.7500" in out
        assert "0.8200" in out

    @patch(_P_REPORT_GEN)
    @patch(_P_GET_ATTACK)
    @patch(_P_TRAINER)
    @patch(_P_BUILD_OPT)
    @patch(_P_GET_MODEL)
    @patch(_P_DATALOADERS)
    def test_audit_report_path_in_output(
        self,
        mock_dataloaders,
        mock_get_model,
        mock_build_opt,
        mock_trainer_cls,
        mock_get_attack,
        mock_report_gen_cls,
        tmp_path,
        capsys,
    ) -> None:
        mock_dataloaders.return_value = _fake_dataloaders()
        mock_get_model.return_value = _fake_model()
        mock_build_opt.return_value = MagicMock()
        mock_trainer_cls.return_value = _fake_trainer()
        mock_get_attack.return_value = _fake_attack()
        mock_report_gen_cls.return_value = MagicMock()

        p = _write_yaml(tmp_path, """\
            attacks:
              - mia_threshold
        """)
        rc = main(["audit", "--config", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Report saved to" in out


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @patch(_P_DATALOADERS)
    def test_handler_exception_returns_1(
        self, mock_dataloaders, tmp_path, capsys,
    ) -> None:
        mock_dataloaders.side_effect = RuntimeError("download failed")
        p = _write_yaml(tmp_path, "training:\n  epochs: 1\n")
        rc = main(["train", "--config", str(p)])
        assert rc == 1
        assert "download failed" in capsys.readouterr().err
