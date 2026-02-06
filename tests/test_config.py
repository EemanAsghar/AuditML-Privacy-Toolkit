"""Tests for the AuditML configuration system."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from auditml.config import (
    AttackType,
    AuditMLConfig,
    DatasetName,
    config_to_dict,
    default_config,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write *content* to a temporary YAML file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# Tests — default_config
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_returns_auditml_config(self) -> None:
        cfg = default_config()
        assert isinstance(cfg, AuditMLConfig)

    def test_default_dataset(self) -> None:
        cfg = default_config()
        assert cfg.data.dataset is DatasetName.CIFAR10

    def test_default_attacks(self) -> None:
        cfg = default_config()
        assert cfg.attacks == [AttackType.MIA_THRESHOLD]

    def test_dp_disabled_by_default(self) -> None:
        cfg = default_config()
        assert cfg.dp.enabled is False


# ---------------------------------------------------------------------------
# Tests — load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_minimal_yaml(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, "experiment_name: test_run\n")
        cfg = load_config(p)
        assert cfg.experiment_name == "test_run"
        # Everything else should be defaults
        assert cfg.data.dataset is DatasetName.CIFAR10

    def test_override_nested(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, """\
            data:
              dataset: mnist
              train_ratio: 0.8
            training:
              epochs: 50
        """)
        cfg = load_config(p)
        assert cfg.data.dataset is DatasetName.MNIST
        assert cfg.data.train_ratio == 0.8
        assert cfg.training.epochs == 50

    def test_attack_list(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, """\
            attacks:
              - mia_shadow
              - model_inversion
        """)
        cfg = load_config(p)
        assert cfg.attacks == [AttackType.MIA_SHADOW, AttackType.MODEL_INVERSION]

    def test_dp_section(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, """\
            dp:
              enabled: true
              epsilon: 0.1
        """)
        cfg = load_config(p)
        assert cfg.dp.enabled is True
        assert cfg.dp.epsilon == 0.1

    def test_empty_file(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, "")
        cfg = load_config(p)
        assert isinstance(cfg, AuditMLConfig)

    def test_default_yaml_loads(self) -> None:
        """The shipped default.yaml must load without errors."""
        cfg = load_config(Path("configs/default.yaml"))
        assert cfg.experiment_name == "audit_cifar10"
        assert AttackType.MIA_THRESHOLD in cfg.attacks
        assert AttackType.MIA_SHADOW in cfg.attacks


# ---------------------------------------------------------------------------
# Tests — validation / error handling
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, "bogus_key: 42\n")
        with pytest.raises(ValueError, match="Unknown config keys"):
            load_config(p)

    def test_bad_dataset_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, """\
            data:
              dataset: imagenet
        """)
        with pytest.raises(ValueError, match="Invalid value"):
            load_config(p)

    def test_bad_attack_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, """\
            attacks:
              - not_a_real_attack
        """)
        with pytest.raises(ValueError, match="Invalid value"):
            load_config(p)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# Tests — round-trip
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    def test_dict_roundtrip(self) -> None:
        cfg = default_config()
        d = config_to_dict(cfg)
        assert isinstance(d, dict)
        assert d["experiment_name"] == "audit"
        assert d["data"]["dataset"] == "cifar10"
        assert d["dp"]["enabled"] is False
