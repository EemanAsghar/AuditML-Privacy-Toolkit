"""Tests for the AuditML CLI."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from auditml.cli import build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return p


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


# ---------------------------------------------------------------------------
# main() integration tests
# ---------------------------------------------------------------------------

class TestMain:
    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main([])
        assert rc == 1
        assert "auditml" in capsys.readouterr().out

    def test_audit_with_config(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
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

    def test_train_with_config(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
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

    def test_train_with_dp(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
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
