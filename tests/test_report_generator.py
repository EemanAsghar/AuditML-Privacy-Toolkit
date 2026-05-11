"""Tests for Task 3.1 — Unified Report Generator.

Tests the ReportGenerator class which orchestrates per-attack reports,
cross-attack comparison, and DP comparison into a single output directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from auditml.attacks.results import AttackResult
from auditml.reporting.report_generator import ReportGenerator

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_result(
    n: int = 100,
    attack_accuracy: float = 0.7,
    attack_name: str = "test",
    seed: int = 0,
) -> AttackResult:
    """Create a synthetic AttackResult."""
    rng = np.random.RandomState(seed)
    gt = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    scores = rng.random(n)
    scores[:n // 2] += (attack_accuracy - 0.5) * 2
    scores[n // 2:] -= (attack_accuracy - 0.5) * 2
    scores = np.clip(scores, 0.01, 0.99)
    preds = (scores >= 0.5).astype(np.int32)
    return AttackResult(
        predictions=preds,
        ground_truth=gt,
        confidence_scores=scores,
        attack_name=attack_name,
    )


def _make_mock_attack(
    attack_name: str,
    n: int = 100,
    accuracy: float = 0.7,
    seed: int = 0,
) -> tuple[MagicMock, AttackResult]:
    """Create a mock attack with evaluate() and generate_report()."""
    result = _make_result(n, accuracy, attack_name, seed)
    attack = MagicMock()
    attack.attack_name = attack_name
    attack.result = result
    attack.evaluate.return_value = {
        "accuracy": accuracy,
        "precision": accuracy - 0.05,
        "recall": accuracy - 0.03,
        "f1": accuracy - 0.04,
        "auc_roc": accuracy + 0.05,
        "auc_pr": accuracy + 0.02,
        "tpr_at_1fpr": accuracy - 0.2,
        "tpr_at_01fpr": accuracy - 0.3,
    }

    def _gen_report(output_dir):
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "metrics.json").write_text(
            json.dumps(attack.evaluate.return_value, indent=2),
        )
        return p

    attack.generate_report.side_effect = _gen_report
    return attack, result


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def single_attack():
    """One attack result."""
    attack, result = _make_mock_attack("mia_threshold", accuracy=0.7, seed=0)
    return {"mia_threshold": (attack, result)}


@pytest.fixture()
def three_attacks():
    """Three attack results with different effectiveness."""
    results = {}
    for i, (name, acc) in enumerate([
        ("mia_threshold", 0.65),
        ("mia_shadow", 0.80),
        ("attribute_inference", 0.55),
    ]):
        attack, result = _make_mock_attack(name, accuracy=acc, seed=i)
        results[name] = (attack, result)
    return results


@pytest.fixture()
def dp_attacks():
    """DP attack results (weaker than baseline)."""
    results = {}
    for i, (name, acc) in enumerate([
        ("mia_threshold", 0.55),
        ("mia_shadow", 0.60),
    ]):
        attack, result = _make_mock_attack(name, accuracy=acc, seed=i + 100)
        results[name] = (attack, result)
    return results


# ── Construction ─────────────────────────────────────────────────────────


class TestReportGeneratorInit:
    def test_creates_generator(self, single_attack) -> None:
        gen = ReportGenerator(
            experiment_name="test_audit",
            attack_results=single_attack,
        )
        assert gen.experiment_name == "test_audit"
        assert len(gen.attack_results) == 1

    def test_default_values(self) -> None:
        gen = ReportGenerator()
        assert gen.experiment_name == "audit"
        assert gen.attack_results == {}
        assert gen.dp_attack_results == {}
        assert gen.epsilon is None
        assert gen.model_accuracy == {}


# ── Single attack report ─────────────────────────────────────────────────


class TestSingleAttackReport:
    def test_creates_output_dir(self, single_attack, tmp_path) -> None:
        gen = ReportGenerator(attack_results=single_attack)
        out = gen.generate(tmp_path / "report")
        assert out.is_dir()

    def test_creates_audit_summary_json(self, single_attack, tmp_path) -> None:
        gen = ReportGenerator(attack_results=single_attack)
        out = gen.generate(tmp_path / "report")
        f = out / "audit_summary.json"
        assert f.exists()
        data = json.loads(f.read_text())
        assert data["num_attacks"] == 1
        assert "mia_threshold" in data["attack_names"]

    def test_creates_summary_txt(self, single_attack, tmp_path) -> None:
        gen = ReportGenerator(attack_results=single_attack)
        out = gen.generate(tmp_path / "report")
        f = out / "summary.txt"
        assert f.exists()
        text = f.read_text()
        assert "Privacy Audit Report" in text
        assert "mia_threshold" in text

    def test_creates_attack_subdir(self, single_attack, tmp_path) -> None:
        gen = ReportGenerator(attack_results=single_attack)
        out = gen.generate(tmp_path / "report")
        assert (out / "attacks" / "mia_threshold").is_dir()

    def test_calls_generate_report(self, single_attack, tmp_path) -> None:
        gen = ReportGenerator(attack_results=single_attack)
        gen.generate(tmp_path / "report")
        attack, _ = single_attack["mia_threshold"]
        attack.generate_report.assert_called_once()

    def test_no_attack_comparison_for_single(self, single_attack, tmp_path) -> None:
        gen = ReportGenerator(attack_results=single_attack)
        out = gen.generate(tmp_path / "report")
        assert not (out / "attack_comparison").exists()

    def test_returns_output_dir(self, single_attack, tmp_path) -> None:
        gen = ReportGenerator(attack_results=single_attack)
        out = gen.generate(tmp_path / "report")
        assert out == tmp_path / "report"


# ── Multi-attack report ─────────────────────────────────────────────────


class TestMultiAttackReport:
    def test_creates_all_attack_subdirs(self, three_attacks, tmp_path) -> None:
        gen = ReportGenerator(attack_results=three_attacks)
        out = gen.generate(tmp_path / "report")
        for name in three_attacks:
            assert (out / "attacks" / name).is_dir()

    def test_creates_attack_comparison(self, three_attacks, tmp_path) -> None:
        gen = ReportGenerator(attack_results=three_attacks)
        out = gen.generate(tmp_path / "report")
        comp_dir = out / "attack_comparison"
        assert comp_dir.is_dir()
        assert (comp_dir / "attack_comparison.json").exists()
        assert (comp_dir / "attack_comparison_bar.png").exists()
        assert (comp_dir / "attack_roc_overlay.png").exists()

    def test_summary_has_all_attacks(self, three_attacks, tmp_path) -> None:
        gen = ReportGenerator(attack_results=three_attacks)
        out = gen.generate(tmp_path / "report")
        data = json.loads((out / "audit_summary.json").read_text())
        assert data["num_attacks"] == 3
        assert data["attack_comparison"] is True

    def test_calls_all_generate_reports(self, three_attacks, tmp_path) -> None:
        gen = ReportGenerator(attack_results=three_attacks)
        gen.generate(tmp_path / "report")
        for name, (attack, _) in three_attacks.items():
            attack.generate_report.assert_called_once()


# ── DP comparison ────────────────────────────────────────────────────────


class TestDPComparisonReport:
    def test_creates_dp_comparison_dir(
        self, three_attacks, dp_attacks, tmp_path,
    ) -> None:
        gen = ReportGenerator(
            attack_results=three_attacks,
            dp_attack_results=dp_attacks,
            epsilon=5.0,
            model_accuracy={"baseline": 0.95, "dp": 0.85},
        )
        out = gen.generate(tmp_path / "report")
        dp_dir = out / "dp_comparison"
        assert dp_dir.is_dir()

    def test_dp_comparison_per_shared_attack(
        self, three_attacks, dp_attacks, tmp_path,
    ) -> None:
        gen = ReportGenerator(
            attack_results=three_attacks,
            dp_attack_results=dp_attacks,
            epsilon=5.0,
        )
        out = gen.generate(tmp_path / "report")
        # dp_attacks has mia_threshold and mia_shadow
        assert (out / "dp_comparison" / "mia_threshold").is_dir()
        assert (out / "dp_comparison" / "mia_shadow").is_dir()
        # attribute_inference not in dp_attacks → no comparison
        assert not (out / "dp_comparison" / "attribute_inference").exists()

    def test_summary_includes_dp_info(
        self, three_attacks, dp_attacks, tmp_path,
    ) -> None:
        gen = ReportGenerator(
            attack_results=three_attacks,
            dp_attack_results=dp_attacks,
            epsilon=5.0,
        )
        out = gen.generate(tmp_path / "report")
        data = json.loads((out / "audit_summary.json").read_text())
        assert data["dp_enabled"] is True
        assert data["epsilon"] == 5.0
        assert data["dp_comparison"] is not None

    def test_summary_txt_mentions_dp(
        self, three_attacks, dp_attacks, tmp_path,
    ) -> None:
        gen = ReportGenerator(
            attack_results=three_attacks,
            dp_attack_results=dp_attacks,
            epsilon=5.0,
        )
        out = gen.generate(tmp_path / "report")
        text = (out / "summary.txt").read_text()
        assert "DP" in text
        assert "5.00" in text


# ── Model accuracy ──────────────────────────────────────────────────────


class TestModelAccuracy:
    def test_model_accuracy_in_summary(self, single_attack, tmp_path) -> None:
        gen = ReportGenerator(
            attack_results=single_attack,
            model_accuracy={"baseline": 0.95, "dp": 0.85},
        )
        out = gen.generate(tmp_path / "report")
        data = json.loads((out / "audit_summary.json").read_text())
        assert data["model_accuracy"]["baseline"] == 0.95

    def test_model_accuracy_in_text(self, single_attack, tmp_path) -> None:
        gen = ReportGenerator(
            attack_results=single_attack,
            model_accuracy={"baseline": 0.9500},
        )
        out = gen.generate(tmp_path / "report")
        text = (out / "summary.txt").read_text()
        assert "0.9500" in text


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_no_attacks(self, tmp_path) -> None:
        gen = ReportGenerator()
        out = gen.generate(tmp_path / "report")
        assert out.is_dir()
        assert (out / "audit_summary.json").exists()

    def test_attack_without_generate_report(self, tmp_path) -> None:
        """Attack that doesn't have generate_report() should still work."""
        result = _make_result(100, 0.7, "simple", seed=0)
        attack = MagicMock()
        attack.evaluate.return_value = {"accuracy": 0.7, "auc_roc": 0.75}
        del attack.generate_report  # remove the method

        gen = ReportGenerator(attack_results={"simple": (attack, result)})
        out = gen.generate(tmp_path / "report")
        # Should fall back to just saving metrics.json
        assert (out / "attacks" / "simple" / "metrics.json").exists()


# ── Imports ──────────────────────────────────────────────────────────────


class TestImports:
    def test_import_from_reporting(self) -> None:
        from auditml.reporting import ReportGenerator
        assert ReportGenerator is not None
