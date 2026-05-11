"""Tests for the high-level public API: audit(), split_loaders(), AuditResults."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import auditml
from auditml import AttackSummary, AuditResults, split_loaders

# ── Helpers ──────────────────────────────────────────────────────────────────

NUM_SAMPLES = 80
NUM_CLASSES = 5
INPUT_DIM = 10


class _LinearNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(INPUT_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture()
def toy_dataset() -> TensorDataset:
    torch.manual_seed(0)
    x = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    return TensorDataset(x, y)


@pytest.fixture()
def toy_model() -> _LinearNet:
    return _LinearNet()


@pytest.fixture()
def member_nonmember(toy_dataset):
    return split_loaders(toy_dataset, batch_size=16, seed=42)


@pytest.fixture()
def audit_results(toy_model, member_nonmember):
    m, nm = member_nonmember
    return auditml.audit(
        toy_model, m, nm,
        attacks=["mia_threshold"],
        device="cpu",
    )


# ── split_loaders ─────────────────────────────────────────────────────────────

class TestSplitLoaders:
    def test_returns_two_dataloaders(self, toy_dataset):
        m, nm = split_loaders(toy_dataset)
        assert isinstance(m, DataLoader)
        assert isinstance(nm, DataLoader)

    def test_default_50_50_split(self, toy_dataset):
        m, nm = split_loaders(toy_dataset)
        assert len(m.dataset) == NUM_SAMPLES // 2
        assert len(nm.dataset) == NUM_SAMPLES // 2

    def test_custom_member_ratio(self, toy_dataset):
        m, nm = split_loaders(toy_dataset, member_ratio=0.8)
        assert len(m.dataset) == 64
        assert len(nm.dataset) == 16

    def test_batch_size_respected(self, toy_dataset):
        m, nm = split_loaders(toy_dataset, batch_size=8)
        assert m.batch_size == 8
        assert nm.batch_size == 8

    def test_reproducible_with_seed(self, toy_dataset):
        m1, _ = split_loaders(toy_dataset, seed=1)
        m2, _ = split_loaders(toy_dataset, seed=1)
        x1 = next(iter(m1))[0]
        x2 = next(iter(m2))[0]
        assert torch.allclose(x1, x2)

    def test_different_seeds_give_different_splits(self, toy_dataset):
        m1, _ = split_loaders(toy_dataset, seed=1)
        m2, _ = split_loaders(toy_dataset, seed=2)
        x1 = next(iter(m1))[0]
        x2 = next(iter(m2))[0]
        assert not torch.allclose(x1, x2)

    def test_total_samples_preserved(self, toy_dataset):
        m, nm = split_loaders(toy_dataset, member_ratio=0.3)
        assert len(m.dataset) + len(nm.dataset) == NUM_SAMPLES


# ── AuditResults.save / load ──────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_file(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            audit_results.save(path)
            assert path.exists()

    def test_save_valid_json(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            audit_results.save(path)
            data = json.loads(path.read_text())
            assert "attacks" in data
            assert "model_info" in data

    def test_load_restores_attack_names(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            audit_results.save(path)
            r2 = AuditResults.load(path)
            assert set(r2.attacks.keys()) == set(audit_results.attacks.keys())

    def test_load_restores_metrics(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            audit_results.save(path)
            r2 = AuditResults.load(path)
            orig = audit_results["mia_threshold"]
            loaded = r2["mia_threshold"]
            assert abs(orig.auc_roc - loaded.auc_roc) < 1e-6
            assert abs(orig.accuracy - loaded.accuracy) < 1e-6

    def test_load_returns_audit_results_instance(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            audit_results.save(path)
            r2 = AuditResults.load(path)
            assert isinstance(r2, AuditResults)

    def test_save_creates_parent_dirs(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "dir" / "results.json"
            audit_results.save(path)
            assert path.exists()

    def test_loaded_is_vulnerable_matches(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            audit_results.save(path)
            r2 = AuditResults.load(path)
            assert r2.is_vulnerable() == audit_results.is_vulnerable()

    def test_load_raw_results_empty(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            audit_results.save(path)
            r2 = AuditResults.load(path)
            assert r2._raw_results == {}


# ── AuditResults.report() ─────────────────────────────────────────────────────

class TestReport:
    def test_report_creates_html(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            html = audit_results.report(tmp, open_browser=False)
            assert Path(html).exists()
            assert Path(html).suffix == ".html"

    def test_report_html_has_content(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            html = audit_results.report(tmp, open_browser=False)
            content = Path(html).read_text()
            assert "AuditML" in content
            assert "mia_threshold" in content.lower() or "Threshold MIA" in content

    def test_report_has_svg_charts(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            html = audit_results.report(tmp, open_browser=False)
            content = Path(html).read_text()
            assert "<svg" in content

    def test_report_has_risk_label(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            html = audit_results.report(tmp, open_browser=False)
            content = Path(html).read_text()
            assert "RISK" in content or "SAFE" in content

    def test_report_custom_experiment_name(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            html = audit_results.report(tmp, open_browser=False, experiment_name="my_exp")
            content = Path(html).read_text()
            assert "my_exp" in content

    def test_report_works_after_load(self, audit_results):
        """report() must not crash when _raw_results is empty (post-load)."""
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "r.json"
            report_dir = Path(tmp) / "report"
            audit_results.save(save_path)
            r2 = AuditResults.load(save_path)
            # Should not raise even with no raw attack data
            html = r2.report(str(report_dir), open_browser=False)
            assert Path(html).exists()

    def test_report_returns_path(self, audit_results):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit_results.report(tmp, open_browser=False)
            assert isinstance(result, Path)


# ── AuditResults general ──────────────────────────────────────────────────────

class TestAuditResults:
    def test_getitem(self, audit_results):
        s = audit_results["mia_threshold"]
        assert isinstance(s, AttackSummary)

    def test_getitem_missing_raises(self, audit_results):
        with pytest.raises(KeyError):
            audit_results["nonexistent_attack"]

    def test_most_vulnerable(self, audit_results):
        mv = audit_results.most_vulnerable()
        assert isinstance(mv, AttackSummary)
        assert mv.auc_roc == max(a.auc_roc for a in audit_results.attacks.values())

    def test_is_vulnerable_threshold(self, audit_results):
        assert isinstance(audit_results.is_vulnerable(), bool)
        assert audit_results.is_vulnerable(threshold=0.0)
        assert not audit_results.is_vulnerable(threshold=1.0)

    def test_to_dict_structure(self, audit_results):
        d = audit_results.to_dict()
        assert "attacks" in d
        assert "model_info" in d
        assert "mia_threshold" in d["attacks"]
        assert "auc_roc" in d["attacks"]["mia_threshold"]

    def test_summary_contains_attack_name(self, audit_results):
        s = audit_results.summary()
        assert "mia_threshold" in s

    def test_repr(self, audit_results):
        r = repr(audit_results)
        assert "AuditResults" in r

    def test_raw_results_populated_after_audit(self, audit_results):
        assert "mia_threshold" in audit_results._raw_results


# ── Shadow MIA through public API ─────────────────────────────────────────────

class TestShadowMIAPublicAPI:
    """Smoke-test mia_shadow end-to-end through auditml.audit()."""

    def test_shadow_mia_runs(self, toy_model, member_nonmember):
        m, nm = member_nonmember
        results = auditml.audit(
            toy_model, m, nm,
            attacks=["mia_shadow"],
            device="cpu",
        )
        assert "mia_shadow" in results.attacks

    def test_shadow_mia_returns_attack_summary(self, toy_model, member_nonmember):
        m, nm = member_nonmember
        results = auditml.audit(toy_model, m, nm, attacks=["mia_shadow"], device="cpu")
        s = results["mia_shadow"]
        assert isinstance(s, AttackSummary)

    def test_shadow_mia_metrics_in_range(self, toy_model, member_nonmember):
        m, nm = member_nonmember
        results = auditml.audit(toy_model, m, nm, attacks=["mia_shadow"], device="cpu")
        s = results["mia_shadow"]
        assert 0.0 <= s.auc_roc <= 1.0
        assert 0.0 <= s.accuracy <= 1.0

    def test_shadow_mia_fallback_no_model_fn(self, toy_model, member_nonmember):
        """Should work without shadow_model_fn — uses the MLP fallback."""
        m, nm = member_nonmember
        results = auditml.audit(
            toy_model, m, nm,
            attacks=["mia_shadow"],
            shadow_model_fn=None,
            device="cpu",
        )
        assert results["mia_shadow"].auc_roc >= 0.0

    def test_shadow_mia_custom_model_fn(self, toy_model, member_nonmember):
        """shadow_model_fn is forwarded correctly."""
        m, nm = member_nonmember

        def make_shadow():
            return _LinearNet()

        results = auditml.audit(
            toy_model, m, nm,
            attacks=["mia_shadow"],
            shadow_model_fn=make_shadow,
            device="cpu",
        )
        assert "mia_shadow" in results.attacks

    def test_shadow_mia_in_report(self, toy_model, member_nonmember):
        """HTML report must mention shadow MIA when it was run."""
        m, nm = member_nonmember
        results = auditml.audit(toy_model, m, nm, attacks=["mia_shadow"], device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            html = results.report(tmp, open_browser=False)
            content = html.read_text()
            assert "shadow" in content.lower() or "Shadow MIA" in content

    def test_shadow_mia_save_load(self, toy_model, member_nonmember):
        """save/load round-trip preserves mia_shadow metrics."""
        m, nm = member_nonmember
        results = auditml.audit(toy_model, m, nm, attacks=["mia_shadow"], device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            results.save(path)
            r2 = AuditResults.load(path)
            assert abs(results["mia_shadow"].auc_roc - r2["mia_shadow"].auc_roc) < 1e-6
