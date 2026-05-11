"""High-level public API for AuditML.

This is the entry point for external users who bring their own model and data.
No YAML config, no dataset names, no internal knowledge required.

Example
-------
>>> import auditml
>>> results = auditml.audit(model, train_loader, test_loader)
>>> print(results.summary())

>>> # Pick specific attacks
>>> results = auditml.audit(
...     model, train_loader, test_loader,
...     attacks=["mia_threshold", "model_inversion"],
... )

>>> # Shadow MIA needs a factory to build shadow models
>>> results = auditml.audit(
...     model, train_loader, test_loader,
...     attacks=["mia_shadow"],
...     shadow_model_fn=lambda: MyCNN(),
... )

>>> # Generate an HTML report and open it in the browser
>>> results.report("./my_report", open_browser=True)

>>> # Save and reload results later
>>> results.save("audit_results.json")
>>> results2 = auditml.AuditResults.load("audit_results.json")
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

VALID_ATTACKS = ["mia_threshold", "mia_shadow", "model_inversion", "attribute_inference"]


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class AttackSummary:
    """Results for a single attack."""
    name: str
    accuracy: float
    auc_roc: float
    precision: float
    recall: float
    f1: float
    tpr_at_1fpr: float
    tpr_at_01fpr: float
    duration_seconds: float
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AttackSummary({self.name}: "
            f"accuracy={self.accuracy:.3f}, auc_roc={self.auc_roc:.3f})"
        )


class AuditResults:
    """Container for all attack results from a single audit run.

    Attributes
    ----------
    attacks : dict[str, AttackSummary]
        Mapping from attack name → metrics.
    model_info : dict
        Basic info about the audited model.

    Examples
    --------
    >>> results = auditml.audit(model, train_loader, test_loader)
    >>> results.summary()                      # print table
    >>> results["mia_threshold"].auc_roc       # access specific attack
    >>> results.most_vulnerable()              # attack with highest AUC
    >>> results.report("./report", open_browser=True)  # HTML report
    >>> results.save("results.json")           # persist to disk
    >>> results2 = AuditResults.load("results.json")   # reload
    """

    def __init__(
        self,
        attacks: dict[str, AttackSummary],
        model_info: dict,
        _raw_results: dict | None = None,
    ) -> None:
        self.attacks = attacks
        self.model_info = model_info
        # Raw (attack_object, AttackResult) pairs — used for HTML report generation.
        # Not persisted to JSON (not serialisable), but available after audit().
        self._raw_results: dict[str, tuple] = _raw_results or {}

    def __getitem__(self, attack_name: str) -> AttackSummary:
        if attack_name not in self.attacks:
            available = list(self.attacks.keys())
            raise KeyError(f"Attack {attack_name!r} not in results. Available: {available}")
        return self.attacks[attack_name]

    def __repr__(self) -> str:
        return f"AuditResults(attacks={list(self.attacks.keys())})"

    def most_vulnerable(self) -> AttackSummary:
        """Return the attack that found the highest AUC-ROC."""
        return max(self.attacks.values(), key=lambda a: a.auc_roc)

    def is_vulnerable(self, threshold: float = 0.55) -> bool:
        """Return True if any attack achieved AUC above *threshold*.

        A model with AUC > 0.55 is showing detectable privacy leakage.
        """
        return any(a.auc_roc > threshold for a in self.attacks.values())

    def summary(self) -> str:
        """Return a formatted table of all attack results."""
        lines = [
            "",
            "=" * 65,
            "  AuditML Privacy Audit Results",
            "=" * 65,
        ]

        if self.model_info:
            lines.append(f"  Model     : {self.model_info.get('num_parameters', '?')} parameters")
            lines.append(f"  Device    : {self.model_info.get('device', '?')}")

        lines += [
            "",
            f"  {'Attack':<22} {'Accuracy':>8} {'AUC-ROC':>8} {'F1':>6} {'TPR@1%FPR':>10}",
            "  " + "-" * 58,
        ]

        for name, att in self.attacks.items():
            vuln = " ⚠" if att.auc_roc > 0.55 else ""
            lines.append(
                f"  {name:<22} {att.accuracy:>8.3f} {att.auc_roc:>8.3f} "
                f"{att.f1:>6.3f} {att.tpr_at_1fpr:>10.3f}{vuln}"
            )

        lines.append("  " + "-" * 58)

        if self.is_vulnerable():
            mv = self.most_vulnerable()
            lines.append(f"\n  ⚠  Leakage detected — highest AUC: {mv.auc_roc:.3f} ({mv.name})")
            lines.append("     Consider Differential Privacy training (epsilon=3–5).")
        else:
            lines.append("\n  ✓  No significant leakage detected (all AUC < 0.55).")

        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary of all results."""
        return {
            "model_info": self.model_info,
            "attacks": {
                name: {
                    "accuracy": att.accuracy,
                    "auc_roc": att.auc_roc,
                    "precision": att.precision,
                    "recall": att.recall,
                    "f1": att.f1,
                    "tpr_at_1fpr": att.tpr_at_1fpr,
                    "tpr_at_01fpr": att.tpr_at_01fpr,
                    "duration_seconds": att.duration_seconds,
                    "metadata": att.metadata,
                }
                for name, att in self.attacks.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        """Persist results to a JSON file.

        Parameters
        ----------
        path:
            File path to write (e.g. ``"results.json"``).

        Returns
        -------
        Path
            The path that was written.

        Notes
        -----
        Only metrics are saved — raw attack objects are not serialisable.
        Use :meth:`load` to restore metrics from the file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return out

    @classmethod
    def load(cls, path: str | Path) -> AuditResults:
        """Restore an :class:`AuditResults` from a JSON file saved by :meth:`save`.

        Parameters
        ----------
        path:
            Path to a JSON file previously written by ``results.save()``.

        Returns
        -------
        AuditResults
            Restored results object. ``_raw_results`` will be empty (raw
            attack objects are not serialisable), but all metrics are intact.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        attacks: dict[str, AttackSummary] = {}
        for name, m in data.get("attacks", {}).items():
            attacks[name] = AttackSummary(
                name=name,
                accuracy=m.get("accuracy", 0.0),
                auc_roc=m.get("auc_roc", 0.0),
                precision=m.get("precision", 0.0),
                recall=m.get("recall", 0.0),
                f1=m.get("f1", 0.0),
                tpr_at_1fpr=m.get("tpr_at_1fpr", 0.0),
                tpr_at_01fpr=m.get("tpr_at_01fpr", 0.0),
                duration_seconds=m.get("duration_seconds", 0.0),
                metadata=m.get("metadata", {}),
            )

        return cls(attacks=attacks, model_info=data.get("model_info", {}))

    def report(
        self,
        output_dir: str | Path = "./auditml_report",
        open_browser: bool = True,
        experiment_name: str | None = None,
    ) -> Path:
        """Generate a self-contained HTML report and optionally open it.

        Parameters
        ----------
        output_dir:
            Directory to write ``report.html`` and supporting files into.
            Created automatically if it does not exist.
        open_browser:
            If ``True`` (default), open the report in the system browser
            immediately after generation.
        experiment_name:
            Title shown in the report header. Defaults to ``"audit"``.

        Returns
        -------
        Path
            Path to the generated ``report.html`` file.

        Examples
        --------
        >>> results = auditml.audit(model, m_loader, nm_loader)
        >>> results.report("./report", open_browser=True)
        """
        from auditml.reporting.html_report import generate_html_report

        out = Path(output_dir)
        name = experiment_name or "audit"

        attack_metrics = {n: {
            "accuracy": s.accuracy,
            "auc_roc": s.auc_roc,
            "precision": s.precision,
            "recall": s.recall,
            "f1": s.f1,
            "tpr_at_1fpr": s.tpr_at_1fpr,
            "tpr_at_01fpr": s.tpr_at_01fpr,
        } for n, s in self.attacks.items()}

        html_path = generate_html_report(
            output_dir=out,
            experiment_name=name,
            attack_results=self._raw_results,
            dp_attack_results=None,
            attack_metrics=attack_metrics,
            dp_attack_metrics=None,
            model_accuracy=self.model_info.get("model_accuracy"),
            epsilon=None,
        )

        if open_browser:
            import webbrowser
            webbrowser.open(html_path.as_uri())

        return html_path


# ---------------------------------------------------------------------------
# Dataset split helper
# ---------------------------------------------------------------------------

def split_loaders(
    dataset: Any,
    member_ratio: float = 0.5,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split a dataset into member and non-member DataLoaders.

    Randomly splits *dataset* into two halves — the "members" (samples the
    model trained on) and "non-members" (held-out samples).  Pass the
    resulting loaders directly to :func:`audit`.

    Parameters
    ----------
    dataset:
        Any ``torch.utils.data.Dataset`` (e.g. your training set).
    member_ratio:
        Fraction of samples to treat as members.  Default ``0.5`` gives
        a balanced 50/50 split.
    batch_size:
        Batch size for both returned loaders.
    seed:
        Random seed for reproducible splits.
    num_workers:
        Number of DataLoader worker processes.

    Returns
    -------
    (member_loader, nonmember_loader)
        Two balanced DataLoaders ready to pass to :func:`audit`.

    Examples
    --------
    >>> member_loader, nonmember_loader = auditml.split_loaders(train_dataset)
    >>> results = auditml.audit(model, member_loader, nonmember_loader)
    """
    from torch.utils.data import random_split

    n = len(dataset)  # type: ignore[arg-type]
    n_members = int(n * member_ratio)
    n_nonmembers = n - n_members

    generator = torch.Generator().manual_seed(seed)
    member_ds, nonmember_ds = random_split(
        dataset, [n_members, n_nonmembers], generator=generator
    )

    member_loader = DataLoader(
        member_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    nonmember_loader = DataLoader(
        nonmember_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return member_loader, nonmember_loader


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def audit(
    model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    attacks: list[str] | None = None,
    shadow_model_fn: Callable[[], nn.Module] | None = None,
    input_shape: tuple[int, ...] | None = None,
    num_classes: int | None = None,
    device: str = "auto",
) -> AuditResults:
    """Audit a trained PyTorch model for privacy leakage.

    Parameters
    ----------
    model:
        Your trained ``nn.Module``. Can be any architecture.
    member_loader:
        ``DataLoader`` of samples the model **was** trained on.
    nonmember_loader:
        ``DataLoader`` of samples the model **was not** trained on.
        Should be the same size as ``member_loader`` for balanced evaluation.
    attacks:
        List of attack names to run. If ``None``, all attacks are run.
        Valid values: ``"mia_threshold"``, ``"mia_shadow"``,
        ``"model_inversion"``, ``"attribute_inference"``.
    shadow_model_fn:
        A zero-argument callable that returns a fresh untrained model of the
        same architecture as *model*. Used only for ``"mia_shadow"``.
        If omitted, a simple MLP is automatically constructed as a fallback.
    input_shape:
        Shape of one input sample, e.g. ``(1, 28, 28)``. Required only for
        ``"model_inversion"`` if it cannot be inferred from the data.
    num_classes:
        Number of output classes. Inferred automatically if not provided.
    device:
        ``"auto"`` picks MPS → CUDA → CPU. Pass ``"cpu"`` etc. to override.

    Returns
    -------
    AuditResults
        Call ``.summary()`` to print a table, ``.report()`` for an HTML
        report, or access individual attacks via ``results["mia_threshold"]``.

    Examples
    --------
    >>> member_loader, nonmember_loader = auditml.split_loaders(train_dataset)
    >>> results = auditml.audit(model, member_loader, nonmember_loader)
    >>> print(results.summary())
    >>> results.report("./report", open_browser=True)
    >>> results.save("results.json")
    """
    from tqdm import tqdm

    # ── Validate inputs ────────────────────────────────────────────────
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(member_loader, DataLoader):
        raise TypeError("member_loader must be a torch.utils.data.DataLoader")
    if not isinstance(nonmember_loader, DataLoader):
        raise TypeError("nonmember_loader must be a torch.utils.data.DataLoader")

    if attacks is None:
        attacks = list(VALID_ATTACKS)

    unknown = [a for a in attacks if a not in VALID_ATTACKS]
    if unknown:
        raise ValueError(
            f"Unknown attack(s): {unknown}. Valid choices: {VALID_ATTACKS}"
        )

    # ── Resolve device ─────────────────────────────────────────────────
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    # ── Infer num_classes / input_shape ────────────────────────────────
    if num_classes is None:
        num_classes = _infer_num_classes(model, member_loader, resolved_device)
    if input_shape is None and "model_inversion" in attacks:
        input_shape = _infer_input_shape(member_loader)

    model_info: dict[str, Any] = {
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(resolved_device),
        "num_classes": num_classes,
    }

    # ── Run each attack with a progress bar ────────────────────────────
    results: dict[str, AttackSummary] = {}
    raw_results: dict[str, tuple] = {}

    pbar = tqdm(attacks, desc="AuditML", unit="attack", ncols=70)
    for attack_name in pbar:
        pbar.set_description(f"AuditML  [{attack_name}]")
        t0 = time.perf_counter()

        try:
            summary, attack_obj = _run_attack(
                attack_name=attack_name,
                model=model,
                member_loader=member_loader,
                nonmember_loader=nonmember_loader,
                num_classes=num_classes,
                input_shape=input_shape,
                shadow_model_fn=shadow_model_fn,
                device=str(resolved_device),
            )
            summary.duration_seconds = time.perf_counter() - t0
            results[attack_name] = summary
            raw_results[attack_name] = (attack_obj, attack_obj.result)
            pbar.write(
                f"  {attack_name}: accuracy={summary.accuracy:.3f}  "
                f"auc_roc={summary.auc_roc:.3f}  ({summary.duration_seconds:.1f}s)"
            )
        except Exception as exc:
            pbar.write(f"  {attack_name} failed: {exc}")
            raise

    return AuditResults(attacks=results, model_info=model_info, _raw_results=raw_results)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _infer_num_classes(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> int:
    batch = next(iter(loader))
    inputs = batch[0].to(device)
    outputs = model(inputs)
    return outputs.shape[-1]


def _infer_input_shape(loader: DataLoader) -> tuple[int, ...]:
    batch = next(iter(loader))
    return tuple(batch[0].shape[1:])


def _run_attack(
    attack_name: str,
    model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    num_classes: int,
    input_shape: tuple[int, ...] | None,
    shadow_model_fn: Callable[[], nn.Module] | None,
    device: str,
) -> tuple[AttackSummary, Any]:
    """Instantiate and run a single attack. Returns (AttackSummary, attack_object)."""
    if attack_name == "mia_threshold":
        from auditml.attacks.mia_threshold import ThresholdMIA
        attack = ThresholdMIA(target_model=model, device=device)

    elif attack_name == "mia_shadow":
        from auditml.attacks.mia_shadow import ShadowMIA
        shadow_ds = _loaders_to_dataset(member_loader, nonmember_loader)
        attack = ShadowMIA(
            target_model=model,
            device=device,
            shadow_dataset=shadow_ds,
            shadow_model_fn=shadow_model_fn,
            num_classes=num_classes,
        )

    elif attack_name == "model_inversion":
        from auditml.attacks.model_inversion import ModelInversion
        attack = ModelInversion(
            target_model=model,
            device=device,
            input_shape=input_shape,
            num_classes=num_classes,
        )

    elif attack_name == "attribute_inference":
        from auditml.attacks.attribute_inference import AttributeInference
        attack = AttributeInference(
            target_model=model,
            device=device,
            num_classes=num_classes,
        )

    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    attack.run(member_loader, nonmember_loader)
    metrics = attack.evaluate()

    summary = AttackSummary(
        name=attack_name,
        accuracy=metrics.get("accuracy", 0.0),
        auc_roc=metrics.get("auc_roc", 0.0),
        precision=metrics.get("precision", 0.0),
        recall=metrics.get("recall", 0.0),
        f1=metrics.get("f1", 0.0),
        tpr_at_1fpr=metrics.get("tpr_at_1fpr", 0.0),
        tpr_at_01fpr=metrics.get("tpr_at_01fpr", 0.0),
        duration_seconds=0.0,
        metadata=attack.result.metadata if attack.result else {},
    )
    return summary, attack


def _loaders_to_dataset(
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
) -> Any:
    """Combine two DataLoaders into a single TensorDataset."""
    from torch.utils.data import TensorDataset

    all_x, all_y = [], []
    for loader in (member_loader, nonmember_loader):
        for x, y in loader:
            all_x.append(x)
            all_y.append(y)

    return TensorDataset(torch.cat(all_x), torch.cat(all_y))
