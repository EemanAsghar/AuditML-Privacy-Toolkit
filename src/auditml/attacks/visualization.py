"""Visualization utilities for AuditML attack results.

Provides reusable plotting functions that any attack can call.
All functions optionally save to disk and return a ``matplotlib.figure.Figure``
so callers can further customise or display interactively.

The module uses the ``Agg`` backend by default so that plots can be generated
on headless servers (e.g. Colab, CI) without requiring a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless-safe — must come before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


# ── ROC Curve ────────────────────────────────────────────────────────────


def plot_roc_curve(
    ground_truth: np.ndarray,
    confidence_scores: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "ROC Curve — Membership Inference Attack",
) -> plt.Figure:
    """Plot the Receiver Operating Characteristic curve.

    The ROC curve shows the trade-off between True Positive Rate (TPR)
    and False Positive Rate (FPR) at every possible threshold.  The
    Area Under the Curve (AUC) summarises overall attack effectiveness:
    0.5 = random guessing, 1.0 = perfect attack.

    Parameters
    ----------
    ground_truth:
        Binary array (1 = member, 0 = non-member).
    confidence_scores:
        Continuous scores where higher = more likely member.
    save_path:
        If given, the plot is saved to this path.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Compute ROC
    fpr, tpr, _ = roc_curve(ground_truth, confidence_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {roc_auc:.4f}")

    # Random baseline
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random (AUC = 0.5)")

    # Mark key FPR thresholds
    for target_fpr, marker, label in [
        (0.01, "o", "TPR @ 1% FPR"),
        (0.001, "s", "TPR @ 0.1% FPR"),
    ]:
        tpr_at_fpr = float(np.interp(target_fpr, fpr, tpr))
        ax.plot(target_fpr, tpr_at_fpr, marker=marker, markersize=8,
                label=f"{label} = {tpr_at_fpr:.3f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ── Score Distribution Histogram ─────────────────────────────────────────


def plot_score_distributions(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    metric_name: str = "loss",
    threshold: float | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot overlapping histograms of member vs non-member scores.

    This is the most intuitive visualisation for threshold MIA: if the
    two distributions overlap a lot, the attack is weak (can't tell
    members from non-members).  If they're well-separated, the attack
    is strong.

    Parameters
    ----------
    member_scores:
        Raw signal values for training members.
    nonmember_scores:
        Raw signal values for non-members.
    metric_name:
        Name of the metric (for axis label).
    threshold:
        If given, a vertical line is drawn at this value.
    save_path:
        If given, saves the figure.
    title:
        Plot title. Auto-generated if ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if title is None:
        title = f"Score Distribution — {metric_name.capitalize()} Metric"

    # Compute shared bin edges for fair comparison
    all_scores = np.concatenate([member_scores, nonmember_scores])
    bins = np.linspace(all_scores.min(), all_scores.max(), 50)

    ax.hist(member_scores, bins=bins, alpha=0.6, color="#2563eb",
            label=f"Members (n={len(member_scores)})", density=True)
    ax.hist(nonmember_scores, bins=bins, alpha=0.6, color="#dc2626",
            label=f"Non-members (n={len(nonmember_scores)})", density=True)

    if threshold is not None:
        ax.axvline(threshold, color="black", lw=2, linestyle="--",
                   label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel(f"{metric_name.capitalize()} Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ── Per-Class Metrics Bar Chart ──────────────────────────────────────────


def plot_per_class_metrics(
    per_class_metrics: dict[int, dict[str, float]],
    metric_key: str = "accuracy",
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Bar chart showing a chosen metric for each class.

    Useful for identifying which classes are most vulnerable to
    membership inference (higher accuracy = more vulnerable).

    Parameters
    ----------
    per_class_metrics:
        Output of ``ThresholdMIA.evaluate_per_class()``.
    metric_key:
        Which metric to plot (default ``"accuracy"``).
    save_path:
        If given, saves the figure.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if title is None:
        title = f"Per-Class Attack {metric_key.replace('_', ' ').title()}"

    classes = sorted(per_class_metrics.keys())
    values = [per_class_metrics[c][metric_key] for c in classes]
    n_samples = [per_class_metrics[c].get("n_samples", 0) for c in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.5), 5))

    bars = ax.bar(range(len(classes)), values, color="#2563eb", alpha=0.8)

    # Add sample counts above each bar
    for i, (bar, n) in enumerate(zip(bars, n_samples)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=7)

    # Draw random baseline at 0.5
    ax.axhline(0.5, color="grey", linestyle="--", lw=1, alpha=0.7,
               label="Random baseline (0.5)")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([str(c) for c in classes], fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
