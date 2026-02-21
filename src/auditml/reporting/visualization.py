"""Visualization functions for DP vs Non-DP comparison reports.

Provides side-by-side plots that make it easy to see how DP training
affects attack effectiveness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


# ── Metric Comparison Bar Chart ──────────────────────────────────────────


def plot_metric_comparison(
    baseline_metrics: dict[str, float],
    dp_metrics: dict[str, float],
    save_path: str | Path | None = None,
    title: str = "Attack Metrics — Baseline vs DP",
) -> plt.Figure:
    """Side-by-side bar chart comparing attack metrics.

    Parameters
    ----------
    baseline_metrics:
        Metrics from the standard (non-DP) model.
    dp_metrics:
        Metrics from the DP model.
    save_path:
        If given, saves the figure.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Select the most informative metrics for the chart
    display_keys = [
        k for k in ("accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr")
        if k in baseline_metrics and k in dp_metrics
    ]

    baseline_vals = [baseline_metrics[k] for k in display_keys]
    dp_vals = [dp_metrics[k] for k in display_keys]

    x = np.arange(len(display_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_base = ax.bar(x - width / 2, baseline_vals, width,
                       label="Baseline (no DP)", color="#2563eb", alpha=0.8)
    bars_dp = ax.bar(x + width / 2, dp_vals, width,
                     label="DP-trained", color="#16a34a", alpha=0.8)

    for bar in bars_base:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_dp:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0.5, color="grey", linestyle="--", lw=1, alpha=0.7,
               label="Random baseline (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in display_keys],
                       fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim([0, 1.15])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ── ROC Curve Comparison ─────────────────────────────────────────────────


def plot_roc_comparison(
    baseline_gt: np.ndarray,
    baseline_scores: np.ndarray,
    dp_gt: np.ndarray,
    dp_scores: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "ROC Curve — Baseline vs DP",
) -> plt.Figure:
    """Overlay ROC curves from baseline and DP models.

    Parameters
    ----------
    baseline_gt:
        Ground truth for baseline attack.
    baseline_scores:
        Confidence scores for baseline attack.
    dp_gt:
        Ground truth for DP attack.
    dp_scores:
        Confidence scores for DP attack.
    save_path:
        If given, saves the figure.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Baseline ROC
    fpr_b, tpr_b, _ = roc_curve(baseline_gt, baseline_scores)
    auc_b = auc(fpr_b, tpr_b)
    ax.plot(fpr_b, tpr_b, color="#2563eb", lw=2,
            label=f"Baseline (AUC = {auc_b:.4f})")

    # DP ROC
    fpr_d, tpr_d, _ = roc_curve(dp_gt, dp_scores)
    auc_d = auc(fpr_d, tpr_d)
    ax.plot(fpr_d, tpr_d, color="#16a34a", lw=2,
            label=f"DP (AUC = {auc_d:.4f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--",
            label="Random (AUC = 0.5)")

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


# ── Score Distribution Comparison ────────────────────────────────────────


def plot_score_comparison(
    baseline_scores: np.ndarray,
    baseline_gt: np.ndarray,
    dp_scores: np.ndarray,
    dp_gt: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Confidence Score Distribution — Baseline vs DP",
) -> plt.Figure:
    """Four-panel histogram comparing member/non-member score distributions.

    Top row: baseline model (members vs non-members).
    Bottom row: DP model (members vs non-members).

    Well-separated distributions indicate a successful attack; overlapping
    distributions indicate the model is more private.

    Parameters
    ----------
    baseline_scores:
        Confidence scores from baseline attack.
    baseline_gt:
        Ground truth for baseline.
    dp_scores:
        Confidence scores from DP attack.
    dp_gt:
        Ground truth for DP.
    save_path:
        If given, saves the figure.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, scores, gt, label in [
        (axes[0], baseline_scores, baseline_gt, "Baseline"),
        (axes[1], dp_scores, dp_gt, "DP"),
    ]:
        member_mask = gt == 1
        nonmember_mask = gt == 0

        all_vals = np.concatenate([scores[member_mask], scores[nonmember_mask]])
        bins = np.linspace(all_vals.min(), all_vals.max(), 40)

        ax.hist(scores[member_mask], bins=bins, alpha=0.6,
                color="#2563eb", label=f"Members (n={member_mask.sum()})",
                density=True)
        ax.hist(scores[nonmember_mask], bins=bins, alpha=0.6,
                color="#dc2626", label=f"Non-members (n={nonmember_mask.sum()})",
                density=True)

        ax.set_xlabel("Confidence Score")
        ax.set_title(f"{label} Model")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Density")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
