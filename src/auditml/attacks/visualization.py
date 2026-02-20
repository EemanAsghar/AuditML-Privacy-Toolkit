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


# ── Reconstructed Images Grid ────────────────────────────────────────────


def plot_reconstructions(
    reconstructions: dict[int, np.ndarray],
    confidences: dict[int, float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Model Inversion — Reconstructed Images",
) -> plt.Figure:
    """Display reconstructed images in a grid, one per class.

    This is the key visual for model inversion: if the images look like
    recognisable digits/objects, the model has leaked training data.

    Parameters
    ----------
    reconstructions:
        Mapping from class label to image array of shape
        ``(1, C, H, W)`` or ``(C, H, W)``.
    confidences:
        Optional mapping from class label to reconstruction confidence.
        Displayed below each image.
    save_path:
        If given, saves the figure.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = sorted(reconstructions.keys())
    n = len(classes)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, cls in enumerate(classes):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        img = reconstructions[cls]
        # Handle (1, C, H, W) or (C, H, W) shapes
        if img.ndim == 4:
            img = img[0]  # remove batch dim -> (C, H, W)

        if img.shape[0] == 1:
            # Grayscale: (1, H, W) -> (H, W)
            ax.imshow(img[0], cmap="gray")
        elif img.shape[0] == 3:
            # RGB: (C, H, W) -> (H, W, C), normalise to [0, 1]
            img_hwc = np.transpose(img, (1, 2, 0))
            img_hwc = np.clip(
                (img_hwc - img_hwc.min()) / (img_hwc.max() - img_hwc.min() + 1e-8),
                0, 1,
            )
            ax.imshow(img_hwc)
        else:
            ax.imshow(img[0], cmap="gray")

        label = f"Class {cls}"
        if confidences and cls in confidences:
            label += f"\nconf={confidences[cls]:.3f}"
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ── Reconstruction Confidence Bar Chart ──────────────────────────────────


def plot_reconstruction_confidence(
    confidences: dict[int, float],
    save_path: str | Path | None = None,
    title: str = "Reconstruction Confidence per Class",
) -> plt.Figure:
    """Bar chart of model confidence on each reconstructed image.

    Higher confidence means the optimisation was more successful at
    producing an image the model strongly associates with that class.

    Parameters
    ----------
    confidences:
        Mapping from class label to confidence (softmax probability).
    save_path:
        If given, saves the figure.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = sorted(confidences.keys())
    values = [confidences[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.5), 5))
    bars = ax.bar(range(len(classes)), values, color="#7c3aed", alpha=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([str(c) for c in classes], fontsize=9)
    ax.set_xlabel("Class")
    ax.set_ylabel("Confidence (Softmax Probability)")
    ax.set_title(title)
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ── Attribute Inference Accuracy per Group ────────────────────────────────


def plot_attribute_accuracy(
    member_accuracy: dict[int, float],
    nonmember_accuracy: dict[int, float],
    save_path: str | Path | None = None,
    title: str = "Attribute Prediction Accuracy — Members vs Non-Members",
) -> plt.Figure:
    """Side-by-side bar chart comparing attribute accuracy for members and non-members.

    A large gap between member and non-member accuracy indicates that the
    model's outputs reveal more about the sensitive attribute for training
    data — a privacy leak.

    Parameters
    ----------
    member_accuracy:
        Mapping from group ID to attribute prediction accuracy on members.
    nonmember_accuracy:
        Mapping from group ID to attribute prediction accuracy on non-members.
    save_path:
        If given, saves the figure.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    groups = sorted(set(member_accuracy.keys()) | set(nonmember_accuracy.keys()))
    mem_vals = [member_accuracy.get(g, 0.0) for g in groups]
    nonmem_vals = [nonmember_accuracy.get(g, 0.0) for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.8), 5))
    bars_mem = ax.bar(x - width / 2, mem_vals, width, label="Members",
                      color="#2563eb", alpha=0.8)
    bars_non = ax.bar(x + width / 2, nonmem_vals, width, label="Non-members",
                      color="#dc2626", alpha=0.8)

    # Add value labels above bars
    for bar in bars_mem:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_non:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(1.0 / max(len(groups), 1), color="grey", linestyle="--", lw=1,
               alpha=0.7, label="Random baseline")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Group {g}" for g in groups], fontsize=8)
    ax.set_xlabel("Sensitive Attribute Group")
    ax.set_ylabel("Prediction Accuracy")
    ax.set_title(title)
    ax.set_ylim([0, 1.15])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
