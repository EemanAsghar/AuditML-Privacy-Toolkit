"""Visualization functions for cross-attack comparison reports.

Provides plots that compare multiple attack types against the same
model, making it easy to see which attacks are most effective.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from auditml.attacks.results import AttackResult

# Consistent colours for each attack type
_ATTACK_COLOURS: dict[str, str] = {
    "mia_threshold": "#2563eb",
    "mia_shadow": "#dc2626",
    "model_inversion": "#16a34a",
    "attribute_inference": "#9333ea",
}

_DEFAULT_COLOURS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def _colour_for(name: str, idx: int) -> str:
    """Return a consistent colour for a given attack name."""
    return _ATTACK_COLOURS.get(name, _DEFAULT_COLOURS[idx % len(_DEFAULT_COLOURS)])


# ── Grouped Bar Chart ────────────────────────────────────────────────────


def plot_attack_comparison_bar(
    metrics: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
    title: str = "Attack Comparison — Key Metrics",
) -> plt.Figure:
    """Grouped bar chart comparing selected metrics across attacks.

    Parameters
    ----------
    metrics:
        Mapping from attack name to metric dict.
    save_path:
        If given, saves the figure.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    display_keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]
    # Only include keys that exist in at least one attack
    display_keys = [
        k for k in display_keys
        if any(k in m for m in metrics.values())
    ]

    attack_names = list(metrics.keys())
    n_attacks = len(attack_names)
    n_metrics = len(display_keys)

    x = np.arange(n_metrics)
    width = 0.8 / max(n_attacks, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), 5))

    for i, name in enumerate(attack_names):
        vals = [metrics[name].get(k, 0.0) for k in display_keys]
        offset = (i - n_attacks / 2 + 0.5) * width
        colour = _colour_for(name, i)
        bars = ax.bar(x + offset, vals, width, label=name, color=colour, alpha=0.8)

        for bar in bars:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=6,
                )

    ax.axhline(0.5, color="grey", linestyle="--", lw=1, alpha=0.7,
               label="Random (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [k.replace("_", " ").title() for k in display_keys], fontsize=9,
    )
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim([0, 1.15])
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ── Overlaid ROC Curves ──────────────────────────────────────────────────


def plot_attack_roc_overlay(
    results: dict[str, AttackResult],
    save_path: str | Path | None = None,
    title: str = "ROC Curves — All Attacks",
) -> plt.Figure:
    """Overlay ROC curves from multiple attacks on a single plot.

    Parameters
    ----------
    results:
        Mapping from attack name to ``AttackResult``.
    save_path:
        If given, saves the figure.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(result.ground_truth, result.confidence_scores)
        roc_auc = auc(fpr, tpr)
        colour = _colour_for(name, i)
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f"{name} (AUC = {roc_auc:.4f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--",
            label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
