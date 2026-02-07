"""Standardised result container for all AuditML attacks.

Every attack's ``run()`` method returns an ``AttackResult``. This ensures
the CLI, report generator, and comparison tools can process results from
any attack type without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AttackResult:
    """Container for the outputs of a single attack run.

    Attributes
    ----------
    predictions:
        Binary array — the attack's guess for each sample.
        For membership inference: 1 = predicted member, 0 = predicted
        non-member. Length equals ``len(ground_truth)``.
    ground_truth:
        Binary array — the true label for each sample.
        For membership inference: 1 = actual member, 0 = actual
        non-member.
    confidence_scores:
        Continuous score per sample indicating how confident the attack
        is.  Higher = more confident the sample is a member (for MIA)
        or more confident in the predicted attribute (for attribute
        inference). Used for ROC curves and threshold-independent
        evaluation.
    attack_name:
        Human-readable name, e.g. ``"mia_threshold"`` or
        ``"model_inversion"``.
    metadata:
        Free-form dict for attack-specific extras (e.g. reconstructed
        images for model inversion, per-class breakdowns, etc.).
    """

    predictions: np.ndarray
    ground_truth: np.ndarray
    confidence_scores: np.ndarray
    attack_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array lengths match."""
        n = len(self.ground_truth)
        if len(self.predictions) != n:
            raise ValueError(
                f"predictions length ({len(self.predictions)}) != "
                f"ground_truth length ({n})"
            )
        if len(self.confidence_scores) != n:
            raise ValueError(
                f"confidence_scores length ({len(self.confidence_scores)}) != "
                f"ground_truth length ({n})"
            )
