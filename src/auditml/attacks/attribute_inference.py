"""Attribute Inference attack.

Given a trained model's output, try to infer a **sensitive attribute**
(e.g. a coarse superclass label) that the model does not explicitly
predict.  The key insight: if the model has memorised training data, its
softmax probability vector leaks information about attributes of the
samples it was trained on.

**Workflow**:

1. Extract the target model's softmax outputs for all samples.
2. Map each sample's class label to a *sensitive attribute* — a
   higher-level grouping such as CIFAR-100's 20 superclasses, or a
   synthetic grouping for MNIST / CIFAR-10.
3. Train a small attack MLP: ``softmax_output → sensitive_attribute``.
   Training uses the **member** (training) data, because the attacker
   typically has access to some labelled examples.
4. Evaluate on both members and non-members.  Members tend to have
   *higher* attribute-prediction confidence because the target model's
   outputs are more informative for data it has seen before.
5. This confidence gap is the privacy-leakage signal.

References
----------
Fredrikson et al., "Privacy in Pharmacogenetics: An End-to-End Case
Study of Personalized Warfarin Dosing", USENIX Security 2014.

Yeom et al., "Privacy Risk in Machine Learning: Analyzing the
Connection to Overfitting", CSF 2018.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from auditml.attacks.base import BaseAttack
from auditml.attacks.results import AttackResult
from auditml.config.schema import AuditMLConfig

logger = logging.getLogger(__name__)


# ── Attack MLP (multi-class) ─────────────────────────────────────────────


class AttributeAttackMLP(nn.Module):
    """Small multi-class classifier: model outputs → sensitive attribute.

    Architecture: input → 64 → 32 → num_groups.  Two hidden layers with
    ReLU and dropout — deliberately simple to avoid overfitting.

    Parameters
    ----------
    input_dim:
        Size of the input feature vector (= ``num_classes`` of the
        target model, i.e. the softmax probability vector length).
    num_groups:
        Number of sensitive-attribute categories to predict.
    hidden_dim:
        Width of the first hidden layer (second is ``hidden_dim // 2``).
    """

    def __init__(
        self,
        input_dim: int,
        num_groups: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(N, num_groups)``."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-class probabilities of shape ``(N, num_groups)``."""
        with torch.no_grad():
            return F.softmax(self.net(x), dim=1)


# ── Default class → group mappings ───────────────────────────────────────

# Sensible semantic groupings for the three supported datasets.
# CIFAR-100 uses modular arithmetic (100 classes → 20 groups of 5).

_DEFAULT_GROUPS: dict[str, dict[int, int] | None] = {
    "mnist": {
        # 10 digits → 5 groups (pairs of visually similar digits)
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3, 7: 3,
        8: 4, 9: 4,
    },
    "cifar10": {
        # 10 classes → 5 semantic groups
        # 0=airplane, 1=automobile → transport
        # 2=bird, 3=cat → small animals
        # 4=deer, 5=dog → medium animals
        # 6=frog, 7=horse → other animals
        # 8=ship, 9=truck → large vehicles
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3, 7: 3,
        8: 4, 9: 4,
    },
    "cifar100": None,  # auto-generate: 100 classes → 20 groups of 5
}


# ── Attribute Inference Attack ────────────────────────────────────────────


class AttributeInference(BaseAttack):
    """Attribute inference attack via model output analysis.

    For each sample the attacker observes the target model's softmax
    probability vector and tries to predict a sensitive attribute that
    the model was *not* designed to reveal.  The attack trains a small
    MLP on the member (training) data, then evaluates whether members'
    attributes are more predictable than non-members'.

    Parameters
    ----------
    target_model:
        The trained model being audited.
    config:
        Full AuditML configuration.
    device:
        Torch device.
    num_groups:
        Override for the number of sensitive-attribute groups.
        If ``None``, determined automatically from the dataset.
    class_to_group:
        Explicit mapping ``{class_label: group_id}``. If ``None``,
        a default mapping is used based on the dataset.
    """

    attack_name = "attribute_inference"

    def __init__(
        self,
        target_model: nn.Module,
        config: AuditMLConfig,
        device: torch.device | str = "cpu",
        num_groups: int | None = None,
        class_to_group: dict[int, int] | None = None,
    ) -> None:
        super().__init__(target_model, config, device)

        params = config.attack_params.attribute_inference
        self.sensitive_attribute = params.sensitive_attribute
        self.attack_model_type = params.attack_model
        self.num_classes = config.model.num_classes
        dataset_name = config.data.dataset.value

        # Build the class → group mapping
        if class_to_group is not None:
            self.class_to_group = class_to_group
            self.num_groups = len(set(class_to_group.values()))
        elif num_groups is not None:
            self.num_groups = num_groups
            self.class_to_group = {
                c: c % num_groups for c in range(self.num_classes)
            }
        elif dataset_name in _DEFAULT_GROUPS and _DEFAULT_GROUPS[dataset_name] is not None:
            self.class_to_group = _DEFAULT_GROUPS[dataset_name]
            self.num_groups = len(set(self.class_to_group.values()))
        else:
            # Fallback: auto-generate with ≈5 classes per group
            self.num_groups = max(2, self.num_classes // 5)
            self.class_to_group = {
                c: c % self.num_groups for c in range(self.num_classes)
            }

        # Populated during run()
        self.attack_model: AttributeAttackMLP | None = None
        self.member_labels: np.ndarray | None = None
        self.nonmember_labels: np.ndarray | None = None
        self.member_confidence: np.ndarray | None = None
        self.nonmember_confidence: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Main attack logic
    # ------------------------------------------------------------------

    def run(
        self,
        member_loader: DataLoader,
        nonmember_loader: DataLoader,
    ) -> AttackResult:
        """Execute the attribute inference attack.

        Steps:

        1. Extract the target model's softmax outputs for all samples.
        2. Map class labels → sensitive attribute (group labels).
        3. Train an attack MLP on *member* outputs → group.
        4. Score every sample by the confidence of the correct group
           prediction.  Members should score higher.

        Parameters
        ----------
        member_loader:
            DataLoader over training (member) samples.
        nonmember_loader:
            DataLoader over non-member samples.

        Returns
        -------
        AttackResult
        """
        # Step 1: Extract model outputs
        member_probs, _, member_true_labels = self.get_model_outputs(member_loader)
        nonmember_probs, _, nonmember_true_labels = self.get_model_outputs(nonmember_loader)

        self.member_labels = member_true_labels
        self.nonmember_labels = nonmember_true_labels

        # Step 2: Sensitive attribute labels (group assignments)
        member_groups = self._labels_to_groups(member_true_labels)
        nonmember_groups = self._labels_to_groups(nonmember_true_labels)

        logger.info(
            "Training attribute attack model: %d groups, %d member samples",
            self.num_groups, len(member_probs),
        )

        # Step 3: Train attack model on member data
        self.attack_model = self._train_attack_model(member_probs, member_groups)

        # Step 4: Predict attribute confidence for both sets
        member_attr_conf = self._predict_attribute_confidence(
            member_probs, member_groups,
        )
        nonmember_attr_conf = self._predict_attribute_confidence(
            nonmember_probs, nonmember_groups,
        )

        self.member_confidence = member_attr_conf
        self.nonmember_confidence = nonmember_attr_conf

        logger.info(
            "Attribute confidence — members: %.4f, non-members: %.4f",
            float(member_attr_conf.mean()), float(nonmember_attr_conf.mean()),
        )

        # Step 5: Build membership inference signal
        ground_truth = np.concatenate([
            np.ones(len(member_attr_conf)),
            np.zeros(len(nonmember_attr_conf)),
        ])
        all_scores = np.concatenate([member_attr_conf, nonmember_attr_conf])

        # Threshold at median for binary predictions
        threshold = float(np.median(all_scores))
        predictions = (all_scores >= threshold).astype(np.int32)

        self.result = AttackResult(
            predictions=predictions,
            ground_truth=ground_truth,
            confidence_scores=all_scores,
            attack_name=self.attack_name,
            metadata={
                "num_groups": self.num_groups,
                "sensitive_attribute": self.sensitive_attribute,
                "mean_member_attr_confidence": float(member_attr_conf.mean()),
                "mean_nonmember_attr_confidence": float(nonmember_attr_conf.mean()),
                # Store probs for per-group evaluation (prefixed with _ to
                # exclude from summary text)
                "_member_probs": member_probs,
                "_nonmember_probs": nonmember_probs,
            },
        )
        return self.result

    # ------------------------------------------------------------------
    # Label → group mapping
    # ------------------------------------------------------------------

    def _labels_to_groups(self, labels: np.ndarray) -> np.ndarray:
        """Convert class labels to sensitive-attribute group IDs.

        Parameters
        ----------
        labels:
            Integer class labels, shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Integer group IDs, shape ``(N,)``.
        """
        return np.array([
            self.class_to_group.get(int(c), 0) for c in labels
        ])

    # ------------------------------------------------------------------
    # Attack model training
    # ------------------------------------------------------------------

    def _train_attack_model(
        self,
        probs: np.ndarray,
        groups: np.ndarray,
        epochs: int = 50,
        lr: float = 0.001,
    ) -> AttributeAttackMLP:
        """Train an MLP to predict the sensitive attribute.

        Parameters
        ----------
        probs:
            Softmax outputs from the target model, shape ``(N, C)``.
        groups:
            Sensitive attribute labels, shape ``(N,)``.
        epochs:
            Number of training epochs.
        lr:
            Learning rate.

        Returns
        -------
        AttributeAttackMLP
            The trained attack model (in eval mode).
        """
        model = AttributeAttackMLP(
            input_dim=probs.shape[1],
            num_groups=self.num_groups,
        )
        model.to(self.device)

        x = torch.tensor(probs, dtype=torch.float32)
        y = torch.tensor(groups, dtype=torch.long)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 25 == 0:
                logger.debug(
                    "Attack model epoch %d/%d — loss: %.4f",
                    epoch + 1, epochs, total_loss / len(loader),
                )

        model.eval()
        return model

    # ------------------------------------------------------------------
    # Attribute prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_attribute_confidence(
        self,
        probs: np.ndarray,
        true_groups: np.ndarray,
    ) -> np.ndarray:
        """Measure how confidently the attack model predicts each sample's attribute.

        For each sample, returns the softmax probability that the attack
        model assigns to the *correct* group.  Higher confidence means
        the model's output is more informative about the sensitive
        attribute — a sign that the sample was in the training data.

        Parameters
        ----------
        probs:
            Softmax outputs from the target model, shape ``(N, C)``.
        true_groups:
            Ground-truth group labels, shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — confidence in the correct group for each
            sample.
        """
        self.attack_model.eval()
        x = torch.tensor(probs, dtype=torch.float32).to(self.device)
        logits = self.attack_model(x)
        pred_probs = F.softmax(logits, dim=1).cpu().numpy()

        # Confidence = probability assigned to the correct group
        confidences = pred_probs[np.arange(len(true_groups)), true_groups]
        return confidences

    # ------------------------------------------------------------------
    # Attribute prediction accuracy
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_attributes(self, probs: np.ndarray) -> np.ndarray:
        """Predict the sensitive attribute for each sample.

        Parameters
        ----------
        probs:
            Softmax outputs from the target model, shape ``(N, C)``.

        Returns
        -------
        np.ndarray
            Predicted group IDs, shape ``(N,)``.
        """
        if self.attack_model is None:
            raise RuntimeError("Call run() before predict_attributes().")

        self.attack_model.eval()
        x = torch.tensor(probs, dtype=torch.float32).to(self.device)
        logits = self.attack_model(x)
        return logits.argmax(dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    # Per-class evaluation (membership inference per original class)
    # ------------------------------------------------------------------

    def evaluate_per_class(self) -> dict[int, dict[str, float]]:
        """Compute evaluation metrics separately for each original class.

        Groups all samples by their original class label and computes the
        full metric suite for each class.  This reveals which classes are
        most vulnerable to the attribute inference attack.

        Returns
        -------
        dict[int, dict[str, float]]
            Mapping from class label to metric dictionary.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called yet.
        """
        if self.result is None:
            raise RuntimeError("Call run() before evaluate_per_class().")

        all_labels = np.concatenate([self.member_labels, self.nonmember_labels])
        unique_classes = np.unique(all_labels)

        per_class: dict[int, dict[str, float]] = {}
        for cls in unique_classes:
            mask = all_labels == cls
            preds_cls = self.result.predictions[mask]
            gt_cls = self.result.ground_truth[mask]
            scores_cls = self.result.confidence_scores[mask]

            if len(gt_cls) < 2 or len(np.unique(gt_cls)) < 2:
                per_class[int(cls)] = {
                    "accuracy": float(np.mean(preds_cls == gt_cls)) if len(gt_cls) > 0 else 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "auc_roc": 0.0,
                    "auc_pr": 0.0,
                    "tpr_at_1fpr": 0.0,
                    "tpr_at_01fpr": 0.0,
                    "n_samples": int(mask.sum()),
                }
                continue

            metrics = self._compute_metrics(preds_cls, gt_cls, scores_cls)
            metrics["n_samples"] = int(mask.sum())
            per_class[int(cls)] = metrics

        return per_class

    # ------------------------------------------------------------------
    # Per-group evaluation (attribute prediction accuracy per group)
    # ------------------------------------------------------------------

    def evaluate_per_group(self) -> dict[str, dict[int, float]]:
        """Compute attribute prediction accuracy for each group.

        Returns separate accuracy dictionaries for members and
        non-members.  A gap between the two signals privacy leakage.

        Returns
        -------
        dict with keys ``"member"`` and ``"nonmember"``, each mapping
        group ID to prediction accuracy on that group.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called yet.
        """
        if self.result is None or self.attack_model is None:
            raise RuntimeError("Call run() before evaluate_per_group().")

        member_groups = self._labels_to_groups(self.member_labels)
        nonmember_groups = self._labels_to_groups(self.nonmember_labels)

        member_preds = self.predict_attributes(
            self._get_stored_probs("member"),
        )
        nonmember_preds = self.predict_attributes(
            self._get_stored_probs("nonmember"),
        )

        member_acc: dict[int, float] = {}
        for g in range(self.num_groups):
            mask = member_groups == g
            if mask.sum() > 0:
                member_acc[g] = float((member_preds[mask] == member_groups[mask]).mean())

        nonmember_acc: dict[int, float] = {}
        for g in range(self.num_groups):
            mask = nonmember_groups == g
            if mask.sum() > 0:
                nonmember_acc[g] = float((nonmember_preds[mask] == nonmember_groups[mask]).mean())

        return {"member": member_acc, "nonmember": nonmember_acc}

    def _get_stored_probs(self, split: str) -> np.ndarray:
        """Re-extract softmax probs from stored confidence data.

        Since we store per-sample confidence (scalar), but need the full
        probability vector for ``predict_attributes``, we store them
        during ``run()``.
        """
        # We need to store the probs during run() — add them to metadata
        key = f"_{split}_probs"
        if key not in self.result.metadata:
            raise RuntimeError(
                f"Probabilities not stored. Ensure run() was called."
            )
        return self.result.metadata[key]

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_dir: str | Path) -> Path:
        """Generate a complete evaluation report with metrics and plots.

        Creates the following files in *output_dir*:

        - ``metrics.json`` — overall evaluation metrics
        - ``per_class_metrics.json`` — per-class breakdown
        - ``per_group_accuracy.json`` — per-group attribute accuracy
        - ``roc_curve.png`` — ROC curve plot
        - ``confidence_distributions.png`` — member vs non-member histogram
        - ``per_class_accuracy.png`` — bar chart of per-class accuracy
        - ``attribute_accuracy.png`` — per-group member vs non-member accuracy
        - ``summary.txt`` — human-readable text summary

        Parameters
        ----------
        output_dir:
            Directory where all report files are saved.

        Returns
        -------
        Path
            The output directory.
        """
        if self.result is None:
            raise RuntimeError("Call run() before generate_report().")

        from auditml.attacks.visualization import (
            plot_attribute_accuracy,
            plot_per_class_metrics,
            plot_roc_curve,
            plot_score_distributions,
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Overall metrics
        metrics = self.evaluate()
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 2. Per-class metrics
        per_class = self.evaluate_per_class()
        per_class_str = {str(k): v for k, v in per_class.items()}
        with open(out / "per_class_metrics.json", "w") as f:
            json.dump(per_class_str, f, indent=2)

        # 3. Per-group attribute accuracy
        per_group = self.evaluate_per_group()
        with open(out / "per_group_accuracy.json", "w") as f:
            json.dump({k: {str(g): v for g, v in d.items()} for k, d in per_group.items()}, f, indent=2)

        # 4. ROC curve
        plot_roc_curve(
            ground_truth=self.result.ground_truth,
            confidence_scores=self.result.confidence_scores,
            title="ROC Curve — Attribute Inference Attack",
            save_path=out / "roc_curve.png",
        )

        # 5. Confidence distribution histogram
        plot_score_distributions(
            member_scores=self.member_confidence,
            nonmember_scores=self.nonmember_confidence,
            metric_name="attribute confidence",
            save_path=out / "confidence_distributions.png",
            title="Attribute Confidence Distribution — Members vs Non-Members",
        )

        # 6. Per-class accuracy bar chart
        plot_per_class_metrics(
            per_class_metrics=per_class,
            save_path=out / "per_class_accuracy.png",
        )

        # 7. Per-group attribute accuracy comparison
        plot_attribute_accuracy(
            member_accuracy=per_group["member"],
            nonmember_accuracy=per_group["nonmember"],
            save_path=out / "attribute_accuracy.png",
        )

        # 8. Summary text
        self._write_summary(out / "summary.txt", metrics, per_class, per_group)

        return out

    def _write_summary(
        self,
        path: Path,
        metrics: dict[str, float],
        per_class: dict[int, dict[str, float]],
        per_group: dict[str, dict[int, float]],
    ) -> None:
        """Write a human-readable text summary of the attack results."""
        lines = [
            "=" * 60,
            "AuditML — Attribute Inference Attack Report",
            "=" * 60,
            "",
            f"Sensitive attribute: {self.sensitive_attribute}",
            f"Number of groups:    {self.num_groups}",
            f"Total samples:       {len(self.result.predictions)}",
            f"  Members:           {int(self.result.ground_truth.sum())}",
            f"  Non-members:       {int((1 - self.result.ground_truth).sum())}",
            "",
            "--- Overall Metrics ---",
        ]
        for key, val in metrics.items():
            lines.append(f"  {key:<20s}: {val:.4f}")

        lines.append("")
        lines.append("--- Per-Group Attribute Accuracy ---")
        for g in sorted(set(per_group.get("member", {}).keys()) | set(per_group.get("nonmember", {}).keys())):
            mem_acc = per_group.get("member", {}).get(g, 0.0)
            nonmem_acc = per_group.get("nonmember", {}).get(g, 0.0)
            gap = mem_acc - nonmem_acc
            lines.append(
                f"  Group {g:>3d}:  member={mem_acc:.3f}  "
                f"non-member={nonmem_acc:.3f}  gap={gap:+.3f}"
            )

        lines.append("")
        lines.append("--- Per-Class Membership Accuracy ---")
        for cls in sorted(per_class.keys()):
            m = per_class[cls]
            lines.append(
                f"  Class {cls:>3d}:  acc={m['accuracy']:.3f}  "
                f"auc={m['auc_roc']:.3f}  n={m['n_samples']}"
            )

        lines.append("")
        lines.append("--- Metadata ---")
        for key, val in self.result.metadata.items():
            if not key.startswith("_"):
                lines.append(f"  {key}: {val}")

        lines.append("")
        path.write_text("\n".join(lines))
