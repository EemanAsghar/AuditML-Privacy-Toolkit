"""Shadow-model Membership Inference Attack.

This is the classic MIA from Shokri et al., "Membership Inference Attacks
Against Machine Learning Models" (IEEE S&P 2017).

**Key idea**: Train multiple *shadow models* that behave like the target
model. Use them to generate a labelled dataset of (model_output, is_member)
pairs.  Then train a small neural network ("attack model") on that dataset
to classify whether a given sample was in the target's training set.

This is more powerful than the threshold approach (Task 2.2) because the
attack model can exploit the **full output probability vector**, capturing
subtle multi-dimensional patterns that a single threshold on loss cannot.

References
----------
Shokri et al., "Membership Inference Attacks Against Machine Learning
Models", IEEE S&P 2017.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from auditml.attacks.base import BaseAttack
from auditml.attacks.results import AttackResult
from auditml.config.schema import AuditMLConfig
from auditml.data.datasets import get_shadow_data_splits
from auditml.models import get_model
from auditml.training.trainer import Trainer, build_optimizer

logger = logging.getLogger(__name__)


# ── Attack MLP ───────────────────────────────────────────────────────────


class AttackMLP(nn.Module):
    """Small binary classifier that predicts membership from model outputs.

    Architecture: input → 64 → 32 → 1 (sigmoid). Two hidden layers with
    ReLU and dropout.  Input dimension = ``num_classes`` of the target
    model (the softmax probability vector).

    This is intentionally simple — we don't want the attack model to be
    more complex than necessary, as that could lead to overfitting on the
    shadow data and poor transfer to the target model.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (shape ``(N, 1)``)."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of being a member (shape ``(N,)``)."""
        with torch.no_grad():
            return torch.sigmoid(self.net(x)).squeeze(-1)


# ── Shadow Model MIA ─────────────────────────────────────────────────────


class ShadowMIA(BaseAttack):
    """Shadow-model Membership Inference Attack.

    Workflow executed by ``run()``:

    1. **Train shadow models** — each on a different random split of the
       same dataset distribution. The number and epochs come from
       ``config.attack_params.mia_shadow``.
    2. **Collect attack data** — for each shadow model, gather its softmax
       outputs on its own members (label 1) and non-members (label 0).
    3. **Train attack MLP** — a small binary classifier on the collected
       (probability_vector, membership_label) dataset.
    4. **Attack the target** — run the target model on the supplied member
       and non-member loaders, then classify each sample with the trained
       attack model.

    Parameters
    ----------
    target_model:
        The trained model being audited.
    config:
        Full AuditML configuration.
    device:
        Torch device.
    shadow_dataset:
        The dataset from which shadow model training data is drawn.
        This should be the **same distribution** as the target's training
        data (e.g. the full CIFAR-10 training set). If ``None``, shadow
        models must be provided manually via ``shadow_models``.
    shadow_models:
        Pre-trained shadow models. If provided, skips the training step.
        Each entry is ``(model, member_loader, nonmember_loader)``.
    """

    attack_name = "mia_shadow"

    def __init__(
        self,
        target_model: nn.Module,
        config: AuditMLConfig,
        device: torch.device | str = "cpu",
        shadow_dataset: Dataset | None = None,
        shadow_models: list[tuple[nn.Module, DataLoader, DataLoader]] | None = None,
    ) -> None:
        super().__init__(target_model, config, device)
        self.shadow_dataset = shadow_dataset
        self.shadow_models = shadow_models

        # Config shortcuts
        params = config.attack_params.mia_shadow
        self.num_shadows = params.num_shadow_models
        self.shadow_epochs = params.shadow_epochs
        self.num_classes = config.model.num_classes

        # Will be populated during run()
        self.attack_model: AttackMLP | None = None
        self.trained_shadows: list[nn.Module] = []

    # ------------------------------------------------------------------
    # Main attack logic
    # ------------------------------------------------------------------

    def run(
        self,
        member_loader: DataLoader,
        nonmember_loader: DataLoader,
    ) -> AttackResult:
        """Execute the full shadow-model MIA pipeline.

        Steps:
            1. Train shadow models (or use pre-trained ones)
            2. Collect (output, membership) pairs from all shadows
            3. Train the attack MLP on shadow data
            4. Use the attack MLP to classify target model outputs
        """
        # Step 1: Get shadow models with their data
        shadow_data = self._get_shadow_data()

        # Step 2: Collect attack training data from shadow models
        attack_features, attack_labels = self._collect_attack_data(shadow_data)
        logger.info(
            "Collected %d attack training samples (%d members, %d non-members)",
            len(attack_labels),
            int(attack_labels.sum()),
            int((1 - attack_labels).sum()),
        )

        # Step 3: Train the attack model
        self.attack_model = self._train_attack_model(attack_features, attack_labels)

        # Step 4: Attack the target model
        member_probs, _, _ = self.get_model_outputs(member_loader)
        nonmember_probs, _, _ = self.get_model_outputs(nonmember_loader)

        # Build ground truth: 1 = member, 0 = non-member
        ground_truth = np.concatenate([
            np.ones(len(member_probs)),
            np.zeros(len(nonmember_probs)),
        ])

        # Get attack model predictions
        all_probs = np.concatenate([member_probs, nonmember_probs])
        confidence_scores = self._attack_predict(all_probs)
        predictions = (confidence_scores >= 0.5).astype(np.int32)

        self.result = AttackResult(
            predictions=predictions,
            ground_truth=ground_truth,
            confidence_scores=confidence_scores,
            attack_name=self.attack_name,
            metadata={
                "num_shadow_models": self.num_shadows,
                "shadow_epochs": self.shadow_epochs,
                "attack_train_samples": len(attack_labels),
                "member_mean_confidence": float(confidence_scores[:len(member_probs)].mean()),
                "nonmember_mean_confidence": float(confidence_scores[len(member_probs):].mean()),
            },
        )
        return self.result

    # ------------------------------------------------------------------
    # Step 1: Shadow model training
    # ------------------------------------------------------------------

    def _get_shadow_data(
        self,
    ) -> list[tuple[nn.Module, DataLoader, DataLoader]]:
        """Return shadow models with their member/non-member loaders.

        If ``shadow_models`` were passed at init, use them directly.
        Otherwise, train new shadow models from ``shadow_dataset``.
        """
        if self.shadow_models is not None:
            return self.shadow_models

        if self.shadow_dataset is None:
            raise ValueError(
                "Either shadow_dataset or shadow_models must be provided. "
                "Pass the full training dataset as shadow_dataset so we can "
                "create independent splits for shadow model training."
            )

        return self._train_shadow_models()

    def _train_shadow_models(
        self,
    ) -> list[tuple[nn.Module, DataLoader, DataLoader]]:
        """Train shadow models from scratch.

        Each shadow model gets its own random member/non-member split of
        ``shadow_dataset``. This mirrors how the target model was trained,
        so the shadow models learn similar decision boundaries.
        """
        batch_size = self.config.training.batch_size
        splits = get_shadow_data_splits(
            self.shadow_dataset,
            n_shadows=self.num_shadows,
            member_ratio=self.config.data.train_ratio,
            seed=self.config.training.seed,
        )

        results: list[tuple[nn.Module, DataLoader, DataLoader]] = []

        for i, (member_set, nonmember_set, _, _) in enumerate(splits):
            logger.info("Training shadow model %d/%d ...", i + 1, self.num_shadows)

            # Create a fresh model with the same architecture as the target
            shadow = get_model(
                arch=self.config.model.arch,
                dataset=self.config.data.dataset.value,
            ).to(self.device)

            # Create data loaders
            train_loader = DataLoader(
                member_set, batch_size=batch_size, shuffle=True,
            )
            val_loader = DataLoader(
                nonmember_set, batch_size=batch_size, shuffle=False,
            )

            # Train the shadow model
            optimizer = build_optimizer(
                shadow,
                name=self.config.training.optimizer,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
            trainer = Trainer(
                model=shadow,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=self.device,
            )
            trainer.train(epochs=self.shadow_epochs, patience=0)

            shadow.eval()
            self.trained_shadows.append(shadow)

            # Create evaluation loaders (no shuffle, for consistent ordering)
            member_eval = DataLoader(
                member_set, batch_size=batch_size, shuffle=False,
            )
            nonmember_eval = DataLoader(
                nonmember_set, batch_size=batch_size, shuffle=False,
            )
            results.append((shadow, member_eval, nonmember_eval))

        return results

    # ------------------------------------------------------------------
    # Step 2: Collect attack training data
    # ------------------------------------------------------------------

    def _collect_attack_data(
        self,
        shadow_data: list[tuple[nn.Module, DataLoader, DataLoader]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gather (softmax_output, membership_label) from all shadows.

        For each shadow model:
        - Run it on its member data → label these outputs as 1 (member)
        - Run it on its non-member data → label these outputs as 0

        Returns
        -------
        (features, labels)
            features: shape ``(total_samples, num_classes)``
            labels: shape ``(total_samples,)`` — 0 or 1
        """
        all_features: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for shadow_model, member_loader, nonmember_loader in shadow_data:
            # Temporarily swap target_model to extract outputs from shadow
            original_model = self.target_model
            self.target_model = shadow_model
            self.target_model.eval()

            member_probs, _, _ = self.get_model_outputs(member_loader)
            nonmember_probs, _, _ = self.get_model_outputs(nonmember_loader)

            # Restore the real target
            self.target_model = original_model

            all_features.append(member_probs)
            all_features.append(nonmember_probs)
            all_labels.append(np.ones(len(member_probs)))
            all_labels.append(np.zeros(len(nonmember_probs)))

        return np.concatenate(all_features), np.concatenate(all_labels)

    # ------------------------------------------------------------------
    # Step 3: Train attack classifier
    # ------------------------------------------------------------------

    def _train_attack_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        lr: float = 0.001,
    ) -> AttackMLP:
        """Train a binary MLP to predict membership from softmax outputs.

        Parameters
        ----------
        features:
            Shape ``(N, num_classes)`` — softmax probability vectors.
        labels:
            Shape ``(N,)`` — 1 for member, 0 for non-member.
        epochs:
            Training epochs for the attack model.
        lr:
            Learning rate.

        Returns
        -------
        AttackMLP
            The trained attack classifier.
        """
        input_dim = features.shape[1]
        model = AttackMLP(input_dim=input_dim).to(self.device)

        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = model(batch_x).squeeze(-1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        logger.info("Attack model trained for %d epochs (final loss: %.4f)",
                     epochs, total_loss / max(len(loader), 1))
        return model

    # ------------------------------------------------------------------
    # Step 4: Predict with attack model
    # ------------------------------------------------------------------

    def _attack_predict(self, probs: np.ndarray) -> np.ndarray:
        """Use the trained attack MLP to predict membership probability.

        Parameters
        ----------
        probs:
            Shape ``(N, num_classes)`` — softmax outputs from the target.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — probability of being a member, in [0, 1].
        """
        self.attack_model.eval()
        x = torch.tensor(probs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            scores = self.attack_model.predict_proba(x)
        return scores.cpu().numpy()
