"""Differentially-private training loop using Opacus.

Extends the standard ``Trainer`` with Opacus's ``PrivacyEngine`` so that
models can be trained with formal (epsilon, delta)-differential privacy
guarantees.  The key changes compared to standard training:

1. **Gradient clipping** — each sample's gradient is clipped to a fixed
   norm (``max_grad_norm``).
2. **Noise injection** — calibrated Gaussian noise is added to the
   clipped gradients before the optimiser step.
3. **Privacy accounting** — Opacus tracks the cumulative privacy budget
   (epsilon) after every batch.

Usage
-----
>>> dp_trainer = DPTrainer(model, train_loader, val_loader, optimizer,
...                        config=cfg)
>>> dp_trainer.train(epochs=20)
>>> print(f"Final epsilon: {dp_trainer.epsilon_history[-1]:.2f}")

Important
---------
- The model **must** be Opacus-compatible (no ``BatchNorm``, no
  ``Dropout2d`` in older Opacus). Use ``validate_and_fix_model()`` or
  ``ModuleValidator.fix()`` before passing the model.
- ``PrivacyEngine.make_private()`` replaces the model, optimizer, and
  data-loader in-place, so the trainer keeps references to the wrapped
  versions.

References
----------
Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from auditml.config.schema import AuditMLConfig, DPConfig
from auditml.training.trainer import Trainer

logger = logging.getLogger(__name__)


# ── Model compatibility helpers ──────────────────────────────────────────


def validate_and_fix_model(model: nn.Module) -> nn.Module:
    """Make a model compatible with Opacus DP training.

    Opacus requires models to pass ``ModuleValidator.validate()``.
    Common incompatible layers include ``BatchNorm`` (replaced with
    ``GroupNorm``) and ``Dropout2d`` (replaced with ``Dropout``).

    Parameters
    ----------
    model:
        The model to validate and fix.

    Returns
    -------
    nn.Module
        A fixed copy of the model (original is not modified).
    """
    from opacus.validators import ModuleValidator

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.info(
            "Model has %d Opacus-incompatible layer(s) — fixing automatically.",
            len(errors),
        )
        model = ModuleValidator.fix(model)
    return model


def is_dp_compatible(model: nn.Module) -> bool:
    """Check whether a model is compatible with Opacus.

    Returns
    -------
    bool
        ``True`` if no incompatible layers are found.
    """
    from opacus.validators import ModuleValidator

    errors = ModuleValidator.validate(model, strict=False)
    return len(errors) == 0


# ── DP Trainer ───────────────────────────────────────────────────────────


class DPTrainer(Trainer):
    """Differentially-private training loop powered by Opacus.

    Wraps the standard ``Trainer`` with Opacus's ``PrivacyEngine``.
    After ``make_private()`` is called the training loop automatically
    clips per-sample gradients and injects calibrated noise.

    Parameters
    ----------
    model:
        The model to train (must be Opacus-compatible — call
        ``validate_and_fix_model`` first if needed).
    train_loader:
        Training data loader.
    val_loader:
        Validation data loader.
    optimizer:
        PyTorch optimizer (will be wrapped by Opacus).
    dp_config:
        Differential privacy parameters.
    criterion:
        Loss function.
    device:
        Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        dp_config: DPConfig,
        criterion: nn.Module | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=None,  # Opacus handles clipping
        )
        self.dp_config = dp_config
        self.privacy_engine = None
        self.epsilon_history: list[float] = []
        self._is_private = False

    def make_private(self) -> None:
        """Attach the Opacus ``PrivacyEngine`` to model/optimizer/loader.

        After this call the training loop will enforce DP guarantees.
        This method must be called **before** ``train()``.

        The method uses either ``noise_multiplier`` (if set in config)
        or calibrates noise from ``(epsilon, delta, epochs)``.
        """
        from opacus import PrivacyEngine

        self.privacy_engine = PrivacyEngine()

        kwargs: dict[str, Any] = {
            "max_grad_norm": self.dp_config.max_grad_norm,
        }

        if self.dp_config.noise_multiplier is not None:
            kwargs["noise_multiplier"] = self.dp_config.noise_multiplier
        else:
            # Let Opacus calibrate noise from target epsilon/delta
            kwargs["noise_multiplier"] = self._calibrate_noise()

        (
            self.model,
            self.optimizer,
            self.train_loader,
        ) = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            **kwargs,
        )

        self._is_private = True
        logger.info(
            "DP training enabled — max_grad_norm=%.2f, noise_multiplier=%.4f",
            self.dp_config.max_grad_norm,
            kwargs["noise_multiplier"],
        )

    def _calibrate_noise(self) -> float:
        """Estimate a noise multiplier from the target epsilon.

        Uses a simple heuristic: higher epsilon → less noise.
        For precise calibration Opacus's ``get_noise_multiplier`` can be
        used, but it requires knowing the number of steps in advance.
        """
        from opacus.accountants.utils import get_noise_multiplier

        # Estimate total steps (assume 20 epochs if unknown)
        steps_per_epoch = len(self.train_loader)
        sample_rate = 1.0 / steps_per_epoch if steps_per_epoch > 0 else 0.01

        noise = get_noise_multiplier(
            target_epsilon=self.dp_config.epsilon,
            target_delta=self.dp_config.delta,
            sample_rate=sample_rate,
            epochs=20,  # default planning horizon
        )
        logger.info(
            "Calibrated noise_multiplier=%.4f for epsilon=%.2f, delta=%.1e",
            noise, self.dp_config.epsilon, self.dp_config.delta,
        )
        return noise

    def train(
        self,
        epochs: int = 20,
        patience: int = 10,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Run the DP training loop.

        If ``make_private()`` has not been called yet, it is called
        automatically before training begins.

        Returns the training history dict (same as ``Trainer.train()``)
        with an additional ``"epsilon"`` key.
        """
        if not self._is_private:
            self.make_private()

        # Add epsilon tracking to history
        self.history["epsilon"] = []

        result = super().train(
            epochs=epochs,
            patience=patience,
            checkpoint_dir=checkpoint_dir,
        )

        return result

    def _train_epoch(self, epoch: int, total_epochs: int) -> tuple[float, float]:
        """Train one epoch with DP noise.

        Identical to the parent except we record epsilon after each epoch.
        Opacus handles the gradient clipping and noise internally via the
        wrapped optimizer.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{total_epochs} (DP)",
            leave=False,
        )
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(1) == targets).sum().item()
            total += batch_size

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Record privacy budget after this epoch
        if self.privacy_engine is not None:
            eps = self.privacy_engine.get_epsilon(self.dp_config.delta)
            self.epsilon_history.append(eps)
            self.history["epsilon"].append(eps)
            tqdm.write(f"  DP epsilon after epoch {epoch}: {eps:.2f}")

        return total_loss / max(total, 1), correct / max(total, 1)

    def get_epsilon(self) -> float:
        """Return the current cumulative privacy budget (epsilon).

        Returns
        -------
        float
            The epsilon spent so far. Returns 0.0 if training
            has not started or no steps have been taken.
        """
        if self.epsilon_history:
            return self.epsilon_history[-1]
        if self.privacy_engine is not None and self._is_private:
            try:
                return self.privacy_engine.get_epsilon(self.dp_config.delta)
            except (ValueError, RuntimeError):
                # No steps taken yet — accountant can't compute epsilon
                return 0.0
        return 0.0

    def save_checkpoint(
        self,
        directory: Path,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save checkpoint with DP-specific metadata.

        Extends the parent to include epsilon in the saved metrics.
        """
        if metrics is None:
            metrics = {}
        metrics["epsilon"] = self.get_epsilon()
        metrics["noise_multiplier"] = (
            self.dp_config.noise_multiplier
            if self.dp_config.noise_multiplier is not None
            else 0.0
        )
        return super().save_checkpoint(directory, epoch, metrics)
