"""Reusable PyTorch training loop for AuditML.

Handles standard training, checkpointing, early stopping, and LR
scheduling. The interface is intentionally kept simple so the DP
training path (Opacus) can reuse or extend it in Phase 3.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Standard (non-private) training loop.

    Parameters
    ----------
    model:
        The network to train.
    train_loader:
        DataLoader for training data.
    val_loader:
        DataLoader for validation/test data.
    optimizer:
        PyTorch optimizer instance.
    criterion:
        Loss function (default ``CrossEntropyLoss``).
    device:
        Device to train on.
    max_grad_norm:
        Optional gradient clipping max-norm. ``None`` disables clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module | None = None,
        device: torch.device | str = "cpu",
        max_grad_norm: float | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = torch.device(device)
        self.max_grad_norm = max_grad_norm

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    # ── public API ───────────────────────────────────────────────────────

    def train(
        self,
        epochs: int = 20,
        patience: int = 10,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Parameters
        ----------
        epochs:
            Maximum number of training epochs.
        patience:
            Stop early if validation loss has not improved for this many
            consecutive epochs. Set to ``0`` to disable.
        checkpoint_dir:
            If provided, save the best model checkpoint here.

        Returns
        -------
        dict
            The training ``history`` dict.
        """
        scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=max(1, patience // 3),
        )

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch(epoch, epochs)
            val_metrics = self.evaluate(self.val_loader)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            scheduler.step(val_loss)

            tqdm.write(
                f"Epoch {epoch}/{epochs} — "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.2%}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.2%}"
            )

            # Early stopping / checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if patience > 0 and no_improve >= patience:
                tqdm.write(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        if checkpoint_dir is not None:
            self.save_checkpoint(
                Path(checkpoint_dir),
                epoch=epoch,
                metrics={"val_loss": best_val_loss, "val_acc": val_acc},
            )

        return self.history

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate model on *loader*.

        Returns
        -------
        dict
            ``{"loss": float, "accuracy": float}``
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, targets).item() * inputs.size(0)
                correct += (outputs.argmax(1) == targets).sum().item()
                total += inputs.size(0)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }

    # ── checkpoint helpers ───────────────────────────────────────────────

    def save_checkpoint(
        self,
        directory: Path,
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save model weights, optimizer state, and metadata.

        Returns the path to the saved ``.pt`` file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        ckpt_path = directory / "model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics or {},
                "history": self.history,
            },
            ckpt_path,
        )

        # Also save a human-readable metrics file
        if metrics:
            (directory / "metrics.json").write_text(
                json.dumps(metrics, indent=2),
            )

        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load a checkpoint and restore model/optimizer state.

        Returns the checkpoint dict (contains ``epoch``, ``metrics``, etc.).
        """
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt.get("history", self.history)
        return ckpt

    # ── internals ────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int, total_epochs: int) -> tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{total_epochs}",
            leave=False,
        )
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(1) == targets).sum().item()
            total += batch_size

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(total, 1), correct / max(total, 1)


def build_optimizer(
    model: nn.Module,
    name: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Create an optimizer from a string name.

    Parameters
    ----------
    model:
        The model whose parameters will be optimised.
    name:
        ``"adam"`` or ``"sgd"``.
    lr:
        Learning rate.
    weight_decay:
        L2 regularisation strength.
    """
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9,
        )
    raise ValueError(f"Unknown optimizer {name!r}. Choose 'adam' or 'sgd'.")
