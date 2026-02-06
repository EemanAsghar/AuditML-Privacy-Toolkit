"""Shared pytest fixtures for AuditML tests.

These fixtures create lightweight synthetic data and models that run
on CPU in seconds — no dataset downloads or GPU required.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


# ── Dimensions ───────────────────────────────────────────────────────────

BATCH_SIZE = 16
NUM_SAMPLES = 100
NUM_CLASSES = 10
IMG_C, IMG_H, IMG_W = 1, 28, 28


# ── Synthetic data ───────────────────────────────────────────────────────

def _make_synthetic_dataset(
    n: int = NUM_SAMPLES,
    channels: int = IMG_C,
    height: int = IMG_H,
    width: int = IMG_W,
    num_classes: int = NUM_CLASSES,
) -> TensorDataset:
    """Create a random image-classification dataset for testing."""
    images = torch.randn(n, channels, height, width)
    labels = torch.randint(0, num_classes, (n,))
    return TensorDataset(images, labels)


@pytest.fixture()
def synthetic_dataset() -> TensorDataset:
    """100-sample random MNIST-like dataset."""
    return _make_synthetic_dataset()


@pytest.fixture()
def synthetic_cifar_dataset() -> TensorDataset:
    """100-sample random CIFAR-like dataset."""
    return _make_synthetic_dataset(channels=3, height=32, width=32)


@pytest.fixture()
def train_loader(synthetic_dataset: TensorDataset) -> DataLoader:
    return DataLoader(synthetic_dataset, batch_size=BATCH_SIZE, shuffle=True)


@pytest.fixture()
def val_loader(synthetic_dataset: TensorDataset) -> DataLoader:
    return DataLoader(synthetic_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ── Models ───────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_cnn():
    """A SimpleCNN configured for MNIST-like input."""
    from auditml.models import SimpleCNN

    return SimpleCNN(input_channels=IMG_C, num_classes=NUM_CLASSES, input_size=IMG_H)


@pytest.fixture()
def cifar_cnn():
    """A SimpleCNN configured for CIFAR-like input."""
    from auditml.models import SimpleCNN

    return SimpleCNN(input_channels=3, num_classes=NUM_CLASSES, input_size=32)


@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")
