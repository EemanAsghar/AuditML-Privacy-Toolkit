"""Dataset loading, splitting, and DataLoader creation for AuditML.

The key abstraction here is the *member / non-member split* — every
privacy attack needs to know which samples were used for training
(members) and which were not (non-members). This module provides
reproducible, seed-controlled splits at the dataset level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets

from auditml.data.transforms import get_transforms


# ─── Dataset metadata ────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetInfo:
    """Lightweight descriptor for a supported dataset."""

    name: str
    num_classes: int
    input_shape: tuple[int, ...]
    num_train: int
    num_test: int


DATASET_INFO: dict[str, DatasetInfo] = {
    "mnist": DatasetInfo("mnist", 10, (1, 28, 28), 60_000, 10_000),
    "cifar10": DatasetInfo("cifar10", 10, (3, 32, 32), 50_000, 10_000),
    "cifar100": DatasetInfo("cifar100", 100, (3, 32, 32), 50_000, 10_000),
}


# ─── Raw dataset loading ────────────────────────────────────────────────

_TV_MAP = {
    "mnist": tv_datasets.MNIST,
    "cifar10": tv_datasets.CIFAR10,
    "cifar100": tv_datasets.CIFAR100,
}


def get_dataset(
    name: str,
    train: bool = True,
    data_dir: str = "./data",
    download: bool = True,
) -> Dataset:
    """Load a torchvision dataset by name.

    Parameters
    ----------
    name:
        ``"mnist"``, ``"cifar10"``, or ``"cifar100"``.
    train:
        Load the training split if ``True``, test split otherwise.
    data_dir:
        Root directory for downloaded data.
    download:
        Download the dataset if not already present.

    Returns
    -------
    Dataset
        A torchvision dataset with the appropriate transforms applied.
    """
    if name not in _TV_MAP:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {sorted(_TV_MAP)}")

    transform = get_transforms(name, train=train)
    return _TV_MAP[name](
        root=data_dir, train=train, download=download, transform=transform,
    )


# ─── Member / non-member splits ─────────────────────────────────────────

def create_member_nonmember_split(
    dataset: Dataset,
    member_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[Subset, Subset, np.ndarray, np.ndarray]:
    """Split *dataset* into disjoint member and non-member subsets.

    Parameters
    ----------
    dataset:
        The full training dataset.
    member_ratio:
        Fraction of samples assigned to the member set.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (member_subset, nonmember_subset, member_indices, nonmember_indices)
    """
    n = len(dataset)  # type: ignore[arg-type]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    split = int(n * member_ratio)
    member_idx = np.sort(indices[:split])
    nonmember_idx = np.sort(indices[split:])

    return (
        Subset(dataset, member_idx.tolist()),
        Subset(dataset, nonmember_idx.tolist()),
        member_idx,
        nonmember_idx,
    )


def get_shadow_data_splits(
    dataset: Dataset,
    n_shadows: int = 5,
    member_ratio: float = 0.5,
    seed: int = 42,
) -> list[tuple[Subset, Subset, np.ndarray, np.ndarray]]:
    """Create *n_shadows* independent member/non-member splits.

    Each shadow model will be trained on its own member subset, providing
    diverse "in" vs "out" examples for the attack classifier.

    Parameters
    ----------
    dataset:
        The full dataset to split.
    n_shadows:
        Number of independent splits.
    member_ratio:
        Fraction of the dataset each shadow's member set uses.
    seed:
        Base seed; each split uses ``seed + i``.

    Returns
    -------
    list of (member_subset, nonmember_subset, member_indices, nonmember_indices)
    """
    return [
        create_member_nonmember_split(dataset, member_ratio, seed=seed + i)
        for i in range(n_shadows)
    ]


# ─── DataLoader helpers ─────────────────────────────────────────────────

def get_dataloaders(
    dataset_name: str,
    batch_size: int = 64,
    member_ratio: float = 0.5,
    num_workers: int = 2,
    seed: int = 42,
    data_dir: str = "./data",
    download: bool = True,
) -> dict[str, DataLoader | np.ndarray]:
    """Convenience loader that returns everything needed for an audit.

    Returns
    -------
    dict with keys:
        ``"train_loader"`` — DataLoader for the member (training) subset
        ``"test_loader"``  — DataLoader for the held-out test split
        ``"member_loader"`` — same data as train_loader (alias)
        ``"nonmember_loader"`` — DataLoader for non-member training data
        ``"member_indices"`` — indices into the full training set
        ``"nonmember_indices"`` — indices into the full training set
    """
    full_train = get_dataset(dataset_name, train=True, data_dir=data_dir, download=download)
    test_set = get_dataset(dataset_name, train=False, data_dir=data_dir, download=download)

    member_set, nonmember_set, mem_idx, nonmem_idx = create_member_nonmember_split(
        full_train, member_ratio=member_ratio, seed=seed,
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return {
        "train_loader": DataLoader(member_set, shuffle=True, **loader_kwargs),
        "test_loader": DataLoader(test_set, shuffle=False, **loader_kwargs),
        "member_loader": DataLoader(member_set, shuffle=False, **loader_kwargs),
        "nonmember_loader": DataLoader(nonmember_set, shuffle=False, **loader_kwargs),
        "member_indices": mem_idx,
        "nonmember_indices": nonmem_idx,
    }
