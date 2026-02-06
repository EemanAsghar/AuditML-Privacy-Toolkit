"""Tests for AuditML data module.

These tests use synthetic data only — no network downloads required.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from auditml.data import (
    DATASET_INFO,
    DatasetInfo,
    create_member_nonmember_split,
    get_shadow_data_splits,
    get_transforms,
)


class TestDatasetInfo:
    def test_mnist_info(self) -> None:
        info = DATASET_INFO["mnist"]
        assert info.num_classes == 10
        assert info.input_shape == (1, 28, 28)

    def test_cifar100_info(self) -> None:
        info = DATASET_INFO["cifar100"]
        assert info.num_classes == 100


class TestTransforms:
    def test_mnist_transform_exists(self) -> None:
        t = get_transforms("mnist", train=True)
        assert t is not None

    def test_cifar_train_has_augmentation(self) -> None:
        t = get_transforms("cifar10", train=True)
        # Should contain RandomHorizontalFlip
        names = [type(tr).__name__ for tr in t.transforms]
        assert "RandomHorizontalFlip" in names

    def test_cifar_test_no_augmentation(self) -> None:
        t = get_transforms("cifar10", train=False)
        names = [type(tr).__name__ for tr in t.transforms]
        assert "RandomHorizontalFlip" not in names


class TestMemberNonmemberSplit:
    def test_disjoint_split(self, synthetic_dataset: TensorDataset) -> None:
        mem, nonmem, mem_idx, nonmem_idx = create_member_nonmember_split(
            synthetic_dataset, member_ratio=0.5, seed=42,
        )
        assert len(set(mem_idx) & set(nonmem_idx)) == 0  # no overlap

    def test_complete_split(self, synthetic_dataset: TensorDataset) -> None:
        mem, nonmem, mem_idx, nonmem_idx = create_member_nonmember_split(
            synthetic_dataset, member_ratio=0.5, seed=42,
        )
        total = len(mem_idx) + len(nonmem_idx)
        assert total == len(synthetic_dataset)

    def test_ratio(self, synthetic_dataset: TensorDataset) -> None:
        _, _, mem_idx, nonmem_idx = create_member_nonmember_split(
            synthetic_dataset, member_ratio=0.8, seed=42,
        )
        assert len(mem_idx) == 80
        assert len(nonmem_idx) == 20

    def test_reproducibility(self, synthetic_dataset: TensorDataset) -> None:
        _, _, idx1, _ = create_member_nonmember_split(synthetic_dataset, seed=123)
        _, _, idx2, _ = create_member_nonmember_split(synthetic_dataset, seed=123)
        np.testing.assert_array_equal(idx1, idx2)

    def test_different_seed_different_split(self, synthetic_dataset: TensorDataset) -> None:
        _, _, idx1, _ = create_member_nonmember_split(synthetic_dataset, seed=1)
        _, _, idx2, _ = create_member_nonmember_split(synthetic_dataset, seed=2)
        assert not np.array_equal(idx1, idx2)


class TestShadowSplits:
    def test_correct_count(self, synthetic_dataset: TensorDataset) -> None:
        splits = get_shadow_data_splits(synthetic_dataset, n_shadows=3, seed=42)
        assert len(splits) == 3

    def test_each_split_is_disjoint(self, synthetic_dataset: TensorDataset) -> None:
        splits = get_shadow_data_splits(synthetic_dataset, n_shadows=3, seed=42)
        for _, _, mem_idx, nonmem_idx in splits:
            assert len(set(mem_idx) & set(nonmem_idx)) == 0

    def test_splits_differ(self, synthetic_dataset: TensorDataset) -> None:
        splits = get_shadow_data_splits(synthetic_dataset, n_shadows=3, seed=42)
        idx_0 = splits[0][2]
        idx_1 = splits[1][2]
        assert not np.array_equal(idx_0, idx_1)
