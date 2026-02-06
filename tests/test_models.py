"""Tests for AuditML model definitions."""

from __future__ import annotations

import torch
import pytest

from auditml.models import SimpleCNN, SmallResNet, count_parameters, get_model


class TestSimpleCNN:
    def test_mnist_output_shape(self, simple_cnn: SimpleCNN) -> None:
        x = torch.randn(4, 1, 28, 28)
        out = simple_cnn(x)
        assert out.shape == (4, 10)

    def test_cifar_output_shape(self) -> None:
        model = SimpleCNN(input_channels=3, num_classes=10, input_size=32)
        x = torch.randn(4, 3, 32, 32)
        assert model(x).shape == (4, 10)

    def test_cifar100_output_shape(self) -> None:
        model = SimpleCNN(input_channels=3, num_classes=100, input_size=32)
        x = torch.randn(2, 3, 32, 32)
        assert model(x).shape == (2, 100)

    def test_get_features_shape(self, simple_cnn: SimpleCNN) -> None:
        x = torch.randn(4, 1, 28, 28)
        feat = simple_cnn.get_features(x)
        assert feat.shape == (4, 128)


class TestSmallResNet:
    def test_cifar10_output_shape(self) -> None:
        model = SmallResNet(input_channels=3, num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        assert model(x).shape == (2, 10)

    def test_cifar100_output_shape(self) -> None:
        model = SmallResNet(input_channels=3, num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        assert model(x).shape == (2, 100)

    def test_get_features_shape(self) -> None:
        model = SmallResNet(input_channels=3, num_classes=10, feature_dim=512)
        x = torch.randn(2, 3, 32, 32)
        assert model.get_features(x).shape == (2, 512)

    def test_mnist_compatible(self) -> None:
        model = SmallResNet(input_channels=1, num_classes=10)
        x = torch.randn(2, 1, 32, 32)
        assert model(x).shape == (2, 10)


class TestGetModel:
    def test_cnn_mnist(self) -> None:
        m = get_model("cnn", "mnist")
        assert isinstance(m, SimpleCNN)
        assert m(torch.randn(1, 1, 28, 28)).shape == (1, 10)

    def test_resnet_cifar100(self) -> None:
        m = get_model("resnet", "cifar100")
        assert isinstance(m, SmallResNet)
        assert m(torch.randn(1, 3, 32, 32)).shape == (1, 100)

    def test_unknown_arch_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_model("transformer", "mnist")

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_model("cnn", "imagenet")


class TestCountParameters:
    def test_returns_dict(self, simple_cnn: SimpleCNN) -> None:
        counts = count_parameters(simple_cnn)
        assert "total" in counts
        assert "trainable" in counts
        assert counts["total"] > 0
        assert counts["total"] == counts["trainable"]
