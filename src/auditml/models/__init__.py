"""AuditML model definitions."""

from __future__ import annotations

from auditml.models.base import BaseModel, count_parameters
from auditml.models.cnn import SimpleCNN
from auditml.models.resnet import SmallResNet

# Maps arch name → model class.
_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "cnn": SimpleCNN,
    "simple_cnn": SimpleCNN,
    "resnet": SmallResNet,
    "small_resnet": SmallResNet,
}

_DATASET_DEFAULTS: dict[str, dict[str, int]] = {
    "mnist": {"input_channels": 1, "num_classes": 10, "input_size": 28},
    "cifar10": {"input_channels": 3, "num_classes": 10, "input_size": 32},
    "cifar100": {"input_channels": 3, "num_classes": 100, "input_size": 32},
}


def get_model(arch: str = "cnn", dataset: str = "cifar10") -> BaseModel:
    """Factory that returns a correctly configured model.

    Parameters
    ----------
    arch:
        Architecture name (``"cnn"`` / ``"simple_cnn"`` / ``"resnet"`` /
        ``"small_resnet"``).
    dataset:
        Dataset name (``"mnist"`` / ``"cifar10"`` / ``"cifar100"``).

    Returns
    -------
    BaseModel
        An uninitialised (random weights) model instance.
    """
    if arch not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture {arch!r}. Choose from: {sorted(_MODEL_REGISTRY)}"
        )
    if dataset not in _DATASET_DEFAULTS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose from: {sorted(_DATASET_DEFAULTS)}"
        )

    cls = _MODEL_REGISTRY[arch]
    kwargs = dict(_DATASET_DEFAULTS[dataset])

    # SmallResNet doesn't use `input_size`
    if cls is SmallResNet:
        kwargs.pop("input_size", None)

    return cls(**kwargs)


__all__ = [
    "BaseModel",
    "SimpleCNN",
    "SmallResNet",
    "count_parameters",
    "get_model",
]
