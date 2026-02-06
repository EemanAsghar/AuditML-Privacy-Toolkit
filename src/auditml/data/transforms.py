"""Standard transforms for each dataset."""

from __future__ import annotations

from torchvision import transforms

# ── MNIST ────────────────────────────────────────────────────────────────
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

mnist_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

mnist_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

# ── CIFAR-10 / CIFAR-100 ────────────────────────────────────────────────
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

cifar_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

cifar_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def get_transforms(
    dataset: str, train: bool = True,
) -> transforms.Compose:
    """Return the appropriate transform for a given dataset and split.

    Parameters
    ----------
    dataset:
        ``"mnist"``, ``"cifar10"``, or ``"cifar100"``.
    train:
        If ``True`` return the training transform (with augmentation for
        CIFAR); otherwise the evaluation transform.
    """
    if dataset == "mnist":
        return mnist_train_transform if train else mnist_test_transform
    if dataset in ("cifar10", "cifar100"):
        return cifar_train_transform if train else cifar_test_transform
    raise ValueError(f"Unknown dataset: {dataset!r}")
