"""Small ResNet variant for CIFAR datasets.

A compact ResNet-18-style architecture adapted for 32 × 32 inputs (no
aggressive stem downsampling). Achieves stronger accuracy than SimpleCNN
on CIFAR-10/100, useful for testing whether a more capable model leaks
more privacy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from auditml.models.base import BaseModel


class _BasicBlock(nn.Module):
    """Standard residual block with two 3×3 convolutions."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.GroupNorm(min(32, planes), planes)  # GroupNorm for Opacus compat
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.GroupNorm(min(32, planes), planes)

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(32, planes), planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class SmallResNet(BaseModel):
    """ResNet-18-style network for 32×32 inputs.

    Uses GroupNorm instead of BatchNorm so the model is compatible with
    Opacus (differential-privacy training) out of the box.

    Parameters
    ----------
    input_channels:
        Number of input channels (1 for MNIST, 3 for CIFAR).
    num_classes:
        Number of output classes.
    feature_dim:
        Width of the penultimate layer (default 512).
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        feature_dim: int = 512,
    ) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = nn.GroupNorm(32, 64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(feature_dim, 2, stride=2)

        self.fc = nn.Linear(feature_dim, num_classes)
        self._feature_dim = feature_dim

    # ----- forward / features --------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._backbone(x)
        return self.fc(feat)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    # ----- internals -----------------------------------------------------

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return torch.flatten(out, 1)
