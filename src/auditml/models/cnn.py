"""Simple CNN architecture for MNIST / CIFAR-10 / CIFAR-100.

The architecture is intentionally compact — two conv blocks followed by
two fully-connected layers. This mirrors common MIA literature baselines
and trains quickly, which matters when training multiple shadow models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from auditml.models.base import BaseModel


class SimpleCNN(BaseModel):
    """Configurable 2-block CNN.

    Parameters
    ----------
    input_channels:
        Number of input channels (1 for MNIST, 3 for CIFAR).
    num_classes:
        Number of output classes.
    input_size:
        Spatial dimension of the input (28 for MNIST, 32 for CIFAR).
    feature_dim:
        Width of the penultimate fully-connected layer.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28,
        feature_dim: int = 128,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)

        # After two pools: spatial size = input_size // 4
        flat_size = 64 * (input_size // 4) ** 2

        self.fc1 = nn.Linear(flat_size, feature_dim)
        self.fc2 = nn.Linear(feature_dim, num_classes)

        self._feature_dim = feature_dim

    # ----- forward / features --------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._backbone(x)
        x = self.fc2(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    # ----- internals -----------------------------------------------------

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return x
