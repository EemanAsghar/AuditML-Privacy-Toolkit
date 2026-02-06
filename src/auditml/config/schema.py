"""Typed configuration schema for AuditML.

Every section of the YAML config maps to a dataclass here. This gives us
runtime validation, IDE autocompletion, and a single source of truth for
what the config accepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DatasetName(str, Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


class AttackType(str, Enum):
    MIA_SHADOW = "mia_shadow"
    MIA_THRESHOLD = "mia_threshold"
    MODEL_INVERSION = "model_inversion"
    ATTRIBUTE_INFERENCE = "attribute_inference"


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset selection and splitting."""

    dataset: DatasetName = DatasetName.CIFAR10
    data_dir: str = "./data"
    train_ratio: float = 0.5
    num_workers: int = 2
    download: bool = True


@dataclass
class ModelConfig:
    """Target and shadow model architecture."""

    arch: str = "cnn"
    num_classes: int = 10
    pretrained: bool = False


@dataclass
class TrainingConfig:
    """Standard (non-private) training parameters."""

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"


@dataclass
class DPConfig:
    """Differential privacy training parameters (Opacus)."""

    enabled: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: Optional[float] = None


@dataclass
class MIAShadowConfig:
    """Membership Inference Attack — shadow model variant."""

    num_shadow_models: int = 3
    shadow_epochs: int = 20
    attack_model: str = "mlp"


@dataclass
class MIAThresholdConfig:
    """Membership Inference Attack — threshold variant."""

    metric: str = "loss"
    percentile: float = 50.0


@dataclass
class ModelInversionConfig:
    """Model Inversion attack parameters."""

    num_iterations: int = 1000
    learning_rate: float = 0.01
    lambda_tv: float = 0.001
    lambda_l2: float = 0.0
    target_class: Optional[int] = None


@dataclass
class AttributeInferenceConfig:
    """Attribute Inference attack parameters."""

    sensitive_attribute: str = "superclass"
    attack_model: str = "mlp"
    known_features: list[str] = field(default_factory=lambda: ["subclass"])


@dataclass
class ReportingConfig:
    """Output and reporting settings."""

    output_dir: str = "./results"
    save_plots: bool = True
    save_json: bool = True
    plot_format: str = "png"


@dataclass
class AttackConfig:
    """Container for all attack-specific settings."""

    mia_shadow: MIAShadowConfig = field(default_factory=MIAShadowConfig)
    mia_threshold: MIAThresholdConfig = field(default_factory=MIAThresholdConfig)
    model_inversion: ModelInversionConfig = field(default_factory=ModelInversionConfig)
    attribute_inference: AttributeInferenceConfig = field(default_factory=AttributeInferenceConfig)


@dataclass
class AuditMLConfig:
    """Top-level configuration for an AuditML run."""

    experiment_name: str = "audit"
    attacks: list[AttackType] = field(
        default_factory=lambda: [AttackType.MIA_THRESHOLD],
    )
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    attack_params: AttackConfig = field(default_factory=AttackConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
