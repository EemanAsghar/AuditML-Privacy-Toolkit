"""AuditML privacy attack implementations.

All attacks inherit from ``BaseAttack`` and return ``AttackResult``.
Use ``get_attack()`` to instantiate an attack by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from auditml.attacks.base import BaseAttack
from auditml.attacks.results import AttackResult
from auditml.config.schema import AttackType, AuditMLConfig

if TYPE_CHECKING:
    pass

# Registry mapping AttackType → concrete class.
# Entries are added as each attack is implemented in Tasks 2.2–2.12.
# Using strings for lazy imports avoids circular-import issues and
# means we don't crash if an attack file has an uninstalled dependency.
_ATTACK_REGISTRY: dict[AttackType, str] = {
    AttackType.MIA_THRESHOLD: "auditml.attacks.mia_threshold.ThresholdMIA",
    AttackType.MIA_SHADOW: "auditml.attacks.mia_shadow.ShadowMIA",
    # AttackType.MODEL_INVERSION: "auditml.attacks.model_inversion.ModelInversion",
    # AttackType.ATTRIBUTE_INFERENCE: "auditml.attacks.attribute_inference.AttributeInference",
}


def get_attack(
    attack_type: AttackType | str,
    target_model: nn.Module,
    config: AuditMLConfig,
    device: str = "cpu",
) -> BaseAttack:
    """Instantiate a concrete attack by type.

    Parameters
    ----------
    attack_type:
        Which attack to create — an ``AttackType`` enum value or its
        string form (e.g. ``"mia_threshold"``).
    target_model:
        The trained model to attack.
    config:
        Full AuditML configuration.
    device:
        Torch device string.

    Returns
    -------
    BaseAttack
        A ready-to-run attack instance.

    Raises
    ------
    ValueError
        If *attack_type* is not recognised or not yet implemented.
    """
    if isinstance(attack_type, str):
        attack_type = AttackType(attack_type)

    if attack_type not in _ATTACK_REGISTRY:
        implemented = [k.value for k in _ATTACK_REGISTRY]
        raise ValueError(
            f"Attack {attack_type.value!r} is not yet implemented. "
            f"Available: {implemented or 'none yet — coming in Tasks 2.2-2.12'}"
        )

    # Lazy import: "auditml.attacks.mia_threshold.ThresholdMIA" → class
    dotted_path = _ATTACK_REGISTRY[attack_type]
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    return cls(target_model=target_model, config=config, device=device)


__all__ = [
    "AttackResult",
    "BaseAttack",
    "get_attack",
]
