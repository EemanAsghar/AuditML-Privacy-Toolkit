"""Load, validate, and merge YAML configuration files."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from auditml.config.schema import AuditMLConfig


def load_config(path: str | Path) -> AuditMLConfig:
    """Load a YAML file and return a validated ``AuditMLConfig``.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    AuditMLConfig
        Fully populated configuration with defaults for missing fields.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML content is invalid or contains unknown keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level mapping in {path}, got {type(raw).__name__}")

    return _dict_to_dataclass(AuditMLConfig, raw)


def default_config() -> AuditMLConfig:
    """Return an ``AuditMLConfig`` with all defaults."""
    return AuditMLConfig()


def config_to_dict(cfg: AuditMLConfig) -> dict[str, Any]:
    """Serialize an ``AuditMLConfig`` back to a plain dict (YAML-friendly)."""
    return _dataclass_to_dict(cfg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a raw dict into a dataclass instance.

    Unknown keys raise ``ValueError`` so typos in configs are caught early.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    known_fields = {f.name: f for f in fields(cls)}
    # get_type_hints() resolves stringified annotations from __future__
    resolved_hints = get_type_hints(cls)
    unknown = set(data.keys()) - set(known_fields.keys())
    if unknown:
        raise ValueError(
            f"Unknown config keys for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known_fields.keys())}"
        )

    kwargs: dict[str, Any] = {}
    for name in known_fields:
        if name not in data:
            continue
        value = data[name]
        kwargs[name] = _coerce_value(resolved_hints[name], value, field_name=name)

    return cls(**kwargs)


def _coerce_value(type_hint: Any, value: Any, *, field_name: str) -> Any:
    """Coerce a raw YAML value to the expected Python type."""

    # --- Enum ---
    if isinstance(type_hint, type) and issubclass(type_hint, Enum):
        try:
            return type_hint(value)
        except ValueError:
            valid = [e.value for e in type_hint]
            raise ValueError(
                f"Invalid value {value!r} for {field_name}. Must be one of {valid}"
            ) from None

    # --- Dataclass (nested section) ---
    if is_dataclass(type_hint):
        if not isinstance(value, dict):
            raise ValueError(f"Expected mapping for {field_name}, got {type(value).__name__}")
        return _dict_to_dataclass(type_hint, value)

    # --- list[...] ---
    origin = get_origin(type_hint)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"Expected list for {field_name}, got {type(value).__name__}")
        args = get_args(type_hint)
        if args:
            inner = args[0]
            return [_coerce_value(inner, v, field_name=f"{field_name}[]") for v in value]
        return list(value)

    # --- Optional[...] ---
    if _is_optional(type_hint):
        if value is None:
            return None
        inner = _unwrap_optional(type_hint)
        return _coerce_value(inner, value, field_name=field_name)

    # --- Primitive passthrough ---
    return value


def _is_optional(tp: Any) -> bool:
    """Check if a type hint is ``Optional[X]`` (i.e. ``Union[X, None]``)."""
    import types
    origin = get_origin(tp)
    if origin is types.UnionType:
        args = get_args(tp)
        return type(None) in args
    # typing.Optional
    if origin is not None:
        args = get_args(tp)
        return type(None) in args
    return False


def _unwrap_optional(tp: Any) -> Any:
    """Return the inner type from ``Optional[X]``."""
    args = get_args(tp)
    return next(a for a in args if a is not type(None))


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass to a plain dict."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name))
            for f in fields(obj)
        }
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return copy.deepcopy(obj)
