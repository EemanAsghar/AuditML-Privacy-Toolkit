"""Device detection and management for AuditML."""

from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Return the best available ``torch.device``.

    Parameters
    ----------
    preference:
        ``"auto"`` (default) picks CUDA > MPS > CPU.
        ``"cuda"``, ``"mps"``, or ``"cpu"`` force a specific backend.

    Returns
    -------
    torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(preference)
    if preference == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    if preference == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return device


def device_info() -> dict[str, str | bool]:
    """Return a dict of device/hardware information."""
    info: dict[str, str | bool] = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory
        info["gpu_memory_gb"] = f"{mem / (1024 ** 3):.1f}"
    info["mps_available"] = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    return info
