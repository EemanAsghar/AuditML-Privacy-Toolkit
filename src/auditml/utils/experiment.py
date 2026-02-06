"""Experiment tracking for AuditML.

``ExperimentLogger`` manages the output directory for a single experiment
run, saving configs, metrics, and system information in a structured
layout::

    results/<experiment_name>/
    ├── config.yaml
    ├── system_info.json
    ├── metrics.csv
    └── logs/
        └── <experiment_name>.log
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

from auditml.utils.logging import setup_logging

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class ExperimentLogger:
    """Manages artefacts for a single experiment run.

    Parameters
    ----------
    experiment_name:
        Human-readable experiment identifier.
    output_dir:
        Root directory under which the experiment folder is created.
    """

    def __init__(
        self,
        experiment_name: str = "audit",
        output_dir: str | Path = "./results",
    ) -> None:
        self.experiment_name = experiment_name
        self.run_dir = Path(output_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.run_dir / "logs"
        self.logger = setup_logging(
            log_dir=self.log_dir,
            experiment_name=experiment_name,
        )

        self._metrics_rows: list[dict[str, Any]] = []

    # ── config ───────────────────────────────────────────────────────────

    def log_config(self, config: dict[str, Any]) -> None:
        """Write the experiment config to ``config.yaml``."""
        path = self.run_dir / "config.yaml"
        path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        self.logger.info("Config saved to %s", path)

    # ── system info ──────────────────────────────────────────────────────

    def log_system_info(self) -> dict[str, Any]:
        """Capture and save system/hardware information."""
        info: dict[str, Any] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
        }
        if _TORCH_AVAILABLE:
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_name"] = torch.cuda.get_device_name(0)

        path = self.run_dir / "system_info.json"
        path.write_text(json.dumps(info, indent=2))
        self.logger.info("System info: %s", json.dumps(info))
        return info

    # ── metrics ──────────────────────────────────────────────────────────

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Record a metrics snapshot.

        Logged to both the Python logger and an in-memory list that can
        be flushed to CSV with ``save_metrics``.
        """
        row = dict(metrics)
        if step is not None:
            row["step"] = step
        self._metrics_rows.append(row)
        self.logger.info("Metrics (step=%s): %s", step, metrics)

    def save_metrics(self) -> Path:
        """Flush accumulated metrics to ``metrics.csv``."""
        import pandas as pd

        path = self.run_dir / "metrics.csv"
        df = pd.DataFrame(self._metrics_rows)
        df.to_csv(path, index=False)
        self.logger.info("Metrics saved to %s (%d rows)", path, len(df))
        return path

    # ── convenience ──────────────────────────────────────────────────────

    def log_model_summary(self, model: Any) -> None:
        """Log architecture and parameter counts."""
        from auditml.models.base import count_parameters

        counts = count_parameters(model)
        self.logger.info(
            "Model: %s | Params: %s total, %s trainable",
            model.__class__.__name__,
            f"{counts['total']:,}",
            f"{counts['trainable']:,}",
        )

    def info(self, msg: str, *args: Any) -> None:
        """Shortcut to ``self.logger.info``."""
        self.logger.info(msg, *args)
