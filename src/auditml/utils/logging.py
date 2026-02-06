"""Structured logging for AuditML experiments."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: str | Path | None = None,
    experiment_name: str = "auditml",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return the ``auditml`` logger.

    Sets up two handlers:
    - **Console** — concise, colourless, INFO-level output.
    - **File** (optional) — detailed, DEBUG-level output written to
      ``<log_dir>/<experiment_name>.log``.

    Parameters
    ----------
    log_dir:
        Directory for the log file. ``None`` disables file logging.
    experiment_name:
        Used as the log file name.
    level:
        Minimum level for the console handler.
    """
    logger = logging.getLogger("auditml")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{experiment_name}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
