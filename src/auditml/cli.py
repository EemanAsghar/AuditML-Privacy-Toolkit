"""AuditML command-line interface.

Entry point registered as ``auditml`` in pyproject.toml.

Usage examples::

    auditml audit   --config configs/default.yaml
    auditml train   --config configs/default.yaml
    auditml show-config --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from auditml import __version__
from auditml.config import AuditMLConfig, config_to_dict, default_config, load_config


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_audit(cfg: AuditMLConfig) -> None:
    """Run the privacy audit pipeline."""
    print(f"[audit] experiment : {cfg.experiment_name}")
    print(f"[audit] dataset    : {cfg.data.dataset.value}")
    print(f"[audit] attacks    : {[a.value for a in cfg.attacks]}")
    print(f"[audit] DP enabled : {cfg.dp.enabled}")
    # TODO: wire to attack runners once implemented
    print("[audit] Attack pipeline not yet implemented.")


def _handle_train(cfg: AuditMLConfig) -> None:
    """Train the target model."""
    print(f"[train] dataset    : {cfg.data.dataset.value}")
    print(f"[train] arch       : {cfg.model.arch}")
    print(f"[train] epochs     : {cfg.training.epochs}")
    print(f"[train] DP enabled : {cfg.dp.enabled}")
    if cfg.dp.enabled:
        print(f"[train] epsilon    : {cfg.dp.epsilon}")
    # TODO: wire to training loop once implemented
    print("[train] Training loop not yet implemented.")


def _handle_show_config(cfg: AuditMLConfig) -> None:
    """Print the fully-resolved configuration as JSON."""
    print(json.dumps(config_to_dict(cfg), indent=2))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="auditml",
        description="AuditML — privacy auditing toolkit for PyTorch models",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"auditml {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # -- audit ---------------------------------------------------------------
    audit_p = subparsers.add_parser(
        "audit",
        help="Run privacy attacks against a trained model",
    )
    audit_p.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    # -- train ---------------------------------------------------------------
    train_p = subparsers.add_parser(
        "train",
        help="Train the target model (optionally with DP)",
    )
    train_p.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    # -- show-config ---------------------------------------------------------
    show_p = subparsers.add_parser(
        "show-config",
        help="Print resolved configuration as JSON",
    )
    show_p.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (omit for defaults)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_HANDLERS = {
    "audit": _handle_audit,
    "train": _handle_train,
    "show-config": _handle_show_config,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns an exit code (0 = success)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Load config
    config_path = getattr(args, "config", None)
    if config_path is not None:
        try:
            cfg = load_config(config_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error loading config: {exc}", file=sys.stderr)
            return 1
    else:
        cfg = default_config()

    # Dispatch
    handler = _HANDLERS[args.command]
    handler(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
