"""AuditML command-line interface.

Entry point registered as ``auditml`` in pyproject.toml.

Usage examples::

    auditml train      --config configs/default.yaml
    auditml audit      --config configs/default.yaml
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


def _resolve_device(cfg: AuditMLConfig) -> str:
    """Pick the training device from config (supports 'auto')."""
    import torch

    device = cfg.training.device
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _handle_train(cfg: AuditMLConfig) -> None:
    """Train the target model (optionally with DP)."""
    import torch

    from auditml.data.datasets import get_dataloaders
    from auditml.models import get_model
    from auditml.training import Trainer, build_optimizer

    device = _resolve_device(cfg)
    dataset = cfg.data.dataset.value

    print(f"[train] dataset    : {dataset}")
    print(f"[train] arch       : {cfg.model.arch}")
    print(f"[train] epochs     : {cfg.training.epochs}")
    print(f"[train] device     : {device}")
    print(f"[train] DP enabled : {cfg.dp.enabled}")

    # Seed for reproducibility
    torch.manual_seed(cfg.training.seed)

    # Data
    data = get_dataloaders(
        dataset_name=dataset,
        batch_size=cfg.training.batch_size,
        member_ratio=cfg.data.train_ratio,
        num_workers=cfg.data.num_workers,
        seed=cfg.training.seed,
        data_dir=cfg.data.data_dir,
        download=cfg.data.download,
    )

    # Model
    model = get_model(arch=cfg.model.arch, dataset=dataset)

    # Optimizer
    optimizer = build_optimizer(
        model,
        name=cfg.training.optimizer,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Train
    out_dir = Path(cfg.reporting.output_dir) / cfg.experiment_name
    ckpt_dir = out_dir / "checkpoints"

    if cfg.dp.enabled:
        from auditml.training import DPTrainer, validate_and_fix_model

        print(f"[train] epsilon    : {cfg.dp.epsilon}")
        model = validate_and_fix_model(model)
        # Rebuild optimizer after fix (parameters may have changed)
        optimizer = build_optimizer(
            model,
            name=cfg.training.optimizer,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        trainer = DPTrainer(
            model=model,
            train_loader=data["train_loader"],
            val_loader=data["test_loader"],
            optimizer=optimizer,
            dp_config=cfg.dp,
            device=device,
        )
    else:
        trainer = Trainer(
            model=model,
            train_loader=data["train_loader"],
            val_loader=data["test_loader"],
            optimizer=optimizer,
            device=device,
        )

    history = trainer.train(
        epochs=cfg.training.epochs,
        checkpoint_dir=ckpt_dir,
    )

    # Final evaluation
    final_metrics = trainer.evaluate(data["test_loader"])
    print(f"[train] Final accuracy: {final_metrics['accuracy']:.4f}")
    print(f"[train] Final loss:     {final_metrics['loss']:.4f}")

    if cfg.dp.enabled and hasattr(trainer, "get_epsilon"):
        print(f"[train] Final epsilon:  {trainer.get_epsilon():.2f}")

    print(f"[train] Checkpoint saved to {ckpt_dir}")


def _handle_audit(cfg: AuditMLConfig) -> None:
    """Run the privacy audit pipeline."""
    import torch

    from auditml.attacks import get_attack
    from auditml.data.datasets import get_dataloaders
    from auditml.models import get_model
    from auditml.reporting import ReportGenerator
    from auditml.training import Trainer, build_optimizer

    device = _resolve_device(cfg)
    dataset = cfg.data.dataset.value

    print(f"[audit] experiment : {cfg.experiment_name}")
    print(f"[audit] dataset    : {dataset}")
    print(f"[audit] attacks    : {[a.value for a in cfg.attacks]}")
    print(f"[audit] DP enabled : {cfg.dp.enabled}")
    print(f"[audit] device     : {device}")

    torch.manual_seed(cfg.training.seed)

    # Data
    data = get_dataloaders(
        dataset_name=dataset,
        batch_size=cfg.training.batch_size,
        member_ratio=cfg.data.train_ratio,
        num_workers=cfg.data.num_workers,
        seed=cfg.training.seed,
        data_dir=cfg.data.data_dir,
        download=cfg.data.download,
    )

    out_dir = Path(cfg.reporting.output_dir) / cfg.experiment_name
    ckpt_path = out_dir / "checkpoints" / "model.pt"

    # ── Train or load the baseline model ──────────────────────────────
    model = get_model(arch=cfg.model.arch, dataset=dataset)
    optimizer = build_optimizer(
        model,
        name=cfg.training.optimizer,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    if ckpt_path.exists():
        print(f"[audit] Loading checkpoint from {ckpt_path}")
        trainer = Trainer(
            model=model,
            train_loader=data["train_loader"],
            val_loader=data["test_loader"],
            optimizer=optimizer,
            device=device,
        )
        trainer.load_checkpoint(ckpt_path)
    else:
        print("[audit] No checkpoint found — training baseline model ...")
        trainer = Trainer(
            model=model,
            train_loader=data["train_loader"],
            val_loader=data["test_loader"],
            optimizer=optimizer,
            device=device,
        )
        trainer.train(epochs=cfg.training.epochs, checkpoint_dir=out_dir / "checkpoints")

    baseline_acc = trainer.evaluate(data["test_loader"])
    print(f"[audit] Baseline model accuracy: {baseline_acc['accuracy']:.4f}")
    model.eval()

    # ── Run attacks on baseline model ─────────────────────────────────
    attack_results: dict[str, tuple] = {}

    for attack_type in cfg.attacks:
        print(f"[audit] Running {attack_type.value} ...")
        attack = get_attack(attack_type, model, cfg, device=device)
        result = attack.run(data["member_loader"], data["nonmember_loader"])
        metrics = attack.evaluate()
        attack_results[attack_type.value] = (attack, result)

        print(
            f"[audit]   accuracy={metrics['accuracy']:.4f}  "
            f"auc_roc={metrics['auc_roc']:.4f}"
        )

    # ── DP model (if enabled) ─────────────────────────────────────────
    dp_attack_results: dict[str, tuple] = {}
    model_accuracy: dict[str, float] = {"baseline": baseline_acc["accuracy"]}
    epsilon = None

    if cfg.dp.enabled:
        from auditml.training import DPTrainer, validate_and_fix_model

        print("[audit] Training DP model ...")
        dp_model = get_model(arch=cfg.model.arch, dataset=dataset)
        dp_model = validate_and_fix_model(dp_model)
        dp_optimizer = build_optimizer(
            dp_model,
            name=cfg.training.optimizer,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        dp_trainer = DPTrainer(
            model=dp_model,
            train_loader=data["train_loader"],
            val_loader=data["test_loader"],
            optimizer=dp_optimizer,
            dp_config=cfg.dp,
            device=device,
        )
        dp_trainer.train(
            epochs=cfg.training.epochs,
            checkpoint_dir=out_dir / "checkpoints_dp",
        )

        dp_acc = dp_trainer.evaluate(data["test_loader"])
        epsilon = dp_trainer.get_epsilon()
        model_accuracy["dp"] = dp_acc["accuracy"]
        print(f"[audit] DP model accuracy: {dp_acc['accuracy']:.4f} (epsilon={epsilon:.2f})")

        dp_model.eval()

        # Re-run attacks on DP model
        # Need fresh data loaders for DP attacks (same splits)
        for attack_type in cfg.attacks:
            print(f"[audit] Running {attack_type.value} on DP model ...")
            dp_attack = get_attack(attack_type, dp_model, cfg, device=device)
            dp_result = dp_attack.run(data["member_loader"], data["nonmember_loader"])
            dp_metrics = dp_attack.evaluate()
            dp_attack_results[attack_type.value] = (dp_attack, dp_result)

            print(
                f"[audit]   accuracy={dp_metrics['accuracy']:.4f}  "
                f"auc_roc={dp_metrics['auc_roc']:.4f}"
            )

    # ── Generate unified report ───────────────────────────────────────
    print("[audit] Generating report ...")
    report_dir = out_dir / "report"
    generator = ReportGenerator(
        experiment_name=cfg.experiment_name,
        attack_results=attack_results,
        dp_attack_results=dp_attack_results if dp_attack_results else None,
        epsilon=epsilon,
        model_accuracy=model_accuracy,
    )
    generator.generate(report_dir)
    print(f"[audit] Report saved to {report_dir}")


def _handle_show_config(cfg: AuditMLConfig) -> None:
    """Print the fully-resolved configuration as JSON."""
    print(json.dumps(config_to_dict(cfg), indent=2))


def _handle_ui(_cfg: AuditMLConfig) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    from pathlib import Path

    home_py = Path(__file__).parent / "ui" / "Home.py"
    if not home_py.exists():
        print(f"Error: UI entry point not found at {home_py}", file=sys.stderr)
        raise RuntimeError("UI files not found.")
    print("[ui] Launching AuditML dashboard → http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(home_py)], check=True)


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

    # -- ui ------------------------------------------------------------------
    subparsers.add_parser(
        "ui",
        help="Launch the Streamlit web dashboard",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_HANDLERS = {
    "audit": _handle_audit,
    "train": _handle_train,
    "show-config": _handle_show_config,
    "ui": _handle_ui,
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
    try:
        handler(cfg)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
