#!/usr/bin/env python
"""Train and save baseline (non-private) models for AuditML.

Usage::

    # Train on a single dataset using a config file
    python scripts/train_baseline.py --config configs/default.yaml

    # Override dataset and epochs from the command line
    python scripts/train_baseline.py --config configs/default.yaml \
        --dataset mnist --epochs 10

    # Train all three datasets in sequence
    python scripts/train_baseline.py --config configs/default.yaml --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from auditml.config import load_config
from auditml.data import get_dataloaders
from auditml.models import get_model, count_parameters
from auditml.training import Trainer, build_optimizer
from auditml.utils import set_seed, get_device, ExperimentLogger


ALL_DATASETS = ["mnist", "cifar10", "cifar100"]


def train_one(
    dataset: str,
    arch: str,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer_name: str,
    weight_decay: float,
    seed: int,
    member_ratio: float,
    device_pref: str,
    output_root: Path,
) -> None:
    """Train a single baseline model and save all artefacts."""

    set_seed(seed)
    device = get_device(device_pref)

    out_dir = output_root / dataset
    logger = ExperimentLogger(experiment_name=f"baseline_{dataset}", output_dir=output_root)
    logger.log_system_info()

    print(f"\n{'='*60}")
    print(f"Training baseline — {dataset} | {arch} | {epochs} epochs")
    print(f"{'='*60}")

    # Data
    loaders = get_dataloaders(
        dataset, batch_size=batch_size, member_ratio=member_ratio,
        seed=seed, download=True,
    )

    # Model
    model = get_model(arch, dataset)
    model = model.to(device)
    params = count_parameters(model)
    print(f"Model: {arch} | Params: {params['total']:,}")
    logger.log_model_summary(model)

    # Optimizer
    optimizer = build_optimizer(model, optimizer_name, lr=lr, weight_decay=weight_decay)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=loaders["train_loader"],
        val_loader=loaders["test_loader"],
        optimizer=optimizer,
        device=device,
    )

    history = trainer.train(epochs=epochs, patience=10, checkpoint_dir=out_dir)

    # Final evaluation
    final = trainer.evaluate(loaders["test_loader"])
    print(f"\nFinal test — loss: {final['loss']:.4f}  acc: {final['accuracy']:.2%}")

    # Save member indices (critical for MIA evaluation)
    np.save(out_dir / "member_indices.npy", loaders["member_indices"])
    np.save(out_dir / "nonmember_indices.npy", loaders["nonmember_indices"])

    # Save training config + final metrics
    meta = {
        "dataset": dataset,
        "arch": arch,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer": optimizer_name,
        "seed": seed,
        "member_ratio": member_ratio,
        "final_test_loss": final["loss"],
        "final_test_accuracy": final["accuracy"],
        "total_params": params["total"],
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

    # Log to experiment logger
    logger.log_metrics(final, step=epochs)
    logger.save_metrics()

    print(f"Artefacts saved to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models for AuditML")
    parser.add_argument("-c", "--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset (mnist/cifar10/cifar100)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--all", action="store_true", help="Train on all three datasets")
    parser.add_argument("--output", type=str, default="models/baselines", help="Output root directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(args.output)

    datasets = ALL_DATASETS if args.all else [args.dataset or cfg.data.dataset.value]

    for ds in datasets:
        train_one(
            dataset=ds,
            arch=cfg.model.arch,
            epochs=args.epochs or cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.training.learning_rate,
            optimizer_name=cfg.training.optimizer,
            weight_decay=cfg.training.weight_decay,
            seed=cfg.training.seed,
            member_ratio=cfg.data.train_ratio,
            device_pref=cfg.training.device,
            output_root=output_root,
        )

    print("\nAll baseline training complete.")


if __name__ == "__main__":
    main()
