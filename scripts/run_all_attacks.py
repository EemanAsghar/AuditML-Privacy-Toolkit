"""Run all privacy attacks against baseline and DP-trained models.

Experiment matrix:
    datasets      : mnist, cifar10, cifar100
    privacy levels: baseline, eps=10, eps=1, eps=0.1
    attacks       : mia_threshold, mia_shadow, model_inversion, attribute_inference

Results are saved to:
    models/{dataset}/{privacy_level}/model.pt
    results/attacks/{dataset}/{privacy_level}/{attack_name}/results.json
    results/master_results.csv

Usage
-----
    # Quick run (fewer epochs, good for testing)
    python scripts/run_all_attacks.py

    # Full validation run
    python scripts/run_all_attacks.py --full

    # Single dataset
    python scripts/run_all_attacks.py --datasets mnist

    # Resume (skips already-completed experiments)
    python scripts/run_all_attacks.py  # reruns automatically skip done combos
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Make sure src/ is importable when run directly
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

import platform

from auditml.attacks.attribute_inference import AttributeInference
from auditml.attacks.mia_shadow import ShadowMIA
from auditml.attacks.mia_threshold import ThresholdMIA
from auditml.attacks.model_inversion import ModelInversion
from auditml.config.schema import (
    AttackConfig,
    AuditMLConfig,
    DataConfig,
    DatasetName,
    DPConfig,
    ModelConfig,
    ReportingConfig,
    TrainingConfig,
)
from auditml.data.datasets import (
    DATASET_INFO,
    get_dataloaders,
    get_dataset,
)
from auditml.models import get_model
from auditml.training.dp_trainer import DPTrainer, validate_and_fix_model
from auditml.training.trainer import Trainer, build_optimizer
from auditml.utils.device import get_device
from auditml.utils.reproducibility import set_seed

# macOS multiprocessing with DataLoader requires num_workers=0
_NUM_WORKERS = 0 if platform.system() == "Darwin" else 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

DATASETS = ["mnist", "cifar10", "cifar100"]

PRIVACY_LEVELS: dict[str, float | None] = {
    "baseline": None,
    "eps10": 10.0,
    "eps1": 1.0,
    "eps01": 0.1,
}

ATTACKS = ["mia_threshold", "mia_shadow", "model_inversion", "attribute_inference"]

# Quick-mode settings (fast, suitable for dev/CI)
QUICK_SETTINGS = {
    "mnist":    {"epochs": 5,  "batch_size": 128, "shadow_epochs": 3,  "n_shadows": 2, "inv_iters": 200},   # noqa: E501
    "cifar10":  {"epochs": 8,  "batch_size": 128, "shadow_epochs": 5,  "n_shadows": 2, "inv_iters": 300},   # noqa: E501
    "cifar100": {"epochs": 8,  "batch_size": 128, "shadow_epochs": 5,  "n_shadows": 2, "inv_iters": 300},   # noqa: E501
}

# Full-mode settings (as per project plan)
FULL_SETTINGS = {
    "mnist":    {"epochs": 20, "batch_size": 64, "shadow_epochs": 20, "n_shadows": 5, "inv_iters": 1000},  # noqa: E501
    "cifar10":  {"epochs": 20, "batch_size": 64, "shadow_epochs": 20, "n_shadows": 5, "inv_iters": 1000},  # noqa: E501
    "cifar100": {"epochs": 20, "batch_size": 64, "shadow_epochs": 20, "n_shadows": 5, "inv_iters": 1000},  # noqa: E501
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_config(
    dataset: str,
    epsilon: float | None,
    settings: dict,
) -> AuditMLConfig:
    """Build an AuditMLConfig for this (dataset, privacy_level) combination."""
    dp = DPConfig(
        enabled=epsilon is not None,
        epsilon=epsilon or 1.0,  # value unused when enabled=False
        delta=1e-5,
        max_grad_norm=1.0,
    )

    from auditml.config.schema import (
        MIAShadowConfig,
        MIAThresholdConfig,
        ModelInversionConfig,
    )

    attack_params = AttackConfig(
        mia_shadow=MIAShadowConfig(
            num_shadow_models=settings["n_shadows"],
            shadow_epochs=settings["shadow_epochs"],
        ),
        mia_threshold=MIAThresholdConfig(metric="loss"),
        model_inversion=ModelInversionConfig(
            num_iterations=settings["inv_iters"],
            learning_rate=0.01,
            lambda_tv=0.001,
        ),
    )

    num_classes = DATASET_INFO[dataset].num_classes
    arch = "cnn"  # SimpleCNN works for all three datasets

    return AuditMLConfig(
        experiment_name=f"{dataset}_{'eps' + str(epsilon) if epsilon else 'baseline'}",
        data=DataConfig(
            dataset=DatasetName(dataset),
            data_dir="./data",
            train_ratio=0.5,
            num_workers=_NUM_WORKERS,
        ),
        model=ModelConfig(arch=arch, num_classes=num_classes),
        training=TrainingConfig(
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            learning_rate=0.001,
            optimizer="adam",
            seed=42,
        ),
        dp=dp,
        attack_params=attack_params,
        reporting=ReportingConfig(output_dir="./results"),
    )


def train_or_load_model(
    dataset: str,
    privacy_label: str,
    epsilon: float | None,
    config: AuditMLConfig,
    dataloaders: dict,
    device: torch.device,
    model_dir: Path,
) -> torch.nn.Module:
    """Return a trained model, loading from checkpoint if it already exists."""
    ckpt_path = model_dir / "model.pt"

    model = get_model(arch=config.model.arch, dataset=dataset)

    if ckpt_path.exists():
        logger.info("Loading cached model from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Strip Opacus '_module.' prefix if present
        state = {
            k.replace("_module.", ""): v
            for k, v in ckpt["model_state_dict"].items()
        }
        model.load_state_dict(state, strict=False)
        model = model.to(device)
        return model

    logger.info(
        "Training %s model on %s (epsilon=%s)…",
        "DP" if epsilon is not None else "baseline",
        dataset,
        epsilon,
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    optimizer = build_optimizer(
        model,
        name=config.training.optimizer,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    if epsilon is not None:
        model = validate_and_fix_model(model)
        trainer = DPTrainer(
            model=model,
            train_loader=dataloaders["train_loader"],
            val_loader=dataloaders["test_loader"],
            optimizer=optimizer,
            dp_config=config.dp,
            device=device,
        )
    else:
        trainer = Trainer(
            model=model,
            train_loader=dataloaders["train_loader"],
            val_loader=dataloaders["test_loader"],
            optimizer=optimizer,
            device=device,
        )

    trainer.train(
        epochs=config.training.epochs,
        patience=config.training.epochs,  # no early stopping — run full epochs
        checkpoint_dir=model_dir,
    )

    val_metrics = trainer.evaluate(dataloaders["test_loader"])
    logger.info(
        "Trained %s/%s — val_acc=%.2f%%",
        dataset, privacy_label, val_metrics["accuracy"] * 100,
    )

    # Save extra metadata alongside the checkpoint
    meta = {"dataset": dataset, "privacy": privacy_label, "val_acc": val_metrics["accuracy"]}
    if epsilon is not None and hasattr(trainer, "get_epsilon"):
        meta["epsilon_achieved"] = trainer.get_epsilon()
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return trainer.model


def run_attack(
    attack_name: str,
    model: torch.nn.Module,
    config: AuditMLConfig,
    dataloaders: dict,
    full_train_dataset,
    device: torch.device,
    out_dir: Path,
) -> dict[str, float]:
    """Run a single attack and return its metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    if results_path.exists():
        logger.info("Skipping %s — results already exist at %s", attack_name, results_path)
        return json.loads(results_path.read_text())

    member_loader = dataloaders["member_loader"]
    nonmember_loader = dataloaders["nonmember_loader"]

    logger.info("Running attack: %s", attack_name)
    t0 = time.time()

    if attack_name == "mia_threshold":
        attack = ThresholdMIA(model, config, device=device)
        attack.run(member_loader, nonmember_loader)

    elif attack_name == "mia_shadow":
        attack = ShadowMIA(
            model, config, device=device, shadow_dataset=full_train_dataset
        )
        attack.run(member_loader, nonmember_loader)

    elif attack_name == "model_inversion":
        input_shape = DATASET_INFO[config.data.dataset.value].input_shape
        attack = ModelInversion(model, config, device=device, input_shape=input_shape)
        attack.run(member_loader, nonmember_loader)

    elif attack_name == "attribute_inference":
        attack = AttributeInference(model, config, device=device)
        attack.run(member_loader, nonmember_loader)

    else:
        raise ValueError(f"Unknown attack: {attack_name!r}")

    metrics = attack.evaluate()
    elapsed = time.time() - t0
    metrics["elapsed_seconds"] = round(elapsed, 1)

    results_path.write_text(json.dumps(metrics, indent=2))
    logger.info(
        "  %s done in %.0fs — acc=%.3f  auc=%.3f",
        attack_name, elapsed, metrics.get("accuracy", 0), metrics.get("auc_roc", 0),
    )
    return metrics


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiments(
    datasets: list[str],
    privacy_levels: list[str],
    attacks: list[str],
    settings_map: dict,
    device: torch.device,
) -> list[dict]:
    """Run the full experiment matrix and return all result rows."""
    results_root = Path("results/attacks")
    models_root = Path("models")

    all_rows: list[dict] = []

    total = len(datasets) * len(privacy_levels) * len(attacks)
    done = 0

    for dataset in datasets:
        full_train_dataset = get_dataset(dataset, train=True, data_dir="./data", download=True)
        settings = settings_map[dataset]

        for privacy_label, epsilon in privacy_levels.items():
            if privacy_label not in ["baseline", "eps10", "eps1", "eps01"]:
                continue

            set_seed(42)
            config = build_config(dataset, epsilon, settings)

            dataloaders = get_dataloaders(
                dataset_name=dataset,
                batch_size=settings["batch_size"],
                member_ratio=0.5,
                num_workers=_NUM_WORKERS,
                seed=42,
                data_dir="./data",
            )

            model_dir = models_root / dataset / privacy_label
            model = train_or_load_model(
                dataset, privacy_label, epsilon, config,
                dataloaders, device, model_dir,
            )
            model.eval()

            for attack_name in attacks:
                done += 1
                logger.info(
                    "[%d/%d] %s | %s | %s",
                    done, total, dataset, privacy_label, attack_name,
                )

                out_dir = results_root / dataset / privacy_label / attack_name
                try:
                    metrics = run_attack(
                        attack_name, model, config, dataloaders,
                        full_train_dataset, device, out_dir,
                    )
                except Exception as exc:
                    logger.error("FAILED %s/%s/%s: %s", dataset, privacy_label, attack_name, exc)
                    metrics = {"error": str(exc)}

                row = {
                    "dataset": dataset,
                    "privacy": privacy_label,
                    "epsilon": epsilon if epsilon is not None else "none",
                    "attack": attack_name,
                    **{k: v for k, v in metrics.items() if k != "elapsed_seconds"},
                }
                all_rows.append(row)

    return all_rows


def write_master_csv(rows: list[dict], path: Path) -> None:
    """Write all experiment results to a flat CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all column names (preserving order)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    logger.info("Master results written to %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all AuditML privacy attacks across datasets and privacy levels."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full training settings (more epochs, more shadows). Default is quick mode.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        choices=DATASETS, metavar="DATASET",
        help="Datasets to evaluate (default: all three).",
    )
    parser.add_argument(
        "--privacy", nargs="+", default=list(PRIVACY_LEVELS),
        choices=list(PRIVACY_LEVELS), metavar="LEVEL",
        help="Privacy levels to evaluate (default: all four).",
    )
    parser.add_argument(
        "--attacks", nargs="+", default=ATTACKS,
        choices=ATTACKS, metavar="ATTACK",
        help="Attacks to run (default: all four).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device string ('auto', 'cpu', 'cuda', 'mps').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    logger.info("Using device: %s", device)

    settings_map = FULL_SETTINGS if args.full else QUICK_SETTINGS
    mode = "FULL" if args.full else "QUICK"
    logger.info(
        "Mode: %s | datasets=%s | privacy=%s | attacks=%s",
        mode, args.datasets, args.privacy, args.attacks,
    )

    selected_privacy = {k: v for k, v in PRIVACY_LEVELS.items() if k in args.privacy}

    t_start = time.time()
    rows = run_experiments(
        datasets=args.datasets,
        privacy_levels=selected_privacy,
        attacks=args.attacks,
        settings_map=settings_map,
        device=device,
    )

    master_csv = Path("results/master_results.csv")
    write_master_csv(rows, master_csv)

    elapsed = time.time() - t_start
    logger.info("All experiments complete in %.0f seconds.", elapsed)

    # Print a quick summary table
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"{'dataset':<10} {'privacy':<10} {'attack':<22} {'accuracy':>8}  {'auc_roc':>7}")
    print("-" * 65)
    for row in rows:
        if "error" in row:
            err_snippet = row["error"][:30]
            ds, pv, atk = row["dataset"], row["privacy"], row["attack"]
            print(f"{ds:<10} {pv:<10} {atk:<22}  ERROR: {err_snippet}")
        else:
            acc = row.get("accuracy", "")
            auc = row.get("auc_roc", "")
            acc_str = f"{float(acc):.3f}" if acc != "" else "  N/A "
            auc_str = f"{float(auc):.3f}" if auc != "" else "  N/A "
            print(
                f"{row['dataset']:<10} {row['privacy']:<10} {row['attack']:<22}"
                f" {acc_str:>8}  {auc_str:>7}"
            )
    print()


if __name__ == "__main__":
    main()
