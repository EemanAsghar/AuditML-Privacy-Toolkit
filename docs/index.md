# AuditML

**AuditML** is a privacy auditing toolkit for PyTorch models. It lets you measure how much private information a trained model leaks through a suite of membership inference, model inversion, and attribute inference attacks — with optional Differential Privacy training to harden your model.

---

## What AuditML does

| Capability | Description |
|---|---|
| **Membership Inference** | Determines whether a given sample was in the training set |
| **Shadow Model MIA** | Trains surrogate models to build a membership classifier |
| **Model Inversion** | Reconstructs representative images for each class |
| **Attribute Inference** | Predicts sensitive attributes from model outputs |
| **DP Training** | Trains with Opacus to provide (ε, δ)-differential privacy |
| **Reporting** | Generates metrics, plots, and human-readable summaries |

---

## Quick example

```python
from auditml import AuditPipeline

pipeline = AuditPipeline.from_yaml("configs/audit_mnist.yaml")
results  = pipeline.run()
print(results.summary())
```

Or from the command line:

```bash
auditml train  --config configs/audit_mnist.yaml
auditml audit  --config configs/audit_mnist.yaml --attack mia_threshold
```

---

## Installation

```bash
pip install auditml
```

See [Installation](guide/installation.md) for the full setup guide, including the optional Rust acceleration module.

---

## Project layout

```
auditml/
├── src/auditml/
│   ├── attacks/        # MIA, shadow, model inversion, attribute inference
│   ├── config/         # YAML schema + typed dataclasses
│   ├── data/           # Dataset loaders (MNIST, CIFAR-10, CIFAR-100)
│   ├── models/         # CNN architectures
│   ├── training/       # Standard + DP trainer
│   ├── reporting/      # Report generator + comparison tools
│   └── utils/          # Device detection, Rust acceleration
├── configs/            # Example YAML configs
├── rust/               # Rust extension (optional speedup)
├── tests/              # pytest suite
└── docs/               # This documentation
```

---

## Navigation

- [Installation](guide/installation.md) — pip, Rust extension, dependencies
- [Quick Start](guide/quickstart.md) — run your first audit in 5 minutes
- [Training Models](guide/training.md) — standard and DP training options
- [Privacy Attacks](guide/attacks.md) — all four attack types explained
- [Differential Privacy](guide/differential_privacy.md) — ε, δ, and Opacus
- [Interpreting Results](guide/interpretation.md) — what the numbers mean
