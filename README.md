# AuditML

**Privacy Auditing Toolkit for PyTorch Models**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blueviolet)](https://eemanasghar.github.io/AuditML-Privacy-Toolkit/)

AuditML lets you measure how much private information leaks from a trained PyTorch model.
One function call audits your model for membership inference, model inversion, and attribute
inference attacks — with an interactive HTML report that opens automatically in your browser.

---

## Features

| | |
|---|---|
| **Threshold MIA** | Exploit loss/confidence/entropy gaps between members and non-members |
| **Shadow Model MIA** | Train surrogate models to build a membership classifier |
| **Model Inversion** | Reconstruct per-class images via gradient ascent |
| **Attribute Inference** | Predict sensitive attributes from model outputs |
| **DP Training** | Opacus DP-SGD with automatic (ε, δ) accounting |
| **HTML Reports** | Interactive browser report — charts, ROC curves, risk level — auto-opens after audit |
| **Rust acceleration** | 11× faster threshold scanning, 3× faster SSIM (optional) |

---

## Quick Start

```bash
pip install auditml
```

```python
import auditml

# Split your training set into members / non-members
member_loader, nonmember_loader = auditml.split_loaders(train_dataset)

# Audit your model — works with any nn.Module
results = auditml.audit(model, member_loader, nonmember_loader)

print(results.summary())
# ⚠  Leakage detected — highest AUC: 0.641 (mia_threshold)

# Open an interactive HTML report in your browser
results.report("./report", open_browser=True)

# Save results to reload later
results.save("audit_results.json")
results2 = auditml.AuditResults.load("audit_results.json")
```

---

## Installation

```bash
pip install auditml
```

Or from source:

```bash
git clone https://github.com/EemanAsghar/AuditML-Privacy-Toolkit.git
cd AuditML-Privacy-Toolkit
pip install -e ".[dev]"
```

### Optional: Rust extension (~11× speedup)

```bash
pip install maturin
cd rust && maturin build --release --out ../dist
pip install ../dist/auditml_rust-*.whl --force-reinstall
```

---

## Python API

```python
import auditml
from torch.utils.data import DataLoader

# 1. Split dataset into members / non-members
member_loader, nonmember_loader = auditml.split_loaders(
    train_dataset,
    member_ratio=0.5,   # 50/50 split
    batch_size=64,
    seed=42,
)

# 2. Run all attacks (or pick specific ones)
results = auditml.audit(
    model,
    member_loader,
    nonmember_loader,
    attacks=["mia_threshold", "model_inversion"],  # omit for all 4
    device="auto",   # auto | cpu | cuda | mps
)

# 3. Inspect results
print(results.summary())
results["mia_threshold"].auc_roc   # → 0.641
results.most_vulnerable()          # → AttackSummary(mia_threshold: ...)
results.is_vulnerable()            # → True

# 4. HTML report (auto-opens in browser)
results.report("./my_report", open_browser=True)

# 5. Save / reload without re-running
results.save("results.json")
results2 = auditml.AuditResults.load("results.json")
```

### Shadow MIA with a custom architecture

```python
results = auditml.audit(
    model, member_loader, nonmember_loader,
    attacks=["mia_shadow"],
    shadow_model_fn=lambda: MyCNN(num_classes=10),  # optional — MLP fallback used if omitted
)
```

---

## CLI

```bash
# Train a model
auditml train --config configs/mnist_baseline.yaml

# Run a full privacy audit (opens HTML report automatically)
auditml audit --config configs/mnist_baseline.yaml

# Run specific attacks
auditml audit --config configs/mnist_baseline.yaml --attack mia_threshold model_inversion

# Print resolved config as JSON
auditml show-config --config configs/mnist_baseline.yaml
```

---

## Config format

```yaml
experiment_name: mnist_baseline

data:
  dataset: mnist          # mnist | cifar10 | cifar100
  train_ratio: 0.5

model:
  arch: cnn               # cnn | resnet

training:
  epochs: 30
  batch_size: 64
  learning_rate: 0.001
  device: auto            # auto | cpu | cuda | mps

attacks:
  - mia_threshold

dp:
  enabled: false
  epsilon: 5.0
  delta: 1.0e-5
  max_grad_norm: 1.0

reporting:
  output_dir: ./outputs
```

---

## Project structure

```
AuditML/
├── src/auditml/
│   ├── attacks/           # MIA, shadow, model inversion, attribute inference
│   ├── config/            # YAML schema → typed dataclasses
│   ├── data/              # Dataset loaders (MNIST, CIFAR-10, CIFAR-100)
│   ├── models/            # CNN + ResNet architectures
│   ├── training/          # Standard trainer + Opacus DP trainer
│   ├── reporting/         # Report generator, HTML report, comparison modules
│   └── utils/             # Device detection, Rust acceleration, logging
├── rust/                  # Rust/PyO3 extension (optional)
├── configs/               # Example YAML configs
├── scripts/               # Experiment runners
├── benchmarks/            # Rust vs NumPy benchmark
├── tests/                 # pytest suite — 380 tests
└── docs/                  # MkDocs documentation
```

---

## Benchmark: Rust acceleration

```
find_best_threshold  (N=10,000)
  NumPy   : 160.00 ms
  Rust    :  14.00 ms
  Speedup : 11.4x  ✅

compute_ssim  (pixels=784)
  NumPy   :  21.0 µs
  Rust    :   7.0 µs
  Speedup :  3.0x  ✅
```

---

## Documentation

Full documentation at **[eemanasghar.github.io/AuditML-Privacy-Toolkit](https://eemanasghar.github.io/AuditML-Privacy-Toolkit/)**

---

## License

MIT © Eeman Asghar,2026
