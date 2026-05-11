# Running AuditML on Google Colab

Google Colab provides free T4/A100 GPUs that are useful for CIFAR-10/100 experiments.

## Setup

Open a new Colab notebook and run:

```python
# Install AuditML and dependencies
!pip install auditml opacus torchvision maturin

# Clone the repo (to get configs and scripts)
!git clone https://github.com/EemanAsghar/AuditML.git
%cd AuditML
```

## Check GPU availability

```python
import torch
print(torch.cuda.is_available())          # True on Colab GPU
print(torch.cuda.get_device_name(0))      # Tesla T4
```

## CIFAR-10 experiment

```python
# Run with the CIFAR-10 config
!auditml train --config configs/audit_cifar10.yaml
!auditml audit --config configs/audit_cifar10.yaml --attack mia_threshold
```

Or use the full experiment script:

```python
!python scripts/run_all_attacks.py --full
```

This runs all 3 datasets × 4 privacy levels × 4 attacks and writes a master CSV.

## Mounting Google Drive (optional)

Save results to your Drive so they persist after the session ends:

```python
from google.colab import drive
drive.mount('/content/drive')

# Update output path in config
import yaml

with open("configs/audit_cifar10.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["output"]["results_dir"] = "/content/drive/MyDrive/AuditML/results"

with open("configs/audit_cifar10_colab.yaml", "w") as f:
    yaml.dump(cfg, f)
```

## Transfer results back to local machine

```python
from google.colab import files

# Zip results
!zip -r results.zip results/

# Download
files.download("results.zip")
```

## Colab-specific tips

- Use `num_workers: 0` in your YAML config on Colab (DataLoader workers can cause issues).
- Colab sessions disconnect after ~90 minutes of inactivity — use Drive to checkpoint.
- For long runs, use Colab Pro or run `scripts/run_all_attacks.py` in quick mode first.

## Quick mode vs full mode

```python
# Quick mode (default) — 1 epoch, fast sanity check
!python scripts/run_all_attacks.py

# Full mode — production-quality results
!python scripts/run_all_attacks.py --full
```

Quick mode is useful to verify everything works before committing to a long full run.
