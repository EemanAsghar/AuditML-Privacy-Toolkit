# Tutorial: Differential Privacy Training

This tutorial shows how to train a model with DP-SGD and verify the privacy guarantee by running an MIA before and after.

## Prerequisites

```bash
pip install auditml opacus
```

## Understanding the parameters

| Parameter | Effect |
|---|---|
| `epsilon` | Lower = stronger privacy. Start at 5, tighten to 1. |
| `delta` | Set to `1/N` where N is training set size. For 5000 samples: `2e-4`. |
| `max_grad_norm` | Clip gradients per sample. 0.5–1.0 is typical. |
| `noise_multiplier` | Amount of Gaussian noise. Auto-computed if null. |

## Step 1: Baseline (no DP)

```yaml
# configs/dp_tutorial_baseline.yaml
experiment_name: dp_tutorial_baseline
data:
  dataset: mnist
  train_size: 5000
  batch_size: 256
training:
  epochs: 20
  learning_rate: 0.001
  device: auto
```

```bash
auditml train --config configs/dp_tutorial_baseline.yaml
auditml audit --config configs/dp_tutorial_baseline.yaml --attack mia_threshold
```

## Step 2: DP training (ε=5)

```yaml
# configs/dp_tutorial_eps5.yaml
experiment_name: dp_tutorial_eps5
data:
  dataset: mnist
  train_size: 5000
  batch_size: 256        # larger batch = better privacy per epoch
training:
  epochs: 20
  learning_rate: 0.001
  device: auto
  dp:
    enabled: true
    epsilon: 5.0
    delta: 2e-4
    max_grad_norm: 1.0
```

```bash
auditml train --config configs/dp_tutorial_eps5.yaml
auditml audit --config configs/dp_tutorial_eps5.yaml --attack mia_threshold
```

After training you'll see:

```
Achieved DP guarantee: ε=4.97, δ=2e-04
```

## Step 3: DP training (ε=1)

```yaml
training:
  dp:
    enabled: true
    epsilon: 1.0
    delta: 2e-4
    max_grad_norm: 0.5
```

Strong privacy at some accuracy cost.

## Step 4: Compare

```python
from auditml.reporting.dp_comparison import DPComparison

cmp = DPComparison(
    baseline_dir="results/dp_tutorial_baseline",
    dp_dir="results/dp_tutorial_eps5",
)
cmp.generate_report("results/dp_comparison")
```

Typical results on MNIST:

| Model | Val Acc | MIA AUC |
|---|---|---|
| No DP | 98.8% | 0.671 |
| ε=5 | 97.8% | 0.558 |
| ε=1 | 96.2% | 0.512 |

## Tips

- **Increase batch size** when using DP — larger batches dilute the noise.
- **Train longer** — DP models converge more slowly; 2–3× more epochs may be needed.
- **Tune `max_grad_norm`** — too small clips valid gradients, too large lets noise dominate.
- **Don't use Batch Normalisation** — it's incompatible with per-sample gradient computation.
