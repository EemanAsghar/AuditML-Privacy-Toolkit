# Differential Privacy

## What is Differential Privacy?

Differential Privacy (DP) is a mathematical guarantee that an algorithm's output doesn't reveal whether any single individual's data was included in the training set. The guarantee is parameterised by:

- **ε (epsilon)** — privacy budget. Smaller = more private, usually at the cost of accuracy.
- **δ (delta)** — failure probability. Typically set to `1/N` where `N` is the dataset size.

A model trained with DP-SGD satisfies `(ε, δ)`-DP, meaning an adversary gains at most `e^ε` additional confidence that a given sample was in the training set.

Common ε guidelines:
| ε | Privacy level |
|---|---|
| < 1 | Strong (used in production systems) |
| 1–10 | Moderate (research baseline) |
| > 10 | Weak (near no protection) |

## How AuditML implements DP

AuditML uses [Opacus](https://opacus.ai/), which wraps a standard PyTorch optimiser and:

1. **Clips per-sample gradients** to `max_grad_norm`.
2. **Adds Gaussian noise** scaled by `noise_multiplier` to the summed gradient.
3. **Tracks the privacy budget** using the Moments Accountant (RDP → (ε, δ)).

## Configuration

```yaml
training:
  dp:
    enabled: true
    epsilon: 5.0            # target privacy budget
    delta: 1e-5             # failure probability (≪ 1/N)
    max_grad_norm: 1.0      # per-sample gradient clip norm
    noise_multiplier: null  # auto-computed if null
```

When `noise_multiplier` is `null`, Opacus computes the multiplier required to achieve the target `epsilon` over the configured number of epochs.

## Python API

```python
from auditml.training.dp_trainer import DPTrainer

trainer = DPTrainer(config, device="cuda")
trainer.train()

epsilon, delta = trainer.get_privacy_spent()
print(f"Achieved ε={epsilon:.2f}, δ={delta:.2e}")
```

## Privacy-utility trade-off

A key insight: **DP training reduces overfitting**, which in turn reduces MIA success. You can verify this directly:

```bash
# Non-DP model
auditml train  --config configs/audit_mnist.yaml
auditml audit  --config configs/audit_mnist.yaml --attack mia_threshold

# DP model (ε=3)
auditml train  --config configs/audit_mnist_dp.yaml
auditml audit  --config configs/audit_mnist_dp.yaml --attack mia_threshold
```

Typical result: MIA AUC drops from ~0.65 (no DP) to ~0.52 (ε=3).

## Limitations

- **Batch Normalisation** is incompatible with per-sample gradients — use Group Norm instead (AuditML's `cnn_small` uses no batch norm for this reason).
- **Larger batches** provide better privacy per epoch but require more memory.
- **The ε bound is worst-case** — practical leakage is usually lower.

See [DP vs Non-DP Comparison](../api/reporting.md) in the API reference for how to compare results programmatically.
