# Quick Start

Run your first privacy audit in five minutes.

## 1. Pick a config

AuditML ships example configs in `configs/`. The simplest is `audit_mnist.yaml`:

```yaml
experiment_name: mnist_baseline
data:
  dataset: mnist
  train_size: 2500
  test_size: 2500
  batch_size: 64
model:
  architecture: cnn_small
  num_classes: 10
training:
  epochs: 10
  learning_rate: 0.001
  device: auto
attack_params:
  mia_threshold:
    metric: loss
output:
  results_dir: results/quickstart
```

## 2. Train a model

```bash
auditml train --config configs/audit_mnist.yaml
```

The checkpoint is saved to `results/quickstart/model.pth`.

## 3. Run an attack

```bash
auditml audit \
  --config configs/audit_mnist.yaml \
  --attack mia_threshold \
  --output results/quickstart/mia
```

This produces:
- `metrics.json` — accuracy, AUC-ROC, precision, recall, F1
- `roc_curve.png` — ROC curve
- `score_distributions.png` — member vs non-member score histograms
- `summary.txt` — human-readable report

## 4. Python API

```python
import auditml

member_loader, nonmember_loader = auditml.split_loaders(train_dataset)

results = auditml.audit(model, member_loader, nonmember_loader)
print(results.summary())

# Open an interactive HTML report in your browser
results.report("./report", open_browser=True)

# Save and reload without re-running
results.save("results.json")
results2 = auditml.AuditResults.load("results.json")
```

## Next steps

- [Training Models](training.md) — DP training, epochs, learning rate
- [Privacy Attacks](attacks.md) — all four attacks with parameter tuning
- [Interpreting Results](interpretation.md) — what AUC 0.6 means
