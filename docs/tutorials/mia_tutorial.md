# Tutorial: Membership Inference Attack on MNIST

This tutorial walks through a complete MIA experiment from scratch.

## Setup

```bash
pip install auditml
```

Create `configs/tutorial_mia.yaml`:

```yaml
experiment_name: tutorial_mia
data:
  dataset: mnist
  train_size: 5000
  test_size: 5000
  batch_size: 128
model:
  architecture: cnn_small
  num_classes: 10
training:
  epochs: 15
  learning_rate: 0.001
  device: auto
attack_params:
  mia_threshold:
    metric: loss
output:
  results_dir: results/tutorial_mia
```

## Step 1: Train the model

```bash
auditml train --config configs/tutorial_mia.yaml
```

Expected output:

```
Epoch 15/15 — loss=0.034  val_acc=0.988
Checkpoint saved → results/tutorial_mia/model.pth
```

## Step 2: Run Threshold MIA

```bash
auditml audit \
  --config configs/tutorial_mia.yaml \
  --attack mia_threshold
```

## Step 3: Inspect the results

```bash
cat results/tutorial_mia/mia_threshold/summary.txt
```

```
AuditML — Threshold MIA Report
============================================================
Metric used:     loss
Threshold:       0.041230
Total samples:   10000
  Members:       5000
  Non-members:   5000

--- Overall Metrics ---
  accuracy             : 0.6340
  auc_roc              : 0.6712
  tpr_at_1fpr          : 0.1420
```

## Step 4: Understand what happened

The model achieved 98.8% test accuracy — high accuracy often correlates with memorisation. The MIA AUC of 0.67 confirms significant leakage.

Open the score distribution plot:

```bash
open results/tutorial_mia/mia_threshold/score_distributions.png
```

You should see two overlapping but separated distributions: members cluster at lower loss values.

## Step 5: Try a different metric

```yaml
attack_params:
  mia_threshold:
    metric: confidence   # instead of loss
```

Confidence and entropy attacks often give similar results; comparing them helps identify which signal is strongest for your model.

## Step 6: Mitigate with DP

Add to your config:

```yaml
training:
  dp:
    enabled: true
    epsilon: 3.0
    delta: 1e-5
    max_grad_norm: 1.0
```

Re-train and re-run the attack. You should see AUC drop toward 0.5.

## Summary

| Experiment | Val Acc | MIA AUC |
|---|---|---|
| No DP | 98.8% | 0.671 |
| DP ε=3 | 97.1% | 0.524 |

A small accuracy cost provides strong privacy protection.
