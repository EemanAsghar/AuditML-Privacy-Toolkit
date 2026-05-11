# Privacy Attacks

AuditML implements four attacks. Each produces an `AttackResult` with standardised metrics so you can compare them directly.

---

## Threshold MIA

The simplest membership inference attack. It exploits the fact that a model assigns lower loss (or higher confidence) to samples it was trained on.

**How it works:**
1. Compute a signal (loss, confidence, or entropy) for every sample.
2. Sweep all unique signal values as potential thresholds.
3. Pick the threshold that maximises attack accuracy.

```yaml
attack_params:
  mia_threshold:
    metric: loss          # loss | confidence | entropy
    percentile: 50        # fallback if optimal scan disabled
```

```bash
auditml audit --config configs/audit_mnist.yaml --attack mia_threshold
```

**When to use:** Quick baseline. If AUC > 0.6 with this attack, your model is leaking.

---

## Shadow Model MIA

A stronger attack that trains several "shadow models" on data with known membership labels, then uses them to train a binary membership classifier.

```yaml
attack_params:
  mia_shadow:
    num_shadow_models: 4
    shadow_epochs: 10
```

**When to use:** More powerful than threshold MIA but requires 4× the training time.

---

## Model Inversion

Reconstructs representative images for each class by gradient ascent in pixel space. If the model has memorised training data, the reconstructions resemble actual training samples.

```yaml
attack_params:
  model_inversion:
    num_iterations: 500
    learning_rate: 0.1
    lambda_tv: 0.01       # Total Variation regularisation
    lambda_l2: 0.001      # L2 regularisation
    target_class: null    # null = all classes
```

**Output:** A grid of reconstructed images per class, plus SSIM quality scores.

---

## Attribute Inference

Predicts sensitive attributes (e.g., gender, age group) from the model's intermediate representations.

```yaml
attack_params:
  attribute_inference:
    target_attribute: label
    attack_epochs: 10
```

---

## Attack metrics

All attacks report the same set of metrics:

| Metric | Meaning |
|---|---|
| `accuracy` | Fraction of correct member/non-member predictions |
| `precision` | Of predicted members, fraction that are truly members |
| `recall` | Of true members, fraction correctly identified |
| `f1` | Harmonic mean of precision and recall |
| `auc_roc` | Area under the ROC curve (0.5 = random, 1.0 = perfect) |
| `auc_pr` | Area under the precision-recall curve |
| `tpr_at_1fpr` | True Positive Rate at 1% False Positive Rate |
| `tpr_at_01fpr` | True Positive Rate at 0.1% False Positive Rate |

See [Interpreting Results](interpretation.md) for guidance on what these numbers mean in practice.
