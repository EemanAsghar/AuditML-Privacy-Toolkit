# Interpreting Results

## The baseline: random chance

A trivial attacker who guesses randomly achieves:
- **Accuracy = 50%** (with balanced member/non-member split)
- **AUC-ROC = 0.5**
- **AUC-PR = 0.5**

Any value significantly above these baselines indicates privacy leakage.

## AUC-ROC

The area under the Receiver Operating Characteristic curve. It measures the attack's ability to rank members above non-members regardless of threshold choice.

| AUC | Interpretation |
|---|---|
| 0.50 | No leakage (random guess) |
| 0.55 | Marginal leakage |
| 0.60 | Moderate leakage — investigate |
| 0.70 | Significant leakage |
| 0.80+ | Severe leakage |

## TPR at low FPR

`tpr_at_1fpr` is the True Positive Rate when the False Positive Rate is capped at 1%. This is the **most practically important** metric because a real attacker accepts few false alarms.

- **TPR@1%FPR > 10%** → serious concern, especially for medical or financial data.
- **TPR@0.1%FPR** — used in the strictest privacy threat models.

## Accuracy

Attack accuracy above 60% on a balanced split is a warning sign. However, a 60% accurate attack can still be dangerous if it achieves high TPR at low FPR.

## Per-class breakdown

Some classes leak more than others. Threshold MIA generates `per_class_metrics.json`. Look for:
- Classes with unusually high attack AUC → the model memorised those samples more.
- Small, rare classes tend to have higher per-class AUC.

## Model inversion quality

The SSIM (Structural Similarity Index) scores in the Model Inversion report measure how diverse the reconstructions are relative to the mean:
- SSIM near 1.0 → all reconstructions look the same (poor inversion).
- SSIM near 0.5 → distinct reconstructions per class (good inversion, higher privacy risk).

## Comparing DP vs no DP

Use the DP comparison module:

```python
from auditml.reporting.dp_comparison import DPComparison

cmp = DPComparison(
    baseline_dir="results/mnist_no_dp",
    dp_dir="results/mnist_eps3",
)
cmp.generate_report("results/dp_comparison")
```

A well-configured DP model should show:
- Attack AUC drops closer to 0.5
- Some accuracy loss (typically 2–5% on MNIST, 5–15% on CIFAR)

## Practical guidance

1. **Start with Threshold MIA** — cheapest, good baseline.
2. **If AUC > 0.6**, try Shadow MIA for a stronger signal.
3. **Enable DP training** with ε=3–5 as a first mitigation.
4. **Re-audit after mitigation** to confirm the leakage dropped.
5. **Check TPR@1%FPR**, not just AUC, before declaring the model safe.
