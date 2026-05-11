# Training Models

## Standard training

```bash
auditml train --config configs/audit_mnist.yaml
```

Key config options:

```yaml
training:
  epochs: 20
  learning_rate: 0.001
  weight_decay: 1e-4
  device: auto        # auto | cpu | cuda | mps
  optimizer: adam     # adam | sgd
```

`device: auto` picks MPS on Apple Silicon, CUDA on Linux/Windows with a GPU, and CPU otherwise.

## DP training

Enable Differential Privacy by adding an `epsilon` value:

```yaml
training:
  epochs: 20
  learning_rate: 0.001
  dp:
    enabled: true
    epsilon: 3.0       # privacy budget
    delta: 1e-5        # failure probability
    max_grad_norm: 1.0 # per-sample gradient clipping
    noise_multiplier: null  # computed from epsilon/delta if null
```

The trainer uses [Opacus](https://opacus.ai/) under the hood. At the end of training, the achieved `(ε, δ)` is logged:

```
Achieved DP guarantee: ε=2.97, δ=1e-05
```

## Python API

```python
from auditml.config.schema import AuditMLConfig
from auditml.training.trainer import Trainer

config = AuditMLConfig.from_yaml("configs/audit_mnist.yaml")
trainer = Trainer(config, device="mps")
history = trainer.train()

# history is a list of dicts: [{"epoch": 1, "loss": ..., "val_acc": ...}, ...]
print(f"Final val accuracy: {history[-1]['val_acc']:.3f}")
```

For DP training, use `DPTrainer`:

```python
from auditml.training.dp_trainer import DPTrainer

trainer = DPTrainer(config, device="cuda")
trainer.train()
epsilon, delta = trainer.get_privacy_spent()
print(f"Privacy spent: ε={epsilon:.2f}, δ={delta}")
```

## Checkpoints

A checkpoint is saved automatically at the end of training:

```
results/<experiment_name>/model.pth
```

Contents:
```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": 20,
    "config": {...},
    "val_accuracy": 0.987,
    "val_loss": 0.043,
}
```

Load a checkpoint:

```python
import torch
ckpt = torch.load("results/mnist_baseline/model.pth")
model.load_state_dict(ckpt["model_state_dict"])
```

## Datasets

| Name | Input shape | Classes | Note |
|---|---|---|---|
| `mnist` | (1, 28, 28) | 10 | Fast, good for local dev |
| `cifar10` | (3, 32, 32) | 10 | Needs GPU for good accuracy |
| `cifar100` | (3, 32, 32) | 100 | Most challenging |

```yaml
data:
  dataset: mnist      # mnist | cifar10 | cifar100
  train_size: 5000    # members (half of training set)
  test_size: 5000     # non-members (test set)
  batch_size: 128
  num_workers: 0      # 0 on macOS, 2-4 on Linux
```
