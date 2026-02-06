# AuditML

A privacy auditing toolkit for PyTorch machine learning models.

AuditML provides implementations of privacy attacks and defenses for evaluating
the privacy risks of trained models. It is designed for researchers and
practitioners who need to quantify how much private information a model leaks.

## Planned Features

- **Membership Inference Attacks** — threshold-based and shadow-model-based
- **Model Inversion Attacks** — reconstruct training data from model access
- **Attribute Inference Attacks** — infer sensitive attributes of training records
- **Differential Privacy Defenses** — training with Opacus (DP-SGD)
- **Unified CLI** — run audits from the command line
- **Report Generation** — produce structured audit reports

## Installation

```bash
# Clone the repository
git clone https://github.com/EemanAsghar/AuditML.git
cd AuditML

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"
```

## Project Structure

```
src/auditml/
├── attacks/      # Privacy attack implementations
├── defenses/     # Differential privacy and other defenses
├── models/       # Target and shadow model definitions
├── data/         # Dataset loading and preprocessing
├── training/     # Training loops and evaluation
├── config/       # Configuration handling
├── reporting/    # Audit report generation
└── utils/        # Shared utilities
```

## Usage

Usage instructions will be added as features are implemented.

## License

MIT
