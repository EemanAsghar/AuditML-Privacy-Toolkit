# Installation

## Requirements

- Python 3.10+
- PyTorch 2.x
- macOS (MPS), Linux (CUDA), or CPU

## Install from PyPI

```bash
pip install auditml
```

## Install from source

```bash
git clone https://github.com/EemanAsghar/AuditML.git
cd AuditML
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, ruff, and mkdocs tooling.

## Optional: Rust acceleration

AuditML ships a Rust extension that speeds up threshold scanning (~11× faster) and SSIM computation (~3× faster). It falls back to NumPy automatically if unavailable.

### Prerequisites

1. Install Rust (one-time, global):

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source "$HOME/.cargo/env"
    ```

2. Install maturin inside your virtual environment:

    ```bash
    pip install maturin
    ```

### Build and install the extension

```bash
cd rust
maturin build --release --out ../dist
pip install ../dist/auditml_rust-*.whl --force-reinstall
```

To verify:

```python
from auditml.utils.rust_accel import rust_status
print(rust_status())
# Rust extension available (auditml_rust vX.Y.Z)
```

## Verify the full install

```bash
python -c "import auditml; print(auditml.__version__)"
auditml --help
```

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model training and inference |
| `torchvision` | Datasets and transforms |
| `opacus` | Differential Privacy |
| `scikit-learn` | Metrics (AUC, F1, …) |
| `numpy` | Array operations |
| `matplotlib` | Plots |
| `tqdm` | Progress bars |
| `pyyaml` | Config parsing |
