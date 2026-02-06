#!/usr/bin/env python
"""Verify that the AuditML development environment is correctly set up.

Run with::

    python scripts/verify_env.py
"""

from __future__ import annotations

import sys


def main() -> None:
    print("=" * 60)
    print("AuditML — Environment Verification")
    print("=" * 60)

    # Python -----------------------------------------------------------
    print(f"\nPython  : {sys.version}")

    # PyTorch ----------------------------------------------------------
    try:
        import torch

        print(f"PyTorch : {torch.__version__}")
    except ImportError:
        sys.exit("ERROR: PyTorch is not installed.")

    # CUDA / MPS -------------------------------------------------------
    if torch.cuda.is_available():
        print(f"CUDA    : {torch.version.cuda}")
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"GPU Mem : {mem_gb:.1f} GB")
        t = torch.randn(2, 3, device="cuda")
        print(f"GPU test: tensor on cuda OK  (shape {tuple(t.shape)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS     : available")
        t = torch.randn(2, 3, device="mps")
        print(f"MPS test: tensor on mps OK   (shape {tuple(t.shape)})")
    else:
        print("GPU     : not available (CPU only)")
        t = torch.randn(2, 3)
        print(f"CPU test: tensor on cpu OK   (shape {tuple(t.shape)})")

    # Key dependencies -------------------------------------------------
    deps = [
        "torchvision", "numpy", "pandas", "sklearn", "opacus",
        "matplotlib", "seaborn", "yaml", "click", "tqdm",
    ]
    print("\nDependencies:")
    all_ok = True
    for name in deps:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "ok")
            print(f"  {name:14s} {ver}")
        except ImportError:
            print(f"  {name:14s} MISSING")
            all_ok = False

    # AuditML itself ---------------------------------------------------
    try:
        import auditml

        print(f"\nauditml : {auditml.__version__}")
    except ImportError:
        print("\nauditml : NOT INSTALLED (run: pip install -e .)")
        all_ok = False

    # Summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed.")
    else:
        print("Some checks failed — see above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
