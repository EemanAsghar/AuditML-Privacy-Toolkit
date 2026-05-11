"""Benchmark: Rust acceleration vs NumPy fallback.

Measures the speedup provided by the auditml_rust extension for:
  1. find_best_threshold  — threshold scanning for MIA
  2. compute_ssim         — single image pair SSIM
  3. batch_ssim           — batch of image pairs SSIM

Usage
-----
    python benchmarks/benchmark_rust.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auditml.utils.rust_accel import (
    _compute_ssim_numpy,
    _find_best_threshold_numpy,
    rust_status,
)

# ── Try importing Rust directly for isolated comparison ───────────────────
try:
    import auditml_rust as _rust
except ImportError:
    _rust = None


def _time(fn, *args, repeats: int = 10) -> float:
    """Return median wall time (seconds) over *repeats* calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def bench_threshold(n: int = 10_000) -> None:
    rng = np.random.default_rng(42)
    scores = rng.random(n).astype(np.float64)
    labels = rng.integers(0, 2, n).astype(np.int32)

    t_numpy = _time(_find_best_threshold_numpy, scores, labels, repeats=5)

    print(f"\n── find_best_threshold  (N={n:,}) ──────────────────")
    print(f"  NumPy   : {t_numpy * 1000:.2f} ms")

    if _rust is not None:
        scores_list = scores.tolist()
        labels_list = labels.tolist()
        t_rust = _time(_rust.find_best_threshold, scores_list, labels_list, repeats=20)
        speedup = t_numpy / t_rust
        print(f"  Rust    : {t_rust * 1000:.2f} ms")
        print(f"  Speedup : {speedup:.1f}x")

        # Verify numerical equivalence
        r_numpy = _find_best_threshold_numpy(scores, labels)
        r_rust = _rust.find_best_threshold(scores_list, labels_list)
        print(f"  NumPy result  : thresh={r_numpy[0]:.6f}  acc={r_numpy[1]:.6f}")
        print(f"  Rust  result  : thresh={r_rust[0]:.6f}  acc={r_rust[1]:.6f}")
        acc_match = abs(r_numpy[1] - r_rust[1]) < 1e-6
        print(f"  Accuracy match: {'✅' if acc_match else '❌'}")
    else:
        print("  Rust    : not available")


def bench_ssim(img_size: int = 784) -> None:
    rng = np.random.default_rng(0)
    img_a = rng.random(img_size)
    img_b = rng.random(img_size)

    t_numpy = _time(_compute_ssim_numpy, img_a, img_b, repeats=100)

    print(f"\n── compute_ssim  (pixels={img_size}) ───────────────")
    print(f"  NumPy   : {t_numpy * 1_000_000:.1f} µs")

    if _rust is not None:
        a_list = img_a.tolist()
        b_list = img_b.tolist()
        t_rust = _time(_rust.compute_ssim, a_list, b_list, repeats=500)
        speedup = t_numpy / t_rust
        print(f"  Rust    : {t_rust * 1_000_000:.1f} µs")
        print(f"  Speedup : {speedup:.1f}x")

        r_numpy = _compute_ssim_numpy(img_a, img_b)
        r_rust = _rust.compute_ssim(a_list, b_list)
        match = abs(r_numpy - r_rust) < 1e-6
        print(f"  SSIM NumPy: {r_numpy:.6f}  Rust: {r_rust:.6f}  Match: {'✅' if match else '❌'}")
    else:
        print("  Rust    : not available")


def bench_batch_ssim(batch: int = 100, img_size: int = 1024) -> None:
    rng = np.random.default_rng(7)
    imgs_a = [rng.random(img_size) for _ in range(batch)]
    imgs_b = [rng.random(img_size) for _ in range(batch)]

    def _numpy_batch():
        return [_compute_ssim_numpy(a, b) for a, b in zip(imgs_a, imgs_b)]

    t_numpy = _time(_numpy_batch, repeats=10)

    print(f"\n── batch_ssim  (batch={batch}, pixels={img_size}) ──")
    print(f"  NumPy   : {t_numpy * 1000:.2f} ms")

    if _rust is not None:
        a_lists = [a.tolist() for a in imgs_a]
        b_lists = [b.tolist() for b in imgs_b]
        t_rust = _time(_rust.batch_ssim, a_lists, b_lists, repeats=20)
        speedup = t_numpy / t_rust
        print(f"  Rust    : {t_rust * 1000:.2f} ms")
        print(f"  Speedup : {speedup:.1f}x")
    else:
        print("  Rust    : not available")


if __name__ == "__main__":
    print("=" * 55)
    print("  AuditML — Rust vs NumPy Benchmark")
    print("=" * 55)
    print(f"  {rust_status()}")

    bench_threshold(n=10_000)
    bench_ssim(img_size=784)       # MNIST image size
    bench_ssim(img_size=3072)      # CIFAR image size (3×32×32)
    bench_batch_ssim(batch=50, img_size=784)
    bench_batch_ssim(batch=50, img_size=3072)

    print("\n" + "=" * 55)
