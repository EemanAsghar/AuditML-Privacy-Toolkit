"""Rust acceleration helpers with transparent NumPy fallback.

AuditML ships an optional Rust extension (``auditml_rust``) that provides
10-15x speedups for two compute-intensive operations:

- ``find_best_threshold`` — scans all possible thresholds to find the one
  that maximises binary classification accuracy (used by ThresholdMIA).
- ``compute_ssim`` / ``batch_ssim`` — Structural Similarity Index between
  pixel arrays (used by ModelInversion to measure reconstruction quality).

If the Rust extension is not available (e.g. fresh clone without building),
this module falls back to equivalent NumPy implementations automatically.
No code outside this file needs to change.

Build the Rust extension
------------------------
    cd rust/
    maturin build --release --out ../dist
    pip install ../dist/auditml_rust-*.whl

Or for development (rebuilds on every change):
    maturin develop --release   # must run from rust/ directory
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Try to import the compiled Rust extension
# ---------------------------------------------------------------------------

try:
    import auditml_rust as _rust
    RUST_AVAILABLE = True
except ImportError:
    _rust = None  # type: ignore[assignment]
    RUST_AVAILABLE = False


def rust_status() -> str:
    """Return a human-readable string indicating whether Rust acceleration is active."""
    if RUST_AVAILABLE:
        return "Rust acceleration: ENABLED (auditml_rust)"
    return "Rust acceleration: DISABLED (using NumPy fallback)"


# ---------------------------------------------------------------------------
# find_best_threshold
# ---------------------------------------------------------------------------

def find_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Find the threshold on *scores* that maximises binary accuracy.

    Parameters
    ----------
    scores:
        1-D float array of per-sample confidence scores.
    labels:
        1-D int array of ground-truth binary labels (1=member, 0=non-member).

    Returns
    -------
    (best_threshold, best_accuracy)
    """
    if RUST_AVAILABLE:
        return _rust.find_best_threshold(
            scores.astype(float).tolist(),
            labels.astype(int).tolist(),
        )
    return _find_best_threshold_numpy(scores, labels)


def _find_best_threshold_numpy(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Pure NumPy fallback for find_best_threshold."""
    thresholds = np.sort(np.unique(scores))
    best_acc = 0.0
    best_thresh = thresholds[0] - 1.0

    n = len(labels)
    for t in thresholds:
        preds = (scores > t).astype(int)
        acc = float(np.sum(preds == labels) / n)
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(t)

    return best_thresh, best_acc


# ---------------------------------------------------------------------------
# compute_ssim
# ---------------------------------------------------------------------------

def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute SSIM between two images (flat float arrays, values in [0,1]).

    Parameters
    ----------
    img_a:
        Reference image, flattened to 1-D, pixel values in [0, 1].
    img_b:
        Reconstructed image, same shape as img_a.

    Returns
    -------
    float
        SSIM in [-1, 1]. 1.0 = identical.
    """
    if RUST_AVAILABLE:
        return float(_rust.compute_ssim(
            img_a.astype(float).flatten().tolist(),
            img_b.astype(float).flatten().tolist(),
        ))
    return _compute_ssim_numpy(img_a.flatten(), img_b.flatten())


def _compute_ssim_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Pure NumPy fallback for compute_ssim."""
    mu_a, mu_b = float(np.mean(a)), float(np.mean(b))
    var_a = float(np.var(a))
    var_b = float(np.var(b))
    cov = float(np.mean((a - mu_a) * (b - mu_b)))

    c1, c2 = 0.0001, 0.0009
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(num / den) if den != 0 else 0.0


# ---------------------------------------------------------------------------
# batch_ssim
# ---------------------------------------------------------------------------

def batch_ssim(
    originals: np.ndarray | list[np.ndarray],
    reconstructed: np.ndarray | list[np.ndarray],
) -> list[float]:
    """Compute SSIM for a batch of (reference, reconstructed) pairs.

    Parameters
    ----------
    originals:
        List of reference images (each a flat float array).
    reconstructed:
        List of reconstructed images (same length as originals).

    Returns
    -------
    list[float]
        SSIM score per pair.
    """
    if RUST_AVAILABLE:
        return _rust.batch_ssim(
            [np.asarray(a).astype(float).flatten().tolist() for a in originals],
            [np.asarray(b).astype(float).flatten().tolist() for b in reconstructed],
        )
    return [
        _compute_ssim_numpy(
            np.asarray(a).flatten(),
            np.asarray(b).flatten(),
        )
        for a, b in zip(originals, reconstructed)
    ]
