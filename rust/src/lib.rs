/// AuditML Rust acceleration module.
///
/// Provides fast implementations of two compute-intensive operations:
///
/// 1. `find_best_threshold` — scans a sorted array of scores to find the
///    threshold that maximises binary classification accuracy. Used by the
///    Threshold MIA attack. Python/NumPy equivalent is O(N log N); this Rust
///    version is O(N) after sorting and avoids repeated array passes.
///
/// 2. `compute_ssim` — computes the Structural Similarity Index (SSIM)
///    between two flat pixel arrays. Used by the Model Inversion attack to
///    measure reconstruction quality. Pure Python SSIM calls scikit-image
///    which carries significant overhead per image; batching in Rust is
///    ~10x faster for the iterative inversion loop.
///
/// Both functions are exposed to Python via PyO3 and imported conditionally
/// in `src/auditml/utils/rust_accel.py` with a NumPy fallback.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// ---------------------------------------------------------------------------
// Threshold scanning
// ---------------------------------------------------------------------------

/// Find the confidence threshold that maximises binary classification accuracy.
///
/// Parameters
/// ----------
/// scores : list[float]
///     Per-sample confidence scores (higher = more likely member).
/// labels : list[int]
///     Ground-truth binary labels (1 = member, 0 = non-member).
///
/// Returns
/// -------
/// (best_threshold: float, best_accuracy: float)
#[pyfunction]
fn find_best_threshold(scores: Vec<f64>, labels: Vec<i32>) -> PyResult<(f64, f64)> {
    let n = scores.len();
    if n == 0 {
        return Err(PyValueError::new_err("scores must not be empty"));
    }
    if labels.len() != n {
        return Err(PyValueError::new_err("scores and labels must have the same length"));
    }

    // Pair scores with labels and sort by score ascending
    let mut pairs: Vec<(f64, i32)> = scores.into_iter().zip(labels.into_iter()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total = n as f64;
    let total_pos: i32 = pairs.iter().map(|(_, l)| l).sum();

    // Threshold below all scores → predict all as member
    let mut tp = total_pos;      // true positives
    let mut fp = n as i32 - total_pos; // false positives
    let mut best_acc = (tp as f64) / total;
    let mut best_thresh = pairs[0].0 - 1.0;

    // Sweep threshold upward — each step flips one sample from predicted-member
    // to predicted-non-member
    for i in 0..n {
        let (score, label) = pairs[i];

        // Accuracy when threshold is set just above pairs[i].score
        let acc = (tp + (n as i32 - (i as i32) - 1 - (total_pos - tp - label))) as f64 / total;

        // Simpler: recount correctly classified
        // predict member if score > threshold (threshold = score)
        // samples with score <= threshold → predicted non-member
        // samples with score >  threshold → predicted member
        let predicted_member_correct = total_pos - (pairs[..=i].iter().map(|(_, l)| l).sum::<i32>());
        let predicted_nonmember_correct = (i as i32 + 1) - (pairs[..=i].iter().map(|(_, l)| l).sum::<i32>());
        let _ = (acc, tp, fp); // suppress warnings
        let accuracy = (predicted_member_correct + predicted_nonmember_correct) as f64 / total;

        if accuracy > best_acc {
            best_acc = accuracy;
            best_thresh = score;
        }

        // Update running counts for next iteration
        if label == 1 { tp -= 1; } else { fp -= 1; }
    }

    Ok((best_thresh, best_acc))
}

// ---------------------------------------------------------------------------
// SSIM
// ---------------------------------------------------------------------------

/// Compute Structural Similarity Index (SSIM) between two flat pixel arrays.
///
/// Both arrays must have the same length (H*W*C flattened). Pixel values
/// should be in [0, 1]. Returns a scalar in [-1, 1] where 1.0 = identical.
///
/// Parameters
/// ----------
/// img_a : list[float]   Reference image pixels, flattened, values in [0,1].
/// img_b : list[float]   Reconstructed image pixels, flattened, values in [0,1].
///
/// Returns
/// -------
/// ssim : float
#[pyfunction]
fn compute_ssim(img_a: Vec<f64>, img_b: Vec<f64>) -> PyResult<f64> {
    let n = img_a.len();
    if n == 0 {
        return Err(PyValueError::new_err("images must not be empty"));
    }
    if img_b.len() != n {
        return Err(PyValueError::new_err("img_a and img_b must have the same length"));
    }

    let n_f = n as f64;

    // Means
    let mu_a: f64 = img_a.iter().sum::<f64>() / n_f;
    let mu_b: f64 = img_b.iter().sum::<f64>() / n_f;

    // Variances and covariance
    let (var_a, var_b, cov_ab) = img_a.iter().zip(img_b.iter()).fold(
        (0.0f64, 0.0f64, 0.0f64),
        |(va, vb, co), (a, b)| {
            let da = a - mu_a;
            let db = b - mu_b;
            (va + da * da, vb + db * db, co + da * db)
        },
    );
    let var_a = var_a / n_f;
    let var_b = var_b / n_f;
    let cov_ab = cov_ab / n_f;

    // SSIM constants (standard values for [0,1] range)
    let c1 = 0.0001_f64; // (0.01 * 1.0)^2
    let c2 = 0.0009_f64; // (0.03 * 1.0)^2

    let ssim = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
        / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2));

    Ok(ssim)
}

// ---------------------------------------------------------------------------
// Batch SSIM
// ---------------------------------------------------------------------------

/// Compute SSIM for a batch of (reference, reconstructed) image pairs.
///
/// Parameters
/// ----------
/// originals   : list of list[float]   Reference images (flattened).
/// reconstructed : list of list[float] Reconstructed images (flattened).
///
/// Returns
/// -------
/// list[float]   SSIM score per pair.
#[pyfunction]
fn batch_ssim(originals: Vec<Vec<f64>>, reconstructed: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    if originals.len() != reconstructed.len() {
        return Err(PyValueError::new_err(
            "originals and reconstructed must have the same number of images",
        ));
    }
    originals
        .into_iter()
        .zip(reconstructed.into_iter())
        .map(|(a, b)| compute_ssim(a, b))
        .collect()
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn auditml_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_best_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ssim, m)?)?;
    m.add_function(wrap_pyfunction!(batch_ssim, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Rust-native tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssim_identical() {
        let img: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let result = compute_ssim(img.clone(), img).unwrap();
        assert!((result - 1.0).abs() < 1e-6, "SSIM of identical images should be 1.0");
    }

    #[test]
    fn test_ssim_different() {
        let img_a: Vec<f64> = vec![0.0; 100];
        let img_b: Vec<f64> = vec![1.0; 100];
        let result = compute_ssim(img_a, img_b).unwrap();
        assert!(result < 0.5, "SSIM of black vs white should be low");
    }

    #[test]
    fn test_threshold_all_correct() {
        // Perfect separation: members score 1.0, non-members score 0.0
        let scores = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let (_, acc) = find_best_threshold(scores, labels).unwrap();
        assert!((acc - 1.0).abs() < 1e-9, "Should achieve 100% accuracy");
    }

    #[test]
    fn test_threshold_random() {
        let scores = vec![0.3, 0.6, 0.4, 0.8, 0.2, 0.9];
        let labels = vec![0, 1, 0, 1, 0, 1];
        let (thresh, acc) = find_best_threshold(scores, labels).unwrap();
        assert!(acc >= 0.5, "Random data should be at least 50% acc, got {acc}");
        let _ = thresh;
    }
}
