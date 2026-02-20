"""Model Inversion attack.

Reconstructs representative images for target classes by optimising in
the input (pixel) space. The idea is: if a model has memorised training
data, then gradient-based optimisation can recover images that resemble
actual training samples.

**Key idea**: Start from random noise and iteratively adjust the pixels
so the model becomes maximally confident the image belongs to a chosen
target class. Total Variation (TV) and L2 regularisation encourage
smooth, realistic reconstructions.

This is a **white-box** attack — it requires gradient access to the
target model.

References
----------
Fredrikson et al., "Model Inversion Attacks that Exploit Confidence
Information and Basic Countermeasures", CCS 2015.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from auditml.attacks.base import BaseAttack
from auditml.attacks.results import AttackResult
from auditml.config.schema import AuditMLConfig
from auditml.data.datasets import DATASET_INFO

logger = logging.getLogger(__name__)


class ModelInversion(BaseAttack):
    """Gradient-based Model Inversion attack.

    For each target class, optimises a synthetic image so that the model
    classifies it with maximum confidence.  The reconstructed images
    reveal what the model has learned — and potentially memorised — about
    each class.

    Parameters
    ----------
    target_model:
        The trained model to attack (white-box — needs gradients).
    config:
        Full AuditML configuration.
    device:
        Torch device.
    input_shape:
        Shape of the model's input, e.g. ``(1, 28, 28)`` for MNIST or
        ``(3, 32, 32)`` for CIFAR. If ``None``, inferred from the
        dataset name in config.
    """

    attack_name = "model_inversion"

    def __init__(
        self,
        target_model: nn.Module,
        config: AuditMLConfig,
        device: torch.device | str = "cpu",
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(target_model, config, device)

        # Config shortcuts
        params = config.attack_params.model_inversion
        self.num_iterations = params.num_iterations
        self.lr = params.learning_rate
        self.lambda_tv = params.lambda_tv
        self.lambda_l2 = params.lambda_l2
        self.target_class = params.target_class
        self.num_classes = config.model.num_classes

        # Determine input shape
        if input_shape is not None:
            self.input_shape = input_shape
        else:
            dataset_name = config.data.dataset.value
            if dataset_name in DATASET_INFO:
                self.input_shape = DATASET_INFO[dataset_name].input_shape
            else:
                raise ValueError(
                    f"Cannot infer input_shape for dataset {dataset_name!r}. "
                    "Pass input_shape explicitly."
                )

        # Populated during run()
        self.reconstructions: dict[int, np.ndarray] = {}
        self.reconstruction_confidences: dict[int, float] = {}
        # Stored during run() for visualization
        self.member_scores: np.ndarray | None = None
        self.nonmember_scores: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Main attack logic
    # ------------------------------------------------------------------

    def run(
        self,
        member_loader: DataLoader,
        nonmember_loader: DataLoader,
    ) -> AttackResult:
        """Execute the model inversion attack.

        For each target class, reconstruct an image and measure how
        confidently the model classifies it. Then use the member and
        non-member loaders to evaluate: does the model assign higher
        confidence to reconstructions of classes it trained on?

        The ``member_loader`` and ``nonmember_loader`` are used to
        compute a membership-like signal: for each sample, we measure
        the similarity between the model's output on that sample and
        the reconstructed class prototype. Members tend to produce
        outputs closer to the reconstruction.
        """
        # Determine which classes to invert
        if self.target_class is not None:
            classes_to_invert = [self.target_class]
        else:
            classes_to_invert = list(range(self.num_classes))

        # Step 1: Reconstruct images for each target class
        for cls in classes_to_invert:
            logger.info("Inverting class %d/%d ...", cls + 1, len(classes_to_invert))
            recon, confidence = self.invert_class(cls)
            self.reconstructions[cls] = recon.detach().cpu().numpy()
            self.reconstruction_confidences[cls] = confidence

        # Step 2: Compute membership signal using reconstruction similarity
        self.member_scores = self._compute_similarity_scores(member_loader)
        self.nonmember_scores = self._compute_similarity_scores(nonmember_loader)
        member_scores = self.member_scores
        nonmember_scores = self.nonmember_scores

        # Build ground truth and combined scores
        ground_truth = np.concatenate([
            np.ones(len(member_scores)),
            np.zeros(len(nonmember_scores)),
        ])
        all_scores = np.concatenate([member_scores, nonmember_scores])

        # Threshold at median for binary predictions
        threshold = float(np.median(all_scores))
        predictions = (all_scores >= threshold).astype(np.int32)

        self.result = AttackResult(
            predictions=predictions,
            ground_truth=ground_truth,
            confidence_scores=all_scores,
            attack_name=self.attack_name,
            metadata={
                "num_classes_inverted": len(classes_to_invert),
                "num_iterations": self.num_iterations,
                "lambda_tv": self.lambda_tv,
                "lambda_l2": self.lambda_l2,
                "reconstruction_confidences": self.reconstruction_confidences,
                "mean_member_similarity": float(member_scores.mean()),
                "mean_nonmember_similarity": float(nonmember_scores.mean()),
            },
        )
        return self.result

    # ------------------------------------------------------------------
    # Core inversion — reconstruct one class
    # ------------------------------------------------------------------

    def invert_class(
        self,
        target_class: int,
        num_iterations: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Reconstruct an image for a single target class.

        Parameters
        ----------
        target_class:
            The class label to reconstruct.
        num_iterations:
            Override the config value. Uses ``self.num_iterations`` if None.

        Returns
        -------
        (reconstructed_image, confidence)
            - reconstructed_image: tensor of shape ``(1, C, H, W)``
            - confidence: model's softmax probability for target_class
        """
        if num_iterations is None:
            num_iterations = self.num_iterations

        # Ensure model is in eval mode but gradients can flow through
        self.target_model.eval()

        # Start from random noise, requires_grad so we can optimise it
        x = torch.randn(1, *self.input_shape, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([x], lr=self.lr)

        best_confidence = 0.0
        best_x = x.detach().clone()

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Forward pass
            logits = self.target_model(x)
            probs = F.softmax(logits, dim=1)

            # Classification loss: maximise P(target_class)
            # Equivalent to minimising -log(P(target_class))
            cls_loss = -torch.log(probs[0, target_class] + 1e-10)

            # Regularisation
            reg_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_tv > 0:
                reg_loss = reg_loss + self.lambda_tv * self._total_variation(x)
            if self.lambda_l2 > 0:
                reg_loss = reg_loss + self.lambda_l2 * torch.norm(x)

            total_loss = cls_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            # Track best reconstruction
            current_confidence = probs[0, target_class].item()
            if current_confidence > best_confidence:
                best_confidence = current_confidence
                best_x = x.detach().clone()

        logger.info(
            "Class %d: confidence=%.4f after %d iterations",
            target_class, best_confidence, num_iterations,
        )
        return best_x, best_confidence

    # ------------------------------------------------------------------
    # Membership signal via reconstruction similarity
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_similarity_scores(self, loader: DataLoader) -> np.ndarray:
        """Compute how similar each sample's output is to its class reconstruction.

        For each sample, we measure the cosine similarity between the
        model's softmax output on that sample and the softmax output
        on the reconstructed image for that sample's class. Members
        tend to have higher similarity because the model has memorised
        patterns specific to training data.

        Parameters
        ----------
        loader:
            DataLoader of samples to score.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — similarity score per sample.
        """
        self.target_model.eval()
        all_scores: list[float] = []

        # Pre-compute reconstruction output vectors for each class
        recon_outputs: dict[int, np.ndarray] = {}
        for cls, recon_img in self.reconstructions.items():
            recon_tensor = torch.tensor(recon_img, dtype=torch.float32).to(self.device)
            logits = self.target_model(recon_tensor)
            recon_outputs[cls] = F.softmax(logits, dim=1).cpu().numpy()[0]

        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            logits = self.target_model(inputs)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            for i in range(len(targets)):
                cls = targets[i].item()
                if cls in recon_outputs:
                    # Cosine similarity between sample output and reconstruction output
                    sample_vec = probs[i]
                    recon_vec = recon_outputs[cls]
                    cos_sim = float(
                        np.dot(sample_vec, recon_vec)
                        / (np.linalg.norm(sample_vec) * np.linalg.norm(recon_vec) + 1e-10)
                    )
                    all_scores.append(cos_sim)
                else:
                    # Class wasn't inverted — use neutral score
                    all_scores.append(0.5)

        return np.array(all_scores)

    # ------------------------------------------------------------------
    # Regularisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _total_variation(x: torch.Tensor) -> torch.Tensor:
        """Compute Total Variation loss for an image tensor.

        TV loss encourages spatial smoothness by penalising large
        differences between neighbouring pixels. This prevents the
        optimisation from producing noisy, unrealistic images.

        Parameters
        ----------
        x:
            Image tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar TV loss.
        """
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]  # vertical differences
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]  # horizontal differences
        return torch.mean(diff_h ** 2) + torch.mean(diff_w ** 2)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_dir: str | Path) -> Path:
        """Generate a complete model inversion report.

        Creates the following files in *output_dir*:

        - ``metrics.json`` — overall evaluation metrics
        - ``reconstructions.png`` — grid of reconstructed images
        - ``reconstruction_confidence.png`` — per-class confidence bar chart
        - ``similarity_distributions.png`` — member vs non-member similarity
        - ``roc_curve.png`` — ROC curve
        - ``summary.txt`` — human-readable text summary

        Parameters
        ----------
        output_dir:
            Directory where all report files are saved.

        Returns
        -------
        Path
            The output directory.
        """
        if self.result is None:
            raise RuntimeError("Call run() before generate_report().")

        from auditml.attacks.visualization import (
            plot_reconstruction_confidence,
            plot_reconstructions,
            plot_roc_curve,
            plot_score_distributions,
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Overall metrics
        metrics = self.evaluate()
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 2. Reconstructed images grid
        plot_reconstructions(
            reconstructions=self.reconstructions,
            confidences=self.reconstruction_confidences,
            save_path=out / "reconstructions.png",
        )

        # 3. Reconstruction confidence bar chart
        plot_reconstruction_confidence(
            confidences=self.reconstruction_confidences,
            save_path=out / "reconstruction_confidence.png",
        )

        # 4. Similarity distribution histogram
        plot_score_distributions(
            member_scores=self.member_scores,
            nonmember_scores=self.nonmember_scores,
            metric_name="cosine similarity",
            save_path=out / "similarity_distributions.png",
            title="Similarity Distribution — Model Inversion",
        )

        # 5. ROC curve
        plot_roc_curve(
            ground_truth=self.result.ground_truth,
            confidence_scores=self.result.confidence_scores,
            title="ROC Curve — Model Inversion",
            save_path=out / "roc_curve.png",
        )

        # 6. Summary text
        self._write_summary(out / "summary.txt", metrics)

        return out

    def _write_summary(
        self,
        path: Path,
        metrics: dict[str, float],
    ) -> None:
        """Write a human-readable text summary."""
        lines = [
            "=" * 60,
            "AuditML — Model Inversion Report",
            "=" * 60,
            "",
            f"Classes inverted: {len(self.reconstructions)}",
            f"Iterations:       {self.num_iterations}",
            f"Learning rate:    {self.lr}",
            f"Lambda TV:        {self.lambda_tv}",
            f"Lambda L2:        {self.lambda_l2}",
            f"Input shape:      {self.input_shape}",
            f"Total samples:    {len(self.result.predictions)}",
            f"  Members:        {int(self.result.ground_truth.sum())}",
            f"  Non-members:    {int((1 - self.result.ground_truth).sum())}",
            "",
            "--- Reconstruction Confidences ---",
        ]
        for cls in sorted(self.reconstruction_confidences.keys()):
            lines.append(f"  Class {cls:>3d}: {self.reconstruction_confidences[cls]:.4f}")

        lines.append("")
        lines.append("--- Overall Metrics ---")
        for key, val in metrics.items():
            lines.append(f"  {key:<20s}: {val:.4f}")

        lines.append("")
        lines.append("--- Metadata ---")
        for key, val in self.result.metadata.items():
            if key != "reconstruction_confidences":
                lines.append(f"  {key}: {val}")

        lines.append("")
        path.write_text("\n".join(lines))
