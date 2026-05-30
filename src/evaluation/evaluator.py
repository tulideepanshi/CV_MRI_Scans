#!/usr/bin/env python3
"""
Evaluation Suite — Clinical metrics + Grad-CAM visualization.

WHAT THIS FILE DOES:
  1. ModelEvaluator: loads a trained checkpoint, runs inference on the test set,
     and computes a full clinical metric suite:
       - Per-class precision, recall (sensitivity), F1
       - Specificity per class (true negative rate — clinical standard)
       - Macro/weighted averages
       - Confusion matrix (absolute + normalized)
       - ROC curves and AUC (one-vs-rest for multiclass)
       - Classification report (text + JSON)

  2. GradCAM: generates class activation heatmaps showing WHERE the model
     looks when making predictions — critical for clinical trust.

  3. Visualization functions: confusion matrix plot, ROC curves, Grad-CAM
     overlays, and a per-class error analysis grid.

WHY THESE METRICS (not just accuracy):
  In clinical settings, a model that's 95% accurate but misses 40% of gliomas
  is dangerous. The metrics here address different clinical questions:

  Sensitivity (recall): "Of all glioma patients, how many did we catch?"
    → False negatives are the danger. Missing a tumor = delayed treatment.

  Specificity: "Of all healthy patients, how many did we correctly clear?"
    → False positives cause unnecessary biopsies and patient anxiety.

  F1 score: Harmonic mean of precision and recall. Balanced view.

  ROC-AUC: How well does the model separate classes across ALL thresholds?
    → AUC = 1.0 means perfect separation. AUC = 0.5 means random guessing.
    → Threshold-independent — useful when you haven't chosen an operating point.

WHY GRAD-CAM:
  "The model says glioma with 94% confidence" means nothing without knowing
  WHY. Grad-CAM shows which pixels drove the prediction:
    - If the heatmap highlights the tumor region → model learned correctly
    - If it highlights the skull or background → model found a shortcut
    - If it highlights scanner artifacts → dataset has confounders

  This is the difference between a black box and a clinical decision support tool.

HOW GRAD-CAM WORKS:
  1. Forward pass: get the predicted class
  2. Backward pass: compute gradients of the predicted class score with
     respect to the last convolutional layer's activations
  3. Global average pool the gradients → per-channel importance weights
  4. Weighted sum of activation maps → raw heatmap
  5. ReLU (only keep positive contributions) → final heatmap
  6. Resize to input image dimensions → overlay on original

  Mathematically:
    α_k = (1/Z) Σ_i Σ_j ∂y^c / ∂A^k_ij     (importance of channel k)
    L_Grad-CAM = ReLU(Σ_k α_k · A^k)          (weighted combination)

Usage:
    from src.evaluation.evaluator import ModelEvaluator

    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=["glioma", "meningioma", "notumor", "pituitary"],
        device=device,
    )
    results = evaluator.evaluate()
    evaluator.plot_confusion_matrix(results, save_path="docs/eval/confusion.png")
    evaluator.plot_roc_curves(results, save_path="docs/eval/roc.png")
    evaluator.generate_gradcam_grid(test_loader, save_dir="docs/eval/gradcam/")
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)

from src.data.dataset import CLASS_TO_IDX, IDX_TO_CLASS


# ============================================================
# Grad-CAM Implementation
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into the forward and backward pass of a target layer to
    capture activations and gradients, then combines them into a heatmap.

    Why hooks (not manual computation):
      PyTorch doesn't store intermediate activations or gradients by default
      (memory optimization). Register_hook callbacks capture them on the fly
      without modifying the model architecture.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        # Storage for captured activations and gradients
        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook — capture layer activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook — capture layer gradients."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            input_tensor: (1, C, H, W) normalized input image
            target_class: Class to explain (None = predicted class)

        Returns:
            heatmap: (H, W) numpy array, values in [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # If no target class, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients, then backward pass for the target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Grad-CAM computation
        # gradients: (1, C, h, w) — gradient of target w.r.t. activations
        # activations: (1, C, h, w) — feature maps at target layer
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        # Global average pool gradients → channel importance weights
        # α_k = mean over spatial dimensions of gradient for channel k
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU — only keep features that positively contribute
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def remove_hooks(self):
        """Clean up hooks when done."""
        self._forward_hook.remove()
        self._backward_hook.remove()


# ============================================================
# Model Evaluator
# ============================================================

class ModelEvaluator:
    """
    Comprehensive evaluation pipeline for brain tumor classification.

    Computes clinical metrics, generates visualizations, and saves
    a complete evaluation report.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader,
        class_names: list[str] = None,
        device: torch.device = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names or list(IDX_TO_CLASS.values())
        self.num_classes = len(self.class_names)

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Run full evaluation on the test set.

        Returns a dict with all metrics, raw predictions, and probabilities.

        Why collect probabilities (not just predictions):
          - ROC curves need continuous scores, not hard labels
          - Calibration analysis needs probability estimates
          - Clinical decision thresholds depend on probability distributions
        """
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in self.test_loader:
            images = images.to(self.device)
            outputs = self.model(images)

            # Softmax → probabilities
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # --- Core Metrics ---
        accuracy = accuracy_score(all_labels, all_preds)

        # Per-class precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=range(self.num_classes)
        )

        # Macro averages (treat all classes equally)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.num_classes))

        # --- Clinical Metrics ---
        # Specificity per class (True Negative Rate)
        # For class c: TN = all samples NOT class c that were correctly NOT predicted as c
        specificity = self._compute_specificity(cm)

        # ROC AUC (One-vs-Rest)
        try:
            roc_auc_ovr = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro"
            )
            roc_auc_per_class = self._compute_per_class_auc(all_labels, all_probs)
        except ValueError:
            # Can happen if a class has 0 samples in test set
            roc_auc_ovr = None
            roc_auc_per_class = {}

        # ROC curve data (for plotting)
        roc_curves = self._compute_roc_curves(all_labels, all_probs)

        # Full classification report (text)
        report_text = classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            digits=4,
        )

        results = {
            # Summary
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "roc_auc_macro": float(roc_auc_ovr) if roc_auc_ovr else None,

            # Per-class
            "per_class": {
                self.class_names[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),        # = sensitivity
                    "sensitivity": float(recall[i]),   # Clinical alias
                    "specificity": float(specificity[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                    "roc_auc": float(roc_auc_per_class.get(i, 0)),
                }
                for i in range(self.num_classes)
            },

            # Raw data (for custom analysis)
            "confusion_matrix": cm.tolist(),
            "roc_curves": roc_curves,
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist(),
            "probabilities": all_probs.tolist(),

            # Text report
            "classification_report": report_text,
        }

        return results

    def _compute_specificity(self, cm: np.ndarray) -> np.ndarray:
        """
        Compute per-class specificity from confusion matrix.

        Specificity for class c = TN_c / (TN_c + FP_c)

        Where:
          TN_c = total samples not in class c that were correctly not predicted as c
          FP_c = total samples not in class c that were incorrectly predicted as c

        In a multiclass confusion matrix:
          FP_c = sum of column c (excluding diagonal) = samples predicted as c that aren't c
          TN_c = total - (TP_c + FP_c + FN_c)
        """
        specificity = np.zeros(self.num_classes)

        for c in range(self.num_classes):
            # True Positives for class c
            tp = cm[c, c]
            # False Positives for class c = everything predicted as c that isn't c
            fp = cm[:, c].sum() - tp
            # False Negatives for class c = everything that is c but predicted as other
            fn = cm[c, :].sum() - tp
            # True Negatives = everything else
            tn = cm.sum() - tp - fp - fn

            if (tn + fp) > 0:
                specificity[c] = tn / (tn + fp)
            else:
                specificity[c] = 0.0

        return specificity

    def _compute_per_class_auc(
        self, labels: np.ndarray, probs: np.ndarray
    ) -> dict[int, float]:
        """Compute AUC for each class (one-vs-rest)."""
        auc_dict = {}
        for c in range(self.num_classes):
            binary_labels = (labels == c).astype(int)
            if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                auc_dict[c] = float(roc_auc_score(binary_labels, probs[:, c]))
        return auc_dict

    def _compute_roc_curves(
        self, labels: np.ndarray, probs: np.ndarray
    ) -> dict[str, dict]:
        """Compute ROC curve data for each class (for plotting)."""
        curves = {}
        for c in range(self.num_classes):
            binary_labels = (labels == c).astype(int)
            if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                fpr, tpr, thresholds = roc_curve(binary_labels, probs[:, c])
                roc_auc_val = auc(fpr, tpr)
                curves[self.class_names[c]] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(roc_auc_val),
                }
        return curves

    # ==============================================================
    # Visualization
    # ==============================================================

    def plot_confusion_matrix(
        self,
        results: dict,
        save_path: str = None,
        normalized: bool = True,
    ) -> None:
        """
        Plot confusion matrix — both absolute counts and normalized.

        Normalized matrix shows WHAT FRACTION of each true class was
        predicted as each class. This is more informative than raw counts
        when classes are imbalanced.

        Reading the matrix:
          Row = true class, Column = predicted class
          Diagonal = correct predictions (want this high)
          Off-diagonal = errors (row tells you what it was, column what it was called)
        """
        cm = np.array(results["confusion_matrix"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Absolute counts ---
        ax = axes[0]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")

        # Add text annotations
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=12, fontweight="bold")

        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(self.class_names, fontsize=9)

        # --- Normalized (row-wise) ---
        ax = axes[1]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im2 = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color=color, fontsize=12, fontweight="bold")

        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(self.class_names, fontsize=9)

        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved confusion matrix → {save_path}")
        plt.close()

    def plot_roc_curves(
        self,
        results: dict,
        save_path: str = None,
    ) -> None:
        """
        Plot ROC curves for all classes (one-vs-rest).

        ROC curve: True Positive Rate vs False Positive Rate at every threshold.

        The diagonal dashed line = random guessing (AUC = 0.5).
        A perfect model hugs the top-left corner (AUC = 1.0).

        Clinical interpretation:
          "At a false alarm rate of 5%, what fraction of tumors do we catch?"
          → Read the TPR at FPR=0.05 on the curve.
        """
        roc_curves = results.get("roc_curves", {})
        if not roc_curves:
            print("  No ROC data available.")
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        # Color palette for 4 classes
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

        for i, (class_name, curve_data) in enumerate(roc_curves.items()):
            color = colors[i % len(colors)]
            ax.plot(
                curve_data["fpr"], curve_data["tpr"],
                color=color, linewidth=2,
                label=f"{class_name} (AUC = {curve_data['auc']:.3f})"
            )

        # Random baseline
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
                label="Random (AUC = 0.500)")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        ax.set_title("ROC Curves — One-vs-Rest", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)

        # Add macro AUC annotation
        macro_auc = results.get("roc_auc_macro")
        if macro_auc:
            ax.text(0.6, 0.15, f"Macro AUC: {macro_auc:.3f}",
                    fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved ROC curves → {save_path}")
        plt.close()

    def plot_per_class_metrics(
        self,
        results: dict,
        save_path: str = None,
    ) -> None:
        """
        Bar chart comparing precision, recall, specificity, F1 per class.

        This is the "dashboard view" — one glance shows which classes
        the model handles well and which need attention.
        """
        metrics_to_plot = ["precision", "recall", "specificity", "f1"]
        n_metrics = len(metrics_to_plot)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(self.num_classes)
        width = 0.18
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

        for i, metric in enumerate(metrics_to_plot):
            values = [results["per_class"][c][metric] for c in self.class_names]
            bars = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                          color=colors[i], alpha=0.85)
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Per-Class Clinical Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels(self.class_names, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved per-class metrics → {save_path}")
        plt.close()

    def generate_gradcam_grid(
        self,
        dataset,
        save_dir: str = "docs/eval/gradcam",
        num_samples: int = 5,
        target_layer: nn.Module = None,
        mean: list = None,
        std: list = None,
    ) -> None:
        """
        Generate Grad-CAM heatmaps for sample images from each class.

        Creates a grid: rows = classes, columns = samples.
        Each cell shows the original image with Grad-CAM overlay.

        Args:
            dataset: BrainTumorDataset (NOT DataLoader — we need per-image access)
            save_dir: Directory to save heatmap images
            num_samples: Number of samples per class to visualize
            target_layer: Conv layer for Grad-CAM (auto-detected if None)
            mean/std: Normalization values for de-normalization
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        # Auto-detect target layer
        if target_layer is None:
            if hasattr(self.model, "get_last_conv_layer"):
                target_layer = self.model.get_last_conv_layer()
            else:
                raise ValueError(
                    "Model has no get_last_conv_layer(). Pass target_layer explicitly."
                )

        # Initialize Grad-CAM
        grad_cam = GradCAM(self.model, target_layer)

        # Group dataset indices by class
        class_indices = {c: [] for c in range(self.num_classes)}
        for idx in range(len(dataset)):
            label = dataset.label_indices[idx]
            class_indices[label].append(idx)

        # Generate heatmaps
        fig, axes = plt.subplots(
            self.num_classes, num_samples,
            figsize=(num_samples * 3, self.num_classes * 3),
        )

        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            indices = class_indices[class_idx][:num_samples]

            for col, sample_idx in enumerate(indices):
                ax = axes[class_idx, col] if self.num_classes > 1 else axes[col]

                # Load image and get prediction
                img_tensor, label = dataset[sample_idx]
                input_tensor = img_tensor.unsqueeze(0).to(self.device)

                # Enable gradients for Grad-CAM
                input_tensor.requires_grad_(True)

                # Forward pass to get prediction
                with torch.enable_grad():
                    output = self.model(input_tensor)
                pred_class = output.argmax(dim=1).item()
                pred_conf = F.softmax(output, dim=1)[0, pred_class].item()

                # Generate heatmap
                with torch.enable_grad():
                    heatmap = grad_cam.generate(input_tensor, target_class=pred_class)

                # De-normalize image for display
                img_display = self._denormalize(img_tensor, mean, std)

                # Resize heatmap to image dimensions
                h, w = img_display.shape[:2]
                heatmap_resized = np.array(
                    Image.fromarray(
                        (heatmap * 255).astype(np.uint8)
                    ).resize((w, h), Image.BILINEAR)
                ) / 255.0

                # Create overlay
                overlay = self._create_overlay(img_display, heatmap_resized)

                # Plot
                ax.imshow(overlay)
                pred_name = self.class_names[pred_class]
                correct = "correct" if pred_class == class_idx else "WRONG"
                color = "green" if pred_class == class_idx else "red"
                ax.set_title(
                    f"Pred: {pred_name} ({pred_conf:.0%})\n[{correct}]",
                    fontsize=8, color=color, fontweight="bold",
                )
                ax.axis("off")

                # Row labels
                if col == 0:
                    ax.set_ylabel(class_name, fontsize=11, fontweight="bold",
                                  rotation=90, labelpad=10)

                # Save individual heatmap
                individual_path = save_dir / f"{class_name}_sample{col}.png"
                plt.imsave(str(individual_path), overlay)

        plt.suptitle("Grad-CAM Visualization — Where the Model Looks",
                      fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        grid_path = save_dir / "gradcam_grid.png"
        plt.savefig(str(grid_path), dpi=150, bbox_inches="tight")
        print(f"  Saved Grad-CAM grid → {grid_path}")
        plt.close()

        # Clean up hooks
        grad_cam.remove_hooks()

    def _denormalize(
        self,
        tensor: torch.Tensor,
        mean: list,
        std: list,
    ) -> np.ndarray:
        """
        Reverse normalization to get displayable image.

        Normalization: (pixel - mean) / std
        Reverse:       pixel * std + mean
        Then clip to [0, 1] and convert to uint8 range.
        """
        img = tensor.cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
        mean = np.array(mean)
        std = np.array(std)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img

    def _create_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.

        Uses jet colormap for the heatmap, then alpha-blends with the original.
        Red/yellow = high activation (model looks here)
        Blue/green = low activation (model ignores)
        """
        # Apply jet colormap to heatmap
        cmap = plt.cm.jet
        heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel

        # Alpha blend
        overlay = (1 - alpha) * image + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        return overlay

    # ==============================================================
    # Report generation
    # ==============================================================

    def save_report(
        self,
        results: dict,
        save_dir: str = "docs/eval",
        model_name: str = "model",
    ) -> None:
        """
        Save complete evaluation report — JSON metrics + text summary.

        Produces:
          - {model_name}_metrics.json: machine-readable metrics
          - {model_name}_report.txt: human-readable clinical report
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- JSON metrics (exclude raw predictions to save space) ---
        metrics_json = {
            k: v for k, v in results.items()
            if k not in ("predictions", "labels", "probabilities", "roc_curves")
        }
        json_path = save_dir / f"{model_name}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        print(f"  Saved metrics JSON → {json_path}")

        # --- Text report ---
        report_lines = [
            "=" * 60,
            f"CLINICAL EVALUATION REPORT: {model_name}",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"  Overall Accuracy:    {results['accuracy']:.4f} ({results['accuracy']:.1%})",
            f"  Macro F1 Score:      {results['f1_macro']:.4f}",
            f"  Weighted F1 Score:   {results['f1_weighted']:.4f}",
            f"  Macro ROC-AUC:       {results['roc_auc_macro']:.4f}" if results.get('roc_auc_macro') else "  Macro ROC-AUC:       N/A",
            "",
            "PER-CLASS METRICS",
            "-" * 40,
            f"  {'Class':<14s} {'Prec':>6s} {'Recall':>7s} {'Spec':>6s} {'F1':>6s} {'AUC':>6s} {'N':>5s}",
            f"  {'-'*14} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*5}",
        ]

        for class_name in self.class_names:
            m = results["per_class"][class_name]
            report_lines.append(
                f"  {class_name:<14s} {m['precision']:>6.3f} {m['recall']:>7.3f} "
                f"{m['specificity']:>6.3f} {m['f1']:>6.3f} {m['roc_auc']:>6.3f} "
                f"{m['support']:>5d}"
            )

        report_lines.extend([
            "",
            "CLINICAL INTERPRETATION",
            "-" * 40,
        ])

        # Flag any class with recall < 0.90 (clinical concern)
        for class_name in self.class_names:
            recall = results["per_class"][class_name]["recall"]
            spec = results["per_class"][class_name]["specificity"]
            if recall < 0.90:
                report_lines.append(
                    f"  WARNING: {class_name} recall = {recall:.3f} (<90%) "
                    f"— risk of missed diagnoses"
                )
            if spec < 0.90:
                report_lines.append(
                    f"  NOTE: {class_name} specificity = {spec:.3f} (<90%) "
                    f"— elevated false positive rate"
                )

        # Check if all metrics are good
        all_recall_good = all(
            results["per_class"][c]["recall"] >= 0.90 for c in self.class_names
        )
        if all_recall_good:
            report_lines.append("  All classes have recall >= 90% — acceptable for screening.")

        report_lines.extend([
            "",
            "SKLEARN CLASSIFICATION REPORT",
            "-" * 40,
            results["classification_report"],
            "",
            "CONFUSION MATRIX",
            "-" * 40,
        ])

        cm = np.array(results["confusion_matrix"])
        header = "  " + " " * 14 + "  ".join(f"{c:>10s}" for c in self.class_names)
        report_lines.append(header)
        for i, class_name in enumerate(self.class_names):
            row = f"  {class_name:<14s}" + "  ".join(f"{cm[i,j]:>10d}" for j in range(self.num_classes))
            report_lines.append(row)

        report_text = "\n".join(report_lines) + "\n"

        txt_path = save_dir / f"{model_name}_report.txt"
        with open(txt_path, "w") as f:
            f.write(report_text)
        print(f"  Saved text report → {txt_path}")

        # Print summary to console
        print(f"\n  {model_name} — Acc: {results['accuracy']:.4f}, "
              f"F1: {results['f1_macro']:.4f}, "
              f"AUC: {results.get('roc_auc_macro', 'N/A')}")

    def generate_error_analysis(
        self,
        dataset,
        results: dict,
        save_dir: str = "docs/eval/errors",
        max_errors: int = 10,
        mean: list = None,
        std: list = None,
    ) -> None:
        """
        Visualize misclassified images — helps understand failure modes.

        Shows the original image, what the model predicted, and what it
        actually was. This is gold for interview discussion — shows you
        understand not just metrics but WHY the model fails.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        preds = np.array(results["predictions"])
        labels = np.array(results["labels"])
        probs = np.array(results["probabilities"])

        # Find misclassified indices
        error_mask = preds != labels
        error_indices = np.where(error_mask)[0]

        if len(error_indices) == 0:
            print("  No misclassified samples!")
            return

        # Sort by confidence (highest confidence errors are most concerning)
        error_confidences = [probs[i, preds[i]] for i in error_indices]
        sorted_errors = sorted(
            zip(error_indices, error_confidences),
            key=lambda x: -x[1],  # Descending confidence
        )

        # Plot top errors
        n = min(max_errors, len(sorted_errors))
        cols = min(5, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for idx, (error_idx, conf) in enumerate(sorted_errors[:n]):
            row, col = divmod(idx, cols)
            ax = axes[row, col]

            img_tensor, _ = dataset[error_idx]
            img_display = self._denormalize(img_tensor, mean, std)

            true_class = self.class_names[labels[error_idx]]
            pred_class = self.class_names[preds[error_idx]]

            ax.imshow(img_display)
            ax.set_title(
                f"True: {true_class}\nPred: {pred_class} ({conf:.0%})",
                fontsize=8, color="red", fontweight="bold",
            )
            ax.axis("off")

        # Hide unused subplots
        for idx in range(n, rows * cols):
            row, col = divmod(idx, cols)
            axes[row, col].axis("off")

        plt.suptitle(
            f"Top {n} Misclassified Images (by confidence)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()

        path = save_dir / "error_analysis.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        print(f"  Saved error analysis → {path}")
        plt.close()
