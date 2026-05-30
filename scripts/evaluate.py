#!/usr/bin/env python3
"""
Evaluation script — run clinical metrics + Grad-CAM on trained models.

Loads a checkpoint, evaluates on the test set, and produces:
  - Clinical metrics report (precision, recall, specificity, F1, ROC-AUC)
  - Confusion matrix plot (absolute + normalized)
  - ROC curves (one-vs-rest)
  - Per-class metric bar chart
  - Grad-CAM heatmap grid
  - Error analysis (top misclassified images by confidence)
  - JSON metrics file (machine-readable)

Usage:
    # Evaluate a single model
    python scripts/evaluate.py --model custom_cnn --augmentation conservative

    # Evaluate all trained models and produce a comparison table
    python scripts/evaluate.py --model all

    # Evaluate with Grad-CAM disabled (faster)
    python scripts/evaluate.py --model efficientnet_pretrained --no-gradcam

    # Custom checkpoint path
    python scripts/evaluate.py --checkpoint models/custom_cnn_conservative/best_model.pth
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.reproducibility import seed_everything, get_device
from src.data.dataset import create_dataloaders, BrainTumorDataset, IDX_TO_CLASS
from src.data.augmentation import get_transforms
from src.models.factory import create_model, list_models
from src.evaluation.evaluator import ModelEvaluator


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_checkpoint(
    model_name: str,
    checkpoint_path: str,
    config: dict,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.

    Checkpoints contain:
      - model_state_dict: the learned weights
      - optimizer_state_dict: optimizer state (for resuming training)
      - scheduler_state_dict: LR scheduler state
      - val_metrics: best validation metrics
      - config: config used during training
      - epoch: training epoch when saved

    We only need model_state_dict for evaluation.
    """
    # Create model architecture (must match checkpoint)
    model_cfg = config.get("models", {}).get(model_name, {})
    num_classes = config.get("data", {}).get("num_classes", 4)
    model = create_model(model_name, model_cfg, num_classes)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    epoch = checkpoint.get("epoch", "?")
    val_metrics = checkpoint.get("val_metrics", {})
    print(f"  Loaded checkpoint from epoch {epoch}")
    if val_metrics:
        print(f"  Val metrics at save: "
              f"acc={val_metrics.get('acc', 'N/A'):.4f}, "
              f"f1={val_metrics.get('f1_macro', 'N/A'):.4f}")

    return model


def evaluate_single_model(
    model_name: str,
    config: dict,
    augmentation_mode: str = "conservative",
    checkpoint_path: str = None,
    enable_gradcam: bool = True,
    num_gradcam_samples: int = 5,
) -> dict:
    """
    Evaluate a single trained model on the test set.

    Steps:
      1. Load model from checkpoint
      2. Build test DataLoader
      3. Run inference and compute metrics
      4. Generate all visualizations
      5. Save report
    """
    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name} ({augmentation_mode})")
    print("=" * 60)

    # Setup
    seed_everything(config.get("training", {}).get("seed", 42))
    device = get_device()

    # Resolve checkpoint path
    if checkpoint_path is None:
        checkpoint_path = f"models/{model_name}_{augmentation_mode}/best_model.pth"

    if not Path(checkpoint_path).exists():
        print(f"  ERROR: Checkpoint not found: {checkpoint_path}")
        print(f"  Has this model been trained? Run: python scripts/train.py --model {model_name}")
        return None

    # Load model
    print("\nLoading model...")
    model = load_checkpoint(model_name, checkpoint_path, config, device)

    # Build test DataLoader
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    print("\nLoading test data...")
    loaders = create_dataloaders(
        split_dir=data_cfg.get("split_dir", "data/splits"),
        augmentation_mode="test",  # No augmentation for evaluation
        image_size=data_cfg.get("image_size", 224),
        mean=data_cfg.get("normalization", {}).get("mean"),
        std=data_cfg.get("normalization", {}).get("std"),
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=min(train_cfg.get("num_workers", 4), 2),  # Fewer workers for eval
        pin_memory=False,  # Safe default
        seed=42,
    )

    if "test" not in loaders:
        print("  ERROR: No test split found.")
        return None

    test_loader = loaders["test"]

    # Also build test dataset (for per-image access in Grad-CAM)
    test_dataset = BrainTumorDataset(
        data_dir=Path(data_cfg.get("split_dir", "data/splits")) / "test",
        transform=get_transforms(
            "test",
            data_cfg.get("image_size", 224),
            data_cfg.get("normalization", {}).get("mean"),
            data_cfg.get("normalization", {}).get("std"),
        ),
    )

    # Output directory
    eval_dir = f"docs/eval/{model_name}_{augmentation_mode}"

    # Evaluate
    print("\nRunning evaluation...")
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
    )
    results = evaluator.evaluate()

    # Print classification report
    print(f"\n{results['classification_report']}")

    # Generate visualizations
    print("Generating visualizations...")
    evaluator.plot_confusion_matrix(results, save_path=f"{eval_dir}/confusion_matrix.png")
    evaluator.plot_roc_curves(results, save_path=f"{eval_dir}/roc_curves.png")
    evaluator.plot_per_class_metrics(results, save_path=f"{eval_dir}/per_class_metrics.png")

    # Grad-CAM
    if enable_gradcam:
        print("\nGenerating Grad-CAM heatmaps...")
        try:
            evaluator.generate_gradcam_grid(
                dataset=test_dataset,
                save_dir=f"{eval_dir}/gradcam",
                num_samples=num_gradcam_samples,
            )
        except Exception as e:
            print(f"  Grad-CAM failed: {e}")
            print("  (This can happen on MPS/CPU — gradients may not be available)")

    # Error analysis
    print("\nGenerating error analysis...")
    evaluator.generate_error_analysis(
        dataset=test_dataset,
        results=results,
        save_dir=f"{eval_dir}/errors",
    )

    # Save report
    evaluator.save_report(results, save_dir=eval_dir, model_name=model_name)

    return {
        "model_name": model_name,
        "augmentation": augmentation_mode,
        "accuracy": results["accuracy"],
        "f1_macro": results["f1_macro"],
        "roc_auc_macro": results.get("roc_auc_macro"),
        "per_class": results["per_class"],
    }


def print_comparison_table(all_results: list[dict]) -> None:
    """Print a comparison table across all evaluated models."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON — TEST SET RESULTS")
    print("=" * 90)
    print(f"{'Model':<28s} {'Augmentation':<15s} {'Accuracy':>9s} {'F1 Macro':>9s} {'ROC-AUC':>8s}")
    print("-" * 90)

    for r in all_results:
        if r is None:
            continue
        auc_str = f"{r['roc_auc_macro']:.4f}" if r.get("roc_auc_macro") else "N/A"
        print(f"{r['model_name']:<28s} {r['augmentation']:<15s} "
              f"{r['accuracy']:>9.4f} {r['f1_macro']:>9.4f} {auc_str:>8s}")

    print("=" * 90)

    # Per-class breakdown
    print(f"\n{'':>28s}", end="")
    class_names = list(all_results[0]["per_class"].keys()) if all_results[0] else []
    for c in class_names:
        print(f"  {c:>12s}", end="")
    print()

    for r in all_results:
        if r is None:
            continue
        # F1 per class
        print(f"  {r['model_name']:<26s}", end="")
        for c in class_names:
            print(f"  {r['per_class'][c]['f1']:>12.3f}", end="")
        print()

    print()

    # Save comparison as JSON
    comparison_path = Path("docs/eval/model_comparison.json")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [r for r in all_results if r is not None]
    with open(comparison_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Comparison saved → {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained brain tumor classifiers")
    parser.add_argument("--model", type=str, default="all",
                        choices=list_models() + ["all"],
                        help="Model to evaluate (or 'all' for comparison)")
    parser.add_argument("--augmentation", type=str, default="conservative",
                        choices=["conservative", "aggressive"],
                        help="Which augmentation variant to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (for single model)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--no-gradcam", action="store_true",
                        help="Skip Grad-CAM generation (faster)")
    parser.add_argument("--gradcam-samples", type=int, default=5,
                        help="Number of Grad-CAM samples per class")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model == "all":
        # Evaluate all models that have checkpoints
        all_results = []
        for model_name in list_models():
            checkpoint = f"models/{model_name}_{args.augmentation}/best_model.pth"
            if Path(checkpoint).exists():
                result = evaluate_single_model(
                    model_name, config, args.augmentation,
                    enable_gradcam=not args.no_gradcam,
                    num_gradcam_samples=args.gradcam_samples,
                )
                all_results.append(result)
            else:
                print(f"\nSkipping {model_name} — no checkpoint at {checkpoint}")

        if all_results:
            print_comparison_table(all_results)
        else:
            print("\nNo trained models found. Run training first:")
            print("  python scripts/train.py --model all")
    else:
        result = evaluate_single_model(
            args.model, config, args.augmentation,
            checkpoint_path=args.checkpoint,
            enable_gradcam=not args.no_gradcam,
            num_gradcam_samples=args.gradcam_samples,
        )
        if result:
            print(f"\nDone. Results in docs/eval/{args.model}_{args.augmentation}/")


if __name__ == "__main__":
    main()
