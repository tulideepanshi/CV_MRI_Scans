#!/usr/bin/env python3
"""
Main training script — run a single model training experiment.

This script wires together: config → data → model → trainer → results.

Usage:
    # Train with defaults (EfficientNet-B0, conservative augmentation)
    python scripts/train.py

    # Train a specific model with specific augmentation
    python scripts/train.py --model resnet_pretrained --augmentation aggressive

    # Train all 5 models sequentially (for full comparison)
    python scripts/train.py --model all

    # Override config from command line
    python scripts/train.py --model custom_cnn --epochs 10 --batch-size 16
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.reproducibility import seed_everything, get_device
from src.data.dataset import create_dataloaders, get_inverse_class_weights
from src.models.factory import create_model, count_parameters, list_models
from src.training.trainer import Trainer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_single_model(
    model_name: str,
    config: dict,
    augmentation_mode: str = "conservative",
    use_mixup: bool = False,
) -> dict:
    """Train a single model and return results."""
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print(f"Augmentation: {augmentation_mode}" + (" + mixup" if use_mixup else ""))
    print("=" * 60)

    # Reproducibility
    seed = config.get("training", {}).get("seed", 42)
    seed_everything(seed)
    device = get_device()

    # Data
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    aug_cfg = config.get("augmentation", {})

    print("\nLoading data...")
    loaders = create_dataloaders(
        split_dir=data_cfg.get("split_dir", "data/splits"),
        augmentation_mode=augmentation_mode,
        image_size=data_cfg.get("image_size", 224),
        mean=data_cfg.get("normalization", {}).get("mean"),
        std=data_cfg.get("normalization", {}).get("std"),
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        seed=seed,
        augmentation_config=aug_cfg,
    )

    # Class weights for loss function
    class_weights = get_inverse_class_weights(data_cfg.get("split_dir", "data/splits"))
    print(f"Class weights: {class_weights.tolist()}")

    # Model
    model_cfg = config.get("models", {}).get(model_name, {})
    num_classes = data_cfg.get("num_classes", 4)
    model = create_model(model_name, model_cfg, num_classes)

    params = count_parameters(model)
    print(f"\nModel: {model_name}")
    print(f"  Parameters: {params['total']:,} ({params['total_mb']:.1f} MB)")
    print(f"  Trainable: {params['trainable']:,}")

    # Output directory — unique per model and augmentation
    output_dir = Path("models") / f"{model_name}_{augmentation_mode}"
    if use_mixup:
        output_dir = Path("models") / f"{model_name}_{augmentation_mode}_mixup"

    # Mixup config
    mixup_alpha = 0.0
    if use_mixup:
        mixup_alpha = aug_cfg.get("aggressive", {}).get("mixup", {}).get("alpha", 0.2)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config,
        device=device,
        output_dir=str(output_dir),
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        class_weights=class_weights,
    )

    # Train
    history = trainer.train()

    return {
        "model_name": model_name,
        "augmentation": augmentation_mode,
        "mixup": use_mixup,
        "best_val_f1": max(history["val_f1_macro"]),
        "best_val_acc": max(history["val_acc"]),
        "epochs_trained": len(history["train_loss"]),
        "params": params,
    }


def main():
    parser = argparse.ArgumentParser(description="Train brain tumor classifier")
    parser.add_argument("--model", type=str, default="efficientnet_pretrained",
                        choices=list_models() + ["all"],
                        help="Model to train (or 'all' for full comparison)")
    parser.add_argument("--augmentation", type=str, default="conservative",
                        choices=["conservative", "aggressive"],
                        help="Augmentation strategy")
    parser.add_argument("--mixup", action="store_true",
                        help="Enable mixup (only with aggressive augmentation)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply command-line overrides
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
        config["training"]["scheduler"]["T_max"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    # Train
    if args.model == "all":
        # Full comparison: train all 5 models
        results = []
        for model_name in list_models():
            result = train_single_model(model_name, config, args.augmentation, args.mixup)
            results.append(result)

        # Print comparison table
        print("\n" + "=" * 80)
        print("FINAL COMPARISON")
        print("=" * 80)
        print(f"{'Model':<28s} {'Params':>10s} {'Best F1':>8s} {'Best Acc':>9s} {'Epochs':>7s}")
        print("-" * 80)
        for r in results:
            print(f"{r['model_name']:<28s} {r['params']['total']:>10,d} "
                  f"{r['best_val_f1']:>8.4f} {r['best_val_acc']:>9.4f} {r['epochs_trained']:>7d}")
        print("=" * 80)
    else:
        train_single_model(args.model, config, args.augmentation, args.mixup)


if __name__ == "__main__":
    main()
