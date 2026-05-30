#!/usr/bin/env python3
"""
Model Factory — Create any model by name from config.

Centralizes model creation so the training loop doesn't need to know
which architecture it's training. Just pass the model name and config,
get back a ready-to-train nn.Module.

Usage:
    from src.models.factory import create_model, list_models

    model = create_model("resnet_pretrained", config["models"]["resnet_pretrained"])
    print(list_models())  # ['custom_cnn', 'resnet_scratch', ...]
"""

import torch.nn as nn

from src.models.custom_cnn import CustomCNN
from src.models.mobilenet_scratch import MobileNetScratch
from src.models.pretrained import PretrainedEfficientNet, PretrainedResNet
from src.models.resnet_scratch import ResNetScratch


# Registry of all available models
MODEL_REGISTRY = {
    "custom_cnn": CustomCNN,
    "resnet_scratch": ResNetScratch,
    "mobilenet_scratch": MobileNetScratch,
    "resnet_pretrained": PretrainedResNet,
    "efficientnet_pretrained": PretrainedEfficientNet,
}


def create_model(
    model_name: str,
    model_config: dict = None,
    num_classes: int = 4,
) -> nn.Module:
    """
    Create a model by name with config-driven parameters.

    Args:
        model_name: Key in MODEL_REGISTRY
        model_config: Model-specific config dict from config.yaml
        num_classes: Number of output classes

    Returns:
        Initialized nn.Module ready for training
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = model_config or {}
    model_class = MODEL_REGISTRY[model_name]

    # Build kwargs based on model type
    if model_name == "custom_cnn":
        model = model_class(
            num_classes=num_classes,
            num_conv_blocks=cfg.get("num_conv_blocks", 4),
            initial_filters=cfg.get("initial_filters", 32),
            dropout=cfg.get("dropout", 0.5),
        )

    elif model_name == "resnet_scratch":
        model = model_class(
            num_classes=num_classes,
            num_blocks=cfg.get("num_blocks", [2, 2, 2, 2]),
            initial_filters=cfg.get("initial_filters", 64),
            dropout=cfg.get("dropout", 0.3),
        )

    elif model_name == "mobilenet_scratch":
        model = model_class(
            num_classes=num_classes,
            width_multiplier=cfg.get("width_multiplier", 1.0),
            dropout=cfg.get("dropout", 0.2),
        )

    elif model_name == "resnet_pretrained":
        model = model_class(
            num_classes=num_classes,
            pretrained=cfg.get("pretrained", True),
            freeze_backbone=cfg.get("freeze_backbone", False),
            dropout=cfg.get("dropout", 0.3),
        )

    elif model_name == "efficientnet_pretrained":
        model = model_class(
            num_classes=num_classes,
            pretrained=cfg.get("pretrained", True),
            freeze_backbone=cfg.get("freeze_backbone", False),
            dropout=cfg.get("dropout", 0.2),
        )

    else:
        # Generic fallback (shouldn't reach here due to registry check)
        model = model_class(num_classes=num_classes)

    return model


def list_models() -> list[str]:
    """Return names of all available models."""
    return list(MODEL_REGISTRY.keys())


def count_parameters(model: nn.Module) -> dict:
    """
    Count model parameters — useful for comparison table.

    Returns:
        {
            "total": int,           # All parameters
            "trainable": int,       # Parameters with requires_grad=True
            "frozen": int,          # Parameters with requires_grad=False
            "total_mb": float,      # Size in megabytes (float32)
        }
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_mb": total * 4 / (1024 * 1024),  # float32 = 4 bytes
    }


def print_model_comparison(num_classes: int = 4) -> None:
    """Print a comparison table of all models — great for interview slides."""
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<28s} {'Total Params':>14s} {'Trainable':>12s} {'Size (MB)':>10s}")
    print("-" * 70)

    for name in list_models():
        model = create_model(name, num_classes=num_classes)
        stats = count_parameters(model)
        print(f"{name:<28s} {stats['total']:>14,d} {stats['trainable']:>12,d} {stats['total_mb']:>10.1f}")

    print("=" * 70)
