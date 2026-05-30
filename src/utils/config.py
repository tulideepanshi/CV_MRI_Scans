"""
Configuration loader — single source of truth for all pipeline settings.

Usage:
    from src.utils.config import load_config
    cfg = load_config("configs/config.yaml")
    print(cfg["data"]["image_size"])  # 224
"""

import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML config and resolve paths relative to project root."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_nested(config: dict, key_path: str, default: Any = None) -> Any:
    """
    Access nested config values with dot notation.

    Example:
        get_nested(cfg, "training.learning_rate")  # 0.001
        get_nested(cfg, "data.dedup.enabled")       # True
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
