"""
Shared fixtures for the brain tumor classification test suite.

Provides:
  - Temporary directory structures mimicking the real dataset layout
  - Dummy images for testing the data pipeline
  - Pre-built model instances for architecture tests
  - Config fixtures matching config.yaml defaults
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


# ============================================================
# Constants
# ============================================================

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = 4
IMAGE_SIZE = 224


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def dummy_image_np():
    """Random 224x224 RGB numpy array (uint8), mimics a brain MRI."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


@pytest.fixture
def dummy_batch():
    """Random batch of 4 images as a tensor (B, C, H, W)."""
    return torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def dummy_labels():
    """Random labels for a batch of 4."""
    return torch.tensor([0, 1, 2, 3], dtype=torch.long)


@pytest.fixture
def dummy_split_dir(tmp_path):
    """
    Create a temporary split directory with tiny dummy images.

    Structure:
      tmp_path/
        train/{glioma,meningioma,notumor,pituitary}/  (8 images each)
        val/{...}/   (2 images each)
        test/{...}/  (2 images each)
    """
    rng = np.random.RandomState(42)
    splits = {"train": 8, "val": 2, "test": 2}

    for split_name, n_per_class in splits.items():
        for cls in CLASSES:
            cls_dir = tmp_path / split_name / cls
            cls_dir.mkdir(parents=True)
            for i in range(n_per_class):
                img_arr = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                img = Image.fromarray(img_arr)
                img.save(cls_dir / f"{cls}_{i:03d}.jpg")

    return tmp_path


@pytest.fixture
def dummy_data_dir(tmp_path):
    """
    Create a raw data directory mimicking the Kaggle dataset layout.

    Structure:
      tmp_path/Training/{class}/  and  tmp_path/Testing/{class}/
    """
    rng = np.random.RandomState(42)

    for split in ["Training", "Testing"]:
        for cls in CLASSES:
            cls_dir = tmp_path / split / cls
            cls_dir.mkdir(parents=True)
            n = 6 if split == "Training" else 2
            for i in range(n):
                img_arr = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                img = Image.fromarray(img_arr)
                img.save(cls_dir / f"{cls}_{split[0]}_{i:03d}.jpg")

    return tmp_path


@pytest.fixture
def sample_config():
    """Minimal config dict matching config.yaml structure."""
    return {
        "data": {
            "image_size": IMAGE_SIZE,
            "num_classes": NUM_CLASSES,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "models": {
            "custom_cnn": {"num_conv_blocks": 4, "initial_filters": 32, "dropout": 0.5},
            "resnet_scratch": {"num_blocks": [2, 2, 2, 2], "initial_filters": 64, "dropout": 0.3},
            "mobilenet_scratch": {"width_multiplier": 1.0, "dropout": 0.2},
            "resnet_pretrained": {"pretrained": False, "freeze_backbone": False, "dropout": 0.3},
            "efficientnet_pretrained": {"pretrained": False, "freeze_backbone": False, "dropout": 0.2},
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 0.001,
        },
        "security": {
            "dicom_anonymization": {
                "fields_to_remove": ["PatientName", "PatientID"],
                "fields_to_hash": ["StudyInstanceUID"],
            },
        },
    }
