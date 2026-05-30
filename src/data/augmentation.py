#!/usr/bin/env python3
"""
Augmentation Pipeline — Conservative vs. Aggressive for ablation study.

WHY AUGMENTATION MATTERS FOR MEDICAL IMAGES:
  Medical datasets are small. Our 7200 images across 4 classes means ~1260
  training images per class. Without augmentation, the model sees the same
  images every epoch and overfits quickly. Augmentation creates realistic
  variations that force the model to learn invariant features.

WHY TWO PIPELINES:
  Not all augmentations are clinically safe. Horizontal flipping a brain MRI
  changes left/right laterality — a glioma in the left temporal lobe becomes
  one in the right. Clinicians care about laterality, so flipping could teach
  the model the wrong spatial priors.

  Conservative: only augmentations that preserve clinical meaning.
  Aggressive:   adds flips, cutout, mixup — stronger regularization but
                potentially less clinically faithful.

  We train with both and compare (ablation study) to see if the aggressive
  augmentations actually help or if they hurt by introducing noise.

IMPLEMENTATION:
  Uses albumentations — faster than torchvision transforms because it
  operates on numpy arrays (no PIL→Tensor→PIL roundtrips) and supports
  medical-imaging-friendly transforms like elastic deformation.

Usage:
    from src.data.augmentation import get_transforms

    train_transforms = get_transforms("conservative", image_size=224, config=config)
    val_transforms = get_transforms("val", image_size=224, config=config)
"""

from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_base_transforms(
    image_size: int = 224,
    mean: list = None,
    std: list = None,
) -> A.Compose:
    """
    Minimal transforms applied to ALL images (train, val, test).

    These are preprocessing steps, not augmentations:
      1. Resize to consistent dimensions (224×224 for ImageNet-pretrained models)
      2. Normalize pixel values using channel-wise mean/std
      3. Convert to PyTorch tensor (HWC uint8 → CHW float32)

    Why Resize and not RandomResizedCrop here?
      Resize is deterministic — every image goes to exactly 224×224.
      We use this for val/test where we want reproducible results.
      RandomResizedCrop is an augmentation used only during training.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet defaults
    if std is None:
        std = [0.229, 0.224, 0.225]

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_conservative_transforms(
    image_size: int = 224,
    mean: list = None,
    std: list = None,
    config: dict = None,
) -> A.Compose:
    """
    Conservative augmentation — clinically safe transforms only.

    What each transform does and WHY it's safe:

    ShiftScaleRotate:
      - Shift: translates the image (patient not perfectly centered)
      - Scale: slight zoom in/out (scanner distance varies)
      - Rotate: ±15° (head tilt during scan is common)
      All three simulate real-world scanner variability.

    RandomBrightnessContrast:
      - Brightness ±15%: MRI intensity varies between scanners/protocols
      - Contrast ±15%: window/level settings differ across institutions
      Simulates the reality that the same tumor looks different on
      different machines.

    ElasticTransform:
      - Applies smooth spatial deformation (like poking a rubber sheet)
      - Alpha=50 (displacement magnitude), sigma=5 (smoothness)
      - Simulates soft tissue deformation between scans
      - Low probability (0.3) to avoid over-distortion

    GaussNoise:
      - Adds slight random noise (var_limit=10)
      - Simulates sensor noise in the MRI receiver coils
      - Very conservative — just enough to prevent pixel-level memorization

    What's EXCLUDED and why:
      - HorizontalFlip: laterality matters (left vs right hemisphere)
      - VerticalFlip: orientation matters (superior vs inferior)
      - ColorJitter/HueSaturation: MRI is grayscale, color shifts are meaningless
      - Cutout/CoarseDropout: could occlude the tumor itself
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    # Pull config values or use defaults
    cfg = config or {}
    rotation = cfg.get("rotation_limit", 15)
    scale = cfg.get("scale_range", [0.9, 1.1])
    brightness = cfg.get("brightness_limit", 0.15)
    contrast = cfg.get("contrast_limit", 0.15)
    elastic_cfg = cfg.get("elastic_transform", {})
    elastic_enabled = elastic_cfg.get("enabled", True)
    elastic_alpha = elastic_cfg.get("alpha", 50)
    elastic_sigma = elastic_cfg.get("sigma", 5)
    elastic_prob = elastic_cfg.get("probability", 0.3)

    transforms_list = [
        # Resize first — all augmentations operate on consistent dimensions
        A.Resize(image_size, image_size),

        # Spatial augmentations (Affine replaces deprecated ShiftScaleRotate)
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(scale[0], scale[1]),
            rotate=(-rotation, rotation),
            mode=0,                    # cv2.BORDER_CONSTANT (black fill)
            cval=0,                    # Fill with black (background)
            p=0.7,                     # Apply 70% of the time
        ),

        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=brightness,
            contrast_limit=contrast,
            p=0.5,
        ),

        # Noise — very subtle
        A.GaussNoise(
            std_range=(0.02, 0.06),    # Normalized noise std range
            p=0.3,
        ),
    ]

    # Elastic transform — optional
    if elastic_enabled:
        transforms_list.append(
            A.ElasticTransform(
                alpha=elastic_alpha,
                sigma=elastic_sigma,
                p=elastic_prob,
            )
        )

    # Always end with normalization and tensor conversion
    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


def get_aggressive_transforms(
    image_size: int = 224,
    mean: list = None,
    std: list = None,
    config: dict = None,
) -> A.Compose:
    """
    Aggressive augmentation — stronger regularization for ablation study.

    Everything from conservative PLUS:

    HorizontalFlip:
      - Doubles effective dataset size
      - Sacrifices laterality information
      - Ablation question: does the accuracy gain outweigh clinical correctness?

    CoarseDropout (Cutout):
      - Randomly blacks out a rectangular patch
      - Forces the model to use ALL features, not just the most discriminative one
      - Risk: could occlude the tumor, but the model learns to be robust to
        partial occlusion (which happens in real clinical images too)

    GaussianBlur:
      - Simulates out-of-focus or motion-blurred scans
      - Common artifact in clinical MRI when patient moves

    CLAHE (Contrast Limited Adaptive Histogram Equalization):
      - Enhances local contrast
      - Common preprocessing step in radiology — some institutions apply
        it before sending images to AI systems

    Note: Mixup is NOT here — it operates on pairs of images and must be
    implemented at the DataLoader level, not per-image. See dataset.py.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    cfg = config or {}
    rotation = cfg.get("rotation_limit", 30)
    scale = cfg.get("scale_range", [0.8, 1.2])
    brightness = cfg.get("brightness_limit", 0.2)
    contrast = cfg.get("contrast_limit", 0.2)
    elastic_cfg = cfg.get("elastic_transform", {})
    elastic_enabled = elastic_cfg.get("enabled", True)
    elastic_alpha = elastic_cfg.get("alpha", 80)
    elastic_sigma = elastic_cfg.get("sigma", 5)
    elastic_prob = elastic_cfg.get("probability", 0.5)
    h_flip = cfg.get("horizontal_flip", True)
    cutout_cfg = cfg.get("cutout", {})
    cutout_enabled = cutout_cfg.get("enabled", True)

    transforms_list = [
        A.Resize(image_size, image_size),

        # Spatial augmentations — wider range than conservative
        A.Affine(
            translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
            scale=(scale[0], scale[1]),
            rotate=(-rotation, rotation),
            mode=0,
            cval=0,
            p=0.8,
        ),
    ]

    # Horizontal flip — the key ablation variable
    if h_flip:
        transforms_list.append(A.HorizontalFlip(p=0.5))

    transforms_list.extend([
        # Intensity augmentations — wider range
        A.RandomBrightnessContrast(
            brightness_limit=brightness,
            contrast_limit=contrast,
            p=0.6,
        ),

        # CLAHE — adaptive histogram equalization
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.3,
        ),

        # Gaussian blur — motion/focus artifact
        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2,
        ),

        # Noise — slightly stronger
        A.GaussNoise(
            std_range=(0.02, 0.1),
            p=0.4,
        ),
    ])

    # Elastic transform
    if elastic_enabled:
        transforms_list.append(
            A.ElasticTransform(
                alpha=elastic_alpha,
                sigma=elastic_sigma,
                p=elastic_prob,
            )
        )

    # Cutout (CoarseDropout)
    if cutout_enabled:
        transforms_list.append(
            A.CoarseDropout(
                max_holes=cutout_cfg.get("num_holes", 1),
                max_height=cutout_cfg.get("max_h_size", 40),
                max_width=cutout_cfg.get("max_w_size", 40),
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,          # Black fill
                p=0.4,
            )
        )

    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


def get_transforms(
    mode: str,
    image_size: int = 224,
    mean: list = None,
    std: list = None,
    config: dict = None,
) -> A.Compose:
    """
    Factory function — get the right transform pipeline by name.

    Args:
        mode: One of:
            "conservative" — safe augmentations for training
            "aggressive"   — strong augmentations for ablation
            "val" / "test" — no augmentation, just resize + normalize
        image_size: Target size (default 224)
        mean: Normalization mean (default ImageNet)
        std: Normalization std (default ImageNet)
        config: Augmentation config dict from config.yaml

    Returns:
        albumentations.Compose pipeline
    """
    if mode in ("val", "test", "inference"):
        return get_base_transforms(image_size, mean, std)
    elif mode == "conservative":
        return get_conservative_transforms(image_size, mean, std, config)
    elif mode == "aggressive":
        return get_aggressive_transforms(image_size, mean, std, config)
    else:
        raise ValueError(
            f"Unknown transform mode: '{mode}'. "
            f"Choose from: 'conservative', 'aggressive', 'val', 'test'"
        )


def mixup_data(
    images,
    labels,
    alpha: float = 0.2,
):
    """
    Mixup augmentation — operates on BATCHES, not individual images.

    Called during training after the DataLoader returns a batch.
    Not part of the albumentations pipeline because it needs pairs.

    How it works:
      1. Sample lambda from Beta(alpha, alpha) — controls blend ratio
      2. Shuffle the batch to create random pairs
      3. Mixed image = lambda * image_A + (1-lambda) * image_B
      4. Mixed label = lambda * onehot_A + (1-lambda) * onehot_B

    Why Beta distribution:
      - alpha=0.2 → lambda clusters near 0 or 1 (subtle blending)
      - alpha=1.0 → lambda is uniform [0,1] (aggressive blending)
      - Low alpha means most mixed images look like one class with a
        ghost of another — gentle regularization.

    Args:
        images: Batch tensor (B, C, H, W)
        labels: Batch tensor (B,) integer class labels
        alpha: Beta distribution parameter

    Returns:
        mixed_images, labels_a, labels_b, lam
        (caller computes loss as: lam * loss(pred, labels_a) + (1-lam) * loss(pred, labels_b))
    """
    import torch

    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam
