#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader — Connecting data to models.

WHAT THIS FILE DOES:
  1. BrainTumorDataset: a PyTorch Dataset that loads images from the split
     directories, applies transforms (augmentations for train, resize+normalize
     for val/test), and returns (image_tensor, label_index) pairs.

  2. create_dataloaders: factory function that builds train/val/test DataLoaders
     with proper batching, shuffling, worker config, and optional class weighting.

WHY A CUSTOM DATASET (not ImageFolder):
  torchvision.datasets.ImageFolder would work for basic loading, but we need:
    - Albumentations integration (ImageFolder expects torchvision transforms)
    - Per-image metadata tracking (useful for Grad-CAM visualization later)
    - Flexible label encoding (string → int mapping we control)
    - Hook point for mixup (needs access to the raw batch)

HOW ALBUMENTATIONS INTEGRATES:
  Albumentations operates on numpy arrays (H, W, C), uint8.
  PyTorch expects tensors (C, H, W), float32.
  The pipeline: PIL.open → np.array → albumentations transform → tensor
  The ToTensorV2() at the end of every pipeline handles the conversion.

Usage:
    from src.data.dataset import create_dataloaders

    loaders = create_dataloaders(
        split_dir="data/splits",
        augmentation_mode="conservative",
        config=config,
    )
    for images, labels in loaders["train"]:
        # images: (B, 3, 224, 224) float32
        # labels: (B,) int64
        pass
"""

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.augmentation import get_transforms


# ============================================================
# Label encoding
# ============================================================

# Fixed label mapping — consistent across all runs and splits.
# Alphabetical order matches sklearn's label encoding convention.
CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)


# ============================================================
# Dataset
# ============================================================

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumor MRI classification.

    Loads images from directory structure:
      split_dir/{class_name}/image.jpg

    Each __getitem__ returns:
      image: tensor (C, H, W), float32, normalized
      label: tensor scalar, int64 (class index)

    Optionally stores metadata for Grad-CAM/error analysis:
      self.image_paths[i] — full path to the i-th image
      self.labels[i]      — string class name
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform=None,
        class_to_idx: dict = None,
    ):
        """
        Args:
            data_dir: Path to split directory (e.g., data/splits/train)
            transform: albumentations.Compose pipeline
            class_to_idx: label mapping (default: CLASS_TO_IDX)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx or CLASS_TO_IDX

        # Scan directory and collect all images with labels
        self.image_paths = []
        self.labels = []
        self.label_indices = []

        extensions = {".jpg", ".jpeg", ".png"}

        for class_name in sorted(self.data_dir.iterdir()):
            if not class_name.is_dir():
                continue
            if class_name.name not in self.class_to_idx:
                print(f"  WARNING: Unknown class directory '{class_name.name}', skipping.")
                continue

            label_idx = self.class_to_idx[class_name.name]

            for img_path in sorted(class_name.iterdir()):
                if img_path.suffix.lower() in extensions:
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_name.name)
                    self.label_indices.append(label_idx)

        self.label_indices = np.array(self.label_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load one image, apply transforms, return (image, label).

        Pipeline:
          1. Open with PIL (handles JPEG, PNG, grayscale, RGB)
          2. Convert to RGB (grayscale → 3-channel for pretrained models)
          3. Convert to numpy array (H, W, C), uint8
          4. Apply albumentations transform → normalized tensor (C, H, W)
          5. Return tensor and label
        """
        # Load image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_np = np.array(img)  # (H, W, 3), uint8

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=img_np)
            img_tensor = transformed["image"]  # (3, H, W), float32
        else:
            # Fallback: just convert to tensor (no resize/normalize — avoid this)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(self.label_indices[idx], dtype=torch.long)

        return img_tensor, label

    def get_class_counts(self) -> dict[str, int]:
        """Return {class_name: count} for this split."""
        return dict(Counter(self.labels))

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for loss function.

        If classes are imbalanced (ours are balanced, but this handles the
        general case), minority classes get higher weight so the model
        doesn't just predict the majority class.

        Formula: weight_c = N_total / (N_classes * N_c)
          - Balanced dataset: all weights ≈ 1.0
          - Imbalanced: rare class gets weight > 1, common class < 1
        """
        counts = Counter(self.label_indices)
        total = len(self.label_indices)
        n_classes = len(self.class_to_idx)

        weights = torch.zeros(n_classes)
        for cls_idx in range(n_classes):
            count = counts.get(cls_idx, 1)  # Avoid division by zero
            weights[cls_idx] = total / (n_classes * count)

        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """
        Per-sample weights for WeightedRandomSampler.

        Each sample gets the weight of its class. The sampler then
        draws samples with probability proportional to weight, so
        minority classes get sampled more often per epoch.

        This is different from class weights in the loss function:
          - Loss weights: penalize mistakes on rare classes more
          - Sample weights: show rare classes more often
        Both address imbalance, but from different angles.
        """
        class_weights = self.get_class_weights()
        return torch.tensor(
            [class_weights[idx].item() for idx in self.label_indices],
            dtype=torch.float64,
        )


# ============================================================
# DataLoader factory
# ============================================================

def _worker_init_fn(worker_id: int) -> None:
    """
    Seed each DataLoader worker for reproducibility.

    Must be a module-level function (not a lambda or local function)
    because Python's multiprocessing pickles everything sent to workers,
    and local functions can't be pickled.
    """
    np.random.seed(42 + worker_id)


def create_dataloaders(
    split_dir: str | Path,
    augmentation_mode: str = "conservative",
    image_size: int = 224,
    mean: list = None,
    std: list = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    use_weighted_sampler: bool = False,
    augmentation_config: dict = None,
) -> dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from split directory.

    Args:
        split_dir: Root of splits (contains train/, val/, test/)
        augmentation_mode: "conservative" or "aggressive" for train
        image_size: Resize target (224 for pretrained, can vary)
        mean/std: Normalization values
        batch_size: Images per batch
        num_workers: Parallel data loading workers
        pin_memory: Pin to GPU memory (faster transfer, uses more RAM)
        seed: For reproducibility
        use_weighted_sampler: Use WeightedRandomSampler for imbalanced data
        augmentation_config: Config dict for augmentation parameters

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}

    DataLoader configuration explained:
        shuffle=True (train only):
            Randomize batch order each epoch. Critical for SGD convergence —
            without it, the model sees the same sequence every epoch and can
            memorize the order. Disabled for val/test for reproducibility.

        drop_last=True (train only):
            If dataset size isn't divisible by batch_size, the last batch is
            smaller. BatchNorm behaves poorly with very small batches (unstable
            statistics). drop_last discards the incomplete batch.
            Val/test keep all samples for complete evaluation.

        pin_memory=True:
            Pre-allocates tensors in page-locked (pinned) CPU memory.
            Transfer to GPU is faster (DMA instead of pageable copy).
            Costs ~batch_size * image_size extra RAM. Worth it on GPU.

        num_workers=4:
            Spawn 4 worker processes for parallel image loading/augmentation.
            While GPU processes batch N, workers prepare batch N+1.
            Rule of thumb: num_workers ≈ 4 × num_GPUs.
            Too many → CPU contention. Too few → GPU starves.

        persistent_workers=True:
            Keep workers alive between epochs. Without this, workers are
            spawned and killed each epoch — the spawn overhead adds up.
    """
    split_path = Path(split_dir)

    # Get augmentation config for the chosen mode
    aug_cfg = None
    if augmentation_config and augmentation_mode in augmentation_config:
        aug_cfg = augmentation_config[augmentation_mode]

    # Build transforms
    train_transform = get_transforms(augmentation_mode, image_size, mean, std, aug_cfg)
    val_transform = get_transforms("val", image_size, mean, std)
    test_transform = get_transforms("test", image_size, mean, std)

    # Build datasets
    datasets = {}
    transform_map = {
        "train": train_transform,
        "val": val_transform,
        "test": test_transform,
    }

    for split_name, transform in transform_map.items():
        split_subdir = split_path / split_name
        if not split_subdir.exists():
            print(f"  WARNING: {split_subdir} not found, skipping.")
            continue

        ds = BrainTumorDataset(
            data_dir=split_subdir,
            transform=transform,
        )
        datasets[split_name] = ds
        counts = ds.get_class_counts()
        total = len(ds)
        print(f"  {split_name}: {total} images — {counts}")

    # Auto-detect device capabilities
    # MPS (Apple Silicon) doesn't support pin_memory
    # Also, multiprocessing workers can have pickling issues on macOS
    is_mps = torch.backends.mps.is_available()
    is_cuda = torch.cuda.is_available()

    if is_mps:
        pin_memory = False  # Not supported on MPS
        # On macOS, 'spawn' start method can cause pickling issues with
        # local functions and lambdas. Use fewer workers or 0 for safety.
        if num_workers > 0:
            num_workers = min(num_workers, 2)  # Reduce to avoid contention

    # Build DataLoaders
    loaders = {}

    for split_name, ds in datasets.items():
        is_train = (split_name == "train")

        # Sampler (mutually exclusive with shuffle)
        sampler = None
        shuffle = is_train

        if is_train and use_weighted_sampler:
            sample_weights = ds.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(ds),     # Sample same number as dataset size
                replacement=True,        # Must be True for weighted sampling
            )
            shuffle = False  # WeightedRandomSampler handles ordering

        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,          # Only drop incomplete batches for train
            worker_init_fn=_worker_init_fn,
            persistent_workers=(num_workers > 0),
        )

    return loaders


def get_inverse_class_weights(split_dir: str | Path) -> torch.Tensor:
    """
    Convenience function to get class weights from the training split.

    Used to initialize CrossEntropyLoss(weight=...) for class-balanced training.
    This is separate from the weighted sampler — you can use both together
    or either alone.
    """
    train_dir = Path(split_dir) / "train"
    ds = BrainTumorDataset(data_dir=train_dir)
    return ds.get_class_weights()
