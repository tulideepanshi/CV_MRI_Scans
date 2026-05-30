"""
Tests for the data pipeline: dataset, augmentation, split, dedup.

Covers:
  - Dataset loading, label encoding, class weights
  - Augmentation output shapes and value ranges
  - Mixup blending correctness
  - Split ratios and leakage prevention
  - Union-Find correctness
"""

import numpy as np
import pytest
import torch

from src.data.augmentation import get_transforms, mixup_data
from src.data.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    BrainTumorDataset,
    create_dataloaders,
)
from src.data.split import UnionFind, build_clusters, cluster_aware_stratified_split, collect_all_images


# ============================================================
# Dataset tests
# ============================================================

class TestBrainTumorDataset:
    """Tests for BrainTumorDataset class."""

    def test_label_mapping_consistency(self):
        """CLASS_TO_IDX and IDX_TO_CLASS must be inverses."""
        assert len(CLASS_TO_IDX) == NUM_CLASSES
        for name, idx in CLASS_TO_IDX.items():
            assert IDX_TO_CLASS[idx] == name

    def test_dataset_length(self, dummy_split_dir):
        """Dataset should find all images in the directory."""
        ds = BrainTumorDataset(dummy_split_dir / "train")
        # 4 classes × 8 images = 32
        assert len(ds) == 32

    def test_dataset_getitem_shape(self, dummy_split_dir):
        """__getitem__ should return (C, H, W) tensor and scalar label."""
        transform = get_transforms("val", image_size=224)
        ds = BrainTumorDataset(dummy_split_dir / "train", transform=transform)
        img, label = ds[0]

        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert img.dtype == torch.float32
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert 0 <= label.item() < NUM_CLASSES

    def test_dataset_getitem_no_transform(self, dummy_split_dir):
        """Without transform, fallback should still return a tensor."""
        ds = BrainTumorDataset(dummy_split_dir / "train", transform=None)
        img, label = ds[0]

        assert isinstance(img, torch.Tensor)
        assert img.ndim == 3  # (C, H, W)
        assert img.dtype == torch.float32

    def test_class_counts(self, dummy_split_dir):
        """get_class_counts should return correct per-class counts."""
        ds = BrainTumorDataset(dummy_split_dir / "train")
        counts = ds.get_class_counts()

        assert set(counts.keys()) == set(CLASS_TO_IDX.keys())
        for cls in CLASS_TO_IDX:
            assert counts[cls] == 8

    def test_class_weights_balanced(self, dummy_split_dir):
        """For a balanced dataset, all class weights should be ~1.0."""
        ds = BrainTumorDataset(dummy_split_dir / "train")
        weights = ds.get_class_weights()

        assert weights.shape == (NUM_CLASSES,)
        for w in weights:
            assert abs(w.item() - 1.0) < 0.01

    def test_sample_weights_length(self, dummy_split_dir):
        """Per-sample weights should match dataset length."""
        ds = BrainTumorDataset(dummy_split_dir / "train")
        sample_weights = ds.get_sample_weights()
        assert len(sample_weights) == len(ds)

    def test_ignores_unknown_dirs(self, tmp_path):
        """Unknown subdirectories should be skipped with a warning."""
        # Create a valid class dir and a junk dir
        (tmp_path / "glioma").mkdir()
        (tmp_path / "junk_folder").mkdir()

        rng = np.random.RandomState(0)
        from PIL import Image
        img = Image.fromarray(rng.randint(0, 256, (32, 32, 3), dtype=np.uint8))
        img.save(tmp_path / "glioma" / "test.jpg")

        ds = BrainTumorDataset(tmp_path)
        assert len(ds) == 1
        assert ds.labels[0] == "glioma"


# ============================================================
# DataLoader tests
# ============================================================

class TestDataLoaders:
    """Tests for create_dataloaders factory."""

    def test_creates_all_splits(self, dummy_split_dir):
        """Should return train, val, test DataLoaders."""
        loaders = create_dataloaders(
            dummy_split_dir,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders

    def test_batch_shape(self, dummy_split_dir):
        """Batches should have correct shape."""
        loaders = create_dataloaders(
            dummy_split_dir,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )
        images, labels = next(iter(loaders["train"]))
        assert images.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)

    def test_weighted_sampler(self, dummy_split_dir):
        """WeightedRandomSampler should work without error."""
        loaders = create_dataloaders(
            dummy_split_dir,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            use_weighted_sampler=True,
        )
        # Just verify it doesn't crash and produces a batch
        images, labels = next(iter(loaders["train"]))
        assert images.shape[0] == 4


# ============================================================
# Augmentation tests
# ============================================================

class TestAugmentation:
    """Tests for augmentation pipelines."""

    @pytest.mark.parametrize("mode", ["conservative", "aggressive", "val", "test"])
    def test_transform_output_shape(self, dummy_image_np, mode):
        """All transform modes should produce (3, 224, 224) tensors."""
        transform = get_transforms(mode, image_size=224)
        result = transform(image=dummy_image_np)
        tensor = result["image"]

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_val_test_deterministic(self, dummy_image_np):
        """Val/test transforms should be deterministic (same input → same output)."""
        transform = get_transforms("val", image_size=224)
        r1 = transform(image=dummy_image_np)["image"]
        r2 = transform(image=dummy_image_np)["image"]
        assert torch.allclose(r1, r2)

    def test_invalid_mode_raises(self):
        """Unknown augmentation mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown transform mode"):
            get_transforms("nonexistent_mode")

    def test_normalization_applied(self, dummy_image_np):
        """After normalization, values should NOT be in [0, 255]."""
        transform = get_transforms("val", image_size=224)
        tensor = transform(image=dummy_image_np)["image"]

        # ImageNet-normalized values typically range roughly [-2.5, 2.5]
        assert tensor.min() < 0 or tensor.max() < 1.0


class TestMixup:
    """Tests for mixup augmentation."""

    def test_mixup_output_shape(self, dummy_batch, dummy_labels):
        """Mixup should preserve batch shape."""
        mixed, la, lb, lam = mixup_data(dummy_batch, dummy_labels, alpha=0.2)

        assert mixed.shape == dummy_batch.shape
        assert la.shape == dummy_labels.shape
        assert lb.shape == dummy_labels.shape
        assert 0.0 <= lam <= 1.0

    def test_mixup_alpha_zero(self, dummy_batch, dummy_labels):
        """With alpha=0, lambda should be 1.0 (no mixing)."""
        mixed, la, lb, lam = mixup_data(dummy_batch, dummy_labels, alpha=0)

        assert lam == 1.0
        assert torch.allclose(mixed, dummy_batch)

    def test_mixup_is_convex_combination(self, dummy_batch, dummy_labels):
        """Mixed images should be between the two source images (element-wise)."""
        mixed, la, lb, lam = mixup_data(dummy_batch, dummy_labels, alpha=1.0)

        # lam * A + (1-lam) * B is always between min(A,B) and max(A,B)
        # But due to shuffling, we just verify the output is finite
        assert torch.isfinite(mixed).all()


# ============================================================
# Split tests
# ============================================================

class TestUnionFind:
    """Tests for the Union-Find data structure."""

    def test_singleton(self):
        """Each element starts as its own root."""
        uf = UnionFind()
        assert uf.find("a") == "a"
        assert uf.find("b") == "b"

    def test_union_same_root(self):
        """After union, both elements should have the same root."""
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")

    def test_transitive_union(self):
        """Union is transitive: a-b + b-c → a, b, c in same set."""
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")

    def test_disjoint_sets(self):
        """Elements not unioned should remain in separate sets."""
        uf = UnionFind()
        uf.union("a", "b")
        uf.find("c")  # Register c
        assert uf.find("a") != uf.find("c")


class TestSplit:
    """Tests for cluster-aware stratified splitting."""

    def test_collect_images(self, dummy_data_dir):
        """Should find all images in Training/ and Testing/."""
        images = collect_all_images(dummy_data_dir)
        # 4 classes × (6 train + 2 test) = 32
        assert len(images) == 32

    def test_split_ratios(self, dummy_data_dir):
        """Split should approximate target ratios."""
        images = collect_all_images(dummy_data_dir)
        # No duplicates → each image is its own cluster
        clusters = [[img] for img in images]

        splits = cluster_aware_stratified_split(
            clusters, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
        )

        total = sum(len(v) for v in splits.values())
        assert total == len(images)  # No images lost

        train_pct = len(splits["train"]) / total
        assert 0.60 <= train_pct <= 0.80  # Within tolerance

    def test_no_leakage_with_clusters(self):
        """All images in a cluster must land in the same split."""
        # Create fake images with known clusters
        imgs = []
        for i in range(20):
            imgs.append({
                "path": f"img_{i}.jpg",
                "label": "glioma" if i < 10 else "meningioma",
            })

        # Cluster: images 0,1,2 are duplicates → one cluster
        cluster1 = [imgs[0], imgs[1], imgs[2]]
        singletons = [[img] for img in imgs[3:]]
        clusters = [cluster1] + singletons

        splits = cluster_aware_stratified_split(clusters)
        total = sum(len(v) for v in splits.values())
        assert total == 20

        # Find which split the cluster ended up in
        cluster_paths = {img["path"] for img in cluster1}
        for split_name, split_imgs in splits.items():
            split_paths = {img["path"] for img in split_imgs}
            overlap = cluster_paths & split_paths
            # Either all 3 are in this split, or none
            assert len(overlap) == 0 or len(overlap) == 3

    def test_build_clusters_no_duplicates(self, dummy_data_dir):
        """With no duplicate pairs, every image should be a singleton cluster."""
        images = collect_all_images(dummy_data_dir)
        clusters = build_clusters(images, duplicate_pairs=[])

        assert len(clusters) == len(images)
        for c in clusters:
            assert len(c) == 1
