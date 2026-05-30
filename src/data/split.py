#!/usr/bin/env python3
"""
Cluster-Aware Data Splitting — Preventing data leakage while preserving all samples.

WHY CLUSTER-AWARE:
  Near-duplicate images (same patient, similar slice) must land in the SAME
  pool (train/val/test). If one copy ends up in train and another in test,
  the model gets an unfair advantage — inflated metrics that won't generalize.

  Naive approach: delete duplicates. Problem: you lose samples.
  Our approach: group duplicates into clusters, split at cluster level.

HOW IT WORKS:
  1. Collect all images from Training/ and Testing/ into a single pool
  2. Load duplicate pairs from data/duplicates.json (produced by dedup.py)
  3. Build connected components using Union-Find:
     - If A≈B and B≈C, then {A, B, C} form one cluster
     - All three MUST go to the same split
  4. Stratified split at CLUSTER level (stratify by cluster's class label)
  5. Expand clusters back to images → copy to split directories

  Result: zero cross-pool leakage, zero samples lost.

Usage:
    # First run dedup to produce the pairs file:
    python -m src.data.dedup --data-dir datasets

    # Then split using those pairs:
    python -m src.data.split --data-dir datasets --output-dir data/splits
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


# ============================================================
# Union-Find (Disjoint Set) — for building duplicate clusters
# ============================================================

class UnionFind:
    """
    Union-Find with path compression and union by rank.

    Why Union-Find instead of, say, BFS on an adjacency list?
    - Nearly O(1) amortized per operation (inverse Ackermann)
    - Simple to implement, no recursion depth issues
    - Perfect for "group these pairs" problems

    How it works:
      - Every element starts as its own parent (singleton cluster)
      - find(x): walk up parent pointers to the root, compress path
      - union(x, y): merge the two roots, attach smaller tree under larger
    """

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        """Find root of x with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Merge clusters containing x and y. Union by rank."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return  # Already in the same cluster

        # Attach smaller tree under larger tree's root
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


# ============================================================
# Core pipeline functions
# ============================================================

def collect_all_images(data_dir: Path) -> list[dict]:
    """Collect all images from all subdirectories."""
    extensions = {".jpg", ".jpeg", ".png"}
    images = []

    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix.lower() in extensions:
            images.append({
                "path": img_path,
                "label": img_path.parent.name,
                "original_split": "train" if "Training" in str(img_path) else "test",
            })

    return images


def load_duplicate_pairs(pairs_path: Path) -> list[dict]:
    """
    Load near-duplicate pairs from JSON (produced by dedup.py).

    Returns list of {"path1": str, "path2": str, "distance": int}
    """
    if not pairs_path.exists():
        print(f"  No duplicate pairs file found at {pairs_path}")
        print("  Running without cluster awareness (naive split).")
        print("  Hint: run `python -m src.data.dedup` first to generate it.")
        return []

    with open(pairs_path) as f:
        pairs = json.load(f)

    print(f"  Loaded {len(pairs)} duplicate pairs from {pairs_path}")
    return pairs


def build_clusters(
    images: list[dict],
    duplicate_pairs: list[dict],
) -> list[list[dict]]:
    """
    Group images into clusters using Union-Find on duplicate pairs.

    Images with no duplicates form singleton clusters (size 1).
    Images linked by near-duplicate relationships form larger clusters.

    Returns:
        List of clusters, where each cluster is a list of image dicts.
        Every image appears in exactly one cluster.
    """
    uf = UnionFind()

    # Index images by path string for lookup
    path_to_img = {}
    for img in images:
        path_str = str(img["path"])
        path_to_img[path_str] = img
        uf.find(path_str)  # Register every image in Union-Find

    # Merge duplicate pairs
    merged_count = 0
    for pair in duplicate_pairs:
        p1, p2 = pair["path1"], pair["path2"]
        # Only merge if both paths exist in our image pool
        if p1 in path_to_img and p2 in path_to_img:
            uf.union(p1, p2)
            merged_count += 1

    print(f"  Merged {merged_count} pairs into clusters")

    # Group images by their cluster root
    clusters_dict = {}
    for path_str, img in path_to_img.items():
        root = uf.find(path_str)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(img)

    clusters = list(clusters_dict.values())

    # Report cluster size distribution
    sizes = [len(c) for c in clusters]
    singletons = sum(1 for s in sizes if s == 1)
    multi = len(clusters) - singletons
    max_size = max(sizes) if sizes else 0

    print(f"  Total clusters: {len(clusters)}")
    print(f"  Singletons (no duplicates): {singletons}")
    print(f"  Multi-image clusters: {multi}")
    print(f"  Largest cluster: {max_size} images")

    return clusters


def get_cluster_label(cluster: list[dict]) -> str:
    """
    Determine the class label for a cluster.

    In a well-formed dataset, all images in a duplicate cluster
    should share the same label (a glioma MRI won't be a near-duplicate
    of a pituitary MRI). We take majority vote as a safety net.
    """
    labels = [img["label"] for img in cluster]
    counter = Counter(labels)
    majority_label = counter.most_common(1)[0][0]

    if len(counter) > 1:
        # This would be suspicious — duplicates across classes
        print(f"  WARNING: Mixed-label cluster detected: {counter}")
        print(f"    Using majority label: {majority_label}")

    return majority_label


def cluster_aware_stratified_split(
    clusters: list[list[dict]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Weighted greedy cluster assignment — balanced by IMAGE count, not cluster count.

    THE PROBLEM WITH NAIVE CLUSTER SPLITTING:
      sklearn's train_test_split treats every cluster as one unit. A cluster
      of 8 images has the same weight as a singleton. If large clusters
      concentrate in one pool, the image-level ratio drifts far from 70/15/15.

    OUR APPROACH — GREEDY BIN PACKING PER CLASS:
      1. Group clusters by class label
      2. For each class independently:
         a. Compute target image counts: e.g., 1800 glioma × 0.70 = 1260 for train
         b. Sort clusters LARGEST FIRST (deterministic tie-breaking by seed)
         c. For each cluster, assign it to the pool whose class is furthest
            below its target (measured as: target - current count)
      3. Merge per-class assignments into final splits

    WHY LARGEST FIRST:
      Large clusters are the hard constraint — they cause the most deviation
      if placed badly. By placing them first while all pools are empty,
      we give the algorithm maximum flexibility. Small clusters (singletons)
      placed last act as "fine-tuning" to close the remaining gaps.

      This is the same intuition as First Fit Decreasing in bin packing.

    GUARANTEES:
      - All images in a cluster stay together (no leakage)
      - Every image is assigned (no loss)
      - Image counts per class approximate the target ratios
      - Deterministic given the same seed
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    pool_names = ["train", "val", "test"]
    pool_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    # Group clusters by their class label
    clusters_by_class = {}
    for cluster in clusters:
        label = get_cluster_label(cluster)
        if label not in clusters_by_class:
            clusters_by_class[label] = []
        clusters_by_class[label].append(cluster)

    # Initialize the split result
    splits = {"train": [], "val": [], "test": []}

    rng = np.random.RandomState(seed)

    for label in sorted(clusters_by_class.keys()):
        class_clusters = clusters_by_class[label]

        # Total images for this class
        total_images = sum(len(c) for c in class_clusters)

        # Target image counts per pool for this class
        targets = {pool: total_images * pool_ratios[pool] for pool in pool_names}

        # Current image counts per pool for this class (starts at 0)
        current = {pool: 0 for pool in pool_names}

        # Sort clusters: largest first, with random tie-breaking for determinism
        # Shuffle first so equal-sized clusters get random order, then stable-sort
        # by size descending. This gives deterministic but varied tie-breaking.
        rng.shuffle(class_clusters)
        class_clusters.sort(key=lambda c: len(c), reverse=True)

        # Greedy assignment: give each cluster to the most underfilled pool
        for cluster in class_clusters:
            size = len(cluster)

            # Find the pool with the largest remaining deficit
            # deficit = target - current (higher deficit = more underfilled)
            best_pool = max(pool_names, key=lambda p: targets[p] - current[p])

            splits[best_pool].extend(cluster)
            current[best_pool] += size

        # Report per-class allocation
        print(f"  {label}: {total_images} images → ", end="")
        parts = []
        for pool in pool_names:
            actual_pct = current[pool] / total_images * 100 if total_images > 0 else 0
            target_pct = pool_ratios[pool] * 100
            deviation = actual_pct - target_pct
            parts.append(f"{pool}={current[pool]} ({actual_pct:.1f}%, Δ{deviation:+.1f}%)")
        print(" | ".join(parts))

    return splits


def copy_split_to_directory(
    splits: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """
    Copy images to organized split directories.

    Structure:
      output_dir/
      ├── train/
      │   ├── glioma/
      │   ├── meningioma/
      │   ├── notumor/
      │   └── pituitary/
      ├── val/
      │   └── ...
      └── test/
          └── ...
    """
    for split_name, images in splits.items():
        for img in images:
            dest_dir = output_dir / split_name / img["label"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / img["path"].name

            # Handle filename collisions (rare but possible from merged dirs)
            if dest_path.exists():
                stem = img["path"].stem
                suffix = img["path"].suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            shutil.copy2(img["path"], dest_path)


def print_split_summary(
    splits: dict[str, list[dict]],
    clusters: list[list[dict]] = None,
    target_ratios: dict[str, float] = None,
) -> None:
    """Print detailed split summary with class distributions, deviation, and cluster stats."""
    if target_ratios is None:
        target_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

    print("\n" + "=" * 60)
    print("DATA SPLIT SUMMARY (CLUSTER-AWARE, WEIGHTED GREEDY)")
    print("=" * 60)

    total = sum(len(imgs) for imgs in splits.values())
    max_deviation = 0.0

    for split_name, images in splits.items():
        labels = [img["label"] for img in images]
        counter = Counter(labels)
        pct = len(images) / total * 100
        target_pct = target_ratios[split_name] * 100
        deviation = pct - target_pct
        max_deviation = max(max_deviation, abs(deviation))

        print(f"\n  {split_name.upper()} ({len(images)} images, {pct:.1f}%, target {target_pct:.0f}%, Δ{deviation:+.1f}%):")
        for cls in sorted(counter.keys()):
            cls_pct = counter[cls] / len(images) * 100
            print(f"    {cls:15s}: {counter[cls]:5d} ({cls_pct:.1f}%)")

    print(f"\n  TOTAL: {total} images")
    print(f"  MAX DEVIATION from target: {max_deviation:.1f}%")

    # Verify no cross-pool leakage
    if clusters:
        split_assignment = {}
        for split_name, images in splits.items():
            for img in images:
                split_assignment[str(img["path"])] = split_name

        leaks = 0
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            pool_set = set(split_assignment[str(img["path"])] for img in cluster)
            if len(pool_set) > 1:
                leaks += 1

        print(f"\n  LEAKAGE CHECK: {leaks} clusters span multiple pools")
        if leaks == 0:
            print("  ALL CLEAR — No cross-pool leakage detected")
        else:
            print("  WARNING: Leakage detected — investigate!")

    print("=" * 60)


# ============================================================
# Main pipeline
# ============================================================

def run_split(
    data_dir: str,
    output_dir: str,
    pairs_path: str = "data/duplicates.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """Run the complete cluster-aware splitting pipeline."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    pairs_file = Path(pairs_path)

    if out_path.exists() and any(out_path.iterdir()) and not force:
        print(f"Split directory {out_path} already exists. Use --force to overwrite.")
        return {}

    if force and out_path.exists():
        shutil.rmtree(out_path)

    # Step 1: Collect all images
    print("1. Collecting all images...")
    images = collect_all_images(data_path)
    print(f"   Found {len(images)} images")

    label_counts = Counter(img["label"] for img in images)
    for cls, count in sorted(label_counts.items()):
        print(f"   {cls}: {count}")

    # Step 2: Load duplicate pairs and build clusters
    print(f"\n2. Building duplicate clusters...")
    duplicate_pairs = load_duplicate_pairs(pairs_file)
    clusters = build_clusters(images, duplicate_pairs)

    # Step 3: Cluster-aware stratified split
    print(f"\n3. Creating cluster-aware stratified split ({train_ratio}/{val_ratio}/{test_ratio})...")
    splits = cluster_aware_stratified_split(
        clusters, train_ratio, val_ratio, test_ratio, seed
    )

    # Step 4: Copy to output directory
    print(f"\n4. Copying images to {out_path}...")
    copy_split_to_directory(splits, out_path)

    # Step 5: Summary with leakage check and deviation report
    target_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    print_split_summary(splits, clusters, target_ratios)

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create cluster-aware stratified data splits"
    )
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--pairs-path", type=str, default="data/duplicates.json",
                        help="Path to duplicate pairs JSON from dedup.py")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing splits")
    args = parser.parse_args()

    run_split(
        args.data_dir, args.output_dir, args.pairs_path,
        args.train_ratio, args.val_ratio, args.test_ratio,
        args.seed, args.force,
    )
