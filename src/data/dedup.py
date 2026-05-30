#!/usr/bin/env python3
"""
Near-Duplicate Detection — Finding potential patient-level data leakage.

WHY THIS MATTERS:
The Kaggle dataset has no patient IDs. The same patient may have multiple
MRI slices in both Training and Testing sets. If the model "memorizes"
a patient's brain shape during training and sees another slice from the
same patient during testing, we get inflated performance metrics that
won't generalize to new patients.

APPROACH:
  1. Perceptual Hashing (pHash) — fast, catches near-identical images
  2. Embedding Similarity — deeper, catches same-patient different-slice

Usage:
    python -m src.data.dedup --data-dir datasets --threshold 8
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_phash(img_path: str, hash_size: int = 16) -> str:
    """
    Compute perceptual hash (pHash) for an image.

    pHash works by:
    1. Resize to small square (hash_size x hash_size)
    2. Convert to grayscale
    3. Apply DCT (discrete cosine transform)
    4. Keep only low-frequency components
    5. Threshold at median → binary hash

    Two images with hamming distance <= threshold are likely near-duplicates.
    """
    try:
        import imagehash
        img = Image.open(img_path).convert("L")  # Grayscale
        return str(imagehash.phash(img, hash_size=hash_size))
    except Exception as e:
        print(f"  Warning: Failed to hash {img_path}: {e}")
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute hamming distance between two hex hash strings."""
    if hash1 is None or hash2 is None:
        return float("inf")
    # Convert hex strings to integers and XOR
    h1 = int(hash1, 16)
    h2 = int(hash2, 16)
    xor = h1 ^ h2
    return bin(xor).count("1")


def find_duplicates_phash(
    data_dir: Path,
    threshold: int = 8,
) -> list[tuple[str, str, int]]:
    """
    Find near-duplicate pairs using perceptual hashing.

    Args:
        data_dir: Root data directory containing Training/ and Testing/
        threshold: Max hamming distance to consider as duplicate

    Returns:
        List of (path1, path2, distance) tuples for suspected duplicates
    """
    print("Phase 1: Computing perceptual hashes...")
    extensions = {".jpg", ".jpeg", ".png"}

    # Collect all image paths with their split info
    images = []
    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix.lower() in extensions:
            split = "train" if "Training" in str(img_path) else "test"
            images.append({"path": str(img_path), "split": split, "label": img_path.parent.name})

    print(f"  Found {len(images)} images")

    # Compute hashes
    for img in tqdm(images, desc="  Hashing"):
        img["phash"] = compute_phash(img["path"])

    # Group by hash for fast lookup (exact duplicates)
    hash_groups = defaultdict(list)
    for img in images:
        if img["phash"] is not None:
            hash_groups[img["phash"]].append(img)

    exact_dupes = [(grp[0]["path"], grp[1]["path"], 0)
                   for grp in hash_groups.values() if len(grp) > 1]
    print(f"  Exact hash matches: {len(exact_dupes)}")

    # Find cross-split near-duplicates (train image similar to test image)
    print("\nPhase 2: Finding cross-split near-duplicates...")
    train_images = [img for img in images if img["split"] == "train" and img["phash"]]
    test_images = [img for img in images if img["split"] == "test" and img["phash"]]

    cross_split_dupes = []
    for test_img in tqdm(test_images, desc="  Comparing test→train"):
        for train_img in train_images:
            dist = hamming_distance(test_img["phash"], train_img["phash"])
            if dist <= threshold:
                cross_split_dupes.append((train_img["path"], test_img["path"], dist))

    print(f"  Cross-split near-duplicates (distance <= {threshold}): {len(cross_split_dupes)}")

    # Phase 3: Find ALL near-duplicate pairs (including within-split)
    # Needed for cluster-aware splitting — we must know which images
    # should stay together regardless of their original split.
    print("\nPhase 3: Finding all near-duplicate pairs (for cluster-aware splitting)...")
    hashed_images = [img for img in images if img["phash"]]
    all_pairs = []

    # Group by label first — duplicates across classes are extremely unlikely
    # and this cuts the O(n²) comparisons dramatically
    by_label = defaultdict(list)
    for img in hashed_images:
        by_label[img["label"]].append(img)

    for label, label_images in by_label.items():
        n = len(label_images)
        print(f"  {label}: comparing {n} images ({n * (n-1) // 2} pairs)...")
        for i in range(n):
            for j in range(i + 1, n):
                dist = hamming_distance(label_images[i]["phash"], label_images[j]["phash"])
                if dist <= threshold:
                    all_pairs.append((label_images[i]["path"], label_images[j]["path"], dist))

    print(f"  Total near-duplicate pairs (all): {len(all_pairs)}")

    return cross_split_dupes, all_pairs


def save_duplicate_pairs(
    all_pairs: list[tuple[str, str, int]],
    output_path: Path,
) -> None:
    """
    Save all near-duplicate pairs as structured JSON for downstream consumption.

    This is the bridge between dedup and split — split.py reads this file
    to build clusters and ensure no near-duplicates cross pool boundaries.

    Format:
      [{"path1": "...", "path2": "...", "distance": 3}, ...]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {"path1": str(p1), "path2": str(p2), "distance": int(d)}
        for p1, p2, d in all_pairs
    ]

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} duplicate pairs to {output_path}")


def generate_dedup_report(
    cross_split_dupes: list[tuple[str, str, int]],
    all_pairs: list[tuple[str, str, int]],
    output_path: Path,
) -> None:
    """
    Generate a detailed deduplication report.

    This report is what you'd show in the interview to demonstrate
    data quality awareness.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("NEAR-DUPLICATE DETECTION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Cross-split near-duplicates: {len(cross_split_dupes)}\n")
        f.write(f"Total near-duplicate pairs (all): {len(all_pairs)}\n\n")

        if not cross_split_dupes:
            f.write("No cross-split duplicates detected. The train/test split\n")
            f.write("appears clean at the image level.\n\n")
            f.write("NOTE: This does NOT guarantee no patient-level leakage.\n")
            f.write("Without patient IDs, we cannot rule out that different\n")
            f.write("slices from the same patient appear in both splits.\n")
        else:
            # Group by distance
            by_distance = defaultdict(list)
            for train_path, test_path, dist in cross_split_dupes:
                by_distance[dist].append((train_path, test_path))

            for dist in sorted(by_distance.keys()):
                pairs = by_distance[dist]
                f.write(f"\n--- Hamming Distance = {dist} ({len(pairs)} pairs) ---\n")
                risk = "EXACT DUPLICATE" if dist == 0 else (
                    "HIGH RISK" if dist <= 3 else "MODERATE RISK"
                )
                f.write(f"Risk Level: {risk}\n\n")
                for train_path, test_path in pairs[:10]:  # Show first 10
                    f.write(f"  TRAIN: {train_path}\n")
                    f.write(f"  TEST:  {test_path}\n\n")
                if len(pairs) > 10:
                    f.write(f"  ... and {len(pairs) - 10} more pairs\n\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("=" * 70 + "\n\n")
        f.write("Action: Use cluster-aware stratified splitting.\n")
        f.write("All near-duplicate images are grouped into clusters\n")
        f.write("(connected components). The split operates at the\n")
        f.write("cluster level — every image in a cluster lands in\n")
        f.write("the same pool (train/val/test). This prevents data\n")
        f.write("leakage while preserving ALL samples.\n")

    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect near-duplicate images")
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--output", type=str, default="docs/dedup_report.txt")
    parser.add_argument("--pairs-output", type=str, default="data/duplicates.json",
                        help="Structured JSON output consumed by split.py")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    cross_split_dupes, all_pairs = find_duplicates_phash(data_path, threshold=args.threshold)

    # Save structured pairs for split.py to consume
    save_duplicate_pairs(all_pairs, Path(args.pairs_output))

    # Save human-readable report
    generate_dedup_report(cross_split_dupes, all_pairs, Path(args.output))
