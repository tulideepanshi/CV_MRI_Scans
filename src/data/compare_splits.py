#!/usr/bin/env python3
"""
Split Comparison — Kaggle original vs. cluster-aware split.

This script answers the interview question:
  "How did your re-splitting change things compared to the original data?"

It compares:
  - Kaggle Training  →  where did those images end up in the new split?
  - Kaggle Testing   →  where did those images end up in the new split?

KEY INSIGHT:
  If the original Kaggle split had no leakage, our re-split would roughly
  preserve Training → train and Testing → test+val. But if near-duplicate
  images existed across Training/Testing, the cluster-aware split would
  have MOVED images — pulling some original Training images into test/val
  (or vice versa) to keep clusters together. The number of "migrations"
  directly quantifies how much leakage the original split had.

Usage:
    python -m src.data.compare_splits \
        --kaggle-dir datasets \
        --new-dir data/splits \
        --pairs-path data/duplicates.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path


def collect_images(data_dir: Path) -> dict[str, dict]:
    """Collect all images and return {filename: metadata} mapping."""
    extensions = {".jpg", ".jpeg", ".png"}
    images = {}

    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix.lower() in extensions:
            images[img_path.name] = {
                "path": str(img_path),
                "label": img_path.parent.name,
            }

    return images


def build_migration_table(
    kaggle_dir: Path,
    new_dir: Path,
) -> dict:
    """
    Build a table showing where each image migrated.

    For every image, we know:
      - Original location: Kaggle Training or Testing
      - New location: train, val, or test

    Returns dict with:
      - migrations: {filename: {original, new, label}}
      - summary: counts of each migration pattern
    """
    # Collect original Kaggle images
    kaggle_train = {}
    kaggle_test = {}

    train_dir = kaggle_dir / "Training"
    test_dir = kaggle_dir / "Testing"

    if train_dir.exists():
        kaggle_train = collect_images(train_dir)
    if test_dir.exists():
        kaggle_test = collect_images(test_dir)

    # Collect new split images
    new_splits = {}
    for split_name in ["train", "val", "test"]:
        split_dir = new_dir / split_name
        if split_dir.exists():
            for fname, meta in collect_images(split_dir).items():
                # Handle collision suffixes added during copy (e.g., img_1.jpg)
                new_splits[fname] = {
                    **meta,
                    "new_split": split_name,
                }

    # Match original → new by filename
    migrations = []

    for fname, meta in kaggle_train.items():
        if fname in new_splits:
            migrations.append({
                "filename": fname,
                "label": meta["label"],
                "original": "Training",
                "new": new_splits[fname]["new_split"],
            })

    for fname, meta in kaggle_test.items():
        if fname in new_splits:
            migrations.append({
                "filename": fname,
                "label": meta["label"],
                "original": "Testing",
                "new": new_splits[fname]["new_split"],
            })

    return migrations


def print_comparison_report(migrations: list[dict]) -> None:
    """Print a detailed comparison report."""

    print("=" * 70)
    print("SPLIT COMPARISON: KAGGLE ORIGINAL vs. CLUSTER-AWARE SPLIT")
    print("=" * 70)

    total = len(migrations)
    print(f"\nTotal images tracked: {total}")

    # ── Overall migration matrix ──
    print("\n" + "-" * 70)
    print("MIGRATION MATRIX (rows = original, columns = new)")
    print("-" * 70)

    # Count migrations
    matrix = defaultdict(lambda: defaultdict(int))
    for m in migrations:
        matrix[m["original"]][m["new"]] += 1

    # Print header
    new_pools = ["train", "val", "test"]
    header = f"{'':>15s} | " + " | ".join(f"{p:>8s}" for p in new_pools) + " | {'TOTAL':>8s}"
    print(header)
    print("-" * len(header))

    for orig in ["Training", "Testing"]:
        row_total = sum(matrix[orig][p] for p in new_pools)
        cells = []
        for p in new_pools:
            count = matrix[orig][p]
            pct = count / row_total * 100 if row_total > 0 else 0
            cells.append(f"{count:>5d} ({pct:4.1f}%)")

        # Determine which column is the "expected" destination
        # Training → train, Testing → test+val
        print(f"{orig:>15s} | " + " | ".join(f"{c:>15s}" for c in cells) + f" | {row_total:>8d}")

    # ── Interpretation ──
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    # Training images that did NOT end up in train
    train_to_val = matrix["Training"]["val"]
    train_to_test = matrix["Training"]["test"]
    train_leaked = train_to_val + train_to_test
    train_total = sum(matrix["Training"][p] for p in new_pools)

    print(f"\n  Kaggle Training → new train: {matrix['Training']['train']}/{train_total}"
          f" ({matrix['Training']['train']/train_total*100:.1f}%)")
    print(f"  Kaggle Training → moved to val/test: {train_leaked}"
          f" ({train_leaked/train_total*100:.1f}%)")

    # Testing images that did NOT end up in val or test
    test_to_train = matrix["Testing"]["train"]
    test_to_valtest = matrix["Testing"]["val"] + matrix["Testing"]["test"]
    test_total = sum(matrix["Testing"][p] for p in new_pools)

    print(f"\n  Kaggle Testing → new val+test: {test_to_valtest}/{test_total}"
          f" ({test_to_valtest/test_total*100:.1f}%)")
    print(f"  Kaggle Testing → moved to train: {test_to_train}"
          f" ({test_to_train/test_total*100:.1f}%)")

    # ── Per-class breakdown ──
    print("\n" + "-" * 70)
    print("PER-CLASS MIGRATION (images that changed boundary)")
    print("-" * 70)

    classes = sorted(set(m["label"] for m in migrations))
    for cls in classes:
        cls_migrations = [m for m in migrations if m["label"] == cls]

        # Training images moved out
        cls_train_out = [m for m in cls_migrations
                         if m["original"] == "Training" and m["new"] != "train"]
        # Testing images moved to train
        cls_test_in = [m for m in cls_migrations
                       if m["original"] == "Testing" and m["new"] == "train"]

        cls_total = len(cls_migrations)
        moved = len(cls_train_out) + len(cls_test_in)

        print(f"\n  {cls} ({cls_total} images):")
        print(f"    Training → val/test: {len(cls_train_out)}")
        print(f"    Testing  → train:    {len(cls_test_in)}")
        print(f"    Total moved:         {moved} ({moved/cls_total*100:.1f}%)")

    # ── Summary verdict ──
    total_moved = train_leaked + test_to_train
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"\n  Total images that crossed the original train/test boundary: {total_moved}")
    print(f"  Percentage of dataset reshuffled: {total_moved/total*100:.1f}%")

    if total_moved == 0:
        print("\n  The cluster-aware split preserved the original boundary exactly.")
        print("  This means no near-duplicate clusters spanned Training/Testing.")
    else:
        print(f"\n  The cluster-aware split moved {total_moved} images to keep")
        print("  near-duplicate clusters intact within a single pool.")
        print("  This is evidence that the original Kaggle split had potential")
        print("  data leakage that would inflate evaluation metrics.")

    print("\n" + "=" * 70)


class TeeOutput:
    """Write to both terminal and a string buffer simultaneously."""

    def __init__(self, original_stdout):
        self.terminal = original_stdout
        self.buffer = StringIO()

    def write(self, text):
        self.terminal.write(text)
        self.buffer.write(text)

    def flush(self):
        self.terminal.flush()

    def get_content(self) -> str:
        return self.buffer.getvalue()


def run_comparison(
    kaggle_dir: str,
    new_dir: str,
    pairs_path: str = None,
    report_output: str = "docs/split_comparison_report.txt",
) -> None:
    """Run the full comparison. Prints to terminal AND saves to report file."""
    kaggle_path = Path(kaggle_dir)
    new_path = Path(new_dir)

    # Capture all print output for saving to file
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        migrations = build_migration_table(kaggle_path, new_path)

        if not migrations:
            print("No images could be matched between original and new splits.")
            print("Check that both directories contain the expected images.")
            return

        print_comparison_report(migrations)

        # If pairs file provided, show cluster-level analysis
        if pairs_path:
            pairs_file = Path(pairs_path)
            if pairs_file.exists():
                print_cluster_migration_analysis(migrations, pairs_file)

    finally:
        # Restore stdout and save report
        sys.stdout = tee.terminal

        report_path = Path(report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(tee.get_content())

        print(f"\nReport saved to {report_path}")


def print_cluster_migration_analysis(
    migrations: list[dict],
    pairs_path: Path,
) -> None:
    """
    Show which clusters caused the reshuffling.

    For each cluster that would have been split across Training/Testing
    in the original data, show how the cluster-aware split resolved it.
    """
    with open(pairs_path) as f:
        pairs = json.load(f)

    if not pairs:
        return

    print("\n" + "=" * 70)
    print("CLUSTER MIGRATION ANALYSIS")
    print("=" * 70)

    # Build a lookup: filename → original split
    fname_to_original = {}
    for m in migrations:
        fname_to_original[m["filename"]] = m["original"]

    # Build Union-Find to reconstruct clusters
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Extract filenames from paths in pairs
    for pair in pairs:
        f1 = Path(pair["path1"]).name
        f2 = Path(pair["path2"]).name
        union(f1, f2)

    # Group into clusters
    clusters = defaultdict(set)
    for m in migrations:
        root = find(m["filename"])
        clusters[root].add(m["filename"])

    # Find clusters that span Training/Testing in original data
    cross_boundary_clusters = []
    for root, members in clusters.items():
        if len(members) <= 1:
            continue
        original_splits = set()
        for fname in members:
            if fname in fname_to_original:
                original_splits.add(fname_to_original[fname])
        if len(original_splits) > 1:
            cross_boundary_clusters.append(members)

    print(f"\n  Multi-image clusters: {sum(1 for m in clusters.values() if len(m) > 1)}")
    print(f"  Clusters that spanned original Training/Testing boundary: {len(cross_boundary_clusters)}")

    if cross_boundary_clusters:
        print(f"\n  These {len(cross_boundary_clusters)} clusters are the reason images moved.")
        print("  Each cluster had members in BOTH Training and Testing.")
        print("  The cluster-aware split forced all members into one pool.")

        # Show size distribution of cross-boundary clusters
        sizes = [len(c) for c in cross_boundary_clusters]
        size_counter = Counter(sizes)
        print(f"\n  Cross-boundary cluster sizes:")
        for size in sorted(size_counter.keys()):
            print(f"    Size {size}: {size_counter[size]} clusters")

        # Show a few examples
        print(f"\n  Example cross-boundary clusters (first 5):")
        for i, members in enumerate(sorted(cross_boundary_clusters, key=len, reverse=True)[:5]):
            print(f"\n    Cluster {i+1} ({len(members)} images):")
            for fname in sorted(members):
                orig = fname_to_original.get(fname, "?")
                # Find new split
                new = "?"
                for m in migrations:
                    if m["filename"] == fname:
                        new = m["new"]
                        break
                moved = " ← MOVED" if (orig == "Training" and new != "train") or \
                                       (orig == "Testing" and new == "train") else ""
                print(f"      {fname}: {orig} → {new}{moved}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Kaggle original split vs. cluster-aware split"
    )
    parser.add_argument("--kaggle-dir", type=str, default="datasets",
                        help="Root dir with Training/ and Testing/ subdirs")
    parser.add_argument("--new-dir", type=str, default="data/splits",
                        help="Root dir with train/ val/ test/ subdirs")
    parser.add_argument("--pairs-path", type=str, default="data/duplicates.json",
                        help="Path to duplicate pairs JSON (optional, for cluster analysis)")
    parser.add_argument("--output", type=str, default="docs/split_comparison_report.txt",
                        help="Path to save the comparison report")
    args = parser.parse_args()

    run_comparison(args.kaggle_dir, args.new_dir, args.pairs_path, args.output)
