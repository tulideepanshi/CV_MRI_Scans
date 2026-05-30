#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) — Understand the dataset before modeling.

This script answers the questions any ML engineer should ask before training:
  1. What's the class distribution? Is it balanced?
  2. What are the image dimensions? Are they consistent?
  3. What are the pixel intensity statistics? (needed for normalization)
  4. Are there any corrupted or unreadable images?
  5. What does the data actually look like? (sample visualizations)

Two modes:
  --mode raw     EDA on original Kaggle data (datasets/Training, datasets/Testing)
  --mode split   EDA on cluster-aware splits (data/splits/train, val, test)

Usage:
    python -m src.data.explore --data-dir datasets --output-dir docs/eda --mode raw
    python -m src.data.explore --data-dir data/splits --output-dir docs/eda_post_split --mode split
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm


def scan_images(data_dir: Path) -> dict:
    """
    Scan all images and collect metadata.

    Returns dict with:
      - paths: list of image paths
      - labels: list of class labels
      - sizes: list of (width, height) tuples
      - corrupted: list of paths that couldn't be read
    """
    result = {"paths": [], "labels": [], "sizes": [], "modes": [], "corrupted": []}
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix.lower() not in extensions:
            continue
        try:
            with Image.open(img_path) as img:
                result["paths"].append(str(img_path))
                result["labels"].append(img_path.parent.name)
                result["sizes"].append(img.size)  # (width, height)
                result["modes"].append(img.mode)
        except Exception as e:
            result["corrupted"].append((str(img_path), str(e)))

    return result


def compute_pixel_stats(data_dir: Path, sample_size: int = 500) -> dict:
    """
    Compute channel-wise mean and std across a sample of images.

    This tells us whether to use ImageNet normalization or dataset-specific values.
    We sample rather than scanning all images to keep this fast.
    """
    extensions = {".jpg", ".jpeg", ".png"}
    all_paths = [p for p in data_dir.rglob("*") if p.suffix.lower() in extensions]

    # Random sample
    rng = np.random.RandomState(42)
    sample_paths = rng.choice(all_paths, size=min(sample_size, len(all_paths)), replace=False)

    pixel_sums = np.zeros(3)
    pixel_sq_sums = np.zeros(3)
    pixel_count = 0

    for path in tqdm(sample_paths, desc="Computing pixel stats"):
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img).astype(np.float64) / 255.0  # Normalize to [0, 1]
            pixel_sums += img_np.reshape(-1, 3).sum(axis=0)
            pixel_sq_sums += (img_np.reshape(-1, 3) ** 2).sum(axis=0)
            pixel_count += img_np.shape[0] * img_np.shape[1]
        except Exception:
            continue

    mean = pixel_sums / pixel_count
    std = np.sqrt(pixel_sq_sums / pixel_count - mean ** 2)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "sample_size": len(sample_paths),
    }


def plot_class_distribution(labels: list, split_name: str, output_dir: Path) -> None:
    """Bar plot of class distribution with counts annotated."""
    counter = Counter(labels)
    classes = sorted(counter.keys())
    counts = [counter[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e05050", "#5090e0", "#50b050", "#e0a030"]
    bars = ax.bar(classes, counts, color=colors[: len(classes)], edgecolor="white", linewidth=0.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title(f"Class Distribution — {split_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Tumor Type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / f"class_distribution_{split_name.lower()}.png", dpi=150)
    plt.close()


def plot_image_sizes(sizes: list, output_dir: Path) -> None:
    """Scatter plot of image dimensions to spot inconsistencies."""
    widths, heights = zip(*sizes)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(widths, heights, alpha=0.3, s=10, c="#5090e0")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Dimensions Distribution", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate unique sizes
    unique_sizes = Counter(sizes)
    for (w, h), count in unique_sizes.most_common(5):
        ax.annotate(f"{w}x{h} (n={count})", (w, h), fontsize=8,
                    xytext=(10, 10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    plt.tight_layout()
    plt.savefig(output_dir / "image_sizes.png", dpi=150)
    plt.close()


def plot_sample_grid(data_dir: Path, output_dir: Path, n_per_class: int = 4) -> None:
    """Grid of sample images — one row per class."""
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(3 * n_per_class, 3 * len(classes)))

    for row, cls in enumerate(classes):
        cls_dir = data_dir / cls
        img_paths = sorted(cls_dir.glob("*.jpg"))[:n_per_class]
        for col, img_path in enumerate(img_paths):
            img = Image.open(img_path).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(cls, fontsize=12, fontweight="bold", loc="left")

    plt.suptitle("Sample Images per Class", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_eda(data_dir: str, output_dir: str) -> dict:
    """Run full EDA and save plots + report."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Scan all images
    print("\n1. Scanning images...")
    all_meta = scan_images(data_path)
    print(f"   Total images: {len(all_meta['paths'])}")
    print(f"   Corrupted: {len(all_meta['corrupted'])}")

    if all_meta["corrupted"]:
        print("   Corrupted files:")
        for path, error in all_meta["corrupted"]:
            print(f"     - {path}: {error}")

    # Class distribution
    print("\n2. Class distribution:")
    counter = Counter(all_meta["labels"])
    for cls, count in sorted(counter.items()):
        pct = count / len(all_meta["labels"]) * 100
        print(f"   {cls:15s}: {count:5d} ({pct:.1f}%)")

    # Check for Training/Testing splits
    for split_name in ["Training", "Testing"]:
        split_dir = data_path / split_name
        if split_dir.exists():
            split_meta = scan_images(split_dir)
            plot_class_distribution(split_meta["labels"], split_name, out_path)

    # Image sizes
    print("\n3. Image dimensions:")
    unique_sizes = Counter(all_meta["sizes"])
    for (w, h), count in unique_sizes.most_common(10):
        print(f"   {w}x{h}: {count} images")
    plot_image_sizes(all_meta["sizes"], out_path)

    # Image modes
    print("\n4. Image modes:")
    mode_counter = Counter(all_meta["modes"])
    for mode, count in mode_counter.most_common():
        print(f"   {mode}: {count} images")

    # Pixel statistics
    print("\n5. Pixel statistics (sampled):")
    stats = compute_pixel_stats(data_path)
    print(f"   Mean (RGB): [{stats['mean'][0]:.4f}, {stats['mean'][1]:.4f}, {stats['mean'][2]:.4f}]")
    print(f"   Std  (RGB): [{stats['std'][0]:.4f}, {stats['std'][1]:.4f}, {stats['std'][2]:.4f}]")
    print(f"   Sample size: {stats['sample_size']} images")
    print(f"\n   Compare with ImageNet defaults:")
    print(f"   ImageNet mean: [0.4850, 0.4560, 0.4060]")
    print(f"   ImageNet std:  [0.2290, 0.2240, 0.2250]")

    # Sample grid
    print("\n6. Generating sample grid...")
    train_dir = data_path / "Training"
    if train_dir.exists():
        plot_sample_grid(train_dir, out_path)

    print(f"\nEDA complete. Plots saved to {out_path}/")
    return {"metadata": all_meta, "pixel_stats": stats}


def plot_split_comparison_bars(
    split_stats: dict[str, dict],
    output_dir: Path,
) -> None:
    """
    Side-by-side grouped bar chart: class counts across train/val/test.

    This is THE key plot for verifying the split is balanced.
    Each class should have roughly the same proportion in every pool.
    """
    splits = list(split_stats.keys())
    classes = sorted(set(
        cls for stats in split_stats.values()
        for cls in Counter(stats["labels"]).keys()
    ))

    x = np.arange(len(classes))
    width = 0.25
    colors = {"train": "#5090e0", "val": "#e0a030", "test": "#e05050"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: absolute counts
    for i, split in enumerate(splits):
        counter = Counter(split_stats[split]["labels"])
        counts = [counter.get(c, 0) for c in classes]
        bars = ax1.bar(x + i * width, counts, width, label=split,
                       color=colors.get(split, "#888888"), edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(count), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(classes)
    ax1.set_title("Class Distribution — Absolute Counts", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Number of Images")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right plot: percentage within each split
    for i, split in enumerate(splits):
        counter = Counter(split_stats[split]["labels"])
        total = sum(counter.values())
        pcts = [counter.get(c, 0) / total * 100 if total > 0 else 0 for c in classes]
        ax2.bar(x + i * width, pcts, width, label=split,
                color=colors.get(split, "#888888"), edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(classes)
    ax2.set_title("Class Distribution — Percentage per Split", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Percentage (%)")
    ax2.axhline(y=25, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="ideal (25%)")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "split_comparison_bars.png", dpi=150)
    plt.close()


def plot_pixel_stats_comparison(
    split_stats: dict[str, dict],
    output_dir: Path,
) -> None:
    """
    Bar chart comparing per-split pixel mean/std.

    If the splits are well-mixed, all three should have nearly
    identical statistics. A big gap means the split introduced
    distributional bias (e.g., all bright images in train, dark in test).
    """
    splits = list(split_stats.keys())
    channels = ["R", "G", "B"]
    colors_ch = ["#e05050", "#50b050", "#5090e0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(splits))
    width = 0.2

    # Mean
    for i, (ch, color) in enumerate(zip(channels, colors_ch)):
        means = [split_stats[s]["pixel_stats"]["mean"][i] for s in splits]
        ax1.bar(x + i * width, means, width, label=ch, color=color, edgecolor="white")
        for xi, val in zip(x + i * width, means):
            ax1.text(xi, val + 0.002, f"{val:.3f}", ha="center", fontsize=7)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(splits)
    ax1.set_title("Channel Mean per Split", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Mean (0-1)")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Std
    for i, (ch, color) in enumerate(zip(channels, colors_ch)):
        stds = [split_stats[s]["pixel_stats"]["std"][i] for s in splits]
        ax2.bar(x + i * width, stds, width, label=ch, color=color, edgecolor="white")
        for xi, val in zip(x + i * width, stds):
            ax2.text(xi, val + 0.002, f"{val:.3f}", ha="center", fontsize=7)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels(splits)
    ax2.set_title("Channel Std per Split", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Std (0-1)")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "pixel_stats_comparison.png", dpi=150)
    plt.close()


def plot_image_sizes_per_split(
    split_stats: dict[str, dict],
    output_dir: Path,
) -> None:
    """Scatter plot of image dimensions colored by split."""
    colors = {"train": "#5090e0", "val": "#e0a030", "test": "#e05050"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for split_name, stats in split_stats.items():
        widths, heights = zip(*stats["sizes"])
        ax.scatter(widths, heights, alpha=0.2, s=8,
                   c=colors.get(split_name, "#888888"), label=split_name)

    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Dimensions by Split", fontsize=14, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "image_sizes_by_split.png", dpi=150)
    plt.close()


def plot_sample_grid_per_split(
    data_dir: Path,
    output_dir: Path,
    n_per_class: int = 3,
) -> None:
    """
    Grid of sample images: rows = classes, column groups = splits.

    Lets you visually verify that each split contains representative
    samples from every class — no class is accidentally empty in any split.
    """
    split_names = ["train", "val", "test"]
    available_splits = [s for s in split_names if (data_dir / s).exists()]
    classes = sorted(set(
        d.name for s in available_splits
        for d in (data_dir / s).iterdir() if d.is_dir()
    ))

    n_cols = n_per_class * len(available_splits)
    n_rows = len(classes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, cls in enumerate(classes):
        for split_idx, split_name in enumerate(available_splits):
            cls_dir = data_dir / split_name / cls
            extensions = ["*.jpg", "*.jpeg", "*.png"]
            img_paths = []
            for ext in extensions:
                img_paths.extend(sorted(cls_dir.glob(ext)))
            img_paths = img_paths[:n_per_class]

            for col_offset, img_path in enumerate(img_paths):
                col = split_idx * n_per_class + col_offset
                img = Image.open(img_path).convert("RGB")
                axes[row, col].imshow(img)
                axes[row, col].axis("off")

                # Label first column of each split group
                if row == 0 and col_offset == 0:
                    axes[row, col].set_title(split_name, fontsize=11,
                                             fontweight="bold", color="#333333")

            # Label class on leftmost column
            if split_idx == 0:
                axes[row, 0].set_ylabel(cls, fontsize=11, fontweight="bold", rotation=0,
                                        labelpad=60, va="center")

            # Fill empty slots
            for col_offset in range(len(img_paths), n_per_class):
                col = split_idx * n_per_class + col_offset
                axes[row, col].axis("off")

    plt.suptitle("Sample Images — train | val | test", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "sample_grid_per_split.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_post_split_eda(data_dir: str, output_dir: str) -> dict:
    """
    Run EDA on the cluster-aware split data (data/splits/).

    This is the EDA you'd run AFTER splitting, to verify:
      1. Class balance is maintained across train/val/test
      2. Image dimensions are consistent across splits
      3. Pixel statistics are similar across splits (no distributional bias)
      4. Each split has representative samples from every class
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("POST-SPLIT EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    split_names = ["train", "val", "test"]
    split_stats = {}

    # ── Per-split analysis ──
    for split_name in split_names:
        split_dir = data_path / split_name
        if not split_dir.exists():
            print(f"\n  WARNING: {split_dir} not found, skipping.")
            continue

        print(f"\n{'─' * 50}")
        print(f"  SPLIT: {split_name.upper()}")
        print(f"{'─' * 50}")

        # Scan images
        meta = scan_images(split_dir)
        print(f"  Total images: {len(meta['paths'])}")
        print(f"  Corrupted: {len(meta['corrupted'])}")

        # Class distribution
        counter = Counter(meta["labels"])
        for cls, count in sorted(counter.items()):
            pct = count / len(meta["labels"]) * 100
            print(f"    {cls:15s}: {count:5d} ({pct:.1f}%)")

        # Individual class distribution plot
        plot_class_distribution(meta["labels"], split_name, out_path)

        # Image sizes
        unique_sizes = Counter(meta["sizes"])
        print(f"  Unique dimensions: {len(unique_sizes)}")
        for (w, h), count in unique_sizes.most_common(3):
            print(f"    {w}x{h}: {count} images")

        # Image modes
        mode_counter = Counter(meta["modes"])
        modes_str = ", ".join(f"{m}: {c}" for m, c in mode_counter.most_common())
        print(f"  Modes: {modes_str}")

        # Pixel statistics
        print(f"  Computing pixel stats...")
        pixel_stats = compute_pixel_stats(split_dir, sample_size=300)
        print(f"    Mean (RGB): [{pixel_stats['mean'][0]:.4f}, {pixel_stats['mean'][1]:.4f}, {pixel_stats['mean'][2]:.4f}]")
        print(f"    Std  (RGB): [{pixel_stats['std'][0]:.4f}, {pixel_stats['std'][1]:.4f}, {pixel_stats['std'][2]:.4f}]")

        split_stats[split_name] = {
            "labels": meta["labels"],
            "sizes": meta["sizes"],
            "modes": meta["modes"],
            "paths": meta["paths"],
            "corrupted": meta["corrupted"],
            "pixel_stats": pixel_stats,
        }

    if len(split_stats) < 2:
        print("\nNot enough splits found for comparison plots.")
        return split_stats

    # ── Cross-split comparison plots ──
    print(f"\n{'─' * 50}")
    print("  GENERATING COMPARISON PLOTS")
    print(f"{'─' * 50}")

    print("  1. Split comparison bar charts...")
    plot_split_comparison_bars(split_stats, out_path)

    print("  2. Pixel statistics comparison...")
    plot_pixel_stats_comparison(split_stats, out_path)

    print("  3. Image sizes by split...")
    plot_image_sizes_per_split(split_stats, out_path)

    print("  4. Sample grid per split...")
    plot_sample_grid_per_split(data_path, out_path)

    # ── Distributional consistency check ──
    print(f"\n{'─' * 50}")
    print("  DISTRIBUTIONAL CONSISTENCY CHECK")
    print(f"{'─' * 50}")

    # Compare means across splits
    means = {s: split_stats[s]["pixel_stats"]["mean"] for s in split_stats}
    all_means = list(means.values())
    for ch_idx, ch_name in enumerate(["R", "G", "B"]):
        ch_values = [m[ch_idx] for m in all_means]
        spread = max(ch_values) - min(ch_values)
        status = "OK" if spread < 0.02 else "DRIFT"
        print(f"  {ch_name} channel mean spread: {spread:.4f} ({status})")

    stds = {s: split_stats[s]["pixel_stats"]["std"] for s in split_stats}
    all_stds = list(stds.values())
    for ch_idx, ch_name in enumerate(["R", "G", "B"]):
        ch_values = [s[ch_idx] for s in all_stds]
        spread = max(ch_values) - min(ch_values)
        status = "OK" if spread < 0.02 else "DRIFT"
        print(f"  {ch_name} channel std spread:  {spread:.4f} ({status})")

    print(f"\n  Threshold: spread < 0.02 = OK, >= 0.02 = potential distributional DRIFT")
    print(f"  (DRIFT means one split has systematically brighter/darker images)")

    print(f"\nPost-split EDA complete. Plots saved to {out_path}/")
    return split_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on Brain Tumor MRI Dataset")
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--output-dir", type=str, default="docs/eda")
    parser.add_argument("--mode", type=str, default="raw", choices=["raw", "split"],
                        help="'raw' for original Kaggle data, 'split' for post-split data")
    args = parser.parse_args()

    if args.mode == "raw":
        run_eda(args.data_dir, args.output_dir)
    else:
        run_post_split_eda(args.data_dir, args.output_dir)
