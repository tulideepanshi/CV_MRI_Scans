#!/usr/bin/env python3
"""
Data Download Script — Fetches the Kaggle Brain Tumor MRI Dataset.

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Get your API key from https://www.kaggle.com/settings
    3. Place kaggle.json in ~/.kaggle/kaggle.json
       OR set environment variables:
       export KAGGLE_USERNAME=your_username
       export KAGGLE_KEY=your_api_key

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --output data/raw
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path


def download_kaggle_dataset(dataset: str, output_dir: str) -> None:
    """Download and extract dataset from Kaggle."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    if any(output_path.iterdir()):
        print(f"Data already exists in {output_path}. Skipping download.")
        print("Delete the directory to re-download.")
        return

    # Verify Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")

    if not kaggle_json.exists() and not has_env:
        print("ERROR: Kaggle credentials not found.")
        print()
        print("Option 1: Place kaggle.json in ~/.kaggle/")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token'")
        print("  3. Move downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("  4. chmod 600 ~/.kaggle/kaggle.json")
        print()
        print("Option 2: Set environment variables")
        print("  export KAGGLE_USERNAME=your_username")
        print("  export KAGGLE_KEY=your_api_key")
        sys.exit(1)

    # Import kaggle after credential check to avoid early auth errors
    from kaggle.api.kaggle_api_extended import KaggleApi

    print(f"Downloading {dataset}...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(output_path), unzip=True)

    print(f"Dataset downloaded and extracted to {output_path}")

    # Print summary
    total_images = 0
    for class_dir in sorted(output_path.rglob("*")):
        if class_dir.is_dir() and class_dir.parent != output_path.parent:
            n_images = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.jpeg")))
            if n_images > 0:
                print(f"  {class_dir.name}: {n_images} images")
                total_images += n_images
    print(f"  Total: {total_images} images")


def download_manual_fallback(output_dir: str) -> None:
    """Instructions for manual download if Kaggle API isn't available."""
    print("=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. Visit: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
    print("2. Click 'Download' (you'll need a Kaggle account)")
    print(f"3. Extract the ZIP file to: {output_dir}")
    print()
    print("Expected structure after extraction:")
    print(f"  {output_dir}/")
    print("  ├── Training/")
    print("  │   ├── glioma/")
    print("  │   ├── meningioma/")
    print("  │   ├── notumor/")
    print("  │   └── pituitary/")
    print("  └── Testing/")
    print("      ├── glioma/")
    print("      ├── meningioma/")
    print("      ├── notumor/")
    print("      └── pituitary/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Brain Tumor MRI Dataset")
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Show manual download instructions instead"
    )
    args = parser.parse_args()

    if args.manual:
        download_manual_fallback(args.output)
    else:
        try:
            download_kaggle_dataset(
                "masoudnickparvar/brain-tumor-mri-dataset",
                args.output
            )
        except Exception as e:
            print(f"Download failed: {e}")
            print()
            download_manual_fallback(args.output)
