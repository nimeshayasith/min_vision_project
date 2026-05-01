"""
split_dataset.py — Phase 1, Step 1.5
Splits annotated dataset into train / val / test sets.

Usage:
    python scripts/split_dataset.py \
        --input_dir data/annotated \
        --output_dir data/processed \
        --train 0.70 \
        --val 0.15 \
        --test 0.15 \
        --seed 42
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(
    input_dir: str,
    output_dir: str,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Split annotated images (and their YOLO labels) into train/val/test.

    Expected input layout:
        input_dir/
            images/  *.png
            labels/  *.txt

    Output layout:
        output_dir/
            train/images/, train/labels/
            val/images/,   val/labels/
            test/images/,  test/labels/

    Args:
        input_dir:  Directory containing images/ and labels/ subdirs
        output_dir: Root directory for processed splits
        train:      Fraction for training set (default: 0.70)
        val:        Fraction for validation set (default: 0.15)
        test:       Fraction for test set (default: 0.15)
        seed:       Random seed for reproducibility (default: 42)
    """
    assert abs(train + val + test - 1.0) < 1e-6, \
        f"Splits must sum to 1.0, got {train + val + test}"

    random.seed(seed)

    images_dir = Path(input_dir) / "images"
    labels_dir = Path(input_dir) / "labels"

    images = (
        list(images_dir.glob("*.png")) +
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.jpeg"))
    )
    random.shuffle(images)

    n = len(images)
    if n == 0:
        print("⚠️  No images found in input_dir/images/. Ensure data_cleaning.py and prepare_labels.py ran first.")
        return

    n_train = int(n * train)
    n_val   = int(n * val)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        for img_path in split_images:
            # Copy image
            dest_img = Path(output_dir) / split_name / "images" / img_path.name
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dest_img)

            # Copy label if it exists
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                dest_lbl = Path(output_dir) / split_name / "labels" / label_path.name
                dest_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(label_path, dest_lbl)

        print(f"  {split_name:5s}: {len(split_images)} images")

    print(f"\n[DONE] Split complete -- Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split annotated dataset into train/val/test.")
    parser.add_argument("--input_dir",  default="data/annotated",  help="Directory with images/ and labels/")
    parser.add_argument("--output_dir", default="data/processed",  help="Output directory for splits")
    parser.add_argument("--train", type=float, default=0.70, help="Training fraction (default: 0.70)")
    parser.add_argument("--val",   type=float, default=0.15, help="Validation fraction (default: 0.15)")
    parser.add_argument("--test",  type=float, default=0.15, help="Test fraction (default: 0.15)")
    parser.add_argument("--seed",  type=int,   default=42,   help="Random seed (default: 42)")
    args = parser.parse_args()

    split_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
    )
