"""
data_cleaning.py — Phase 1, Step 1.3
Removes duplicate and low-quality images from raw dataset folders.

Usage:
    python scripts/data_cleaning.py \
        --input_dir data/raw/genuine_site_0 \
        --output_dir data/cleaned/genuine_site_0 \
        --min_size 100 \
        --remove_duplicates

    python scripts/data_cleaning.py \
        --input_dir data/raw/phishing_site_1 \
        --output_dir data/cleaned/phishing_site_1 \
        --min_size 100 \
        --remove_duplicates
"""

import os
import cv2
import hashlib
import argparse
from pathlib import Path


def is_low_quality(image_path: Path, min_size: int = 100) -> bool:
    """Reject images smaller than min_size px in either dimension."""
    img = cv2.imread(str(image_path))
    if img is None:
        return True  # Unreadable file
    h, w = img.shape[:2]
    return h < min_size or w < min_size


def get_hash(image_path: Path) -> str:
    """MD5 hash for duplicate detection."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def clean_dataset(
    input_dir: str,
    output_dir: str,
    min_size: int = 100,
    remove_duplicates: bool = True
) -> None:
    """
    Clean a directory of images by removing:
    - Unreadable or corrupted files
    - Images smaller than min_size in either dimension
    - Exact duplicate images (by MD5 hash)

    Args:
        input_dir:         Source directory containing .png images
        output_dir:        Destination directory for cleaned images
        min_size:          Minimum pixel size in width or height (default: 100)
        remove_duplicates: If True, remove exact duplicate images
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set = set()
    kept, removed = 0, 0

    # Collect all image files
    image_files = list(input_dir.rglob("*.png")) + list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.jpeg"))
    print(f"[INFO] Found {len(image_files)} image(s) in {input_dir}")

    for img_path in image_files:
        # Skip low-quality / unreadable
        if is_low_quality(img_path, min_size):
            print(f"  [SKIP] Low-quality/unreadable: {img_path.name}")
            removed += 1
            continue

        # Skip duplicates
        if remove_duplicates:
            h = get_hash(img_path)
            if h in seen_hashes:
                print(f"  [SKIP] Duplicate: {img_path.name}")
                removed += 1
                continue
            seen_hashes.add(h)

        # Copy to output (preserving uniqueness for nested directories)
        rel_path = img_path.relative_to(input_dir)
        dest_name = "_".join(rel_path.parts)
        dest = output_dir / dest_name
        dest.write_bytes(img_path.read_bytes())
        kept += 1

    print(f"\n[DONE] Cleaning complete -- Kept: {kept} | Removed: {removed}")
    print(f"   Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean dataset: remove low-quality and duplicate images.")
    parser.add_argument("--input_dir", required=True, help="Input directory with raw images")
    parser.add_argument("--output_dir", required=True, help="Output directory for cleaned images")
    parser.add_argument("--min_size", type=int, default=100, help="Minimum image dimension in pixels (default: 100)")
    parser.add_argument("--remove_duplicates", action="store_true", help="Remove exact duplicate images")
    args = parser.parse_args()

    clean_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_size=args.min_size,
        remove_duplicates=args.remove_duplicates,
    )
