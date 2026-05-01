"""
prepare_labels.py — Phase 1, Step 1.4 (Automated label generation)

Since this dataset is organized by folder name (genuine_site_0 / phishing_site_1),
we auto-generate image-level labels WITHOUT bounding boxes.

Each label file contains one line:
    <class_id> 0.5 0.5 1.0 1.0
    (whole-image bounding box in YOLO normalized format)

Label map:
    0 = legitimate  (from genuine_site_0/)
    1 = phishing    (from phishing_site_1/)
    2 = clickbait   (reserved for future self-collected data)

Usage:
    python scripts/prepare_labels.py \
        --cleaned_dir data/cleaned \
        --annotated_dir data/annotated
"""

import argparse
import shutil
from pathlib import Path


# Folder name → class ID mapping
FOLDER_TO_CLASS = {
    "genuine_site_0": 0,   # legitimate
    "phishing_site_1": 1,  # phishing
}


def prepare_labels(cleaned_dir: str, annotated_dir: str) -> None:
    """
    Copy cleaned images to data/annotated/images/ and generate
    whole-image YOLO label files in data/annotated/labels/.

    Args:
        cleaned_dir:   Root directory with class subdirectories (data/cleaned/)
        annotated_dir: Output directory for annotated dataset (data/annotated/)
    """
    cleaned_dir = Path(cleaned_dir)
    annotated_dir = Path(annotated_dir)
    images_out = annotated_dir / "images"
    labels_out = annotated_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    total = 0
    for folder_name, class_id in FOLDER_TO_CLASS.items():
        src_dir = cleaned_dir / folder_name
        if not src_dir.exists():
            print(f"  [SKIP] Missing folder: {src_dir}")
            continue

        images = list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg"))
        print(f"  [CLASS {class_id}] {folder_name}  ({len(images)} images)")

        for img_path in images:
            # Copy image
            dest_img = images_out / img_path.name
            shutil.copy(img_path, dest_img)

            # Write whole-image YOLO label
            label_file = labels_out / (img_path.stem + ".txt")
            with open(label_file, "w") as f:
                # class x_center y_center width height  (all normalized to 0-1)
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

            total += 1

    print(f"\n[DONE] Labels prepared -- {total} image-label pairs saved to {annotated_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLO labels from folder-organized cleaned dataset.")
    parser.add_argument("--cleaned_dir", default="data/cleaned", help="Root directory with class subdirectories")
    parser.add_argument("--annotated_dir", default="data/annotated", help="Output directory for annotated data")
    args = parser.parse_args()

    prepare_labels(args.cleaned_dir, args.annotated_dir)
