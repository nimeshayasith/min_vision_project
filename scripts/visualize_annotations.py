"""
visualize_annotations.py — Preview bounding box labels on images.

Usage:
    python scripts/visualize_annotations.py \
        --images_dir data/annotated/images \
        --labels_dir data/annotated/labels \
        --num_samples 5
"""

import cv2
import random
import argparse
import numpy as np
from pathlib import Path

CLASS_NAMES = {0: "legitimate", 1: "phishing", 2: "clickbait"}
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 0, 255), 2: (0, 165, 255)}  # BGR


def draw_yolo_boxes(image_path: Path, label_path: Path) -> np.ndarray:
    """Draw YOLO bounding boxes on an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if not label_path.exists():
        cv2.putText(img, "No label", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            color = CLASS_COLORS.get(cls, (255, 255, 0))
            label = CLASS_NAMES.get(cls, f"class_{cls}")
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def visualize(images_dir: str, labels_dir: str, num_samples: int = 5) -> None:
    images = list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg"))
    samples = random.sample(images, min(num_samples, len(images)))

    for img_path in samples:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        vis = draw_yolo_boxes(img_path, label_path)
        if vis is not None:
            cv2.imshow(f"Annotation — {img_path.name}", vis)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO annotations on images.")
    parser.add_argument("--images_dir", default="data/annotated/images")
    parser.add_argument("--labels_dir", default="data/annotated/labels")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    visualize(args.images_dir, args.labels_dir, args.num_samples)
