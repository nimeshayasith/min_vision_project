"""
test_dataset.py — Phase 1 Tests
Verifies that train/val/test splits exist, are non-empty, and labels match images.
"""

import pytest
from pathlib import Path

DATA_DIR = Path("data/processed")


def test_splits_exist():
    """All three split directories must exist with images/ and labels/ subdirs."""
    for split in ["train", "val", "test"]:
        assert (DATA_DIR / split / "images").exists(), f"Missing {split}/images/"
        assert (DATA_DIR / split / "labels").exists(), f"Missing {split}/labels/"


def test_no_empty_splits():
    """Each split must contain at least one image."""
    for split in ["train", "val", "test"]:
        images = (
            list((DATA_DIR / split / "images").glob("*.png")) +
            list((DATA_DIR / split / "images").glob("*.jpg"))
        )
        assert len(images) > 0, f"{split} split has no images!"


def test_labels_match_images():
    """Every image must have a corresponding label file."""
    for split in ["train", "val", "test"]:
        images = set(
            p.stem for p in (DATA_DIR / split / "images").glob("*.png")
        )
        images |= set(
            p.stem for p in (DATA_DIR / split / "images").glob("*.jpg")
        )
        labels = set(p.stem for p in (DATA_DIR / split / "labels").glob("*.txt"))
        assert images == labels, (
            f"Mismatch in {split}: "
            f"{images.symmetric_difference(labels)}"
        )


def test_label_format():
    """YOLO format: each line must be 'class x_center y_center width height'."""
    for lbl_file in (DATA_DIR / "train" / "labels").glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                assert len(parts) == 5, f"Bad label in {lbl_file}: {line!r}"
                cls = int(parts[0])
                assert cls in [0, 1, 2], f"Unknown class {cls} in {lbl_file}"
                for coord in parts[1:]:
                    val = float(coord)
                    assert 0.0 <= val <= 1.0, f"Coord out of [0,1] in {lbl_file}: {coord}"
