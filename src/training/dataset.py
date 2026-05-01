"""
dataset.py — Custom PyTorch Dataset for train/val/test splits.
Reads images from data/processed/{split}/images/ and labels from .../labels/
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from src.preprocessing.preprocess import preprocess_pipeline


class ClickbaitDataset(Dataset):
    """
    Reads YOLO-labeled image splits.

    Expected layout:
        root/
            images/ *.png
            labels/ *.txt   (YOLO format: class xc yc w h)

    We use only the class from the first line of each label file.
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.image_paths = sorted(
            list((self.root / "images").glob("*.png")) +
            list((self.root / "images").glob("*.jpg")) +
            list((self.root / "images").glob("*.jpeg"))
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label_path = self.root / "labels" / (img_path.stem + ".txt")

        # Preprocess image
        try:
            img = preprocess_pipeline(str(img_path))  # (224, 224, 3) float32
        except FileNotFoundError:
            img = np.zeros((224, 224, 3), dtype=np.float32)

        # HWC → CHW
        tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (3, 224, 224)

        # Read class label
        label = 0  # default
        if label_path.exists():
            with open(label_path) as f:
                first_line = f.readline().strip()
                if first_line:
                    label = int(first_line.split()[0])

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label
