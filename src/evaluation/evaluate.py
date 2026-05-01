"""
evaluate.py — Phase 8: Model Evaluation & Testing
Owner: All | Tech: scikit-learn, matplotlib, seaborn

Run evaluation:
    python -m src.evaluation.evaluate
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.model.model import ClickbaitDetector
from src.training.config import CONFIG, CLASS_NAMES
from src.training.dataset import ClickbaitDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Step 8.1 — Classification Evaluation
# ─────────────────────────────────────────────

def evaluate_model(model: ClickbaitDetector, test_loader: DataLoader) -> dict:
    """
    Run inference on the test set and compute all metrics.

    Returns dict with: precision, recall, f1, confusion_matrix
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Only report classes that actually appear in the data
    unique_classes = sorted(set(all_labels.tolist()) | set(all_preds.tolist()))
    target_names   = [CLASS_NAMES[c] for c in unique_classes]

    # ── Classification Report ──
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "confusion_matrix.png")
    print(f"📁 Confusion matrix saved to {results_dir / 'confusion_matrix.png'}")
    plt.show()

    # ── Summary Metrics ──
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy  = (all_preds == all_labels).mean()

    print(f"\n✅ Accuracy              : {accuracy:.4f}")
    print(f"✅ Weighted Precision    : {precision:.4f}")
    print(f"✅ Weighted Recall       : {recall:.4f}")
    print(f"✅ Weighted F1-Score     : {f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────
# Step 8.2 — IoU Evaluation (Bounding Boxes)
# ─────────────────────────────────────────────

def compute_iou(box_pred: list, box_true: list) -> float:
    """
    Compute Intersection over Union for two bounding boxes.
    Boxes format: [x_min, y_min, x_max, y_max]

    IoU > 0.5 is considered a successful detection (standard COCO threshold).
    """
    x1 = max(box_pred[0], box_true[0])
    y1 = max(box_pred[1], box_true[1])
    x2 = min(box_pred[2], box_true[2])
    y2 = min(box_pred[3], box_true[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_true = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])
    union = area_pred + area_true - intersection

    return intersection / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ckpt_path = Path(CONFIG["checkpoint_dir"]) / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Run training first:\n"
            "  python -m src.training.train"
        )

    model = ClickbaitDetector(backbone=CONFIG["backbone"]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"✅ Loaded checkpoint from {ckpt_path}")

    test_ds     = ClickbaitDataset(f"{CONFIG['data_root']}/test")
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    metrics = evaluate_model(model, test_loader)
