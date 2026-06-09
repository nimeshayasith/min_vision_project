"""
train_yolo.py — YOLOv8 Clickbait Element Detector Training
Owner: All | Tech: Ultralytics YOLOv8, OpenCV

Trains a YOLOv8n (nano) object detection model on the Roboflow-annotated
clickbait dataset to detect individual clickbait UI elements:
  - Buttons
  - Computer-vision (deceptive visual elements)
  - ad_banner
  - close_button
  - fake_download_button

CPU-optimized: uses YOLOv8n (nano) with 50 epochs.
Expected training time: ~30-60 minutes on CPU.

Run:
    python -m src.training.train_yolo
"""

from pathlib import Path
import torch
from ultralytics import YOLO

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_YAML    = PROJECT_ROOT / "data" / "roboflow" / "data.yaml"
CKPT_DIR     = PROJECT_ROOT / "models" / "checkpoints"
YOLO_MODEL   = CKPT_DIR / "yolo_clickbait.pt"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (GPU-aware) ─────────────────────────────────────────
YOLO_TRAIN_CONFIG = {
    "model":         "yolov8m.pt",   # Larger model for better recall and detection quality
    "data":          str(DATA_YAML),
    "epochs":        100,              # Full long training
    "imgsz":         1024,             # Preserve small UI elements
    "batch":         8,                # Increase batch if GPU memory allows
    "device":        0,
    "workers":       8,                # More workers for faster loading on Linux
    "patience":      20,
    "lr0":           0.01,
    "lrf":           0.1,
    "warmup_epochs": 3,
    "freeze":        10,               # Freeze backbone for first 10 epochs
    "augment":       True,
    "auto_augment":  "randaugment",
    "mosaic":        1.0,
    "copy_paste":    0.0,
    "mixup":         0.0,
    "flipud":        0.0,
    "fliplr":        0.5,
    "degrees":       0.0,
    "scale":         0.5,
    "hsv_h":         0.1,
    "hsv_s":         0.9,
    "hsv_v":         0.6,
    "conf":          0.15,
    "iou":           0.5,
    "project":       str(CKPT_DIR / "yolo_runs"),
    "name":          "clickbait_detector",
    "exist_ok":      True,
    "verbose":       True,
}


def train():
    print("=" * 60)
    print("  YOLOv8 Clickbait Element Detector — Training")
    print("=" * 60)
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Output  : {YOLO_MODEL}")
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {YOLO_TRAIN_CONFIG['epochs']}")
    print("=" * 60)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {DATA_YAML}\n"
            "Make sure the Roboflow dataset is in data/roboflow/"
        )

    # Load pretrained YOLOv8n (downloads ~6MB weights on first run)
    model = YOLO(YOLO_TRAIN_CONFIG["model"])

    # Train
    results = model.train(**{k: v for k, v in YOLO_TRAIN_CONFIG.items()
                             if k != "model"})

    # Copy best weights to our standard checkpoint path
    best_weights = (
        Path(YOLO_TRAIN_CONFIG["project"])
        / YOLO_TRAIN_CONFIG["name"]
        / "weights"
        / "best.pt"
    )
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, YOLO_MODEL)
        print(f"\n✅ Best model saved to: {YOLO_MODEL}")
    else:
        print(f"\n⚠️  Could not find best.pt at {best_weights}")

    print("\n[Validation Metrics]")
    print(f"  mAP50   : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")

    return results


if __name__ == "__main__":
    train()
