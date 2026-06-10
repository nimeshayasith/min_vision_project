import os
from pathlib import Path
import torch
from ultralytics import YOLO

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_YAML    = PROJECT_ROOT / "data" / "roboflow" / "data.yaml"
CKPT_DIR     = PROJECT_ROOT / "models" / "checkpoints"
YOLO_MODEL   = CKPT_DIR / "yolo_clickbait.pt"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (HPC RTX 5090 32GB Optimized) ────────────────────────
YOLO_TRAIN_CONFIG = {
    "model":         "yolov8x.pt",     # X-Large model for absolute maximum precision!
    "data":          str(DATA_YAML),
    "epochs":        200,              # Increased epochs to ensure it fully converges
    "imgsz":         1280,             # Max resolution to catch the tiniest UI buttons
    "batch":         16,               # 32GB VRAM can handle batch 16 even at 1280p
    "device":        DEVICE,           # Dynamic device mapping (safe for laptop and HPC)
    "workers":       16 if os.name != "nt" else 0, # Maximize Intel i7 14th Gen cores on Linux safely
    "patience":      30,               # Give it more time to improve before early stopping
    "lr0":           0.01,
    "lrf":           0.01,             # Standard final learning rate fraction
    "warmup_epochs": 5,                # Longer warmup for the massive X-Large model
    "freeze":        15,               # Freeze backbone for 15 epochs (amazing for Transfer Learning)
    "augment":       True,
    "auto_augment":  "randaugment",
    "mosaic":        1.0,
    "copy_paste":    0.1,              # Introduces slight synthetic button cloning
    "mixup":         0.1,              # Advanced regularization
    "flipud":        0.0,
    "fliplr":        0.5,
    "degrees":       0.0,              # Keep zero (UI elements are always horizontal)
    "scale":         0.6,              # Aggressive scale changes
    "hsv_h":         0.1,
    "hsv_s":         0.9,
    "hsv_v":         0.6,
    "conf":          0.15,
    "iou":           0.5,
    "project":       str(CKPT_DIR / "yolo_runs"),
    "name":          "clickbait_detector_hpc",
    "exist_ok":      True,
    "verbose":       True,
    "optimizer":     "auto",           # Let Ultralytics pick the best optimizer (usually AdamW)
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
