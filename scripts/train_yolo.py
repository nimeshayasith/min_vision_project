"""
train_yolo.py — YOLOv8 Training on Roboflow Dataset
Detects deceptive UI elements: fake_download_button, ad_banner, close_button, etc.

Usage:
    python scripts/train_yolo.py                     # 25 epochs, 416px (fast first run)
    python scripts/train_yolo.py --quick             # 10 epochs, 320px  (pipeline test)
    python scripts/train_yolo.py --full              # 80 epochs, 640px  (best accuracy)
    python scripts/train_yolo.py --model yolov8s     # larger model
    python scripts/train_yolo.py --device cpu        # force CPU
    python scripts/train_yolo.py --device 0          # force GPU (if ROCm/CUDA installed)

AMD GPU Note:
    Your current PyTorch is CPU-only. To use your RX 5500M, install ROCm PyTorch:

    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

    After that, torch.cuda.is_available() will return True for AMD GPUs.
    Training will be ~5-10x faster.
"""

import shutil
import yaml
import argparse
import time
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_YAML      = PROJECT_ROOT / "data" / "data.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
FINAL_WEIGHTS  = CHECKPOINT_DIR / "yolo_best.pt"


def detect_device(requested: str = "auto") -> str:
    """
    Returns the best available device string for ultralytics.
    Prints a clear message about what was found.
    """
    if requested != "auto":
        print(f"  Device   : {requested} (forced by --device)")
        return requested

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Device   : GPU — {name} ({vram:.1f} GB VRAM)")
        return "0"

    # Check for AMD GPU present even without ROCm PyTorch
    try:
        import subprocess
        r = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"],
                           capture_output=True, text=True, timeout=3)
        if "Radeon" in r.stdout or "AMD" in r.stdout:
            print("  Device   : CPU  ⚠️  AMD GPU detected but ROCm PyTorch not installed.")
            print()
            print("  ╔══ HOW TO ENABLE AMD GPU ══════════════════════════════════╗")
            print("  ║  pip uninstall torch torchvision torchaudio -y            ║")
            print("  ║  pip install torch torchvision torchaudio \\               ║")
            print("  ║      --index-url https://download.pytorch.org/whl/rocm6.2 ║")
            print("  ║  Then re-run this script — GPU will be used automatically ║")
            print("  ╚═══════════════════════════════════════════════════════════╝")
            print()
        else:
            print("  Device   : CPU")
    except Exception:
        print("  Device   : CPU")

    return "cpu"


def build_fixed_yaml() -> Path:
    """
    Roboflow exports data.yaml with relative paths that break when running
    from the project root. Rewrite with absolute paths into data_fixed.yaml.
    """
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)

    data_dir = DATA_YAML.parent
    cfg["train"] = str((data_dir / "train" / "images").resolve())
    cfg["val"]   = str((data_dir / "valid" / "images").resolve())
    cfg["test"]  = str((data_dir / "test"  / "images").resolve())
    cfg.pop("roboflow", None)

    fixed_path = data_dir / "data_fixed.yaml"
    with open(fixed_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return fixed_path


def train(
    model_size : str  = "yolov8n",
    epochs     : int  = 25,
    imgsz      : int  = 416,
    batch      : int  = 16,
    device     : str  = "auto",
    cache      : bool = True,
) -> None:

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    fixed_yaml = build_fixed_yaml()

    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)

    resolved_device = detect_device(device)

    # amp (mixed precision) only helps on GPU; disable on CPU to avoid warnings
    use_amp = (resolved_device != "cpu")

    # Cache in RAM if dataset fits (271 train images ~ 100-300 MB — fine)
    cache_mode = True if cache else False

    print("=" * 60)
    print("  YOLOv8 Clickbait Detector — Training")
    print("=" * 60)
    print(f"  Classes  : {cfg['names']}")
    print(f"  Samples  : train=271  val=29  test=29")
    print(f"  Model    : {model_size}.pt  (pretrained COCO weights)")
    print(f"  Epochs   : {epochs}")
    print(f"  Img size : {imgsz}px  (640px = max quality, 416px = faster)")
    print(f"  Batch    : {batch}")
    print(f"  AMP      : {use_amp}  (mixed precision — GPU only)")
    print(f"  Cache    : {cache_mode}  (images preloaded into RAM)")
    print("=" * 60)
    print()

    model = YOLO(f"{model_size}.pt")

    t0 = time.time()

    model.train(
        data       = str(fixed_yaml),
        epochs     = epochs,
        imgsz      = imgsz,
        batch      = batch,
        device     = resolved_device,
        amp        = use_amp,
        cache      = cache_mode,
        project    = str(PROJECT_ROOT / "models"),
        name       = "yolo_run",
        exist_ok   = True,
        save       = True,
        plots      = True,
        patience   = 15,    # Early stop after 15 epochs with no improvement
        val        = True,
        workers    = 0,     # 0 avoids multiprocessing issues on Windows
        optimizer  = "AdamW",
        lr0        = 0.001,
        cos_lr     = True,  # Cosine LR schedule — better convergence
    )

    elapsed = time.time() - t0
    print(f"\n⏱  Training time: {elapsed/60:.1f} minutes")

    # Copy best weights to standard location
    best = PROJECT_ROOT / "models" / "yolo_run" / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, FINAL_WEIGHTS)
        print(f"✅ Best weights → {FINAL_WEIGHTS}")
        print(f"\nStart the server:  python -m src.server.app")
    else:
        print(f"⚠️  Best weights not found at {best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for deceptive UI element detection"
    )

    # Preset modes
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="Quick test: 10 epochs @ 320px — checks the pipeline works")
    mode.add_argument("--full",  action="store_true",
                      help="Full quality: 80 epochs @ 640px — best accuracy, slowest")

    # Manual overrides
    parser.add_argument("--model", default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
                        help="Model size: n=fastest, l=most accurate (default: yolov8n)")
    parser.add_argument("--epochs", type=int,  default=None,
                        help="Override epoch count")
    parser.add_argument("--imgsz",  type=int,  default=None,
                        help="Override image size in pixels")
    parser.add_argument("--batch",  type=int,  default=16,
                        help="Batch size — reduce to 8 or 4 if running out of memory")
    parser.add_argument("--device", type=str,  default="auto",
                        help="Device: auto | cpu | 0 (GPU). Default: auto-detect")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable RAM caching (use if you get memory errors)")

    args = parser.parse_args()

    # Resolve preset modes
    if args.quick:
        epochs, imgsz = 10, 320
        print("⚡ QUICK mode — pipeline test (10 epochs @ 320px)")
    elif args.full:
        epochs, imgsz = 80, 640
        print("🏆 FULL mode — best accuracy (80 epochs @ 640px)")
    else:
        epochs = args.epochs or 25
        imgsz  = args.imgsz  or 416
        print(f"🚀 FIRST RUN mode ({epochs} epochs @ {imgsz}px) — balanced speed/quality")

    print()

    train(
        model_size = args.model,
        epochs     = epochs,
        imgsz      = imgsz,
        batch      = args.batch,
        device     = args.device,
        cache      = not args.no_cache,
    )
