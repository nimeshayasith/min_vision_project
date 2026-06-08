"""
app.py — Phase 7: Flask Backend Server
Receives base64 screenshots from the Chrome extension, runs YOLOv8 inference,
and returns the page classification + bounding boxes of deceptive UI elements.

Start server:
    python -m src.server.app
"""

import base64
import io
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)   # Required: Chrome extension is a cross-origin caller

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

YOLO_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "yolo_best.pt"

# Classes from the Roboflow dataset that indicate deceptive intent
DECEPTIVE_CLASSES = {"fake_download_button", "ad_banner", "close_button"}

# Minimum YOLO confidence to include a detection
CONF_THRESHOLD = 0.20   # Low threshold — small dataset model has lower raw confidence scores

_yolo_model = None


def get_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    if not YOLO_CHECKPOINT.exists():
        print(f"⚠️  No YOLO checkpoint at {YOLO_CHECKPOINT}")
        print("    Run:  python scripts/train_yolo.py")
        return None

    from ultralytics import YOLO
    _yolo_model = YOLO(str(YOLO_CHECKPOINT))
    print(f"✅ YOLO model loaded from {YOLO_CHECKPOINT}")
    return _yolo_model


# ─────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Request JSON:
        { "image": "data:image/png;base64,...", "url": "https://..." }

    Response JSON:
        {
            "label":           "legitimate" | "clickbait",
            "confidence":      0.87,
            "probabilities":   {"legitimate": 0.13, "clickbait": 0.87},
            "bounding_boxes":  [
                { "class": "fake_download_button", "confidence": 0.87,
                  "x": 0.12, "y": 0.45, "w": 0.08, "h": 0.03,
                  "deceptive": true }
            ],
            "all_detections":  [...]   // includes neutral detections too
        }

    Bounding box coordinates are NORMALIZED [0.0 – 1.0] relative to the
    screenshot dimensions. The extension scales them to viewport CSS pixels.
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    # ── Decode base64 image ──────────────────────────────
    try:
        image_b64 = data["image"]
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        image_bytes = base64.b64decode(image_b64)
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = pil_image.size
    except Exception as exc:
        return jsonify({"error": f"Failed to decode image: {exc}"}), 400

    # ── Run YOLO inference ───────────────────────────────
    model = get_model()
    if model is None:
        return jsonify({
            "label":          "unknown",
            "confidence":     0.0,
            "probabilities":  {"legitimate": 0.0, "clickbait": 0.0},
            "bounding_boxes": [],
            "all_detections": [],
            "warning":        "Model not trained yet. Run: python scripts/train_yolo.py",
        })

    results = model.predict(pil_image, conf=CONF_THRESHOLD, verbose=False)

    # ── Parse detections ─────────────────────────────────
    all_detections  = []
    deceptive_confs = []

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id   = int(boxes.cls[i].item())
            conf     = float(boxes.conf[i].item())
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            is_deceptive = cls_name in DECEPTIVE_CLASSES
            detection = {
                "class":      cls_name,
                "confidence": round(conf, 3),
                # Normalized [0,1] relative to screenshot dimensions
                "x": round(x1 / img_w, 4),
                "y": round(y1 / img_h, 4),
                "w": round((x2 - x1) / img_w, 4),
                "h": round((y2 - y1) / img_h, 4),
                "deceptive":  is_deceptive,
            }
            all_detections.append(detection)
            if is_deceptive:
                deceptive_confs.append(conf)

    # ── Page-level classification ─────────────────────────
    if deceptive_confs:
        label      = "clickbait"
        confidence = round(float(max(deceptive_confs)), 3)
    else:
        label = "legitimate"
        if not all_detections:
            # YOLO found nothing — we cannot be certain; cap at 65%
            confidence = 0.65
        else:
            # Only neutral elements (Buttons etc.) detected — reasonably clean
            max_neutral = max(d["confidence"] for d in all_detections)
            confidence  = round(min(0.82, 0.65 + max_neutral * 0.18), 3)

    bounding_boxes = [d for d in all_detections if d["deceptive"]]

    return jsonify({
        "label":          label,
        "confidence":     confidence,
        "probabilities":  {
            "legitimate": round(1.0 - confidence, 3) if label == "clickbait" else confidence,
            "clickbait":  confidence if label == "clickbait" else round(1.0 - confidence, 3),
        },
        "bounding_boxes": bounding_boxes,
        "all_detections": all_detections,
    })


@app.route("/health", methods=["GET"])
def health():
    model_ready = YOLO_CHECKPOINT.exists()
    return jsonify({
        "status":      "ok",
        "device":      str(DEVICE),
        "model_ready": model_ready,
    })


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Clickbait Detector API (YOLOv8)",
        "endpoints": {
            "POST /analyze": "Send base64 screenshot → label + bounding boxes",
            "GET  /health":  "Server health check",
        }
    })


if __name__ == "__main__":
    print("🚀 Starting Clickbait Detector server on http://localhost:5000")
    # Pre-load model at startup
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
