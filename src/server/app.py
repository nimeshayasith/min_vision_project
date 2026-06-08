"""
app.py — Flask Backend Server (Rebuilt for Element-Level Detection)
Tech: Flask, YOLOv8 (Ultralytics), OpenCV

Pipeline:
  1. Receive base64 screenshot from Chrome extension
  2. Run YOLOv8 object detector → get clickbait element bounding boxes
  3. Run CV feature validator → score each region visually (human-like)
  4. Return filtered boxes with coordinates + confidence

Start server:
    python -m src.server.app
"""

import base64
import io
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.preprocessing.cv_features import filter_clickbait_boxes, CLASS_NAMES

app = Flask(__name__)
CORS(app)   # Required: Chrome extension is cross-origin

# ── Model path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "yolo_clickbait.pt"

_yolo_model = None


def get_yolo_model():
    """Load YOLOv8 model lazily (once at first request)."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            if YOLO_MODEL_PATH.exists():
                _yolo_model = YOLO(str(YOLO_MODEL_PATH))
                print(f"✅ YOLOv8 model loaded from {YOLO_MODEL_PATH}")
            else:
                # No trained model yet — use base pretrained for structure only
                print(f"⚠️  No trained model at {YOLO_MODEL_PATH}")
                print("   Run: python -m src.training.train_yolo")
                print("   Using YOLOv8n base model (untrained for clickbait) for testing")
                _yolo_model = YOLO("yolov8n.pt")
        except ImportError:
            print("❌ ultralytics not installed. Run: pip install ultralytics")
            return None
    return _yolo_model


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Receives a base64-encoded webpage screenshot from the Chrome extension.

    Request JSON:
        {
            "image": "data:image/png;base64,...",
            "url":   "https://example.com",
            "dpr":   1.0
        }

    Response JSON:
        {
            "label":         "clickbait" | "clean",
            "confidence":    0.87,
            "bounding_boxes": [
                {
                    "x": 100, "y": 200, "w": 300, "h": 60,
                    "confidence": 0.82,
                    "class_name": "ad_banner",
                    "cv_score": 0.74
                },
                ...
            ]
        }
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_b64 = data["image"]
    dpr       = float(data.get("dpr", 1.0))

    # ── Decode screenshot ─────────────────────────────────────────────────
    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        image_bytes = base64.b64decode(image_b64)
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb     = np.array(pil_image)
        img_bgr     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    # ── Stage 1: YOLOv8 Element Detection ────────────────────────────────
    model = get_yolo_model()
    raw_boxes = []

    if model is not None:
        try:
            results = model.predict(
                source=img_rgb,
                conf=0.25,        # Minimum YOLO confidence threshold
                iou=0.45,         # NMS IoU threshold
                device="cpu",
                verbose=False,
            )

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_xyxy  = result.boxes.xyxy.cpu().numpy()    # [x1,y1,x2,y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids   = result.boxes.cls.cpu().numpy().astype(int)

                    for (x1, y1, x2, y2), conf, cls_id in zip(
                            boxes_xyxy, confidences, class_ids):
                        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown"
                        raw_boxes.append({
                            "x":          int(x1),
                            "y":          int(y1),
                            "w":          int(x2 - x1),
                            "h":          int(y2 - y1),
                            "confidence": float(conf),
                            "class_name": cls_name,
                        })
        except Exception as e:
            print(f"[YOLO Error] {e}")

    # ── Stage 2: CV Feature Validation ────────────────────────────────────
    # Score each YOLO box using pure OpenCV visual features
    # This is the "human-like" judgment layer
    validated_boxes = filter_clickbait_boxes(
        yolo_boxes=raw_boxes,
        img_bgr=img_bgr,
    )

    # ── Scale down by DPR for viewport coordinate mapping ─────────────────
    clickbait_boxes = []
    for box in validated_boxes:
        clickbait_boxes.append({
            "x":          int(round(box["x"] / dpr)),
            "y":          int(round(box["y"] / dpr)),
            "w":          int(round(box["w"] / dpr)),
            "h":          int(round(box["h"] / dpr)),
            "confidence": box["confidence"],
            "class_name": box["class_name"],
            "cv_score":   box["cv_score"],
            "type":       box["type"],
        })

    # ── Determine overall page label ──────────────────────────────────────
    if clickbait_boxes:
        max_conf = max(b["confidence"] for b in clickbait_boxes)
        label    = "clickbait"
    else:
        max_conf = 0.0
        label    = "clean"

    return jsonify({
        "label":         label,
        "confidence":    round(max_conf, 3),
        "bounding_boxes": clickbait_boxes,
        "detection_count": len(clickbait_boxes),
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model_loaded = _yolo_model is not None
    model_trained = YOLO_MODEL_PATH.exists()
    return jsonify({
        "status":        "ok",
        "model_loaded":  model_loaded,
        "model_trained": model_trained,
        "model_path":    str(YOLO_MODEL_PATH),
    })


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Clickbait Element Detector API (YOLOv8 + CV Features)",
        "endpoints": {
            "POST /analyze": "Send base64 screenshot → get clickbait bounding boxes",
            "GET  /health":  "Server health check",
        },
        "classes": CLASS_NAMES,
    })


if __name__ == "__main__":
    print("🚀 Starting Clickbait Detector server on http://localhost:5000")
    # Pre-load model at startup
    get_yolo_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
