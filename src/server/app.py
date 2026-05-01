"""
app.py — Phase 7: Flask Backend Server
Owner: Wanigasooriya | Tech: Flask, PyTorch

Receives base64-encoded webpage screenshots from the Chrome extension
and returns classification results.

Start server:
    python -m src.server.app
"""

import base64
import io
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.model.model import ClickbaitDetector, CLASS_NAMES
from src.preprocessing.preprocess import preprocess_pipeline_from_array
from src.training.config import CONFIG

app = Flask(__name__)
CORS(app)   # Required: Chrome extension is cross-origin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load trained model at startup ──
import os
from pathlib import Path

_model = None

def get_model() -> ClickbaitDetector:
    global _model
    if _model is None:
        ckpt = Path(CONFIG["checkpoint_dir"]) / "best_model.pth"
        _model = ClickbaitDetector(backbone=CONFIG["backbone"])
        if ckpt.exists():
            _model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
            print(f"✅ Model loaded from {ckpt}")
        else:
            print(f"⚠️  No checkpoint found at {ckpt}. Using untrained model for testing.")
        _model.to(DEVICE)
        _model.eval()
    return _model


# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Receives a base64-encoded webpage screenshot from the Chrome extension.

    Request JSON:
        {
            "image": "data:image/png;base64,...",
            "url":   "https://example.com"
        }

    Response JSON:
        {
            "label":         "legitimate" | "phishing" | "clickbait",
            "confidence":    0.95,
            "probabilities": {"legitimate": 0.95, "phishing": 0.04, "clickbait": 0.01},
            "bounding_boxes": []
        }
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_b64 = data["image"]

    # Decode base64 → PIL Image → NumPy RGB array
    try:
        # Handle data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        image_bytes = base64.b64decode(image_b64)
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array   = np.array(pil_image)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    # Preprocess
    preprocessed = preprocess_pipeline_from_array(img_array)
    tensor = (
        torch.from_numpy(preprocessed.transpose(2, 0, 1))
        .unsqueeze(0)
        .to(DEVICE)
    )

    # Predict
    model = get_model()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    predicted_class = int(np.argmax(probs))
    confidence      = float(probs[predicted_class])
    label           = CLASS_NAMES[predicted_class]

    return jsonify({
        "label":          label,
        "confidence":     confidence,
        "probabilities":  {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
        "bounding_boxes": [],  # TODO: integrate object detection for precise box output
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "device": str(DEVICE)})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Clickbait Detector API",
        "endpoints": {
            "POST /analyze": "Send base64 screenshot, get label + confidence",
            "GET  /health":  "Server health check",
        }
    })


if __name__ == "__main__":
    print("🚀 Starting Clickbait Detector server on http://localhost:5000")
    # Pre-load model at startup
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
