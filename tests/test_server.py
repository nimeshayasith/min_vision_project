"""
test_server.py — Flask Server Tests
Uses Flask test client to verify API endpoints without running a real server.
"""

import pytest
import base64
import json
import numpy as np
from PIL import Image
import io


@pytest.fixture
def client():
    """Create Flask test client."""
    from src.server.app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def make_test_image_b64() -> str:
    """Create a small test image encoded as base64 data URL."""
    img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_health_endpoint(client):
    """GET /health should return 200 with status=ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "ok"


def test_index_endpoint(client):
    """GET / should return endpoint documentation."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "endpoints" in data


def test_analyze_returns_label(client):
    """POST /analyze should return label and confidence fields."""
    payload = {
        "image": make_test_image_b64(),
        "url": "https://example.com",
    }
    resp = client.post(
        "/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "label" in data
    assert "confidence" in data
    assert data["label"] in ["legitimate", "phishing", "clickbait"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_analyze_missing_image(client):
    """POST /analyze without image should return 400."""
    resp = client.post(
        "/analyze",
        data=json.dumps({"url": "https://example.com"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_analyze_probabilities_sum_to_one(client):
    """Probabilities must sum to approximately 1.0."""
    payload = {"image": make_test_image_b64(), "url": "https://example.com"}
    resp = client.post("/analyze", data=json.dumps(payload), content_type="application/json")
    data = json.loads(resp.data)
    if "probabilities" in data:
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 1e-3, f"Probabilities sum to {total}, expected ~1.0"
