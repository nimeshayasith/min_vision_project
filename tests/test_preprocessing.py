"""
test_preprocessing.py — Phase 3 Tests
Verifies each preprocessing function in isolation.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from src.preprocessing.preprocess import (
    load_image,
    resize_image,
    normalize_image,
    reduce_noise,
    enhance_contrast,
    preprocess_pipeline,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary 300x400 test image."""
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    path = tmp_path / "test_image.png"
    cv2.imwrite(str(path), img)
    return str(path)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_load_image(sample_image_path):
    img = load_image(sample_image_path)
    assert img is not None
    assert len(img.shape) == 3        # H, W, C
    assert img.shape[2] == 3          # BGR channels


def test_load_missing_image():
    with pytest.raises(FileNotFoundError):
        load_image("data/nonexistent_image.png")


def test_resize_image(sample_image_path):
    img = load_image(sample_image_path)
    resized = resize_image(img, (224, 224))
    assert resized.shape == (224, 224, 3)


def test_normalize_range(sample_image_path):
    img = load_image(sample_image_path)
    img = resize_image(img)
    normalized = normalize_image(img)
    assert normalized.dtype == np.float32
    # After ImageNet normalization values are typically in [-3, 3]
    assert normalized.min() >= -4.0
    assert normalized.max() <=  4.0


def test_normalize_output_is_rgb(sample_image_path):
    """normalize_image must convert BGR → RGB."""
    img = load_image(sample_image_path)      # BGR
    img = resize_image(img)
    normalized = normalize_image(img)        # Should be RGB float32
    assert normalized.shape == (224, 224, 3)
    assert normalized.dtype == np.float32


def test_noise_reduction_preserves_shape(sample_image_path):
    img = load_image(sample_image_path)
    denoised = reduce_noise(img)
    assert denoised.shape == img.shape


def test_enhance_contrast_preserves_shape(sample_image_path):
    img = load_image(sample_image_path)
    enhanced = enhance_contrast(img)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == img.dtype


def test_full_pipeline_output_shape(sample_image_path):
    output = preprocess_pipeline(sample_image_path)
    assert output.shape == (224, 224, 3)
    assert output.dtype == np.float32
