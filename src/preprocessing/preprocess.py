"""
preprocess.py — Phase 3: Image Preprocessing
Owner: Dassanayaka | Tech: OpenCV, NumPy

Full pipeline: load → denoise → enhance contrast → resize → normalize
"""

import cv2
import numpy as np
from pathlib import Path

TARGET_SIZE = (224, 224)   # Required by ResNet-50

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 3.1 — Load & Resize & Normalize
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk. Returns BGR NumPy array."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def resize_image(img: np.ndarray, size: tuple = TARGET_SIZE) -> np.ndarray:
    """Resize image to target size using high-quality interpolation."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] then apply ImageNet mean/std.
    Input:  BGR uint8 image (H, W, 3)
    Output: RGB float32 tensor-ready array (H, W, 3)
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    return normalized


# ---------------------------------------------------------------------------
# Step 3.2 — Noise Reduction
# ---------------------------------------------------------------------------

def reduce_noise(img: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur to remove screenshot compression artifacts.
    Kernel (3,3) is sufficient — stronger blur loses UI detail.
    """
    return cv2.GaussianBlur(img, (3, 3), sigmaX=0)


# ---------------------------------------------------------------------------
# Step 3.3 — Region of Interest Detection
# ---------------------------------------------------------------------------

def detect_ui_regions(img: np.ndarray) -> list:
    """
    Use Canny edge detection to find candidate UI regions (buttons, banners).
    Returns list of bounding boxes: [(x, y, w, h), ...]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Dilate to connect nearby edges into regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Filter: keep reasonably-sized regions (not noise, not whole page)
        if 500 < area < (img.shape[0] * img.shape[1] * 0.5):
            regions.append((x, y, w, h))
    return regions


# ---------------------------------------------------------------------------
# Step 3.4 — Contrast Enhancement (CLAHE)
# ---------------------------------------------------------------------------

def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to make UI element boundaries more visible.
    Works on the L channel in LAB color space.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline: load → denoise → enhance → resize → normalize.
    Returns: normalized float32 array ready for model input (H=224, W=224, C=3).
    """
    img = load_image(image_path)
    img = reduce_noise(img)
    img = enhance_contrast(img)
    img = resize_image(img)
    img = normalize_image(img)
    return img


def preprocess_pipeline_from_array(img_array: np.ndarray) -> np.ndarray:
    """
    Same pipeline but starting from a NumPy RGB array (e.g. from PIL).
    Input:  RGB uint8 array (H, W, 3)
    Output: normalized float32 (224, 224, 3)
    """
    # Create debug directory
    debug_dir = Path("debug_preprocessing")
    debug_dir.mkdir(exist_ok=True)

    # Convert RGB to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(debug_dir / "01_input.png"), img_bgr)

    # Denoise
    img_denoised = reduce_noise(img_bgr)
    cv2.imwrite(str(debug_dir / "02_denoised.png"), img_denoised)

    # Contrast enhance
    img_enhanced = enhance_contrast(img_denoised)
    cv2.imwrite(str(debug_dir / "03_enhanced.png"), img_enhanced)

    # Region of Interest (for visualization only)
    try:
        regions = detect_ui_regions(img_enhanced)
        img_regions = img_enhanced.copy()
        for x, y, w, h in regions:
            cv2.rectangle(img_regions, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "04_ui_regions.png"), img_regions)
    except Exception as e:
        print(f"Error saving UI regions debug image: {e}")

    # Resize
    img_resized = resize_image(img_enhanced)
    cv2.imwrite(str(debug_dir / "05_resized.png"), img_resized)

    # Normalize
    img_out = normalize_image(img_resized)   # normalize_image converts back to RGB
    return img_out

