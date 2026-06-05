"""
cv_features.py — Pure Computer Vision Feature Validator
Owner: All | Tech: OpenCV, NumPy

This module implements HUMAN-LIKE visual reasoning to validate and score
clickbait element candidates detected by YOLOv8.

A human identifies clickbait by looking at:
  1. Color — bright red, orange, yellow → aggressive/attention-grabbing
  2. Saturation — highly saturated = designed to stand out
  3. Contrast — extreme contrast between text and background
  4. Size — unusually large buttons/banners
  5. Aspect ratio — wide banner-like shapes or square button shapes
  6. Edge density — busy/cluttered regions = ad content
  7. Position — centered or top of page = prime clickbait placement

CV Techniques Used:
  - HSV color space analysis (hue, saturation, value histograms)
  - CLAHE for contrast measurement
  - Canny edge detection for edge density
  - Morphological operations
  - Statistical analysis (mean, std of pixel distributions)
"""

import cv2
import numpy as np


# ── Class metadata: visual weight for clickbait scoring ──
CLASS_CLICKBAIT_WEIGHT = {
    "Buttons":          0.7,   # Generic buttons — moderate risk
    "Computer-vision":  0.8,   # Deceptive visual elements — high risk
    "ad_banner":        0.9,   # Ad banners — very high risk
    "close_button":     0.6,   # Close buttons — moderate (may be fake)
    "fake_download_button": 1.0,  # Fake downloads — highest risk
}

CLASS_NAMES = ['Buttons', 'Computer-vision', 'ad_banner',
               'close_button', 'fake_download_button']


# ── Color ranges for "alert colors" in HSV ──
# These are colors commonly used in clickbait: red, orange, yellow, green
ALERT_HUE_RANGES = [
    (0,   15),    # Red (lower)
    (160, 180),   # Red (upper wraparound)
    (15,  35),    # Orange
    (35,  55),    # Yellow-green
]


def _compute_color_features(crop_bgr: np.ndarray) -> dict:
    """
    Analyze the color distribution of a region.

    Returns:
        alert_color_ratio: fraction of pixels in "alert" hues (red/orange/yellow)
        mean_saturation:   average saturation (0-255) — high = vivid/artificial
        mean_brightness:   average brightness (0-255)
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Alert color mask
    alert_mask = np.zeros(h.shape, dtype=np.uint8)
    for lo, hi in ALERT_HUE_RANGES:
        alert_mask |= ((h >= lo) & (h <= hi)).astype(np.uint8)

    total_pixels     = h.size
    alert_pixels     = int(np.sum(alert_mask))
    alert_color_ratio = alert_pixels / max(total_pixels, 1)

    mean_saturation  = float(np.mean(s))
    mean_brightness  = float(np.mean(v))

    return {
        "alert_color_ratio": alert_color_ratio,
        "mean_saturation":   mean_saturation / 255.0,   # Normalize to [0, 1]
        "mean_brightness":   mean_brightness / 255.0,
    }


def _compute_contrast_features(crop_bgr: np.ndarray) -> dict:
    """
    Measure local contrast — high contrast = designed to grab attention.

    Uses standard deviation of grayscale pixel values.
    High std = high contrast = suspicious.
    """
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))

    # CLAHE enhanced contrast
    clahe         = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced      = clahe.apply(gray)
    enhanced_std  = float(np.std(enhanced))

    return {
        "contrast_std":          std_dev / 128.0,       # Normalize: 128 = moderate
        "enhanced_contrast_std": enhanced_std / 128.0,
    }


def _compute_edge_features(crop_bgr: np.ndarray) -> dict:
    """
    Edge density — how 'busy' or 'cluttered' a region is.

    Ads and clickbait tend to have high edge density (many borders,
    text overlays, graphic elements). Clean content areas have low density.

    Uses Canny edge detection.
    """
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    edge_density = float(np.sum(edges > 0)) / max(edges.size, 1)

    return {"edge_density": edge_density}


def _compute_shape_features(x: int, y: int, w: int, h: int,
                             img_w: int, img_h: int) -> dict:
    """
    Spatial & shape features — size, position, aspect ratio.

    These are the most 'human-like' features:
    - A human notices that a button is UNUSUALLY large
    - A human notices that an element is CENTERED to attract attention
    - A human recognizes banner-like aspect ratios

    Args:
        x, y, w, h:   Bounding box in pixel coordinates
        img_w, img_h: Full image dimensions
    """
    area_ratio    = (w * h) / max(img_w * img_h, 1)
    aspect_ratio  = w / max(h, 1)

    # Normalized position (0=top-left, 1=bottom-right)
    center_x_norm = (x + w / 2) / max(img_w, 1)
    center_y_norm = (y + h / 2) / max(img_h, 1)

    # Is it near the top or center? (prime clickbait real estate)
    # Score 1.0 if centered horizontally and in upper 60% vertically
    horizontal_centrality = 1.0 - abs(center_x_norm - 0.5) * 2
    vertical_prominence   = 1.0 - center_y_norm  # Higher score for top placement

    # Banner-like aspect ratio: wide (>3:1) or square-ish (0.8-1.5)
    is_banner_like = 1.0 if aspect_ratio > 3.0 else 0.0
    is_button_like = 1.0 if 1.5 <= aspect_ratio <= 6.0 else 0.0

    return {
        "area_ratio":             area_ratio,
        "aspect_ratio":           aspect_ratio,
        "horizontal_centrality":  horizontal_centrality,
        "vertical_prominence":    vertical_prominence,
        "is_banner_like":         is_banner_like,
        "is_button_like":         is_button_like,
    }


def compute_clickbait_visual_score(
    crop_bgr: np.ndarray,
    x: int, y: int, w: int, h: int,
    img_w: int, img_h: int,
    class_name: str = "Buttons",
) -> tuple[float, dict]:
    """
    Compute a holistic visual clickbait score for a detected region.

    This replicates how a human judges a UI element:
    - Is it bright and alarming in color?
    - Is it high-contrast and attention-grabbing?
    - Is it unusually large for what it is?
    - Is it placed where you can't miss it?
    - Is it visually cluttered like an ad?

    Args:
        crop_bgr:    Cropped BGR image of the detected region
        x,y,w,h:     Bounding box in full-image pixel coordinates
        img_w,img_h: Full screenshot dimensions
        class_name:  Detected class from YOLO

    Returns:
        score:    Float in [0.0, 1.0] — higher = more likely clickbait
        features: Dict of individual feature values (for debugging/logging)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0, {}

    # Ensure minimum crop size for meaningful analysis
    min_dim = 10
    if crop_bgr.shape[0] < min_dim or crop_bgr.shape[1] < min_dim:
        return 0.5, {}   # Give benefit of the doubt for tiny regions

    try:
        color_feats   = _compute_color_features(crop_bgr)
        contrast_feats = _compute_contrast_features(crop_bgr)
        edge_feats    = _compute_edge_features(crop_bgr)
        shape_feats   = _compute_shape_features(x, y, w, h, img_w, img_h)
    except Exception as e:
        print(f"[cv_features] Error computing features: {e}")
        return 0.5, {}

    all_features = {**color_feats, **contrast_feats, **edge_feats, **shape_feats}

    # ── Weighted Score Computation ───────────────────────────────────────
    # Each feature contributes to the final score with a calibrated weight.
    # These weights encode human visual judgment about what makes something
    # look like clickbait.

    score = (
        color_feats["alert_color_ratio"]         * 0.20 +   # Red/orange = alarm
        color_feats["mean_saturation"]           * 0.15 +   # Vivid = artificial
        contrast_feats["contrast_std"]           * 0.15 +   # High contrast = designed
        edge_feats["edge_density"]               * 0.10 +   # Cluttered = ad
        shape_feats["horizontal_centrality"]     * 0.10 +   # Centered = bait
        shape_feats["vertical_prominence"]       * 0.10 +   # Top-of-page = prominent
        min(shape_feats["area_ratio"] * 10, 1.0) * 0.10 +  # Oversized = suspicious
        shape_feats["is_button_like"]            * 0.05 +   # Button shape
        shape_feats["is_banner_like"]            * 0.05     # Banner shape
    )

    # Apply class-based multiplier
    class_weight = CLASS_CLICKBAIT_WEIGHT.get(class_name, 0.7)
    score = min(score * class_weight * 1.5, 1.0)   # Scale up, cap at 1.0

    return float(score), all_features


def filter_clickbait_boxes(
    yolo_boxes: list[dict],
    img_bgr: np.ndarray,
    min_cv_score: float = 0.15,
) -> list[dict]:
    """
    Given YOLO detection results, apply CV feature scoring to filter
    and enrich the bounding boxes.

    Args:
        yolo_boxes:   List of dicts with keys: x,y,w,h,confidence,class_name
        img_bgr:      Full BGR screenshot image
        min_cv_score: Minimum CV visual score to keep a box (default: 0.15)

    Returns:
        Filtered list of boxes with added 'cv_score' and enriched 'confidence'
    """
    img_h, img_w = img_bgr.shape[:2]
    filtered = []

    for box in yolo_boxes:
        x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
        class_name  = box.get("class_name", "Buttons")

        # Clamp to image bounds
        x  = max(0, x)
        y  = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x or y2 <= y:
            continue

        crop = img_bgr[y:y2, x:x2]
        cv_score, features = compute_clickbait_visual_score(
            crop, x, y, (x2 - x), (y2 - y), img_w, img_h, class_name
        )

        if cv_score >= min_cv_score:
            # Combine YOLO confidence with CV score for final confidence
            yolo_conf    = float(box.get("confidence", 0.5))
            final_conf   = (yolo_conf * 0.7 + cv_score * 0.3)

            filtered.append({
                "x":          x,
                "y":          y,
                "w":          x2 - x,
                "h":          y2 - y,
                "confidence": round(final_conf, 3),
                "cv_score":   round(cv_score, 3),
                "class_name": class_name,
                "features":   features,   # For debugging
            })

    return filtered
