# 🛡️ Visual Detection of Deceptive Clickbait UI Elements in Websites

> **Course:** EE7204 / EC7205 — Image Processing & Computer Vision
> **Department:** Electrical and Information Engineering, University of Ruhuna
> **Group:** 10 | **Submission Date:** 23/01/2026

---

## 👥 Team Members

| Name | Index |
|------|-------|
| M.P.S. Koshala | EG/2021/4617 |
| W.A.L.N. Wanigasooriya | EG/2021/4848 |
| D.M.D.P. Dassanayaka | EG/2021/4456 |
| K.N.Y. Kumara | EG/2021/4624 |

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Environment Setup](#4-environment-setup)
5. [Phase 1 — Dataset Collection & Preparation](#5-phase-1--dataset-collection--preparation)
6. [Phase 2 — Browser Extension (Input Layer)](#6-phase-2--browser-extension-input-layer)
7. [Phase 3 — Image Preprocessing](#7-phase-3--image-preprocessing)
8. [Phase 4 — Visual Feature Extraction](#8-phase-4--visual-feature-extraction)
9. [Phase 5 — Multi-Component Reasoning Pipeline](#9-phase-5--multi-component-reasoning-pipeline)
10. [Phase 6 — Classification Layer](#10-phase-6--classification-layer)
11. [Phase 7 — Decision & Alert System](#11-phase-7--decision--alert-system)
12. [Phase 8 — Model Evaluation & Testing](#12-phase-8--model-evaluation--testing)
13. [Running the Full Pipeline](#13-running-the-full-pipeline)
14. [Expert Suggestions & Notes](#14-expert-suggestions--notes)

---

## 1. Project Overview

### Problem Statement
Modern websites increasingly use visually deceptive UI elements — fake download buttons, misleading pop-ups, and system-like security messages — to manipulate users into unintended interactions. Existing browser-level protections rely on URL analysis, HTML inspection, or rule-based blocking, which fail to detect deceptions rooted in **visual appearance and layout manipulation**.

### Our Solution
This project builds a **computer vision-driven pipeline** that:
1. Captures a screenshot of any webpage through a Chrome browser extension
2. Preprocesses and analyzes the screenshot using deep learning
3. Classifies the page and highlights deceptive UI elements in real-time

> **Key Differentiator:** The system detects deception through **visual features only** (color contrast, spatial layout, button geometry, brand impersonation) — not through URL or text analysis.

### Target Classes
| Label | Description |
|-------|-------------|
| ✅ `legitimate` | Normal, safe website with no deceptive elements |
| ⚠️ `phishing` | Website impersonating a trusted brand or service |
| ❌ `clickbait` | Website with fake buttons, misleading ads, or deceptive pop-ups |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYSTEM PIPELINE                          │
│                                                                 │
│  [User Browses]──►[Chrome Extension]──►[Screenshot Capture]    │
│                                               │                 │
│                                               ▼                 │
│                              [Image Preprocessing - OpenCV]     │
│                          (Resize, Normalize, Denoise, Crop)     │
│                                               │                 │
│                                               ▼                 │
│                     [Visual Feature Extraction - ResNet/ViT]    │
│                   (Layout, Color, Logo, Button Design Patterns) │
│                                               │                 │
│                     ┌─────────────────────────┘                 │
│                     ▼                                           │
│          [Multi-Component Reasoning - CNN + Attention]          │
│            (Visual Similarity + UI Deception Patterns)          │
│                     │              (Optional: RNN/LSTM)         │
│                     ▼                                           │
│           [Classification - Softmax/Sigmoid]                    │
│        legitimate | phishing | clickbait                        │
│                     │                                           │
│                     ▼                                           │
│        [Decision & Alert — Browser Extension UI]                │
│     (Warn User | Highlight Elements | Block Page)               │
└─────────────────────────────────────────────────────────────────┘
```

**Technology Stack**

| Component | Technology |
|-----------|------------|
| Browser Extension | Chrome API, JavaScript |
| Backend Server | Python (Flask) |
| Image Preprocessing | OpenCV, NumPy |
| Feature Extraction | ResNet-50 / ViT (PyTorch or TensorFlow) |
| Multi-Component Reasoning | CNN + Attention Layers |
| Temporal Analysis (Optional) | RNN / LSTM |
| Classification | Softmax / Sigmoid |
| Annotation Tool | LabelImg or Roboflow |
| Dataset | Kaggle Phishing Sites + Self-collected |

---

## 3. Project Structure

```
clickbait-detector/
│
├── CLAUDE.md                        ← This file (read first!)
├── README.md                        ← Quick-start guide for running
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── data/                            ← All dataset files (DO NOT commit raw images to git)
│   ├── raw/
│   │   ├── kaggle_phishing/         ← Downloaded from Kaggle (unzipped)
│   │   └── self_collected/          ← Manually captured screenshots
│   ├── annotated/                   ← Labelled images + bounding box XML/JSON files
│   │   ├── images/
│   │   └── labels/
│   └── processed/                   ← Train/Val/Test splits (generated by scripts)
│       ├── train/
│       ├── val/
│       └── test/
│
├── scripts/                         ← Standalone utility scripts
│   ├── data_cleaning.py             ← Remove duplicates, low-quality images
│   ├── split_dataset.py             ← Create train/val/test splits
│   ├── augment_data.py              ← Data augmentation pipeline
│   └── visualize_annotations.py    ← Preview bounding box labels
│
├── src/                             ← Core source code
│   ├── preprocessing/
│   │   └── preprocess.py            ← Phase 3: Image preprocessing functions
│   ├── model/
│   │   ├── feature_extractor.py     ← Phase 4: CNN feature extraction
│   │   ├── mcp_pipeline.py          ← Phase 5: Multi-component reasoning
│   │   ├── classifier.py            ← Phase 6: Classification head
│   │   └── model.py                 ← Full end-to-end model definition
│   ├── training/
│   │   ├── train.py                 ← Training loop
│   │   ├── config.py                ← Hyperparameters and config
│   │   └── callbacks.py             ← Early stopping, checkpointing
│   ├── evaluation/
│   │   └── evaluate.py              ← Phase 8: Metrics and evaluation
│   └── server/
│       └── app.py                   ← Flask API server for the extension
│
├── extension/                       ← Chrome browser extension
│   ├── manifest.json
│   ├── background.js                ← Background service worker
│   ├── content.js                   ← Content script (screenshot + UI overlay)
│   └── popup/
│       ├── popup.html
│       └── popup.js
│
├── models/                          ← Saved model checkpoints
│   └── checkpoints/
│
├── notebooks/                       ← Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_tests.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
└── tests/                           ← Unit and integration tests
    ├── test_preprocessing.py
    ├── test_model.py
    ├── test_evaluation.py
    └── test_server.py
```

> **💡 Note for Team:** Each team member should work in their own git branch named `feature/<your-name>-<feature>`. Example: `feature/koshala-preprocessing`. Merge to `main` only after peer review.

---

## 4. Environment Setup

### 4.1 Prerequisites

Make sure the following are installed on your machine:

- Python 3.10+
- pip
- Git
- Google Chrome (for testing the extension)
- CUDA-compatible GPU (optional but strongly recommended for training)

### 4.2 Clone the Repository

```bash
git clone https://github.com/your-org/clickbait-detector.git
cd clickbait-detector
```

### 4.3 Create a Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate — Linux/Mac
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 4.4 Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` should contain:**

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0
Pillow>=9.4.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
Flask>=2.3.0
flask-cors>=3.0.10
tqdm>=4.65.0
albumentations>=1.3.0
timm>=0.9.0
jupyter>=1.0.0
pytest>=7.3.0
```

### 4.5 Verify Installation

```bash
python -c "import torch; import cv2; import timm; print('✅ All packages installed successfully')"
```

---

## 5. Phase 1 — Dataset Collection & Preparation

> **Owner:** All team members contribute | **Status:** Start here first

---

### Step 1.1 — Download the Kaggle Dataset

The primary dataset is the **Kaggle Phishing Sites Screenshot** dataset.

**How to download:**

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Go to: [https://www.kaggle.com/datasets/zackyzac/phishing-sites-screenshot/data](https://www.kaggle.com/datasets/zackyzac/phishing-sites-screenshot/data)
3. Click **Download** → You will get a `.zip` file
4. Unzip it into `data/raw/kaggle_phishing/`

**Using Kaggle CLI (faster):**

```bash
# Install Kaggle CLI
pip install kaggle

# Download your kaggle.json API token from kaggle.com/settings
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
kaggle datasets download -d zackyzac/phishing-sites-screenshot -p data/raw/kaggle_phishing/ --unzip
```

**How to use this dataset:**

| Folder/Label in Dataset | How We Treat It |
|--------------------------|-----------------|
| Phishing site screenshots | Label as `phishing` |
| Non-phishing screenshots | Label as `legitimate` (benign) |

> **📌 Current Plan:** We are starting with this Kaggle dataset only. Additional self-collected images for the `clickbait` class will be added in a later phase (see Step 1.2).

---

### Step 1.2 — Self-Collect Additional Screenshots (Later Phase)

> **⚠️ Note:** This step is planned for a later phase. Skip for now and focus on the Kaggle dataset.

When you are ready to collect additional data, manually browse real websites and take screenshots of pages containing:

- Fake **Download Now** / **Play Now** buttons
- Misleading pop-ups and overlays
- Deceptive banner advertisements
- Fake security notification prompts

**Screenshot Guidelines:**
- Use full-page screenshots (not just the viewport)
- Cover multiple domains: banking, e-commerce, social media, education
- Vary layouts, languages, and screen resolutions (1920×1080, 1366×768, 375×812)
- Target at least **200–300 screenshots** per category

**Recommended Tools for Screenshots:**
- [GoFullPage Chrome Extension](https://chrome.google.com/webstore/detail/gofullpage-full-page-scre/fdpohaocaechababfibfhkpohealbc) (full-page capture)
- Python + Selenium (automated capture):

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
driver.set_window_size(1920, 1080)
driver.get("https://example.com")
driver.save_screenshot("data/raw/self_collected/example_com.png")
driver.quit()
```

Save all self-collected screenshots to: `data/raw/self_collected/`

---

### Step 1.3 — Data Cleaning

Run the cleaning script to remove duplicates and low-quality images:

```bash
python scripts/data_cleaning.py \
  --input_dir data/raw/kaggle_phishing/ \
  --output_dir data/cleaned/ \
  --min_size 100 \
  --remove_duplicates
```

**`scripts/data_cleaning.py` — What it does:**

```python
import os
import cv2
import hashlib
from pathlib import Path

def is_low_quality(image_path, min_size=100):
    """Reject images smaller than min_size px in either dimension."""
    img = cv2.imread(str(image_path))
    if img is None:
        return True  # Unreadable file
    h, w = img.shape[:2]
    return h < min_size or w < min_size

def get_hash(image_path):
    """MD5 hash for duplicate detection."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def clean_dataset(input_dir, output_dir, min_size=100, remove_duplicates=True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    kept, removed = 0, 0

    for img_path in input_dir.rglob("*.png"):
        # Skip low-quality
        if is_low_quality(img_path, min_size):
            removed += 1
            continue
        # Skip duplicates
        if remove_duplicates:
            h = get_hash(img_path)
            if h in seen_hashes:
                removed += 1
                continue
            seen_hashes.add(h)
        # Copy to output
        dest = output_dir / img_path.name
        dest.write_bytes(img_path.read_bytes())
        kept += 1

    print(f"✅ Kept: {kept} | Removed: {removed}")
```

---

### Step 1.4 — Annotate / Label the Data

**Tool Recommendation:** Use [LabelImg](https://github.com/HumanSignal/labelImg) (free, offline) or [Roboflow](https://roboflow.com) (online, easier).

#### Option A — LabelImg (Offline)

```bash
pip install labelImg
labelImg
```

1. Open `data/cleaned/` as the image directory
2. Set save format to **YOLO** (preferred) or Pascal VOC XML
3. Draw bounding boxes around:
   - Fake buttons (`deceptive`)
   - Misleading advertisements (`deceptive`)
   - Legitimate navigation elements (`legitimate`)
4. Save label files to `data/annotated/labels/`

#### Option B — Roboflow (Online, Recommended for Team)

1. Create a free Roboflow workspace
2. Upload all cleaned images
3. Annotate collaboratively (multiple team members can annotate simultaneously)
4. Export annotations in **YOLO format**
5. Download to `data/annotated/`

**Label Classes:**
```
0 = legitimate
1 = phishing
2 = clickbait
```

---

### Step 1.5 — Create Train / Val / Test Splits

```bash
python scripts/split_dataset.py \
  --input_dir data/annotated/ \
  --output_dir data/processed/ \
  --train 0.70 \
  --val 0.15 \
  --test 0.15 \
  --seed 42
```

**`scripts/split_dataset.py`:**

```python
import os
import shutil
import random
from pathlib import Path

def split_dataset(input_dir, output_dir, train=0.70, val=0.15, test=0.15, seed=42):
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
    random.seed(seed)

    images = list(Path(input_dir, "images").glob("*.png"))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train)
    n_val   = int(n * val)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:]
    }

    for split_name, split_images in splits.items():
        for img_path in split_images:
            # Copy image
            dest_img = Path(output_dir, split_name, "images", img_path.name)
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dest_img)
            # Copy corresponding label
            label_path = Path(input_dir, "labels", img_path.stem + ".txt")
            if label_path.exists():
                dest_lbl = Path(output_dir, split_name, "labels", label_path.name)
                dest_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(label_path, dest_lbl)

        print(f"  {split_name}: {len(split_images)} images")

split_dataset("data/annotated", "data/processed")
```

#### ✅ Test — Phase 1

```bash
python -m pytest tests/test_dataset.py -v
```

**`tests/test_dataset.py`:**

```python
import pytest
from pathlib import Path

DATA_DIR = Path("data/processed")

def test_splits_exist():
    for split in ["train", "val", "test"]:
        assert (DATA_DIR / split / "images").exists(), f"Missing {split}/images/"
        assert (DATA_DIR / split / "labels").exists(), f"Missing {split}/labels/"

def test_no_empty_splits():
    for split in ["train", "val", "test"]:
        images = list((DATA_DIR / split / "images").glob("*.png"))
        assert len(images) > 0, f"{split} split has no images!"

def test_labels_match_images():
    for split in ["train", "val", "test"]:
        images = set(p.stem for p in (DATA_DIR / split / "images").glob("*.png"))
        labels = set(p.stem for p in (DATA_DIR / split / "labels").glob("*.txt"))
        assert images == labels, f"Mismatch in {split}: {images.symmetric_difference(labels)}"

def test_label_format():
    """YOLO format: each line must be 'class x_center y_center width height'"""
    for lbl_file in (DATA_DIR / "train" / "labels").glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                assert len(parts) == 5, f"Bad label in {lbl_file}: {line}"
                cls = int(parts[0])
                assert cls in [0, 1, 2], f"Unknown class {cls} in {lbl_file}"
```

---

## 6. Phase 2 — Browser Extension (Input Layer)

> **Owner:** Wanigasooriya | **Tech:** JavaScript, Chrome API

---

### Step 2.1 — Chrome Extension Structure

Create the following files inside `extension/`:

**`extension/manifest.json`:**

```json
{
  "manifest_version": 3,
  "name": "Clickbait Detector",
  "version": "1.0",
  "description": "Detects deceptive UI elements on websites using computer vision.",
  "permissions": ["activeTab", "scripting", "storage"],
  "host_permissions": ["<all_urls>"],
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"],
      "run_at": "document_idle"
    }
  ],
  "action": {
    "default_popup": "popup/popup.html",
    "default_title": "Clickbait Detector"
  }
}
```

---

### Step 2.2 — Screenshot Capture & Send to Backend

**`extension/content.js`:**

```javascript
// Runs automatically on every page load
(async () => {
  try {
    // Capture visible area as base64 image
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.runtime.sendMessage({ action: "captureTab" }, async (response) => {
      if (!response || !response.imageData) return;

      // Send screenshot to backend Flask server
      const result = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: response.imageData,  // base64 PNG
          url: window.location.href
        })
      });

      const data = await result.json();
      handleDetectionResult(data);
    });
  } catch (error) {
    console.error("[ClickbaitDetector] Error:", error);
  }
})();

function handleDetectionResult(data) {
  const { label, confidence, bounding_boxes } = data;

  if (label === "clickbait" || label === "phishing") {
    // Draw alert banner at top of page
    showAlertBanner(label, confidence);
    // Highlight detected elements with red boxes
    drawBoundingBoxes(bounding_boxes);
  }
}

function showAlertBanner(label, confidence) {
  const banner = document.createElement("div");
  banner.id = "clickbait-alert-banner";
  banner.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%; z-index: 999999;
    background: ${label === "phishing" ? "#FF4444" : "#FF8C00"};
    color: white; padding: 12px; font-size: 16px;
    font-family: Arial, sans-serif; text-align: center;
  `;
  banner.textContent = `⚠️ WARNING: This page may contain ${label} content (${(confidence * 100).toFixed(1)}% confidence)`;
  document.body.prepend(banner);
}

function drawBoundingBoxes(boxes) {
  boxes.forEach(box => {
    const overlay = document.createElement("div");
    overlay.style.cssText = `
      position: fixed;
      left: ${box.x}px; top: ${box.y}px;
      width: ${box.w}px; height: ${box.h}px;
      border: 3px solid red; z-index: 999998;
      pointer-events: none;
      background: rgba(255, 0, 0, 0.1);
    `;
    document.body.appendChild(overlay);
  });
}
```

**`extension/background.js`:**

```javascript
// Listens for screenshot requests from content.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "captureTab") {
    chrome.tabs.captureVisibleTab(null, { format: "png" }, (imageData) => {
      sendResponse({ imageData });
    });
    return true; // Required for async response
  }
});
```

#### ✅ Test — Extension

1. Go to `chrome://extensions/`
2. Enable **Developer Mode** (top right toggle)
3. Click **Load unpacked** → Select the `extension/` folder
4. Open any website
5. Open Chrome DevTools Console and verify there are no errors
6. Check that the backend server receives the POST request

---

## 7. Phase 3 — Image Preprocessing

> **Owner:** Dassanayaka | **Tech:** OpenCV, NumPy

---

### Step 3.1 — Resize & Normalize

**`src/preprocessing/preprocess.py`:**

```python
import cv2
import numpy as np
from pathlib import Path

TARGET_SIZE = (224, 224)  # Required by ResNet-50

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk. Returns BGR NumPy array."""
    img = cv2.imread(image_path)
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
```

---

### Step 3.2 — Noise Reduction

```python
def reduce_noise(img: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur to remove screenshot compression artifacts.
    Kernel (3,3) is sufficient — stronger blur loses UI detail.
    """
    return cv2.GaussianBlur(img, (3, 3), sigmaX=0)
```

---

### Step 3.3 — Region of Interest Cropping

```python
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
```

---

### Step 3.4 — Contrast Enhancement

```python
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


def preprocess_pipeline(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline: load → denoise → enhance → resize → normalize.
    Returns: normalized float32 array ready for model input.
    """
    img = load_image(image_path)
    img = reduce_noise(img)
    img = enhance_contrast(img)
    img = resize_image(img)
    img = normalize_image(img)
    return img
```

#### ✅ Test — Preprocessing

```bash
python -m pytest tests/test_preprocessing.py -v
```

**`tests/test_preprocessing.py`:**

```python
import pytest
import numpy as np
import cv2
from pathlib import Path
from src.preprocessing.preprocess import (
    load_image, resize_image, normalize_image,
    reduce_noise, enhance_contrast, preprocess_pipeline
)

# Use any test image from the dataset
SAMPLE_IMAGE = str(next(Path("data/processed/test/images").glob("*.png")))


def test_load_image():
    img = load_image(SAMPLE_IMAGE)
    assert img is not None
    assert len(img.shape) == 3  # H, W, C
    assert img.shape[2] == 3   # 3 channels (BGR)

def test_resize_image():
    img = load_image(SAMPLE_IMAGE)
    resized = resize_image(img, (224, 224))
    assert resized.shape == (224, 224, 3)

def test_normalize_range():
    img = load_image(SAMPLE_IMAGE)
    img = resize_image(img)
    normalized = normalize_image(img)
    assert normalized.dtype == np.float32
    # After ImageNet normalization, values are typically in [-3, 3]
    assert normalized.min() >= -4.0
    assert normalized.max() <= 4.0

def test_noise_reduction_preserves_shape():
    img = load_image(SAMPLE_IMAGE)
    denoised = reduce_noise(img)
    assert denoised.shape == img.shape

def test_full_pipeline_output_shape():
    output = preprocess_pipeline(SAMPLE_IMAGE)
    assert output.shape == (224, 224, 3)
    assert output.dtype == np.float32

def test_load_missing_image():
    with pytest.raises(FileNotFoundError):
        load_image("data/nonexistent_image.png")
```

---

## 8. Phase 4 — Visual Feature Extraction

> **Owner:** Koshala | **Tech:** PyTorch, timm (ResNet-50 / ViT)

---

### Step 4.1 — Load Pre-trained Model

**`src/model/feature_extractor.py`:**

```python
import torch
import torch.nn as nn
import timm
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    """
    Wraps a pre-trained ResNet-50 or ViT and removes the final
    classification head to output a feature vector.

    Output feature sizes:
      - resnet50:     2048-dimensional vector
      - vit_base_patch16_224: 768-dimensional vector
    """
    def __init__(self, model_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        # Load model without classification head (num_classes=0)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # Remove classifier head
            global_pool="avg"    # Global average pooling
        )
        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 3, 224, 224) — normalized images
        Returns:
            features: Tensor of shape (batch, feature_dim)
        """
        return self.backbone(x)


def extract_features_from_image(image_array: np.ndarray,
                                 extractor: FeatureExtractor) -> np.ndarray:
    """
    Helper to extract features from a single preprocessed image.

    Args:
        image_array: float32 NumPy array (224, 224, 3) — normalized
        extractor: FeatureExtractor model

    Returns:
        Feature vector as NumPy array
    """
    # Convert HWC → CHW and add batch dimension
    tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    extractor.eval()
    with torch.no_grad():
        features = extractor(tensor)
    return features.cpu().numpy().squeeze()
```

---

### Step 4.2 — Data Augmentation

**`scripts/augment_data.py`:**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    """
    Augmentation pipeline for training only.
    Simulates real-world conditions: low bandwidth, different devices, lighting.
    """
    return A.Compose([
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.Resize(224, 224),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def get_val_transforms():
    """Validation/Test: only resize and convert. No augmentation."""
    return A.Compose([
        A.Resize(224, 224),
        ToTensorV2()
    ])
```

#### ✅ Test — Feature Extraction

```bash
python -m pytest tests/test_model.py::test_feature_extraction -v
```

**`tests/test_model.py` (partial):**

```python
import torch
import numpy as np
from src.model.feature_extractor import FeatureExtractor, extract_features_from_image

def test_feature_extractor_output_shape():
    extractor = FeatureExtractor(model_name="resnet50", pretrained=False)
    dummy_input = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        output = extractor(dummy_input)
    assert output.shape == (1, 2048), f"Expected (1, 2048), got {output.shape}"

def test_feature_extraction_from_numpy():
    extractor = FeatureExtractor(model_name="resnet50", pretrained=False)
    dummy_image = np.zeros((224, 224, 3), dtype=np.float32)
    features = extract_features_from_image(dummy_image, extractor)
    assert features.shape == (2048,)
    assert features.dtype == np.float32
```

---

## 9. Phase 5 — Multi-Component Reasoning Pipeline

> **Owner:** Kumara | **Tech:** PyTorch (CNN + Attention)

---

### Step 5.1 — Attention-Based Fusion

**`src/model/mcp_pipeline.py`:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    """
    Self-attention to weight different feature dimensions by importance.
    Helps the model focus on features most associated with deceptive elements.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights


class MCPPipeline(nn.Module):
    """
    Multi-Component Reasoning Pipeline.
    Takes raw feature vectors from the extractor and applies:
      1. Self-attention weighting
      2. Two-layer MLP for non-linear reasoning
      3. Dropout regularization
    """
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.attention = SelfAttentionLayer(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_dim = hidden_dim // 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attended = self.attention(features)
        return self.mlp(attended)
```

### Step 5.2 — Optional: Temporal Analysis (RNN/LSTM)

> This is an optional extension. Implement only after the main pipeline is working.

```python
class TemporalAnalyzer(nn.Module):
    """
    Analyses a sequence of screenshots from the same website over time.
    Detects behavioral patterns: frequent pop-ups, aggressive UI changes.

    Input:  Sequence of feature vectors (batch, time_steps, feature_dim)
    Output: Context-aware feature vector (batch, hidden_dim)
    """
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, _) = self.lstm(x)
        return hidden[-1]  # Last layer's hidden state
```

---

## 10. Phase 6 — Classification Layer

> **Owner:** Koshala + Kumara | **Tech:** PyTorch

---

**`src/model/model.py` — Full End-to-End Model:**

```python
import torch
import torch.nn as nn
from src.model.feature_extractor import FeatureExtractor
from src.model.mcp_pipeline import MCPPipeline

NUM_CLASSES = 3  # legitimate=0, phishing=1, clickbait=2


class ClickbaitDetector(nn.Module):
    """
    Full end-to-end model:
    Screenshot → Feature Extraction → MCP Reasoning → Classification
    """
    def __init__(self, backbone: str = "resnet50"):
        super().__init__()
        self.extractor = FeatureExtractor(model_name=backbone, pretrained=True)
        self.mcp = MCPPipeline(
            input_dim=self.extractor.feature_dim,
            hidden_dim=512
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.mcp.output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES)
            # Note: No Softmax here — use CrossEntropyLoss which applies it internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)    # (batch, 2048)
        reasoned = self.mcp(features)   # (batch, 256)
        logits   = self.classifier(reasoned)  # (batch, 3)
        return logits

    def predict(self, x: torch.Tensor) -> tuple:
        """Returns (predicted_class, probabilities)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            predicted = torch.argmax(probs, dim=-1)
        return predicted, probs
```

---

**`src/training/train.py`:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.model import ClickbaitDetector
from src.training.config import CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["legitimate", "phishing", "clickbait"]


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train():
    model = ClickbaitDetector(backbone=CONFIG["backbone"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc     = validate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/checkpoints/best_model.pth")
            print(f"  ✅ Saved new best model (Val Acc: {best_val_acc:.4f})")
```

**`src/training/config.py`:**

```python
CONFIG = {
    "backbone":   "resnet50",   # Options: "resnet50", "vit_base_patch16_224"
    "epochs":     30,
    "batch_size": 32,
    "lr":         1e-4,
    "num_classes": 3,
    "img_size":   224,
    "seed":       42
}
```

**Run training:**

```bash
python -m src.training.train
```

---

## 11. Phase 7 — Decision & Alert System

> **Owner:** Wanigasooriya | **Tech:** Flask, JavaScript

---

### Step 7.1 — Flask Backend Server

**`src/server/app.py`:**

```python
import base64
import io
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.model.model import ClickbaitDetector
from src.preprocessing.preprocess import preprocess_pipeline

app = Flask(__name__)
CORS(app)  # Allow requests from Chrome extension

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["legitimate", "phishing", "clickbait"]

# Load trained model
model = ClickbaitDetector(backbone="resnet50")
model.load_state_dict(torch.load("models/checkpoints/best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Receives a base64-encoded webpage screenshot from the extension.
    Returns: { label, confidence, bounding_boxes }
    """
    data = request.get_json()
    image_b64 = data["image"]

    # Decode base64 → PIL Image → NumPy array
    image_bytes = base64.b64decode(image_b64.split(",")[1])  # Remove "data:image/png;base64,"
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(pil_image)

    # Preprocess
    preprocessed = preprocess_pipeline_from_array(img_array)
    tensor = torch.from_numpy(preprocessed.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    label = CLASS_NAMES[predicted_class]

    return jsonify({
        "label":         label,
        "confidence":    confidence,
        "probabilities": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
        "bounding_boxes": []  # TODO: Integrate object detection for box output
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
```

**Start the server:**

```bash
python -m src.server.app
```

---

## 12. Phase 8 — Model Evaluation & Testing

> **Owner:** All team members | **When:** After training is complete

---

### Step 8.1 — Run Evaluation

**`src/evaluation/evaluate.py`:**

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from src.model.model import ClickbaitDetector

CLASS_NAMES = ["legitimate", "phishing", "clickbait"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Classification Report ---
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    # --- Summary Metrics ---
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall    = recall_score(all_labels, all_preds, average="weighted")
    f1        = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n✅ Weighted Precision : {precision:.4f}")
    print(f"✅ Weighted Recall    : {recall:.4f}")
    print(f"✅ Weighted F1-Score  : {f1:.4f}")

    return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}
```

**Run evaluation:**

```bash
python -c "from src.evaluation.evaluate import evaluate_model; evaluate_model(...)"
```

---

### Step 8.2 — IoU Evaluation (Bounding Box Accuracy)

```python
def compute_iou(box_pred: list, box_true: list) -> float:
    """
    Compute Intersection over Union for two bounding boxes.
    Boxes format: [x_min, y_min, x_max, y_max]

    IoU > 0.5 is considered a successful detection (standard COCO threshold).
    """
    x1 = max(box_pred[0], box_true[0])
    y1 = max(box_pred[1], box_true[1])
    x2 = min(box_pred[2], box_true[2])
    y2 = min(box_pred[3], box_true[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_true = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])
    union = area_pred + area_true - intersection

    return intersection / union if union > 0 else 0.0
```

---

#### ✅ Test — Full Model & Evaluation

```bash
python -m pytest tests/ -v
```

**`tests/test_evaluation.py`:**

```python
import numpy as np
import pytest
from src.evaluation.evaluate import compute_iou

def test_iou_perfect_overlap():
    box = [0, 0, 100, 100]
    assert compute_iou(box, box) == pytest.approx(1.0)

def test_iou_no_overlap():
    box_a = [0, 0, 50, 50]
    box_b = [100, 100, 150, 150]
    assert compute_iou(box_a, box_b) == pytest.approx(0.0)

def test_iou_threshold():
    """IoU > 0.5 should be a successful detection."""
    box_pred = [0, 0, 100, 100]
    box_true = [10, 10, 110, 110]
    iou = compute_iou(box_pred, box_true)
    assert iou > 0.5, f"Expected IoU > 0.5, got {iou:.4f}"

def test_iou_partial_overlap():
    box_pred = [0, 0, 100, 100]
    box_true = [50, 50, 150, 150]
    iou = compute_iou(box_pred, box_true)
    assert 0.0 < iou < 1.0
```

---

## 13. Running the Full Pipeline

After all phases are complete, run the system end-to-end:

```bash
# 1. Prepare dataset
python scripts/data_cleaning.py
python scripts/split_dataset.py

# 2. Train the model
python -m src.training.train

# 3. Evaluate on test set
python -m src.evaluation.evaluate

# 4. Start Flask backend server
python -m src.server.app

# 5. Load the Chrome extension
# → chrome://extensions → Load Unpacked → select extension/

# 6. Run all tests
python -m pytest tests/ -v --tb=short
```

**Expected terminal output after training:**
```
Epoch 1/30  | Train Loss: 0.8921, Acc: 0.6234 | Val Loss: 0.7812, Acc: 0.6891
Epoch 2/30  | Train Loss: 0.7204, Acc: 0.7012 | Val Loss: 0.6543, Acc: 0.7245
...
Epoch 28/30 | Train Loss: 0.2341, Acc: 0.9102 | Val Loss: 0.3012, Acc: 0.8734
✅ Saved new best model (Val Acc: 0.8734)
```

---

## 14. Expert Suggestions & Notes

> These are recommendations from an image processing and software engineering perspective. Please read carefully before starting.

---

### 🔵 On the Dataset

- **Start with the Kaggle dataset only.** Do not try to collect everything at once. Build a working pipeline first, then expand the dataset.
- **Class imbalance is a real problem.** The Kaggle dataset has many more legitimate samples than phishing ones. Use `WeightedRandomSampler` in PyTorch to handle this, or apply class weights in your loss function.
- **Annotation is the most time-consuming step.** Divide annotation work equally among team members. Use Roboflow for collaborative online annotation — it saves significant time vs. LabelImg.
- **Minimum 200 images per class** is needed for meaningful training. If the Kaggle dataset is not enough for clickbait class, collect manually.

### 🔵 On the Model

- **Start with ResNet-50, not ViT.** ResNet-50 is faster to train, easier to debug, and performs very well on image classification tasks. Switch to ViT only if accuracy is still unsatisfactory.
- **Freeze backbone layers initially.** For the first few training runs, freeze the ResNet backbone and only train the MCP + classifier head. This is called fine-tuning and prevents destroying ImageNet-learned features.
  ```python
  for param in model.extractor.parameters():
      param.requires_grad = False  # Freeze backbone
  ```
  After a few epochs, unfreeze and train end-to-end with a lower learning rate.
- **Use Transfer Learning.** Do not train from scratch. Always initialize from ImageNet weights (pretrained=True). Your dataset is too small for training from scratch.

### 🔵 On Software Engineering

- **Use Git branches for every feature.** Never work directly on `main`. Use: `git checkout -b feature/your-name-feature`
- **Write tests as you go.** Don't leave testing to the end. Write a test for every function you write.
- **Log everything during training.** Use `print()` statements minimally; prefer logging to a file. Consider using TensorBoard:
  ```bash
  pip install tensorboard
  tensorboard --logdir=runs/
  ```
- **Save model checkpoints after every epoch** that improves validation accuracy. You may lose hours of training if the machine crashes.
- **Config file is law.** All hyperparameters live in `config.py`. Never hardcode learning rate or batch size inside training scripts.

### 🔵 On Evaluation

- **Do not report only accuracy.** With class imbalance, a model predicting "legitimate" for everything gets ~80% accuracy but is useless. Always report Precision, Recall, and F1-Score per class.
- **IoU threshold of 0.5 is the standard** for object detection benchmarks (COCO). Use this as your pass criterion for bounding box detection.
- **Always evaluate on the held-out test set** — never on validation. The validation set is only for hyperparameter tuning.

### 🔵 On the Extension

- **The Flask server must be running** before the extension can work. Make sure to document this clearly for the demo.
- **CORS must be enabled** on Flask (already included in `app.py`) or the browser will block extension requests.
- **Test with both legitimate and deceptive pages** during the demo to show contrast.

### 🔵 Performance Targets to Aim For

| Metric | Minimum Target | Good Target |
|--------|---------------|-------------|
| Classification Accuracy | 75% | 85%+ |
| Precision (deceptive class) | 0.70 | 0.85+ |
| Recall (deceptive class) | 0.70 | 0.80+ |
| F1-Score | 0.72 | 0.82+ |
| IoU (bounding boxes) | > 0.50 | > 0.65 |

---

> **📌 Final Note:** This project is entirely achievable within the semester. Focus on getting a clean working pipeline with the Kaggle dataset before adding complexity. A simple model that works reliably is far more valuable than a complex model that fails unpredictably.
>
> **Good luck, Group 10! 🚀**
