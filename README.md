# min_vision_project

## Train & Evaluate Model

**Train Model:**
```bash
python -m src.training.train_yolo
```

**Get Evaluation Metrics:**
```bash
python -m src.training.evaluate_yolo
```

## Quick: Load the Chrome extension & run the demo

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Start the Flask backend (loads the trained model):

```bash
python -m src.server.app
```

3. In Chrome, open Extensions (chrome://extensions), enable **Developer mode**, click **Load unpacked**, and select the `extension/` folder from the project root.

4. Open any web page, click the extension icon, and use **Re-scan This Page** in the popup. The extension captures a screenshot and sends it to the backend.

5. Results are shown in the popup (label, confidence, probabilities). If the page is detected as `phishing` or `clickbait`, an in-page alert banner will appear.

Notes:
- Ensure the Flask server is running on `http://localhost:5000` (default). If you change the host/port update `extension/popup/popup.js` and `extension/content.js` `BACKEND_URL` accordingly.
- For local testing without a trained checkpoint, the server will start with an untrained model (see `models/checkpoints/`).

## How it Works (Hybrid Architecture)

This project uses a two-stage pipeline to detect deceptive UI:

1. **Spatial Deep Learning (YOLOv8)**
   - Scans the webpage screenshot to identify the structural boundaries of all UI elements (Buttons, Banners).
   
2. **Classical Computer Vision Validation (OpenCV)**
   - Filters the detected elements to separate *Real* elements from *Fake/Clickbait* elements using pure mathematical Image Processing algorithms:
     - **Expanded ROI Analysis (Canny Edge):** Measures the edge density surrounding the button. High density indicates a messy ad space, while low density indicates clean UI padding.
     - **Page-Level Color Outliers (K-Means Clustering):** Extracts the 3 dominant colors of the website. If a button's color is a massive Euclidean outlier in the LAB color space, it is flagged as visually disruptive.
     - **Internal Complexity (Harris Corner Detection):** Counts the corners inside the button. Fake buttons pack high complexity (icons, badges), resulting in dense corner maps compared to flat real buttons.
     - **Contrast & Saturation (CLAHE / HSV):** Measures local contrast and vividness to detect attention-grabbing tactics.