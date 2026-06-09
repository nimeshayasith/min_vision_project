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


     yolo dataset- https://app.roboflow.com/join/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3Jrc3BhY2VJZCI6Im5BbTB4Vno5Qm9XWkpWc0t0eWwzZmg3SmVSdDEiLCJyb2xlIjoib3duZXIiLCJpbnZpdGVyIjoibmltZXNoYXlhc2l0aEBnbWFpbC5jb20iLCJpYXQiOjE3ODA1NjE5NTh9.NJ1vhemCN0rZWt6RHJvbJvHM-WqmnTViryljDIbsyQc


     outputs


Ultralytics 8.4.61 🚀 Python-3.10.13 torch-2.12.0+cu130 CUDA:0 (NVIDIA GeForce RTX 5090, 32102MiB)
Model summary (fused): 113 layers, 68,124,531 parameters, 0 gradients, 257.4 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 504.1±387.4 MB/s, size: 300.8 KB)
val: Scanning /home/sdvn_cluster_timing/Downloads/min_vision_project/data/roboflow/test/labels... 29 images, 2 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 29/29 993.7it/s 0.0s
val: New cache created: /home/sdvn_cluster_timing/Downloads/min_vision_project/data/roboflow/test/labels.cache
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4it/s 1.4s
                   all         29         39      0.669      0.846      0.768      0.449
Speed: 2.2ms preprocess, 39.4ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /home/sdvn_cluster_timing/.pyenv/runs/detect/val-10

============================================================
📈 YOLOv8 MODEL EVALUATION RESULTS 📈
============================================================
✅ Precision (Accuracy of predictions): 0.6690  (66.90%)
✅ Recall (Ability to find all objects): 0.8462  (84.62%)
✅ mAP@0.50 (Mean Average Precision):    0.7679  (76.79%)
✅ mAP@0.50:0.95 (Strict mAP):           0.4488  (44.88%)
🎯 F1-Score (Overall Balance):           0.7472  (74.72%)
============================================================
Ultralytics 8.4.61 🚀 Python-3.10.13 torch-2.12.0+cu130 CUDA:0 (NVIDIA GeForce RTX 5090, 32102MiB)
Model summary (fused): 113 layers, 68,124,531 parameters, 0 gradients, 257.4 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 504.1±387.4 MB/s, size: 300.8 KB)
val: Scanning /home/sdvn_cluster_timing/Downloads/min_vision_project/data/roboflow/test/labels... 29 images, 2 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 29/29 993.7it/s 0.0s
val: New cache created: /home/sdvn_cluster_timing/Downloads/min_vision_project/data/roboflow/test/labels.cache
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 1.4it/s 1.4s
                   all         29         39      0.669      0.846      0.768      0.449
Speed: 2.2ms preprocess, 39.4ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /home/sdvn_cluster_timing/.pyenv/runs/detect/val-10

============================================================
📈 YOLOv8 MODEL EVALUATION RESULTS 📈
============================================================
✅ Precision (Accuracy of predictions): 0.6690  (66.90%)
✅ Recall (Ability to find all objects): 0.8462  (84.62%)
✅ mAP@0.50 (Mean Average Precision):    0.7679  (76.79%)
✅ mAP@0.50:0.95 (Strict mAP):           0.4488  (44.88%)
🎯 F1-Score (Overall Balance):           0.7472  (74.72%)
============================================================