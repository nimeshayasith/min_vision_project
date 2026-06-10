import os
from pathlib import Path
from ultralytics import YOLO

def evaluate_model():
    # 1. Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "yolo_clickbait.pt"
    DATA_YAML = PROJECT_ROOT / "data" / "roboflow" / "data.yaml"

    if not MODEL_PATH.exists():
        print(f"❌ Error: Trained model not found at {MODEL_PATH}")
        print("Please train the model first using: python -m src.training.train_yolo")
        return

    if not DATA_YAML.exists():
        print(f"❌ Error: Dataset config not found at {DATA_YAML}")
        return

    print("🚀 Loading model for evaluation...")
    model = YOLO(str(MODEL_PATH))

    print("\n📊 Running final evaluation on the unseen TEST dataset...\n")
    # 2. Run validation
    # imgsz=1280 must match your new high-resolution HPC training!
    # conf=0.40 ignores weak guesses, which massively boosts Precision
    metrics = model.val(data=str(DATA_YAML), split='test', imgsz=1280, batch=16, conf=0.40)

    # 3. Extract metrics
    # YOLOv8 stores bounding box metrics in metrics.box
    precision = metrics.box.mp    # Mean Precision
    recall = metrics.box.mr       # Mean Recall
    map50 = metrics.box.map50     # mAP at IoU=0.50
    map50_95 = metrics.box.map    # mAP at IoU=0.50:0.95

    # 4. Display results beautifully in terminal
    print("\n" + "="*60)
    print("📈 YOLOv8 MODEL EVALUATION RESULTS 📈")
    print("="*60)
    print(f"✅ Precision (Accuracy of predictions): {precision:.4f}  ({precision*100:.2f}%)")
    print(f"✅ Recall (Ability to find all objects): {recall:.4f}  ({recall*100:.2f}%)")
    print(f"✅ mAP@0.50 (Mean Average Precision):    {map50:.4f}  ({map50*100:.2f}%)")
    print(f"✅ mAP@0.50:0.95 (Strict mAP):           {map50_95:.4f}  ({map50_95*100:.2f}%)")
    
    # Calculate F1 Score mathematically (Harmonic mean of Precision and Recall)
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"🎯 F1-Score (Overall Balance):           {f1_score:.4f}  ({f1_score*100:.2f}%)")
        
    print("="*60)

if __name__ == "__main__":
    evaluate_model()
