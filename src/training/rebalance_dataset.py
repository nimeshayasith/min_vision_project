import os
import shutil
import random
from pathlib import Path

def rebalance_dataset():
    # 1. Paths
    base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "roboflow"
    splits = ["train", "valid", "test"]
    
    # 2. Gather all files
    all_images = []
    for split in splits:
        img_dir = base_dir / split / "images"
        if img_dir.exists():
            for img_file in img_dir.glob("*.*"):
                # We need the corresponding label file
                label_file = base_dir / split / "labels" / (img_file.stem + ".txt")
                if label_file.exists():
                    all_images.append((img_file, label_file))

    total = len(all_images)
    print(f"Found {total} complete image-label pairs.")
    if total == 0:
        return

    # 3. Shuffle
    random.seed(42)  # For reproducibility
    random.shuffle(all_images)

    # 4. Calculate splits (80/10/10)
    train_count = int(total * 0.8)
    valid_count = int(total * 0.1)
    
    train_set = all_images[:train_count]
    valid_set = all_images[train_count:train_count+valid_count]
    test_set = all_images[train_count+valid_count:]

    # 5. Create temporary staging area to avoid conflicts while moving
    staging_dir = base_dir / "temp_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(exist_ok=True)
    
    print("Moving files... this may take a moment.")
    for split_name, dataset in [("train", train_set), ("valid", valid_set), ("test", test_set)]:
        split_img_dir = staging_dir / split_name / "images"
        split_lbl_dir = staging_dir / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in dataset:
            shutil.copy(img_path, split_img_dir / img_path.name)
            shutil.copy(lbl_path, split_lbl_dir / lbl_path.name)
            
    # 6. Replace old directories
    for split in splits:
        old_split_dir = base_dir / split
        if old_split_dir.exists():
            shutil.rmtree(old_split_dir)
        # Move from staging
        shutil.move(str(staging_dir / split), str(base_dir / split))
        
    # Clean up staging
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    print(f"✅ Rebalanced! Train: {len(train_set)} | Valid: {len(valid_set)} | Test: {len(test_set)}")

if __name__ == "__main__":
    rebalance_dataset()
