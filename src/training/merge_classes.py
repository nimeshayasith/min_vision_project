import os
from pathlib import Path

def merge_classes():
    # Automatically find the roboflow data directory relative to this script
    base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "roboflow"
    splits = ["train", "valid", "test"]

    if not base_dir.exists():
        print(f"❌ Error: Could not find dataset directory at {base_dir}")
        return

    count = 0
    for split in splits:
        labels_dir = base_dir / split / "labels"
        if not labels_dir.exists():
            continue
            
        for txt_file in labels_dir.glob("*.txt"):
            with open(txt_file, "r") as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                # Force the class ID (first column) to be '0' (fake_button)
                parts[0] = '0'
                new_lines.append(" ".join(parts) + "\n")
                
            with open(txt_file, "w") as f:
                f.writelines(new_lines)
                
            count += 1

    print(f"✅ Successfully merged {count} annotation files to class ID 0 ('fake_button').")

if __name__ == "__main__":
    merge_classes()


# python -m src.training.merge_classes
