"""
Visualize YOLO labels overlaid on images for random samples.
Saves outputs to debug_preprocessing/label_checks/.

Usage: python scripts/visualize_label_samples.py --split train --n 10
"""
import argparse
import random
from pathlib import Path
import cv2

BASE = Path('data/roboflow')
OUT_DIR = Path('debug_preprocessing/label_checks')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_boxes(image_path: Path, label_path: Path, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print('Could not read', image_path)
        return False
    h, w = img.shape[:2]
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            try:
                x_c, y_c, bw, bh = map(float, parts[1:5])
            except:
                continue
            x1 = int((x_c - bw / 2.0) * w)
            y1 = int((y_c - bh / 2.0) * h)
            x2 = int((x_c + bw / 2.0) * w)
            y2 = int((y_c + bh / 2.0) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, str(cls), (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imwrite(str(out_path), img)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train','valid','test'])
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()

    img_dir = BASE / args.split / 'images'
    lbl_dir = BASE / args.split / 'labels'
    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in ('.png','.jpg','.jpeg')]
    if not imgs:
        print('No images found in', img_dir)
        raise SystemExit(1)
    random.seed(0)
    selected = random.sample(imgs, min(args.n, len(imgs)))
    saved = []
    for p in selected:
        lbl = lbl_dir / (p.stem + '.txt')
        if not lbl.exists():
            # try matching full filename: some labels include suffixes with .png.rf...
            possible = [lbl_dir / f for f in lbl_dir.iterdir() if f.name.startswith(p.stem)]
            if possible:
                lbl = possible[0]
            else:
                print('No label for', p.name)
                continue
        outp = OUT_DIR / (p.stem + '_check.png')
        ok = draw_boxes(p, lbl, outp)
        if ok:
            saved.append(outp)
            print('Saved', outp)
    print('Done. Saved', len(saved), 'images to', OUT_DIR)
