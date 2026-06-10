import os
from pathlib import Path

BASE = Path('data/roboflow')
SPLITS = ['train','valid','test']

def fix_file(path: Path) -> bool:
    changed = False
    text = path.read_text().strip().splitlines()
    new_lines = []
    for line in text:
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        if len(parts) == 5:
            new_lines.append(line.strip())
            continue
        # Attempt polygon -> bbox conversion
        try:
            cls = parts[0]
            coords = list(map(float, parts[1:]))
            if len(coords) < 4 or len(coords) % 2 != 0:
                print(f"Skipping {path}: unsupported coords length {len(coords)}")
                new_lines.append(line.strip())
                continue
            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_c = (x_min + x_max) / 2.0
            y_c = (y_min + y_max) / 2.0
            w = x_max - x_min
            h = y_max - y_min
            # clamp
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            new_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            changed = True
        except Exception as e:
            print(f"Error processing {path}: {e}")
            new_lines.append(line.strip())
    if changed:
        bak = path.with_suffix(path.suffix + '.bak')
        if not bak.exists():
            path.replace(bak)
        # write new file
        bak.write_text('\n'.join(new_lines) + '\n')
        # move bak back to original path
        bak.replace(path)
    return changed

if __name__ == '__main__':
    total_changed = 0
    for split in SPLITS:
        lbl_dir = BASE / split / 'labels'
        for fn in sorted(os.listdir(lbl_dir)):
            p = lbl_dir / fn
            if p.suffix != '.txt':
                continue
            if fix_file(p):
                total_changed += 1
                print(f"Fixed: {p}")
    print(f"Done. Files changed: {total_changed}")
