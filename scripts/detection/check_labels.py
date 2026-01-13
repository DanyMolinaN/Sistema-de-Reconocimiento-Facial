from pathlib import Path
from PIL import Image
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = PROJECT_ROOT / "yolo" / "labels"
IMAGES_DIR = PROJECT_ROOT / "yolo" / "images"

REPORT = []

if not LABELS_DIR.exists():
    print(f"Labels folder not found: {LABELS_DIR}")
    sys.exit(1)

for split in ("train", "val"):
    labels_path = LABELS_DIR / split
    images_path = IMAGES_DIR / split
    if not labels_path.exists():
        continue
    for txt in labels_path.glob("*.txt"):
        stem = txt.stem
        # try common image extensions
        img_file = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            p = images_path / (stem + ext)
            if p.exists():
                img_file = p
                break
        if img_file is None:
            REPORT.append((split, txt.name, "MISSING_IMAGE", None))
            continue
        try:
            with Image.open(img_file) as im:
                w, h = im.size
        except Exception as e:
            REPORT.append((split, txt.name, "BAD_IMAGE", str(e)))
            continue
        # read labels
        with open(txt, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
            REPORT.append((split, txt.name, "NO_LABELS", None))
            continue
        for i, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) < 5:
                REPORT.append((split, txt.name, f"BAD_FORMAT_line_{i}", line))
                continue
            try:
                cls = parts[0]
                x_c = float(parts[1])
                y_c = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except Exception:
                REPORT.append((split, txt.name, f"BAD_VALUES_line_{i}", line))
                continue
            area = bw * bh
            if area > 0.5 or bw > 0.9 or bh > 0.9:
                REPORT.append((split, txt.name, "LARGE_BOX", {"line": i, "area": area, "w_norm": bw, "h_norm": bh}))

# summary
if not REPORT:
    print("Labels check OK: no obvious issues found.")
else:
    print("Labels check report:")
    for item in REPORT:
        print(item)
    # write to file
    out = PROJECT_ROOT / "label_check_report.txt"
    with open(out, "w", encoding="utf-8") as f:
        for item in REPORT:
            f.write(str(item) + "\n")
    print(f"Wrote report to {out}")
