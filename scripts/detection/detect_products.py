from ultralytics import YOLO
import cv2
from pathlib import Path
import sys


# PATHS (do not change structure)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_PROD_DIR = PROJECT_ROOT / "models" / "yolo" / "products"


def find_latest_weight(models_dir: Path):
    candidates = []
    if not models_dir.exists():
        return None
    for run in models_dir.iterdir():
        weights_dir = run / "weights"
        if not weights_dir.exists():
            continue
        for name in ("best.pt", "last.pt"):
            p = weights_dir / name
            if p.exists():
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_image_path():
    test_img = PROJECT_ROOT / "test_images" / "test.jpg"
    if test_img.exists():
        return test_img
    for folder in (PROJECT_ROOT / "yolo" / "images" / "val", PROJECT_ROOT / "yolo" / "images" / "train"):
        if folder.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                files = list(folder.glob(ext))
                if files:
                    return files[0]
    return None


MODEL_PATH = find_latest_weight(MODELS_PROD_DIR)
if MODEL_PATH is None:
    print(f"No se encontró un peso entrenado en {MODELS_PROD_DIR}. Ejecuta `train_products.py` primero.")
    sys.exit(1)

IMAGE_PATH = find_image_path()
if IMAGE_PATH is None:
    print("No se encontró imagen de prueba en test_images/ ni en yolo/images/[val|train]. Añade una imagen o genera etiquetas de prueba.")
    sys.exit(1)

print(f"Usando modelo: {MODEL_PATH}")
print(f"Usando imagen: {IMAGE_PATH}")

# Load model
model = YOLO(str(MODEL_PATH))

# Read image
img = cv2.imread(str(IMAGE_PATH))
if img is None:
    print(f"No se pudo leer la imagen {IMAGE_PATH}")
    sys.exit(1)

# Ultralytics expects RGB input
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = model(img_rgb, imgsz=640, conf=0.4, iou=0.5)

if len(results) > 0:
    annotated = results[0].plot()
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imshow("Product Detection", annotated_bgr)
else:
    cv2.imshow("Product Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
