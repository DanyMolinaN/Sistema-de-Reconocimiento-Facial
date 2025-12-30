from ultralytics import YOLO
import cv2
from pathlib import Path


# PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "products" / "products" / "weights" / "best.pt"
IMAGE_PATH = PROJECT_ROOT / "test_images" / "test.jpg"

# LOAD MODEL

model = YOLO(str(MODEL_PATH))


# DETECT

results = model(str(IMAGE_PATH), conf=0.4)

img = cv2.imread(str(IMAGE_PATH))

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imshow("Product Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
