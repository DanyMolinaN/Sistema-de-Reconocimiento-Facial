from ultralytics import YOLO
import cv2
from pathlib import Path


# PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "products" / "products" / "weights" / "best.pt"


# LOAD MODEL

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("[INFO] Detección de productos iniciada (q para salir)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Product Detection Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
