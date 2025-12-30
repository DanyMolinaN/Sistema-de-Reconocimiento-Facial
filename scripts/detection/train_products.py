from ultralytics import YOLO
from pathlib import Path


# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "yolo" / "data.yaml"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "yolo" / "products"

MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)


# TRAIN

def train():
    model = YOLO("yolov8n.pt")  # modelo base liviano

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        name="products",
        project=str(MODEL_OUTPUT),
        device=0  # usa GPU si está disponible
    )

if __name__ == "__main__":
    train()

#El mejor modelo quedará en:
#models/yolo/products/products/weights/best.pt
