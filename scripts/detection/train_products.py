from ultralytics import YOLO
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# data.yaml is under yolo/labels in this repo
DATA_YAML = PROJECT_ROOT / "yolo" / "labels" / "data.yaml"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "yolo" / "products"

MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)


def train():
    model = YOLO("yolov8s.pt")

    # seleccionar dispositivo automáticamente
    use_cuda = torch.cuda.is_available()
    device_arg = 0 if use_cuda else "cpu"

    # ajustamos algunos parámetros según si hay GPU
    batch_arg = 16 if use_cuda else 8
    workers_arg = 8 if use_cuda else 0
    amp_arg = True if use_cuda else False

    print(f"Starting training. device={device_arg}, batch={batch_arg}, workers={workers_arg}, amp={amp_arg}")

    # quick dataset sanity checks
    labels_train_dir = PROJECT_ROOT / "yolo" / "labels" / "train"
    labels_val_dir = PROJECT_ROOT / "yolo" / "labels" / "val"
 

    txts_train = list(labels_train_dir.glob("*.txt")) if labels_train_dir.exists() else []
    txts_val = list(labels_val_dir.glob("*.txt")) if labels_val_dir.exists() else []

    if not txts_train or not txts_val:
        print("Dataset labels missing:")
        print(f" - labels in {labels_train_dir}: {len(txts_train)}")
        print(f" - labels in {labels_val_dir}: {len(txts_val)}")
        print("Training requires YOLO-format .txt label files next to images. See https://docs.ultralytics.com/datasets for formatting.")
        raise RuntimeError("Missing label files for train/val. Create .txt labels or update data.yaml to point to labeled dataset.")

    model.train(
        data=str(DATA_YAML),
        epochs=200,
        imgsz=1280,
        batch=batch_arg,
        workers=workers_arg,
        cache=True,
        name="products",
        project=str(MODEL_OUTPUT),
        device=device_arg,
        patience=30,
        optimizer="AdamW",
        cos_lr=True,
        amp=amp_arg,
        augment=True,
    )


if __name__ == "__main__":
    train()
