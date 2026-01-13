from ultralytics import YOLO
import os
import torch
import cv2
from pathlib import Path
import sys
import argparse
import time
import statistics
import traceback
import logging


# Project paths (keep repo structure unchanged)
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


MODEL_PATH = find_latest_weight(MODELS_PROD_DIR)
if MODEL_PATH is None:
    print(f"No se encontró un peso entrenado en {MODELS_PROD_DIR}. Ejecuta `train_products.py` primero.")
    sys.exit(1)


# Load model
model = YOLO(str(MODEL_PATH))

# Setup logger for camera errors
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=str(LOG_DIR / 'camera_error.log'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


def main(device=0, imgsz=1280, conf=0.25, iou=0.5, model_device='auto', half=False):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    print(f"[INFO] Detección en cámara con modelo {MODEL_PATH} (pulsa 'q' para salir)")
    print(f"[INFO] parámetros: imgsz={imgsz} conf={conf} iou={iou} device={model_device} half={half}")

    # Resolve model device (avoid Ultralytics error when CUDA not available)
    try:
        if not torch.cuda.is_available():
            # sanitize environment variable if set to invalid value like 'auto'
            if os.environ.get('CUDA_VISIBLE_DEVICES') in ("auto", "None"):
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            # force CPU if no CUDA
            if model_device == 'auto' or str(model_device).lower().startswith('cuda'):
                model_device = 'cpu'
        else:
            # if user passed numeric device, convert to cuda index
            if model_device != 'auto' and str(model_device).isdigit():
                model_device = f"cuda:{int(model_device)}"
    except Exception:
        # don't fail startup on device probe
        pass

    # Try to fuse model conv+bn for speed if available
    try:
        model.model.fuse()
    except Exception:
        pass

    frame_idx = 0
    latencies = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            # defensive check: ensure frame is valid
            if frame is None:
                logging.warning(f'Received empty frame at index {frame_idx}')
                continue
            # Resize to reduce inference cost when frame is larger than imgsz
            h0, w0 = frame.shape[:2]
            scale = imgsz / max(h0, w0)
            if scale < 1.0:
                new_w, new_h = int(w0 * scale), int(h0 * scale)
                frame_proc = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_proc = frame

            # Convert BGR->RGB for Ultralytics
            frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)

            # Run inference (Ultralytics will handle letterbox/preprocessing)
            try:
                t0 = time.time()
                results = model(frame_rgb, imgsz=imgsz, conf=conf, iou=iou, device=model_device, half=half)
                t1 = time.time()
                latency = (t1 - t0) * 1000.0
            except Exception:
                logging.exception(f'Inference error on frame {frame_idx}')
                continue

            # diagnostics every 10 frames
            if frame_idx % 10 == 0:
                h, w = frame.shape[:2]
                if len(results) > 0:
                    r = results[0]
                    num = len(r.boxes)
                    boxes_info = []
                    for i, box in enumerate(r.boxes[:3]):
                        xyxy = box.xyxy[0].tolist()
                        # normalize relative to original frame size
                        nx1 = xyxy[0] / float(w)
                        ny1 = xyxy[1] / float(h)
                        nx2 = xyxy[2] / float(w)
                        ny2 = xyxy[3] / float(h)
                        conf_v = float(box.conf[0])
                        boxes_info.append({'norm': [nx1, ny1, nx2, ny2], 'conf': conf_v})
                    print(f"[DEBUG] frame={frame_idx} size=({w}x{h}) detections={num} sample={boxes_info} latency_ms={latency:.1f}")
                else:
                    print(f"[DEBUG] frame={frame_idx} size=({w}x{h}) detections=0 latency_ms={latency:.1f}")

            # track latency stats
            latencies.append(latency)
            if len(latencies) > 30:
                latencies.pop(0)
            avg_latency = statistics.mean(latencies) if latencies else latency

            # results[0].plot() returns an RGB annotated image
            try:
                annotated_rgb = results[0].plot() if len(results) > 0 else frame_rgb
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                logging.exception(f'Plotting error on frame {frame_idx}')
                continue

            # overlay latency
            info_text = f"latency={latency:.0f}ms avg={avg_latency:.0f}ms"
            cv2.putText(annotated_bgr, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Product Detection Live", annotated_bgr)

            try:
                key = cv2.waitKey(1) & 0xFF
            except Exception:
                logging.exception('cv2.waitKey/display failed')
                break
            if key == ord("q"):
                break
            if key == ord("s"):
                # save debug frame
                debug_out = Path.cwd() / f"camera_debug_{frame_idx}.jpg"
                cv2.imwrite(str(debug_out), annotated_bgr)
                print(f"[DEBUG] saved {debug_out}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live product detection (camera)")
    parser.add_argument("--device", default=0, help="camera device index or path")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="nms iou threshold")
    parser.add_argument("--model-device", default="auto", help="model device: 'auto', 'cpu' or GPU index like '0'")
    parser.add_argument("--half", action="store_true", help="use mixed precision (fp16) if supported")
    args = parser.parse_args()

    dev = int(args.device) if str(args.device).isdigit() else args.device
    # decide model device
    md = args.model_device
    if md != "auto" and str(md).isdigit():
        model_device = f"cuda:{md}"
    else:
        model_device = md

    try:
        main(device=dev, imgsz=args.imgsz, conf=args.conf, iou=args.iou, model_device=model_device, half=args.half)
    except Exception:
        tb = traceback.format_exc()
        logging.critical('Unhandled exception in camera main:\n' + tb)
        print('Se produjo un error grave. Revisa', LOG_DIR / 'camera_error.log')
        raise
