from pathlib import Path
import sys
import cv2
import torch
import time
import threading
import pickle
import numpy as np
from collections import deque, Counter
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tracking"))
from deep_sort.deep_sort import DeepSort

GALLERY_FILE = PROJECT_ROOT / "data" / "face_gallery.pkl"

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMERA_INDEX = 0

DIST_THRESHOLD = 0.85          # RECONOCIMIENTO
LEARN_THRESHOLD = 0.75         # AUTO-LEARNING (más estricto)

MEMORY_SIZE = 20
MAX_AGE = 40

DETECT_EVERY = 6
RECOGNIZE_EVERY = 10

AUTO_LEARN = True
MIN_FRAMES_CONFIRM = 8
MAX_NEW_EMB_PER_ID = 50

# MODELS
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=DEVICE)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
tracker = DeepSort(max_age=MAX_AGE)

# LOAD GALLERY
with open(GALLERY_FILE, "rb") as f:
    gallery = pickle.load(f)

for k in gallery:
    gallery[k] = [emb / np.linalg.norm(emb) for emb in gallery[k]]

print(f"[INFO] Gallery cargada: {len(gallery)} empleados")

# SHARED DATA (THREAD SAFE)
lock = threading.Lock()
shared_detections = []

# UTILS
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter)

def recognize_embedding(embedding):
    embedding = embedding / np.linalg.norm(embedding)
    best_name = "Desconocido"
    best_dist = float("inf")

    for name, refs in gallery.items():
        for ref in refs:
            d = np.linalg.norm(embedding - ref)
            if d < best_dist:
                best_dist = d
                best_name = name

    return best_name, best_dist

# DETECTION THREAD
def detection_thread(cap):
    global shared_detections
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % DETECT_EVERY != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        boxes, _ = mtcnn.detect(pil)
        faces = mtcnn(pil)

        detections = []

        if boxes is not None and faces is not None:
            for box, face in zip(boxes, faces):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1

                with torch.no_grad():
                    emb = facenet(face.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]

                detections.append(([x1, y1, w, h], emb))

        with lock:
            shared_detections = detections

# MAIN
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cámara no disponible")
        return

    memory = {}
    track_learning = {}
    fps_hist = deque(maxlen=30)
    frame_id = 0

    threading.Thread(target=detection_thread, args=(cap,), daemon=True).start()
    print("[INFO] Sistema iniciado (q para salir)")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        with lock:
            detections = shared_detections.copy()

        tracks = tracker.update_tracks(detections)

        for tr in tracks:
            if not tr.is_confirmed():
                continue

            tid = tr.track_id
            x1, y1, x2, y2 = map(int, tr.to_ltrb())

            # ================= EMBEDDING MATCH =================
            best_iou = 0
            embedding = None

            for det, emb in detections:
                dx, dy, dw, dh = det
                score = iou(
                    [dx, dy, dx + dw, dy + dh],
                    [x1, y1, x2, y2]
                )
                if score > best_iou:
                    best_iou = score
                    embedding = emb

            if embedding is None:
                continue

            memory.setdefault(tid, deque(maxlen=MEMORY_SIZE))

            # ================= RECOGNITION GATE =================
            if frame_id % RECOGNIZE_EVERY == 0:
                name_pred, dist = recognize_embedding(embedding)

                if dist < DIST_THRESHOLD:
                    memory[tid].append(name_pred)
                else:
                    memory[tid].append("Desconocido")
            else:
                dist = 1.0

            name = (
                Counter(memory[tid]).most_common(1)[0][0]
                if memory[tid] else "Desconocido"
            )

            color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ID:{tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # ================= SAFE AUTO-LEARNING =================
            if AUTO_LEARN and name != "Desconocido" and dist < LEARN_THRESHOLD:
                tl = track_learning.setdefault(tid, {"name": name, "count": 0})

                if tl["name"] == name:
                    tl["count"] += 1
                    if tl["count"] >= MIN_FRAMES_CONFIRM:
                        if len(gallery[name]) < MAX_NEW_EMB_PER_ID:
                            gallery[name].append(
                                embedding / np.linalg.norm(embedding)
                            )
                            print(f"[AUTO-LEARN] Embedding agregado a {name}")
                        del track_learning[tid]
                else:
                    del track_learning[tid]

        fps = 1 / (time.time() - t0)
        fps_hist.append(fps)
        avg_fps = sum(fps_hist) / len(fps_hist)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Face Recognition DeepSort PRO", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
