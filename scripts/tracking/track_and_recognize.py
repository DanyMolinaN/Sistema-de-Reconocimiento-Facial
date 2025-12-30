import json
from pathlib import Path

import cv2
import os
import sys
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# ESTE NO ESTA VINCULADO AL PROYECTO FASE TEMPRANAMENTE, PERO SE USA PARA
# DEMOSTRAR EL TRACKING Y RECONOC
# CONFIGURACIÓN GENERAL
# =============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_FILE = PROJECT_ROOT / "embeddings" / "face_embeddings.json"

THRESHOLD = 0.9
CAMERA_INDEX = 0
RECOGNITION_INTERVAL = 5

# =============================
# MODELOS Y DISPOSITIVO
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Usando dispositivo: {device}")

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=True,
    device=device
)

model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# =============================
# CARGAR EMBEDDINGS
# =============================

with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    embeddings_db = json.load(f)

# Normalizar embeddings almacenados
for person in embeddings_db:
    emb = torch.tensor(embeddings_db[person])
    emb = torch.nn.functional.normalize(emb, p=2, dim=0)
    embeddings_db[person] = emb

# =============================
# FUNCIÓN DE RECONOCIMIENTO
# =============================

def reconocer_rostro(face_tensor):
    if face_tensor is None:
        return "Desconocido", float("inf")

    # Normalizar entrada a tensor y manejar batches
    if isinstance(face_tensor, list):
        # tomar la primera cara si viene como lista
        face_tensor = face_tensor[0] if len(face_tensor) > 0 else None

    if face_tensor is None:
        return "Desconocido", float("inf")

    if not torch.is_tensor(face_tensor):
        try:
            face_tensor = torch.as_tensor(face_tensor)
        except Exception:
            raise RuntimeError("No se pudo convertir la entrada de la cara a tensor")

    # face_tensor puede ser 3D (C,H,W) o 4D (N,C,H,W)
    if face_tensor.ndim == 4 and face_tensor.shape[0] == 1:
        face_tensor = face_tensor.squeeze(0)

    if face_tensor.ndim == 3:
        batch = face_tensor.unsqueeze(0).to(device)
    elif face_tensor.ndim == 4:
        batch = face_tensor.to(device)
    else:
        raise RuntimeError(f"Tensor de entrada con dimensiones no soportadas: {face_tensor.ndim}")

    with torch.no_grad():
        emb = model(batch).squeeze(0).cpu()

    emb = torch.nn.functional.normalize(emb, p=2, dim=0)

    identidad = "Desconocido"
    min_dist = float("inf")

    for person, ref_emb in embeddings_db.items():
        dist = torch.norm(emb - ref_emb)
        if dist < min_dist:
            min_dist = dist
            identidad = person

    if min_dist > THRESHOLD:
        identidad = "Desconocido"

    return identidad, min_dist.item()

# =============================
# MAIN
# =============================

def main():
    # Diagnóstico rápido de OpenCV / backend GUI
    try:
        print(f"[INFO] cv2 version: {cv2.__version__}")
        print(f"[INFO] cv2 file: {getattr(cv2, '__file__', 'unknown')}")
        build_info = cv2.getBuildInformation()
        # Mostrar solo la sección de GUI para no saturar
        gui_lines = [l for l in build_info.splitlines() if 'GUI' in l or 'GTK' in l or 'Win32' in l or 'QT' in l]
        print("[INFO] OpenCV build GUI info:")
        for l in gui_lines:
            print("  ", l)
    except Exception as e:
        print(f"[WARN] No se pudo obtener información del build de OpenCV: {e}")

    # Mostrar versión de NumPy para diagnóstico
    try:
        import numpy as _np
        print(f"[INFO] numpy version: {_np.__version__}")
    except Exception:
        print("[WARN] No se pudo obtener la versión de numpy")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return

    trackers = []
    identities = []
    distances = []

    frame_count = 0
    print("[INFO] Presiona 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Asegurar que pasamos un array contiguo uint8 a mtcnn.detect
        img_for_detect = np.ascontiguousarray(frame_rgb, dtype=np.uint8)


        # DETECCIÓN INICIAL

        if len(trackers) == 0:
            try:
                boxes, _ = mtcnn.detect(img_for_detect)
            except Exception as e:
                print(f"[ERROR] mtcnn.detect falló con array: {e}")
                try:
                    # Intentar con PIL como fallback
                    boxes, _ = mtcnn.detect(image)
                except Exception as e2:
                    print(f"[ERROR] mtcnn.detect fallback con PIL también falló: {e2}")
                    continue

            if boxes is not None:
                # Obtener los tensores de las caras usando PIL (mtcnn acepta PIL)
                faces = mtcnn(image)

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1

                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x1, y1, w, h))

                    trackers.append(tracker)

                    identidad, dist = reconocer_rostro(faces[i])
                    identities.append(identidad)
                    distances.append(dist)


        # ACTUALIZAR TRACKERS

        to_remove = []

        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(frame)

            if not success:
                to_remove.append(i)
                continue

            x, y, w, h = map(int, bbox)

            # Validar bounding box
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue

            # Re-reconocimiento
            if frame_count % RECOGNITION_INTERVAL == 0:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                face_tensor = mtcnn(face_pil)

                if face_tensor is not None:
                    identidad, dist = reconocer_rostro(face_tensor)
                    identities[i] = identidad
                    distances[i] = dist

            color = (0, 255, 0) if identities[i] != "Desconocido" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                f"{identities[i]} ({distances[i]:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Limpiar trackers fallidos
        for i in sorted(to_remove, reverse=True):
            trackers.pop(i)
            identities.pop(i)
            distances.pop(i)

        win_name = "Tracking y Reconocimiento Facial"
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, frame)
        except Exception as e:
            print(f"[ERROR] cv2.imshow falló: {e}")
            print("[HINT] Probablemente estás usando una build 'headless' de OpenCV. Instala 'opencv-python' o 'opencv-contrib-python'.")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
