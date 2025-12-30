import os
import pickle
import cv2
import numpy as np

IMAGE_DIR = "data/employees/augmented"
OUTPUT_FILE = "data/face_gallery.pkl"


# Try to use facenet-pytorch (recommended). If not available, fall back
# to keras-facenet (existing behaviour).
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image
    import torch
    from torchvision import transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Embedder:
        def __init__(self):
            self.mtcnn = MTCNN(keep_all=True, device=device)
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        def extract(self, image_rgb, threshold=0.95):
            # image_rgb: numpy array in RGB
            pil = Image.fromarray(image_rgb)
            boxes, probs = self.mtcnn.detect(pil)
            if boxes is None:
                return []
            faces = []
            tensors = []
            valid_idx = []
            for i, (box, p) in enumerate(zip(boxes, probs)):
                if p is None or p < threshold:
                    continue
                x1, y1, x2, y2 = [int(x) for x in box]
                w, h = x2 - x1, y2 - y1
                crop = pil.crop((x1, y1, x2, y2))
                t = self.transform(crop).unsqueeze(0).to(device)
                tensors.append(t)
                valid_idx.append((x1, y1, w, h))

            if not tensors:
                return []

            batch = torch.cat(tensors, dim=0)
            with torch.no_grad():
                emb = self.model(batch).cpu().numpy()

            for det_box, e in zip(valid_idx, emb):
                dete = { 'box': det_box }
                faces.append({ **dete, 'embedding': e })

            return faces

    embedder = Embedder()
    print('[INFO] Usando facenet-pytorch como extractor de embeddings')
except Exception:
    # fallback to keras-facenet
    try:
        from keras_facenet import FaceNet
        embedder = FaceNet()
        print('[INFO] Usando keras-facenet como extractor de embeddings')
    except Exception as e:
        print('[ERROR] No se pudo inicializar ningún extractor de embeddings:', e)
        raise

gallery = {}


def build_gallery():
    if not os.path.isdir(IMAGE_DIR):
        print(f"[ERROR] No existe el directorio de imágenes: {IMAGE_DIR}")
        return

    for file in os.listdir(IMAGE_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(IMAGE_DIR, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] No se pudo leer la imagen: {img_path}")
            continue

        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[WARN] Error procesando imagen {img_path}: {e}")
            continue

        try:
            faces = embedder.extract(img_rgb, threshold=0.95)
        except Exception as e:
            print(f"[ERROR] Falló el extractor de embeddings en {img_path}: {e}")
            continue

        if not faces:
            continue

        embedding = faces[0].get("embedding")
        if embedding is None:
            print(f"[WARN] No se obtuvo embedding para {img_path}")
            continue

        employee_id = file.split("_")[0]
        gallery.setdefault(employee_id, []).append(embedding)

    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(gallery, f)

    print(f"[OK] Gallery creada con {len(gallery)} empleados")


if __name__ == "__main__":
    build_gallery()

