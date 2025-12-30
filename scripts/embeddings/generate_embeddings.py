import json
from pathlib import Path

import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

# CONFIGURACIÓN GENERAL DEL PROYECTO

# Raíz del proyecto (face-recognition-system/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset de rostros (una carpeta por identidad)
DATASET_DIR = PROJECT_ROOT / "dataset" / "faces"

# Carpeta y archivo de salida para embeddings
OUTPUT_DIR = PROJECT_ROOT / "embeddings"
OUTPUT_FILE = OUTPUT_DIR / "face_embeddings.json"

# Extensiones de imagen permitidas
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# INICIALIZACIÓN DE MODELOS

# Selección automática de GPU o CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Usando dispositivo: {device}")

# Detector facial MTCNN
mtcnn = MTCNN(
    image_size=160,          # Tamaño requerido por FaceNet
    margin=20,               # Margen alrededor del rostro
    select_largest=True,     # Selecciona el rostro más grande
    post_process=True,       # Normaliza y alinea
    device=device
)

# Modelo FaceNet preentrenado
model = InceptionResnetV1(pretrained="vggface2")
model.eval()
model.to(device)
# FUNCIÓN PARA EXTRAER EMBEDDING FACIAL

def extract_embedding(image_path: Path):
    """
    Detecta un rostro en la imagen y retorna su embedding facial (512 dimensiones).
    Devuelve None si no se detecta ningún rostro.
    """

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] No se pudo abrir la imagen {image_path.name}: {e}")
        return None

    # Detección del rostro
    face = mtcnn(image)

    if face is None:
        print(f"[WARNING] No se detectó rostro en {image_path.name}")
        return None

    # Preparar tensor para el modelo
    face = face.unsqueeze(0).to(device)

    # Inferencia sin gradientes (más rápido y seguro)
    with torch.no_grad():
        embedding = model(face)

    # Retornar embedding en CPU
    return embedding.squeeze(0).cpu()
# PIPELINE PRINCIPAL DE GENERACIÓN DE EMBEDDINGS

def generate_embeddings():

    # Validar existencia del dataset
    if not DATASET_DIR.exists():
        print(f"[ERROR] El directorio del dataset no existe: {DATASET_DIR}")
        return

    # Crear carpeta de salida si no existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Base de datos final de embeddings
    embeddings_db = {}

    # Obtener carpetas de identidades
    persons = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    print(f"[INFO] Encontradas {len(persons)} identidades.")

    # Procesar cada identidad
    for person_dir in persons:
        person_name = person_dir.name
        print(f"\n[PROCESANDO] Identidad: {person_name}")

        embeddings = []

        # Procesar imágenes de la identidad
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            emb = extract_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
                print(f"[OK] Embedding extraído de {img_path.name}")
            else:
                print(f"[FALLO] No se pudo extraer embedding de {img_path.name}")

        # Validar que se hayan obtenido embeddings
        if len(embeddings) == 0:
            print(f"[WARNING] No se obtuvieron embeddings para {person_name}. Se omite.")
            continue

        # Promedio de embeddings (representación robusta)
        embeddings_tensor = torch.stack(embeddings)
        avg_embedding = embeddings_tensor.mean(dim=0)

        # Normalización L2 (CLAVE para reconocimiento)
        avg_embedding = torch.nn.functional.normalize(avg_embedding, p=2, dim=0)

        # Guardar embedding final
        embeddings_db[person_name] = avg_embedding.tolist()
        print(f"[FINALIZADO] {person_name}: {len(embeddings)} imágenes procesadas")

    # GUARDAR BASE BIOMÉTRICA FINAL

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings_db, f, indent=2)

    print(f"\n[SAVED] Base de embeddings guardada en:")
    print(f"        {OUTPUT_FILE}")

# PUNTO DE ENTRADA

if __name__ == "__main__":
    generate_embeddings()
