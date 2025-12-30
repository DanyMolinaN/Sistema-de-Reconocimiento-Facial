import cv2
import os
import numpy as np

INPUT_DIR = "data/employees/raw"
OUTPUT_DIR = "data/employees/augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment_image(img):
    variants = []

    # Rotaciones
    for angle in [-15, -8, 8, 15]:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        variants.append(rotated)

    # Cambios de iluminación
    for alpha in [0.7, 1.3]:
        variants.append(cv2.convertScaleAbs(img, alpha=alpha, beta=20))

    # Simulación de mascarilla
    mask = img.copy()
    h = img.shape[0]
    cv2.rectangle(mask, (0, h//2), (img.shape[1], h), (0,0,0), -1)
    variants.append(mask)

    return variants

def generate_variants():
    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".jpg"):
            continue

        img = cv2.imread(os.path.join(INPUT_DIR, file))
        base_name = file.split(".")[0]

        augmented = augment_image(img)
        for i, aug in enumerate(augmented):
            out_path = os.path.join(OUTPUT_DIR, f"{base_name}_aug{i}.jpg")
            cv2.imwrite(out_path, aug)

        print(f"[OK] Variantes generadas para {file}")

if __name__ == "__main__":
    generate_variants()
