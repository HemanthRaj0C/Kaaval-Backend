import os
import cv2
import numpy as np
from ..modules.embed import get_embedding

ALIGNED_DIR = "backend/data/aligned"
EMB_DIR = "backend/data/embeddings"

def ensure_dir():
    os.makedirs(EMB_DIR, exist_ok=True)

def process_all():
    ensure_dir()
    images = os.listdir(ALIGNED_DIR)

    for idx, filename in enumerate(images):
        img_path = os.path.join(ALIGNED_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        emb = get_embedding(img)
        np.save(os.path.join(EMB_DIR, filename.replace(".jpg", ".npy")), emb)

        if idx % 200 == 0:
            print(f"[Embedded] Processed {idx} images...")

    print("✅ Embedding generation complete.")

if __name__ == "__main__":
    process_all()
