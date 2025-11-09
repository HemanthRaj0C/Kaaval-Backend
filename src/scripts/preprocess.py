import os
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align

RAW_DIR = "backend/data/raw"
ALIGNED_DIR = "backend/data/aligned"

def ensure_dirs():
    os.makedirs(ALIGNED_DIR, exist_ok=True)

def init_detector():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def process_all():
    ensure_dirs()
    app = init_detector()

    images = os.listdir(RAW_DIR)

    for idx, filename in enumerate(images):
        img_path = os.path.join(RAW_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        face = faces[0]

        # ✅ Correct face alignment call
        aligned = face_align.norm_crop(img, landmark=face.kps, image_size=112)

        save_path = os.path.join(ALIGNED_DIR, filename)
        cv2.imwrite(save_path, aligned)

        if idx % 500 == 0:
            print(f"[Aligned] Processed {idx} images...")

    print("✅ Face alignment complete.")

if __name__ == "__main__":
    process_all()
