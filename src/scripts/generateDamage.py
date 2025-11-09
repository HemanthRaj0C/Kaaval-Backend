#!/usr/bin/env python3
import os, csv
import cv2
import numpy as np

RAW_DIR = "backend/data/raw"
ALIGNED_DIR = "backend/data/aligned"
OUT_DIR = "backend/data/damaged"

def ensure(path):
    os.makedirs(path, exist_ok=True)

# ===== DAMAGE METHODS =====
def light(img):
    img = cv2.GaussianBlur(img, (7,7), 1.5)
    img = cv2.convertScaleAbs(img, alpha=0.85, beta=-10)
    return img

def severe(img):
    blur = cv2.GaussianBlur(img, (21,21), 7)
    noise = np.random.normal(0, 25, blur.shape).astype(np.int16)
    noisy = np.clip(blur.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def cctv(img):
    small = cv2.resize(img, (64,64))
    restored = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    restored = cv2.convertScaleAbs(restored, alpha=0.75, beta=-5)
    return restored

RECIPES = {
    "light": light,
    "severe": severe,
    "cctv": cctv,
}


def main():
    ensure(OUT_DIR)

    aligned_files = [f for f in os.listdir(ALIGNED_DIR) if f.lower().endswith(("jpg", "png", "jpeg"))]

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["raw_file", "damaged_file", "mode"])

        for fname in aligned_files:
            raw_path = os.path.join(RAW_DIR, fname)

            if not os.path.exists(raw_path):
                print(f"⚠️ Raw image missing for aligned entry: {fname}")
                continue

            img = cv2.imread(raw_path)
            if img is None:
                print(f"⚠️ Could not read image: {raw_path}")
                continue

            for mode, fn in RECIPES.items():
                damaged = fn(img.copy())
                out_name = f"{os.path.splitext(fname)[0]}_{mode}.jpg"
                out_path = os.path.join(OUT_DIR, out_name)
                cv2.imwrite(out_path, damaged)
                writer.writerow([fname, out_name, mode])

    print("\n✅ Damaged dataset created:")
    print(f"   Output directory : {OUT_DIR}")
    print(f"   Manifest         : {manifest_path}\n")


if __name__ == "__main__":
    main()
