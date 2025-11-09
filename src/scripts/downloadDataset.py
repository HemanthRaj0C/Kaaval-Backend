import os
from datasets import load_dataset
from PIL import Image

RAW_DIR = "backend/data/raw/"

def ensure_dirs():
    os.makedirs("backend/data/raw", exist_ok=True)
    os.makedirs("backend/data/aligned", exist_ok=True)
    os.makedirs("backend/data/embeddings", exist_ok=True)
    os.makedirs("backend/data/cache", exist_ok=True)

def download():
    ensure_dirs()
    dataset = load_dataset("nielsr/CelebA-faces", split="train")

    for idx, example in enumerate(dataset):
        img: Image.Image = example["image"]
        img.save(os.path.join(RAW_DIR, f"{idx}.jpg"))
        if idx % 500 == 0:
            print(f"Saved {idx} images...")

if __name__ == "__main__":
    download()
