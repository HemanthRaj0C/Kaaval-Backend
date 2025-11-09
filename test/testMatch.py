import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.src.modules.embed import get_embedding
from backend.src.modules.matcher import search
import cv2

# Load any image from the aligned folder
ALIGNED_DIR = "backend/data/aligned"
files = os.listdir(ALIGNED_DIR)
first_image = os.path.join(ALIGNED_DIR, files[0])

print(f"Testing with: {first_image}")

img = cv2.imread(first_image)

if img is None:
    raise Exception("Image failed to load. Check paths!")

query_emb = get_embedding(img)
results = search(query_emb, k=5)

print("\nMatch Results:")
for filename, score in results:
    print(f"{filename}   similarity={score:.4f}")
