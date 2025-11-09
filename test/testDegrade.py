import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np

from backend.src.modules.embed import get_embedding
from backend.src.modules.matcher import search

IMG = "backend/data/aligned/4.jpg"  # use known aligned image

img = cv2.imread(IMG)

# Low-res degradation
lowres = cv2.resize(img, (56, 56))
lowres = cv2.resize(lowres, (112, 112), interpolation=cv2.INTER_LINEAR)

emb_original = get_embedding(img)
emb_lowres   = get_embedding(lowres)

# ✅ Flatten for similarity
sim = float(np.dot(emb_original.flatten(), emb_lowres.flatten()))
print(f"Similarity original ↔ lowres = {sim:.4f}")

print("\nSearch results for lowres:")
results = search(emb_lowres.flatten(), k=5)
for f, s in results:
    print(f"{f}   |   {s:.4f}")
