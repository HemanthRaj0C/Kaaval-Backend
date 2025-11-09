import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import cv2
import numpy as np
from backend.src.modules.fusion import fusion_embedding
from backend.src.modules.matcher import search

# =============================
# Load a known aligned face (same dataset)
# =============================
IMG = "backend/data/aligned/4.jpg"     # <-- change this to any aligned sample you want
img = cv2.imread(IMG)
if img is None:
    raise Exception(f"Failed to read: {IMG}")

# =============================
# Create degraded versions
# =============================
# Mild decomposition (blur + contrast loss)
light = cv2.GaussianBlur(img, (7, 7), 2)
light = cv2.convertScaleAbs(light, alpha=0.85, beta=-10)

# Heavy decomposition (aggressive blur + noise)
severe = cv2.GaussianBlur(img, (15, 15), 8)
noise = np.random.normal(0, 20, severe.shape).astype(np.int16)
severe = np.clip(severe.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Resize both back to correct embedding input size
light = cv2.resize(light, (112, 112))
severe = cv2.resize(severe, (112, 112))
img_resized = cv2.resize(img, (112, 112))

# =============================
# Embeddings - PASS already_aligned=True
# =============================
emb_original = fusion_embedding(img_resized, already_aligned=True)
emb_light = fusion_embedding(light, already_aligned=True)
emb_severe = fusion_embedding(severe, already_aligned=True)

# =============================
# Check for None embeddings
# =============================
if emb_original is None:
    raise Exception("Failed to generate embedding for original image")
if emb_light is None:
    raise Exception("Failed to generate embedding for light decomposition")
if emb_severe is None:
    raise Exception("Failed to generate embedding for severe decomposition")

# =============================
# Similarity Comparison
# =============================
sim_light = float(np.dot(emb_original, emb_light))
sim_severe = float(np.dot(emb_original, emb_severe))

print(f"\nSimilarity(original ↔ light decomposition) = {sim_light:.4f}")
print(f"Similarity(original ↔ severe decomposition) = {sim_severe:.4f}\n")

# =============================
# Search Identity via FAISS
# =============================
print("Search results - LIGHT:")
results_light = search(emb_light, k=5)
for f, s in results_light:
    print(f"{f}   |   {s:.4f}")

print("\nSearch results - SEVERE:")
results_severe = search(emb_severe, k=5)
for f, s in results_severe:
    print(f"{f}   |   {s:.4f}")