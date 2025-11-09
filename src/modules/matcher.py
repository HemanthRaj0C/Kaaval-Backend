import os
import numpy as np
import faiss

EMB_DIR = "backend/data/embeddings"

def load_embeddings():
    vectors = []
    filenames = []

    for file in os.listdir(EMB_DIR):
        if file.endswith(".npy"):
            vec = np.load(os.path.join(EMB_DIR, file))

            # ✅ Convert (1,512) → (512,)
            if len(vec.shape) == 2 and vec.shape[0] == 1:
                vec = vec.squeeze(0)

            # ✅ Skip completely invalid shapes
            if vec.shape != (512,):
                print(f"[SKIP] Invalid vector shape {vec.shape} in {file}")
                continue

            vectors.append(vec.astype("float32"))
            filenames.append(file)

    vectors = np.vstack(vectors).astype("float32")  # (N, 512)
    return vectors, filenames

def build_index():
    vectors, filenames = load_embeddings()
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (dot product on normalized vectors)
    index.add(vectors)
    return index, filenames

def search(query_emb, k=5, threshold=0.48):
    index, filenames = build_index()
    D, I = index.search(query_emb.reshape(1, -1), k)
    results = [(filenames[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

    # Apply threshold filter
    filtered = [r for r in results if r[1] >= threshold]

    if len(filtered) == 0:
        return [("NO MATCH", 0.0)]

    return filtered

