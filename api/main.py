import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

# ✅ Use fusion embedding instead of normal embedding
from backend.src.modules.fusion import fusion_embedding
from backend.src.modules.matcher import search

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ SAME face detector used during dataset preprocess
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))


def preprocess_same_pipeline(img_bytes):
    """
    EXACT SAME PIPELINE AS TRAINING:
        raw input → detect → align (norm_crop using 5 keypoints)
    """
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    faces = face_app.get(img)
    if len(faces) == 0:
        return None  # No face detected

    face = faces[0]
    aligned = norm_crop(img, face.kps, image_size=112)  # ✅ same alignment

    return aligned


@app.post("/match")
async def match_face(file: UploadFile = File(...)):
    img_bytes = await file.read()

    aligned = preprocess_same_pipeline(img_bytes)

    if aligned is None:
        return {"error": "No face detected — try a clearer raw face image."}

    # ✅ Use fusion embedding
    query_emb = fusion_embedding(aligned, already_aligned=True)
    if query_emb is None:
        return {"error": "Embedding failed — try different image."}

    # ✅ FAISS search
    results = search(query_emb, k=5)

    return {
        "matches": [
            {"filename": filename, "similarity": float(sim)}
            for filename, sim in results
        ]
    }
