import cv2
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

# ✅ Use fusion embedding instead of normal embedding
from backend.src.modules.fusion import fusion_embedding
from backend.src.modules.matcher import search, load_embeddings

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
            {
                "filename": filename, 
                "similarity": float(sim),
                "imageUrl": f"/api/image/{filename.replace('.npy', '.jpg')}"
            }
            for filename, sim in results
        ]
    }


@app.get("/api/image/{filename}")
async def serve_image(filename: str):
    """Serve original images from data/raw folder"""
    try:
        # Clean the filename - remove .npy if present and ensure .jpg extension
        clean_filename = filename.replace('.npy', '').replace('.jpg', '') + '.jpg'
        
        RAW_DIR = "backend/data/raw"
        image_path = os.path.join(RAW_DIR, clean_filename)
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {clean_filename}")
        
        return FileResponse(image_path, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        # Count embeddings
        EMB_DIR = "backend/data/embeddings"
        ALIGNED_DIR = "backend/data/aligned"
        RAW_DIR = "backend/data/raw"
        
        emb_count = len([f for f in os.listdir(EMB_DIR) if f.endswith('.npy')]) if os.path.exists(EMB_DIR) else 0
        aligned_count = len([f for f in os.listdir(ALIGNED_DIR) if f.endswith('.jpg')]) if os.path.exists(ALIGNED_DIR) else 0
        raw_count = len([f for f in os.listdir(RAW_DIR) if f.endswith('.jpg')]) if os.path.exists(RAW_DIR) else 0
        
        return {
            "embeddings": emb_count,
            "aligned_faces": aligned_count,
            "raw_images": raw_count,
            "total_database": emb_count
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/database/files")
async def get_database_files(limit: int = 50):
    """Get list of files in database"""
    try:
        EMB_DIR = "backend/data/embeddings"
        files = [f.replace('.npy', '') for f in os.listdir(EMB_DIR) if f.endswith('.npy')]
        return {
            "files": files[:limit],
            "total": len(files)
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "buffalo_l",
        "embedding_dim": 512
    }


@app.post("/training/preprocess")
async def run_preprocess():
    """Run face detection and alignment on raw images"""
    try:
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "backend/src/scripts/preprocess.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "Process timed out after 5 minutes"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@app.post("/training/embed")
async def run_embedding():
    """Generate embeddings from aligned faces"""
    try:
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "backend/src/scripts/batchEmbed.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "Process timed out after 10 minutes"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
