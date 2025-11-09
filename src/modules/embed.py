# backend/src/embed.py
import numpy as np
from insightface.app import FaceAnalysis

def load_model():
    # create a FaceAnalysis to access the recognition model (assumes buffalo_l models are downloaded)
    app = FaceAnalysis(name="buffalo_l")
    # no det_size here because we only want the recognition model
    app.prepare(ctx_id=0, det_size=(640,640))
    rec_model = app.models.get('recognition', None)
    if rec_model is None:
        # Some insightface versions expose rec model differently
        raise RuntimeError("Recognition model not found inside FaceAnalysis.models")
    return rec_model

model = load_model()

def get_embedding(img):
    """
    Input: BGR aligned face image (112x112 recommended)
    Output: normalized 1D numpy array (512,) float32
    """
    if img is None:
        return None

    # Ensure correct size
    import cv2
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)

    # model.get_feat returns shape (1, D)
    feat = model.get_feat(img)
    if feat is None:
        return None

    emb = np.array(feat).astype(np.float32)
    # flatten & normalize to unit length
    emb = emb.flatten()
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm
