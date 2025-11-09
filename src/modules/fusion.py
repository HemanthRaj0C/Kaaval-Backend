# backend/src/fusionEmbed.py
import cv2
import numpy as np

from .enhance.enhance import enhance_face
from .enhance.shape import normalize_shape
from .enhance.texture import enhance_texture
from .embed import get_embedding

def _norm_vec(x):
    x = x.flatten()
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n

def _contrast_img(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)
    return out

def _edges_img(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 40, 120)
    return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)

def fusion_embedding(img, already_aligned=False):
    """
    img: BGR image (if already_aligned=True, pass 112x112 face crop)
    Returns: normalized embedding (512,) or None
    """
    if img is None:
        return None

    # 1) shape normalize (mild) - returns 112x112
    img_norm = normalize_shape(img)

    # 2) if not already_aligned, attempt detection+alignment using insightface (simple fallback)
    if not already_aligned:
        # try to detect & align using insightface to keep consistency with dataset pipeline
        from insightface.app import FaceAnalysis
        from insightface.utils.face_align import norm_crop
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        faces = face_app.get(img_norm)
        if len(faces) == 0:
            # if detection fails, attempt to resize and continue
            aligned = cv2.resize(img_norm, (112,112))
        else:
            face = faces[0]
            aligned = norm_crop(img_norm, face.kps, image_size=112)
    else:
        aligned = img_norm if img_norm.shape[:2] == (112,112) else cv2.resize(img_norm, (112,112))

    # 3) Different variants
    base_img = aligned.copy()
    enhanced_img = enhance_face(base_img.copy())
    contrast_img = _contrast_img(base_img.copy())
    edges_img = _edges_img(base_img.copy())

    # 4) Get embeddings (may return None)
    emb1 = get_embedding(base_img)
    emb2 = get_embedding(enhanced_img)
    emb3 = get_embedding(contrast_img)
    emb4 = get_embedding(edges_img)

    # If any are None, drop them
    embs = [e for e in (emb1, emb2, emb3, emb4) if e is not None]
    if len(embs) == 0:
        return None

    # Stack and average then normalize
    stacked = np.stack(embs, axis=0)
    fused = np.mean(stacked, axis=0)
    # normalize
    fused = fused / (np.linalg.norm(fused) + 1e-12)
    return fused.astype(np.float32)
