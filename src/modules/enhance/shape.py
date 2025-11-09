# backend/src/shapeNormalize.py
import cv2
import numpy as np

def normalize_shape(img):
    """
    Minimal shape normalization:
    - Resize to 112x112 if not already
    - Apply a small histogram equalization on luminance
    """
    if img is None:
        return img

    # If already 112x112, keep size; else resize preserving aspect by pad+resize
    h, w = img.shape[:2]
    if (h, w) != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)

    # mild CLAHE on L channel
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    except Exception:
        # fallback keep original
        pass

    return img
