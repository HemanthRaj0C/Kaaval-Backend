# backend/src/textureRecover.py
import cv2
import numpy as np

def enhance_texture(img):
    """
    Lightweight texture recovery:
     - detect approximate skin region in HSV
     - increase luminance (CLAHE) for skin region
     - slight sharpen overall
    """
    if img is None or img.size == 0:
        return img

    img_h = img.copy()
    hsv = cv2.cvtColor(img_h, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Simple skin mask heuristic in HSV (works reasonably for many faces)
    mask = cv2.inRange(hsv, (0, 10, 30), (25, 255, 255))
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    mask = (mask / 255.0).astype(np.float32)  # 0..1

    # CLAHE on L channel (LAB)
    lab = cv2.cvtColor(img_h, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    enhanced_lab = cv2.merge([L2, A, B])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Blend only skin regions (soft mask)
    mask_3 = np.stack([mask, mask, mask], axis=2)
    out = (img_h.astype(np.float32) * (1 - mask_3) + enhanced.astype(np.float32) * mask_3)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # gentle sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    out = cv2.filter2D(out, -1, kernel)
    return out
