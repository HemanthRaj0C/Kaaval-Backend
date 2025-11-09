# backend/src/enhance.py
import cv2
import numpy as np

class SimpleFaceEnhancer:
    """Lightweight face enhancer - no GFPGAN needed"""
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # mild sharpen kernel scaled down
        self.sharpen_kernel = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]], dtype=np.float32) * 0.3

    def enhance(self, img, has_aligned=True, only_center_face=True):
        enhanced = self._enhance_face(img)
        cropped_faces = [img]
        restored_faces = [enhanced]
        restored_img = enhanced
        return cropped_faces, restored_faces, restored_img

    def _enhance_face(self, img):
        if img is None or img.size == 0:
            return img

        # Convert to LAB color space and apply CLAHE to L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Fast denoise
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        # Sharpen & blend
        sharpened = cv2.filter2D(enhanced, -1, self.sharpen_kernel)
        enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

        # Bilateral filter for skin smoothing (keeps edges)
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        enhanced = cv2.addWeighted(enhanced, 0.6, bilateral, 0.4, 0)

        return enhanced

# single global restorer to mimic GFPGANer interface
restorer = SimpleFaceEnhancer()

def enhance_face(img):
    """Public helper used by embed/fusion: returns enhanced face image (BGR)."""
    if img is None:
        return img
    _, restored_faces, _ = restorer.enhance(img, has_aligned=True, only_center_face=True)
    if restored_faces and len(restored_faces) > 0:
        return restored_faces[0]
    return img
