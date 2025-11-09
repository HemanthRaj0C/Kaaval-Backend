import cv2
import numpy as np
from insightface.utils import face_align

def geo_normalize(img, kps):
    # Normalize geometry to standard ArcFace alignment model
    aligned = face_align.norm_crop(img, landmark=kps, image_size=112)
    return aligned
