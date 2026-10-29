"""Simple eyelid (conjunctiva) segmentation using classic OpenCV - no deep learning needed."""
import numpy as np, cv2
from PIL import Image

def segment_eyelid(img):
    """Return (RGBA PIL image with background transparent, mask).
    - If the image already has transparency (like the dataset), reuse it.
    - Otherwise: the inner eyelid is the reddest region, so threshold the Lab 'a' channel (Otsu),
      clean with morphology and keep the largest blob."""
    rgba = np.array(img.convert("RGBA"))
    if (rgba[..., 3] == 0).mean() > 0.2:
        mask = (rgba[..., 3] > 0).astype(np.uint8)
    else:
        rgb = rgba[..., :3]
        a = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 1]
        a = cv2.GaussianBlur(a, (7, 7), 0)
        _, mask = cv2.threshold(a, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    rgba[..., 3] = mask * 255
    return Image.fromarray(rgba), mask
