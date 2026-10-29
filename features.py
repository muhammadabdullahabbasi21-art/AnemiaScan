import numpy as np, cv2
from PIL import Image

def load_pixels(img):
    """Return RGB pixels of the eyelid only (alpha>0 = segmented region)."""
    a = np.array(img.convert("RGBA"))
    mask = a[..., 3] > 0
    return a[..., :3], mask

def extract_features(img):
    rgb, mask = load_pixels(img)
    px = rgb[mask].astype(np.float32)
    if len(px) < 20:
        raise ValueError("Not enough eyelid pixels found")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[mask].astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask].astype(np.float32)
    f = []
    for ch in (px, hsv, lab):
        f += list(ch.mean(0)) + list(ch.std(0))
    r, g, b = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    f += [r / (r + g + b + 1e-6), g / (r + g + b + 1e-6), r / (g + 1e-6)]
    return np.array(f, dtype=np.float32)

FEATURE_NAMES = [f"{s}_{c}_{k}" for s, cs in (("RGB","RGB"),("HSV","HSV"),("LAB","LAB")) for k in ("mean","std") for c in cs]
