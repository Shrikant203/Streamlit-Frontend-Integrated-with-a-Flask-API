"""
preprocessing.py
================
Frontend-side image preparation (copied unchanged from the Task 7 application).

Hand-drawn or photographed digits are arbitrary sizes and positions, while the
Flask API (and the CNN behind it) expects an 8x8 image in the same style as the
scikit-learn Digits dataset. These helpers crop to the ink, scale, centre by
mass and rescale to 0-16 *before* the image is sent to the API, so the API
itself stays a thin, model-only service.

tta_variants() builds the 11 slightly shifted / thickened / thinned copies of an
image used for test-time augmentation: the frontend sends them to the API in ONE
POST /predict/batch call and averages the returned probabilities.
"""
import numpy as np
from PIL import Image


def pixels_to_display_image(p, size=192):
    arr = np.clip(p, 0, 16) / 16.0 * 255
    return Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.NEAREST)


def crop_to_ink(gray, pad_frac=0.07):
    arr = gray.astype(np.float64)
    if arr.max() <= arr.min():
        return None
    norm = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    mask = norm > max(25.0, norm.max() * 0.18)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    py, px = max(1, int((y1 - y0 + 1) * pad_frac)), max(1, int((x1 - x0 + 1) * pad_frac))
    y0, y1 = max(0, y0 - py), min(norm.shape[0] - 1, y1 + py)
    x0, x1 = max(0, x0 - px), min(norm.shape[1] - 1, x1 + px)
    return norm[y0:y1 + 1, x0:x1 + 1]


def shift_2d(arr, sy, sx):
    out = np.zeros_like(arr)
    h, w = arr.shape
    out[max(0, sy):min(h, h + sy), max(0, sx):min(w, w + sx)] = \
        arr[max(0, -sy):min(h, h - sy), max(0, -sx):min(w, w - sx)]
    return out


def center_by_mass(c):
    total = c.sum()
    if total <= 0:
        return c
    yy, xx = np.mgrid[0:8, 0:8]
    cy, cx = (yy * c).sum() / total, (xx * c).sum() / total
    return shift_2d(c, int(round(3.5 - cy)), int(round(3.5 - cx)))


def dilate3(a):
    return np.maximum.reduce([shift_2d(a, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


def erode3(a):
    return np.minimum.reduce([shift_2d(a, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


def smart_preprocess_to_8x8(gray):
    cropped = crop_to_ink(gray)
    if cropped is None:
        return np.zeros((8, 8))
    ch, cw = cropped.shape
    scale = 7.0 / max(ch, cw)
    nh = int(np.clip(int(round(ch * scale)), 3.2, 8))
    nw = int(np.clip(int(round(cw * scale)), 3.2, 8))
    resized = np.array(Image.fromarray(cropped.astype(np.uint8)).resize((nw, nh), Image.LANCZOS)).astype(np.float64)
    canvas = np.zeros((8, 8))
    oy, ox = (8 - nh) // 2, (8 - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    canvas = center_by_mass(canvas)
    return canvas / canvas.max() * 16.0 if canvas.max() > 0 else canvas


def tta_variants(pixels_8x8):
    """11 variants: the original, 8 one-pixel shifts, one thicker and one thinner copy."""
    base = np.clip(pixels_8x8, 0, 16)
    variants = [base]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                variants.append(shift_2d(base, dy, dx))
    variants.append(np.clip(dilate3(base), 0, 16))
    variants.append(np.clip(erode3(base), 0, 16))
    return variants

