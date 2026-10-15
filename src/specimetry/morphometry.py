"""Calibrated specimen-mask measurements."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import regionprops


def _feret(hull: np.ndarray) -> tuple[float, float, float]:
    """Return maximum and minimum Feret diameters and the maximum diameter angle."""
    pts = hull.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    fmax = float(d[i, j])
    ang = float(np.degrees(np.arctan2(pts[j, 1] - pts[i, 1], pts[j, 0] - pts[i, 0])) % 180)
    fmin = np.inf
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        e = b - a
        n = np.hypot(*e)
        if n < 1e-9:
            continue
        d = pts - a
        w = np.abs(e[0] * d[:, 1] - e[1] * d[:, 0]) / n       # Compute perpendicular distances from each hull edge.
        fmin = min(fmin, float(w.max()))
    return fmax, float(fmin), ang


def measure_mask(mask: np.ndarray, px_per_mm: float, rel_unc: float) -> dict:
    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
        raise ValueError("px_per_mm must be finite and positive")
    if not np.isfinite(rel_unc) or rel_unc < 0:
        raise ValueError("relative uncertainty must be finite and nonnegative")
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("measurement requires a nonempty 2-D mask")
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (labels == largest).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    major_px, minor_px = max(w, h), min(w, h)
    fmax, fmin, fang = _feret(hull)
    area = float(m.sum())
    per = float(cv2.arcLength(c, True))
    rp = regionprops(m)[0]
    s = px_per_mm
    # Combine calibration uncertainty with a 3-pixel span error.
    def unc(px: float) -> float:
        return float(np.hypot(rel_unc, 3.0 / max(px, 1.0)))
    ar = major_px / max(minor_px, 1e-6)
    circ = 4 * np.pi * area / max(per ** 2, 1.0)
    if ar >= 2.2:
        shape = "elongate"
    elif ar < 1.3 and circ > 0.80:
        shape = "equant"
    else:
        shape = "tabular_or_irregular"
    return {
        "major_mm": major_px / s, "minor_mm": minor_px / s,
        "major_rel_unc": unc(major_px), "minor_rel_unc": unc(minor_px),
        "feret_max_mm": fmax / s, "feret_min_mm": fmin / s, "feret_max_angle_deg": fang,
        "area_mm2": area / s ** 2, "perimeter_mm": per / s,
        "aspect_ratio": float(ar), "solidity": float(rp.solidity),
        "circularity": float(min(circ, 1.0)), "eccentricity": float(rp.eccentricity),
        "shape_class": shape, "centroid_px": (float(cx), float(cy)),
    }
