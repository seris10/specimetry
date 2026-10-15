"""Specimen appearance descriptors."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


@dataclass
class Descriptor:
    mean_lab: np.ndarray
    std_lab: np.ndarray
    lab_hist: np.ndarray
    chroma_hist: np.ndarray            # Lab chroma histogram.
    lbp_hist: np.ndarray
    kp_xy: np.ndarray                 # Working-frame keypoint coordinates, shape N x 2.
    kp_desc: np.ndarray               # SIFT descriptors, shape N x 128.
    source: str                       # Descriptor region: mask or scene.
    embedding: np.ndarray | None = None
    n_pixels: int = 0
    notes: list[str] = field(default_factory=list)


_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.02)
    return _SIFT


def compute_descriptor(bgr: np.ndarray, mask: np.ndarray, source: str = "mask",
                       embed_fn=None) -> Descriptor | None:
    ys, xs = np.where(mask)
    if ys.size < 500:
        return None
    s = min(1.0, 1400 / max(bgr.shape[:2]))
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    mk = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    core = cv2.erode(mk, np.ones((7, 7), np.uint8))                  # Exclude boundary pixels from color statistics.
    if core.sum() < 300:
        core = mk
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[core > 0]
    mean, std = px.mean(axis=0), px.std(axis=0)
    # Use 6 x 5 x 5 bins for the Lab histogram.
    h = cv2.calcHist([lab], [0, 1, 2], core, [6, 5, 5], [0, 256, 0, 256, 0, 256]).flatten()
    h = h / max(h.sum(), 1e-9)
    # Build the lightness-independent chroma histogram.
    ch = cv2.calcHist([lab], [1, 2], core, [24, 24], [104, 168, 104, 168]).flatten()
    ch = cv2.GaussianBlur(ch.reshape(24, 24), (0, 0), 1.0).flatten()
    ch = ch / max(ch.sum(), 1e-9)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    lh = np.bincount(lbp[core > 0].astype(int), minlength=10)[:10].astype(np.float64)
    lh /= max(lh.sum(), 1e-9)
    kps, des = _sift().detectAndCompute(gray, core)
    if des is None or len(kps) < 8:
        xy, des = np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    else:
        xy = np.array([k.pt for k in kps], np.float32)
    emb = None
    if embed_fn is not None:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        emb = embed_fn(bgr[y0:y1 + 1, x0:x1 + 1])
    return Descriptor(mean, std, h.astype(np.float32), ch.astype(np.float32), lh.astype(np.float32), xy, des.astype(np.float32),
                      source, emb, int(ys.size))


def sift_match(a: Descriptor, b: Descriptor, min_inliers: int = 12) -> tuple[int, float]:
    """Return SIFT inlier count and ratio after affine RANSAC."""
    if len(a.kp_desc) < min_inliers or len(b.kp_desc) < min_inliers:
        return 0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(a.kp_desc, b.kp_desc, k=2)
    good = [m for m, n in (r for r in raw if len(r) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return 0, 0.0
    pa = a.kp_xy[[m.queryIdx for m in good]]
    pb = b.kp_xy[[m.trainIdx for m in good]]
    # Fit an affine transform between matched keypoints.
    M, inl = cv2.estimateAffine2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=3000)
    if M is None:
        return 0, 0.0
    n = int(inl.sum())
    return n, n / len(good)


def color_affinity(a: Descriptor, b: Descriptor) -> float:
    """Return chroma similarity from Lab means and histogram overlap."""
    dc = float(np.linalg.norm(a.mean_lab[1:] - b.mean_lab[1:]))
    bc = float(np.sum(np.sqrt(a.chroma_hist * b.chroma_hist)))
    return float(0.6 * np.exp(-dc / 6.0) + 0.4 * bc ** 4)


def texture_affinity(a: Descriptor, b: Descriptor) -> float:
    chi = 0.5 * float(np.sum((a.lbp_hist - b.lbp_hist) ** 2 / (a.lbp_hist + b.lbp_hist + 1e-9)))
    return float(np.exp(-60.0 * chi))


def make_dinov2_embedder(model_id: str = "facebook/dinov2-small"):
    """Create a normalized DINOv2 crop-embedding function."""
    import torch
    from transformers import AutoImageProcessor, AutoModel
    proc, model = AutoImageProcessor.from_pretrained(model_id), AutoModel.from_pretrained(model_id).eval()

    def embed(crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            out = model(**proc(images=rgb, return_tensors="pt"))
        v = out.last_hidden_state[:, 0].squeeze(0).numpy()
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)
    return embed
