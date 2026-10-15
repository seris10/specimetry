"""Foreground segmentation and mask scoring."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from .config import SegmentationConfig


@dataclass
class Instance:
    mask: np.ndarray                  # Boolean mask in the rectified frame.
    score: float = 1.0
    flags: list[str] = field(default_factory=list)



class _Backend:
    name = "base"

    def probability(self, rect_bgr: np.ndarray, exclude: np.ndarray, valid: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RembgBackend(_Backend):
    name = "rembg"

    def __init__(self, model: str = "u2net"):
        from rembg import new_session
        self.model = model
        self.session = new_session(model)

    def probability(self, rect_bgr, exclude, valid):
        from rembg import remove
        work = rect_bgr.copy()
        fill = np.median(rect_bgr[valid & ~exclude].reshape(-1, 3), axis=0).astype(np.uint8)
        work[exclude] = fill                      # Replace excluded ruler pixels with background color.
        work[~valid] = fill
        a = remove(cv2.cvtColor(work, cv2.COLOR_BGR2RGB), session=self.session, only_mask=True)
        p = np.asarray(a, dtype=np.float32) / 255.0
        p[exclude | ~valid] = 0
        return p


class ClassicalBackend(_Backend):
    """Segment foreground with a background-color model and GrabCut."""
    name = "classical"

    def probability(self, rect_bgr, exclude, valid):
        from sklearn.mixture import GaussianMixture
        s = 800 / max(rect_bgr.shape[:2])
        sm = cv2.resize(rect_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ex = cv2.resize(exclude.astype(np.uint8), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        va = cv2.resize(valid.astype(np.uint8), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = va.shape
        frame = np.zeros_like(va)
        b = max(6, int(0.06 * min(h, w)))
        frame[:b], frame[-b:], frame[:, :b], frame[:, -b:] = True, True, True, True
        bg_px = lab[frame & va & ~ex]
        if len(bg_px) < 200:
            return np.zeros(rect_bgr.shape[:2], np.float32)
        gmm = GaussianMixture(3, covariance_type="full", random_state=0).fit(bg_px[::7])
        ll = gmm.score_samples(lab.reshape(-1, 3)).reshape(h, w)
        thr = np.percentile(gmm.score_samples(bg_px[::11]), 3)
        mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        mask[ll < thr] = cv2.GC_PR_FGD
        mask[ll < thr - 12] = cv2.GC_FGD
        mask[(~va) | ex | frame] = cv2.GC_BGD
        bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
        try:
            cv2.grabCut(sm, mask, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return np.zeros(rect_bgr.shape[:2], np.float32)
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.float32)
        p = cv2.resize(fg, (rect_bgr.shape[1], rect_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        p[exclude | ~valid] = 0
        return p


class Sam3Backend(_Backend):
    """Segment rock instances with the SAM 3 model."""
    name = "sam3"

    def __init__(self, model_id: str = "facebook/sam3", prompt: str = "rock"):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch = torch
        self.prompt = prompt
        self.model = Sam3Model.from_pretrained(model_id, device_map="auto")
        self.processor = Sam3Processor.from_pretrained(model_id)

    def probability(self, rect_bgr, exclude, valid):
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, text=self.prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_instance_segmentation(
            out, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
        p = np.zeros(rect_bgr.shape[:2], np.float32)
        for m, sc in zip(res["masks"], res["scores"]):
            mm = m.cpu().numpy().astype(bool)
            p[mm] = np.maximum(p[mm], float(sc))
        p[exclude | ~valid] = 0
        return p


def make_backend(cfg: SegmentationConfig) -> _Backend:
    name = cfg.backend.lower()
    if name == "rembg":
        try:
            return RembgBackend()
        except ImportError as exc:
            raise ValueError("rembg requires the rembg extra; install '.[rembg]' or use --backend classical") from exc
    if name == "sam3":
        return Sam3Backend(cfg.sam3_model, cfg.sam3_prompt)
    if name == "classical":
        return ClassicalBackend()
    raise ValueError(f"unknown segmentation backend: {cfg.backend}")



def split_instances(prob: np.ndarray, cfg: SegmentationConfig, px_per_mm: float | None,
                    multi: bool) -> list[Instance]:
    """Split a foreground probability map into specimen masks."""
    H, W = prob.shape
    # Apply hysteresis to the foreground probability map.
    fg = apply_hysteresis_threshold(prob, 0.25, 0.6).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = ndi.binary_fill_holes(fg).astype(np.uint8)
    min_area = cfg.min_area_frac * H * W
    if not multi:
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
        out = []
        for i in range(1, n):
            if min_area <= st[i][4] <= cfg.max_area_frac * H * W:
                out.append(Instance(mask=lab == i, score=float(prob[lab == i].mean())))
        return sorted(out, key=lambda ins: -ins.mask.sum())
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (0, 0), 3)
    hmax = h_maxima(dist, max(2.0, 0.12 * float(dist.max())))
    markers, _ = ndi.label(hmax)
    lab = watershed(-dist, markers, mask=fg.astype(bool))
    out = []
    for rp in regionprops(lab):
        if rp.area < max(min_area * 0.15, 400):
            continue
        m = lab == rp.label
        out.append(Instance(mask=m, score=float(prob[m].mean())))
    return out


def mask_quality(mask: np.ndarray, valid: np.ndarray, ruler_box: tuple[int, int, int, int] | None,
                 px_per_mm: float | None) -> tuple[float, list[str], dict]:
    """Score mask geometry and return quality flags and statistics."""
    flags: list[str] = []
    H, W = mask.shape
    area = float(mask.sum())
    frac = area / (H * W)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, ["empty_mask"], {}
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    solidity = area / max(cv2.contourArea(hull), 1.0)
    ys, xs = np.where(mask)
    edge_touch = (not valid[ys.min(), xs.min()] or ys.min() <= 1 or xs.min() <= 1 or ys.max() >= H - 2 or xs.max() >= W - 2)
    # Detect mask contact with rotated padding.
    inv = ~valid
    near_pad = bool(cv2.dilate(inv.astype(np.uint8), np.ones((5, 5), np.uint8))[mask].any())
    q = 1.0
    if edge_touch or near_pad:
        flags.append("touches_frame_edge")
        q -= 0.35
    if (edge_touch or near_pad) and frac > 0.30:
        # Penalize large masks touching the frame.
        flags.append("large_mask_touching_frame_probably_background")
        q -= 0.3
    if solidity < 0.80:
        flags.append("ragged_or_merged_mask")
        q -= 0.2
    gap_mm = None
    if ruler_box is not None and px_per_mm:
        x0, y0, x1, y1 = ruler_box
        d = np.inf
        sub = mask[:, max(0, x0):x1 + 1]
        rows = np.where(sub.any(axis=1))[0]
        if rows.size:
            above = rows[rows < y0]
            below = rows[rows > y1]
            d = min(y0 - above.max() if above.size else np.inf, below.min() - y1 if below.size else np.inf)
        gap_mm = float(d / px_per_mm) if np.isfinite(d) else None
        if gap_mm is None or gap_mm > 25:
            flags.append("far_from_ruler")
            q -= 0.3
    if frac < 0.004:
        flags.append("tiny_mask")
        q -= 0.3
    return float(np.clip(q, 0.0, 1.0)), flags, {"solidity": float(solidity), "area_frac": float(frac), "gap_to_ruler_mm": gap_mm}


def pile_metrics(rect_bgr: np.ndarray, valid: np.ndarray, ruler_box, px_per_mm: float) -> dict | None:
    """Estimate edge density and foreground coverage above the ruler."""
    from skimage.filters import threshold_otsu
    x0, y0, x1, y1 = ruler_box
    roi = valid.copy()
    roi[max(0, y0 - int(1.0 * px_per_mm)):, :] = False
    if roi.sum() < 10000:
        return None
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 2), 30, 90) > 0
    L = cv2.GaussianBlur(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), 3)
    fg = (L > threshold_otsu(L[roi])) & roi
    return {"edge_density": float(edges[roi].mean()), "fg_fraction": float(fg.sum() / roi.sum()),
            "roi": roi}
