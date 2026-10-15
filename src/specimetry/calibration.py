"""Ruler detection and pixel-to-millimeter calibration."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from .config import RulerConfig
from .imaging import rotate_bound

MM_PER_INCH = 25.4


@dataclass
class RulerCalibration:
    px_per_mm: float
    rel_uncertainty: float
    angle_deg: float
    M: np.ndarray                      # Working-image to rectified-frame transform.
    rect_shape: tuple[int, int]
    box: tuple[int, int, int, int]     # Ruler bounds in rectified pixels: x0, y0, x1, y1.
    units: str                         # Detected or assumed ruler units.
    bands: list[dict]                  # Tick bands ordered from top to bottom.
    periodicity: float
    n_ticks: int
    confidence: float
    flags: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def scale_near(self, y_rect: float) -> tuple[float, float]:
        """Return the nearest tick band scale and relative uncertainty."""
        b = min(self.bands, key=lambda d: abs(d["y"] - y_rect))
        unc = self.rel_uncertainty
        if len(self.bands) == 2:
            a, c = self.bands[0]["px_per_mm"], self.bands[1]["px_per_mm"]
            unc = float(np.hypot(unc, 0.5 * abs(a / c - 1.0)))
        return float(b["px_per_mm"]), unc

    def to_dict(self) -> dict:
        return {
            "px_per_mm": round(float(self.px_per_mm), 4),
            "bands": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()} for b in self.bands],
            "rel_uncertainty": round(float(self.rel_uncertainty), 4),
            "angle_deg": round(float(self.angle_deg), 2),
            "box": [int(v) for v in self.box],
            "rectified_dimensions": [int(self.rect_shape[1]), int(self.rect_shape[0])],
            "working_to_rectified": self.M.tolist(),
            "units": self.units,
            "periodicity": round(float(self.periodicity), 3),
            "n_ticks": int(self.n_ticks),
            "confidence": round(float(self.confidence), 3),
            "flags": list(self.flags),
            "detail": self.detail,
        }



def _thin_strokes(gray: np.ndarray, k: int, thr: float) -> np.ndarray:
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return (bh > thr).astype(np.uint8) * 255


def _orientation_score(bw: np.ndarray, angle: float, vlen: int, box: tuple[int, int]) -> float:
    rot, _ = rotate_bound(bw, angle, border=cv2.BORDER_CONSTANT)
    v = cv2.morphologyEx(rot, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen)))
    dens = cv2.boxFilter((v > 0).astype(np.float32), -1, box, normalize=True)
    return float(dens.max())


def _hough_angle_candidates(gray: np.ndarray, n_cand: int = 3) -> list[float]:
    """Estimate tick orientations from a length-weighted Hough histogram."""
    s = 1600 / max(gray.shape)
    sm = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bw = _thin_strokes(sm, 9, 22)
    lines = cv2.HoughLinesP(bw, 1, np.pi / 360, threshold=12, minLineLength=14, maxLineGap=2)
    if lines is None:
        return []
    L = np.asarray(lines, dtype=float).reshape(-1, 4)
    dx, dy = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
    phi = np.degrees(np.arctan2(dy, dx)) % 180.0
    hist = np.bincount(phi.astype(int) % 180, weights=np.hypot(dx, dy), minlength=180)
    sm_h = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, "valid")
    out: list[float] = []
    for i in np.argsort(-sm_h):
        alpha = ((float(i) - 90.0 + 90.0) % 180.0) - 90.0       # Rotate the tick direction to vertical.
        if all(min(abs(alpha - o), 180 - abs(alpha - o)) >= 9 for o in out):
            out.append(alpha)
        if len(out) == n_cand:
            break
    return out


def _angle_candidates(gray: np.ndarray, cfg: RulerConfig, n_cand: int = 4) -> list[float]:
    """Refine ruler angles with a coarse-to-fine density search."""
    s = cfg.angle_search_px / max(gray.shape)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bw = _thin_strokes(small, 7, 22)
    box = (max(25, int(small.shape[1] * 0.12)), 7)
    angs = np.arange(-90, 90, cfg.angle_step_coarse)
    scores = np.array([_orientation_score(bw, float(a), 7, box) for a in angs])
    order = np.argsort(-scores)
    picked: list[float] = []
    for i in order:
        if all(min(abs(angs[i] - p), 180 - abs(angs[i] - p)) >= 9 for p in picked):
            picked.append(float(angs[i]))
        if len(picked) == n_cand:
            break
    s2 = 1400 / max(gray.shape)
    mid = cv2.resize(gray, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
    bw2 = _thin_strokes(mid, 9, 22)
    box2 = (max(40, int(mid.shape[1] * 0.12)), 9)
    out = []
    for a0 in picked:
        grid = np.arange(a0 - cfg.angle_step_coarse, a0 + cfg.angle_step_coarse + 1e-6, cfg.angle_step_fine)
        sc = [_orientation_score(bw2, float(a), 9, box2) for a in grid]
        out.append(float(grid[int(np.argmax(sc))]))
    return out


def _analyse(img_bgr: np.ndarray, angle: float, cfg: RulerConfig):
    """Rotate the image and compute periodicity across candidate tick bands."""
    rect, M = rotate_bound(img_bgr, angle)
    g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))).astype(np.float32)
    v = cv2.morphologyEx((bh > 22).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))
    cnt = uniform_filter1d(v.sum(axis=1).astype(np.float32), 9)
    rows = np.flatnonzero(cnt > 0.15 * cnt.max()) if cnt.max() > 0 else np.array([], dtype=int)
    if rows.size > 900:
        rows = np.flatnonzero(cnt > np.sort(cnt)[-900])
    P = np.zeros(g.shape[0], dtype=np.float32)
    lag = np.zeros(g.shape[0], dtype=int)
    if rows.size:
        p, l = _row_periodicity(bh[rows], cfg.lag_range_px)
        P[rows], lag[rows] = p, l
    Ps = uniform_filter1d(P, 5)
    return rect, M, g, bh, P, lag, Ps



def _row_periodicity(bh: np.ndarray, lag_range: tuple[int, int]):
    """Return row periodicity strengths and dominant tick lags."""
    H, W = bh.shape
    L1, L2 = lag_range
    L2 = min(L2, (W - 1) // 3)
    if L2 < L1:
        return np.zeros(H, np.float32), np.zeros(H, int)
    x = np.minimum(bh, np.percentile(bh, 99.5))
    x = x - gaussian_filter1d(x, 9, axis=1, mode="nearest")
    n = 1 << int(np.ceil(np.log2(2 * W)))
    F = np.fft.rfft(x, n=n, axis=1)
    acf = np.fft.irfft(F * np.conj(F), n=n, axis=1)[:, : 3 * L2 + 2]
    acf0 = np.maximum(acf[:, 0], 1e-6)
    overlap = (W - np.arange(acf.shape[1])).astype(np.float32) / W
    r = acf / acf0[:, None] / overlap[None, :]
    lags = np.arange(L1, L2 + 1)
    score = (r[:, lags] + r[:, 2 * lags] + r[:, 3 * lags]) / 3.0
    # Select local autocorrelation maxima.
    peak = (r[:, lags] > r[:, lags - 1]) & (r[:, lags] > r[:, lags + 1])
    score = np.where(peak, score, -1)
    best = score.argmax(axis=1)
    strength = np.take_along_axis(score, best[:, None], axis=1)[:, 0]
    strength = np.clip(strength, 0, 1)
    weak = acf[:, 0] < 0.05 * np.median(acf[:, 0])
    strength[weak] = 0
    return strength.astype(np.float32), (best + L1).astype(int)


def _segments(mask: np.ndarray, merge_gap: int = 12, min_len: int = 6):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    segs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap:
            segs.append((start, prev))
            start = i
        prev = i
    segs.append((start, prev))
    return [(a, b) for a, b in segs if b - a + 1 >= min_len]



def _fit_ticks(signal: np.ndarray, lag_est: float, min_ticks: int):
    """Fit tick positions to a linear spacing model."""
    s = gaussian_filter1d(signal.astype(np.float64), 0.8)
    base = np.median(s)
    spread = np.percentile(s, 97) - base
    if spread <= 1e-6:
        return None
    peaks, _ = find_peaks(s, prominence=0.22 * spread, distance=max(2, int(0.55 * lag_est)))
    if len(peaks) < min_ticks:
        return None
    # Refine peak positions with a parabola.
    xs = []
    for p in peaks:
        if 0 < p < len(s) - 1:
            a, b, c = s[p - 1], s[p], s[p + 1]
            d = (a - c) / (2 * (a - 2 * b + c) + 1e-12)
            xs.append(p + float(np.clip(d, -0.5, 0.5)))
        else:
            xs.append(float(p))
    xs = np.asarray(xs)
    d = np.diff(xs)
    ok = (d > 0.6 * lag_est) & (d < 1.5 * lag_est)
    if ok.sum() < min_ticks - 1:
        return None
    p_cur = float(np.median(d[ok]))
    # Assign integer tick indices from successive peak gaps.
    idx, keep = [0], [0]
    for i in range(1, len(xs)):
        steps = (xs[i] - xs[keep[-1]]) / p_cur
        n = round(steps)
        if n >= 1 and abs(steps - n) < 0.22:
            idx.append(idx[-1] + n)
            keep.append(i)
    xs_k, n_k = xs[keep], np.asarray(idx, dtype=float)
    if len(xs_k) < min_ticks:
        return None
    for _ in range(3):
        A = np.vstack([n_k, np.ones_like(n_k)]).T
        (p, x0), *_ = np.linalg.lstsq(A, xs_k, rcond=None)
        res = xs_k - (p * n_k + x0)
        good = np.abs(res) < max(0.18 * p, 3 * res.std())
        if good.all() or good.sum() < min_ticks:
            break
        xs_k, n_k = xs_k[good], n_k[good]
    A = np.vstack([n_k, np.ones_like(n_k)]).T
    (p, x0), *_ = np.linalg.lstsq(A, xs_k, rcond=None)
    res = xs_k - (p * n_k + x0)
    cov_sigma = res.std(ddof=2) / (np.sqrt(((n_k - n_k.mean()) ** 2).sum()) + 1e-9) if len(xs_k) > 3 else p
    half = len(xs_k) // 2
    p_l = np.polyfit(n_k[:half], xs_k[:half], 1)[0] if half >= 4 else p
    p_r = np.polyfit(n_k[half:], xs_k[half:], 1)[0] if len(xs_k) - half >= 4 else p
    return {"pitch": float(p), "xs": xs_k, "n_idx": n_k, "x_first": float(xs_k.min()), "x_last": float(xs_k.max()),
            "n": int(len(xs_k)), "fit_rel_err": float(cov_sigma / p),
            "halves_rel_diff": float(abs(p_l - p_r) / p / 2), "resid_px": float(res.std())}


def _valid_columns(orig_shape, M, rect_shape, rows):
    """Return columns containing source-image pixels across the selected rows."""
    ones = np.full(orig_shape[:2], 255, np.uint8)
    valid = cv2.warpAffine(ones, M, (rect_shape[1], rect_shape[0]), flags=cv2.INTER_NEAREST)
    y0, y1 = max(0, rows[0]), min(rect_shape[0] - 1, rows[1])
    return valid[y0:y1 + 1].mean(axis=0) > 200


def _ruler_extent(rect_bgr, rows, tick_x, valid):
    """Estimate the ruler extent from color continuity."""
    y0, y1 = max(0, rows[0]), min(rect_bgr.shape[0] - 1, rows[1])
    lab = cv2.cvtColor(rect_bgr[y0:y1 + 1], cv2.COLOR_BGR2LAB).astype(np.float32)
    col = np.median(lab, axis=0)                                   # Per-column Lab color.
    ref = np.median(col[max(0, tick_x[0]):tick_x[1] + 1], axis=0)
    dist = np.linalg.norm(col - ref, axis=1)
    dist = uniform_filter1d(dist, 9)
    ok = (dist < 28) & valid
    cx = int(np.clip((tick_x[0] + tick_x[1]) // 2, 0, len(ok) - 1))
    if not ok[cx]:
        return tick_x
    l = cx
    while l > 0 and ok[l - 1]:
        l -= 1
    r = cx
    while r < len(ok) - 1 and ok[r + 1]:
        r += 1
    return min(l, tick_x[0]), max(r, tick_x[1])



def _band_groups(Ps, lag, cfg):
    segs = _segments(Ps > max(cfg.min_periodicity, 0.55 * Ps.max()))
    info = []
    for a, b in segs:
        rows = np.arange(a, b + 1)
        pk = Ps[rows].max()
        info.append({"rows": (int(a), int(b)), "peak": float(pk),
                     "lag": float(np.median(lag[rows[Ps[rows] > 0.8 * pk]]))})
    # Merge adjacent fragments with similar tick lags.
    info.sort(key=lambda d: d["rows"][0])
    merged: list[dict] = []
    for d in info:
        if merged and d["rows"][0] - merged[-1]["rows"][1] < 60 and abs(d["lag"] / merged[-1]["lag"] - 1) < 0.15:
            m = merged[-1]
            m["rows"] = (m["rows"][0], d["rows"][1])
            m["peak"] = max(m["peak"], d["peak"])
        else:
            merged.append(dict(d))
    return merged


def detect_ruler(img_bgr: np.ndarray, cfg: RulerConfig | None = None) -> RulerCalibration | None:
    cfg = cfg or RulerConfig()
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("ruler detection expects a BGR image")
    if min(img_bgr.shape[:2]) < 48:
        return None
    gray0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    def quality(res) -> float:
        Ps_, lag_ = res[-1], res[-2]
        if Ps_.max() < cfg.min_periodicity:
            return float(Ps_.max())
        nb = len([b for b in _band_groups(Ps_, lag_, cfg) if b["peak"] > 0.55 * Ps_.max()])
        good_rows = min(200, int((Ps_ > 0.6 * Ps_.max()).sum()))
        return float(Ps_.max() + 0.06 * min(2, nb) + 0.0004 * good_rows)

    best = None
    cands = _hough_angle_candidates(gray0) or _angle_candidates(gray0, cfg)
    for ang in cands:
        res = _analyse(img_bgr, ang, cfg)
        sc = quality(res)
        if best is None or sc > best[0]:
            best = (sc, ang, res)
    sc, angle, _ = best
    # Refine orientation using tick periodicity.
    for half, step in ((5.0, 1.0), (1.0, 0.25)):
        centre = best[1]
        for ang in np.arange(centre - half, centre + half + 1e-6, step):
            if abs(ang - centre) < 1e-6:
                continue
            res = _analyse(img_bgr, float(ang), cfg)
            s2 = quality(res)
            if s2 > best[0] + 1e-4:
                best = (s2, float(ang), res)
    _, angle, (rect, M, g, bh, P, lag, Ps) = best
    if Ps.max() < cfg.min_periodicity:
        return None
    bands = _band_groups(Ps, lag, cfg)
    if not bands:
        return None
    bands.sort(key=lambda d: -d["peak"])
    keep = [bands[0]]
    for s_ in bands[1:]:
        far = min(abs(s_["rows"][0] - b["rows"][1]) for b in keep) > 30
        if far and s_["peak"] > max(cfg.min_periodicity, 0.35 * keep[0]["peak"]):
            keep.append(s_)
        if len(keep) == 2:
            break
    bands = keep
    fits = []
    for b in bands:
        a0, b0 = b["rows"]
        rr = np.arange(a0, b0 + 1)
        # Average rows containing the densest tick coverage.
        ink = bh[rr].sum(axis=1)
        dense = rr[ink >= 0.75 * ink.max()]
        sig = bh[dense].mean(axis=0)
        best_fit = None
        for k in (1, 2, 3, 4, 5, 8, 10, 16):
            lag_k = b["lag"] / k
            if lag_k < cfg.lag_range_px[0] * 0.8:
                continue
            f = _fit_ticks(sig, lag_k, cfg.min_ticks)
            if f is None or abs(f["pitch"] / lag_k - 1) > 0.12 or f["resid_px"] > 0.15 * f["pitch"]:
                continue
            # Select the fit containing the most ticks.
            if best_fit is None or f["n"] > best_fit["n"]:
                best_fit = f
        if best_fit is None:                                         # Use the periodic-row average for the fallback fit.
            best_fit = _fit_ticks(bh[rr[Ps[rr] > 0.8 * b["peak"]]].mean(axis=0), b["lag"], cfg.min_ticks)
        if best_fit:
            best_fit["rows"], best_fit["peak"] = b["rows"], b["peak"]
            fits.append(best_fit)
    if not fits:
        return None
    flags: list[str] = []
    fits.sort(key=lambda f: f["rows"][0])          # Order tick bands from top to bottom.
    kinds = ["mm"] * len(fits)
    units = None
    mm_step = 1.0
    inch_step = MM_PER_INCH / cfg.inch_subdivision
    if len(fits) == 2:
        p_a, p_b = fits[0]["pitch"], fits[1]["pitch"]
        ratio = max(p_a, p_b) / min(p_a, p_b)
        candidates = []
        for mm_i in (0, 1):
            for step in (1.0, 2.0, 5.0, 10.0):
                mm_scale = fits[mm_i]["pitch"] / step
                in_scale = fits[1 - mm_i]["pitch"] / inch_step
                error = abs(np.log(mm_scale / in_scale))
                candidates.append((error, step, mm_i))
        err, mm_step, mm_i = min(candidates, key=lambda c: c[0] + .003 * np.log(c[1]))
        if err <= np.log1p(cfg.ratio_tolerance):
            units = "mm+in"
        else:
            err, mm_step, mm_i = min((c for c in candidates if c[1] == 1), key=lambda c: c[0])
            if err <= -np.log(.8):
                units = "mm+in"
                flags.append("scale_gradient_between_ruler_edges")
            else:
                flags.append(f"scale_ratio_mismatch_{ratio:.3f}")
                fits = [max(fits, key=lambda f: f["n"])]
                kinds = ["mm"]
                mm_step = 1.0
        if units == "mm+in":
            kinds = ["mm" if i == mm_i else "in" for i in range(2)]
            flags.append(f"inch_scale_subdivision_{cfg.inch_subdivision}_assumed")
            if mm_step != 1:
                flags.append(f"metric_ticks_every_{mm_step:g}_mm_inferred")
    if units is None:
        if cfg.default_units == "mm":
            units = "mm (assumed)"
            flags.append("single_scale_unit_assumed_mm")
        else:
            units = "in (assumed)"
            kinds = ["in"]
            flags.append("single_scale_unit_assumed_inch")
    band_scales = []
    for f, k in zip(fits, kinds):
        sc_ = f["pitch"] / (mm_step if k == "mm" else inch_step)
        band_scales.append({"y": 0.5 * (f["rows"][0] + f["rows"][1]), "px_per_mm": float(sc_), "kind": k,
                            "tick_step_mm": mm_step if k == "mm" else inch_step})
    mm_fit = fits[kinds.index("mm")] if "mm" in kinds else fits[0]
    p_mm = float(np.mean([b["px_per_mm"] for b in band_scales]))
    px_per_mm = p_mm
    if not (cfg.px_per_mm_min <= px_per_mm <= cfg.px_per_mm_max):
        flags.append("implausible_scale")
    rel = float(np.hypot(mm_fit["fit_rel_err"], mm_fit["halves_rel_diff"]) + 0.002)
    # Expand the outer tick rows to estimate the ruler body.
    y0 = min(f["rows"][0] for f in fits) - int(1.5 * p_mm)
    y1 = max(f["rows"][1] for f in fits) + int(1.5 * p_mm)
    if len(fits) == 1:                              # Extend a single-edge ruler box by its estimated width.
        y1 = max(y1, y0 + int(25 * px_per_mm * 1.0))
    Hh, Ww = g.shape
    tx0 = int(min(f["x_first"] for f in fits))
    tx1 = int(max(f["x_last"] for f in fits))
    x0, x1 = _ruler_extent(rect, (y0, y1), (tx0, tx1), valid=_valid_columns(img_bgr.shape, M, (Hh, Ww), (y0, y1)))
    box = (max(0, x0), max(0, y0), min(Ww - 1, x1), min(Hh - 1, y1))
    conf = float(np.clip(0.4 * min(1.0, mm_fit["n"] / 40) + 0.35 * min(1.0, bands[0]["peak"] / 0.7)
                         + (0.25 if units == "mm+in" else 0.0) - 0.4 * (rel > 0.02), 0.0, 1.0))
    return RulerCalibration(px_per_mm=px_per_mm, rel_uncertainty=rel, angle_deg=angle, M=M,
                            rect_shape=g.shape, box=box, units=units, bands=band_scales,
                            periodicity=bands[0]["peak"],
                            n_ticks=mm_fit["n"], confidence=conf, flags=flags,
                            detail={"fits": [{k: (round(v, 4) if isinstance(v, float) else v)
                                              for k, v in f.items() if k not in ("xs", "n_idx")} for f in fits]})
