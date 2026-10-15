"""Seven-segment scale display decoding."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import BalanceConfig

# Segment order: top, upper right, lower right, bottom, lower left, upper left, middle.
_PATTERNS = {
    (1, 1, 1, 1, 1, 1, 0): "0", (0, 1, 1, 0, 0, 0, 0): "1", (1, 1, 0, 1, 1, 0, 1): "2",
    (1, 1, 1, 1, 0, 0, 1): "3", (0, 1, 1, 0, 0, 1, 1): "4", (1, 0, 1, 1, 0, 1, 1): "5",
    (1, 0, 1, 1, 1, 1, 1): "6", (1, 1, 1, 0, 0, 0, 0): "7", (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 1, 0, 1, 1): "9", (1, 1, 1, 0, 0, 1, 1): "9", (0, 0, 1, 1, 1, 1, 1): "6",
}


@dataclass
class BalanceReading:
    value: float
    unit: str
    text: str
    confidence: float
    flags: list[str]
    lcd_box: tuple[int, int, int, int]

    @property
    def weight_g(self) -> float | None:
        factors = {"g": 1.0, "oz": 28.349523125, "lb": 453.59237, "ct": 0.2}
        if self.unit not in factors or not np.isfinite(self.value) or self.value < 0:
            return None
        return float(self.value * factors[self.unit])

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit, "text": self.text,
                "confidence": round(self.confidence, 3), "flags": self.flags,
                "weight_g": self.weight_g,
                "unit_source": "unknown" if self.unit == "unknown" else "operator"}


def _find_lcd(bgr: np.ndarray):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 85) & (h <= 112) & (s >= 110) & (v >= 120)).astype(np.uint8) * 255
    k = max(9, int(0.012 * max(bgr.shape[:2])))                 # Bridge digit gaps in the panel mask.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, w, hh = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if a < 0.004 * bgr.shape[0] * bgr.shape[1] or not (1.4 <= w / hh <= 7):
            continue
        if a / float(w * hh) < 0.75:
            continue
        if best is None or a > best[4]:
            best = (x, y, w, hh, a)
    return None if best is None else best[:4]


def _deskew(binary: np.ndarray) -> np.ndarray:
    """Remove digit slant by maximizing column-projection sharpness."""
    h, w = binary.shape
    best, best_k = -1.0, 0.0
    for k in np.linspace(-0.35, 0.35, 29):
        M = np.float32([[1, k, -k * h / 2], [0, 1, 0]])
        sh = cv2.warpAffine(binary, M, (w + int(abs(k) * h) + 2, h))
        proj = sh.sum(axis=0).astype(np.float64)
        sc = float((proj ** 2).sum())
        if sc > best:
            best, best_k = sc, float(k)
    M = np.float32([[1, best_k, -best_k * h / 2], [0, 1, 0]])
    return cv2.warpAffine(binary, M, (w + int(abs(best_k) * h) + 2, h))


def _segments_on(d: np.ndarray):
    h, w = d.shape
    def frac(x0, x1, y0, y1):
        r = d[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1), int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]
        return float((r > 0).mean()) if r.size else 0.0
    f = [frac(.22, .78, 0.00, .17), frac(.65, 1.0, .08, .46), frac(.65, 1.0, .54, .92),
         frac(.22, .78, .83, 1.0), frac(0.0, .35, .54, .92), frac(0.0, .35, .08, .46),
         frac(.22, .78, .41, .59)]
    return f


def read_balance(bgr: np.ndarray, cfg: BalanceConfig | None = None) -> BalanceReading | None:
    cfg = cfg or BalanceConfig()
    box = _find_lcd(bgr)
    if box is None:
        return None
    x, y, w, h = box
    crop = bgr[y:y + h, x:x + w]
    # Threshold overlapping horizontal display bands.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.addWeighted(255 - hsv[..., 1], 0.7, hsv[..., 2], 0.3, 0)
    white = cv2.GaussianBlur(white, (0, 0), max(1.0, h / 180))
    binary = np.zeros_like(white)
    nb = 4
    for i in range(nb):
        r0, r1 = int(i * h / nb), int((i + 1) * h / nb)
        lo, hi = max(0, r0 - h // 8), min(h, r1 + h // 8)
        _, t = cv2.threshold(white[lo:hi], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary[r0:r1] = t[r0 - lo:r1 - lo]
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    sk = _deskew(binary)
    # Split digits by column projection.
    col = (sk > 0).sum(axis=0)
    ink = col > 0
    spans, start = [], None
    for i, on_ in enumerate(np.r_[ink, False]):
        if on_ and start is None:
            start = i
        elif not on_ and start is not None:
            spans.append([start, i - 1])
            start = None
    merged: list[list[int]] = []
    for sp in spans:                                   # Bridge narrow gaps within each digit.
        if merged and sp[0] - merged[-1][1] <= 0.03 * h:
            merged[-1][1] = sp[1]
        else:
            merged.append(sp)
    items = []
    for x0, x1 in merged:
        ys = np.flatnonzero((sk[:, x0:x1 + 1] > 0).any(axis=1))
        if not ys.size:
            continue
        top, hgt = int(ys.min()), int(ys.max() - ys.min() + 1)
        touches_frame = (x0 == 0) or (x1 >= sk.shape[1] - 2) or top <= 1 or top + hgt >= h - 1 or hgt >= 0.97 * h
        if not touches_frame:
            items.append((x0, top, x1 - x0 + 1, hgt))
    if not items:
        return None
    ref_h = max(it[3] for it in items)
    if ref_h < cfg.min_digit_height_px or ref_h < 0.45 * h:
        return None
    digits = [it for it in items if it[3] >= 0.7 * ref_h]
    comps = [it for it in items if it[3] < 0.7 * ref_h]
    text, flags, margins = "", ["weight_unit_unconfirmed" if cfg.unit == "unknown" else "weight_unit_operator_supplied"], []
    last_x1 = None
    for bx, by, bw, bh in digits:
        sub = (sk[by:by + bh, bx:bx + bw] > 0).astype(np.uint8) * 255
        if bw < 0.22 * bh:                                   # Decode narrow right-hand segments as 1.
            ch, marg = "1", 0.3
        else:
            f = _segments_on(sub)
            on = tuple(int(v > 0.42) for v in f)
            ch = _PATTERNS.get(on)
            marg = min(abs(v - 0.42) for v in f)
            if ch is None:
                ch, marg = "?", 0.0
        if last_x1 is not None:                              # Insert decimal points from small baseline components.
            for px, py, pw, ph in comps:
                if last_x1 - 2 <= px <= bx and ph < 0.25 * ref_h and py > by + 0.7 * bh:
                    text += "."
        text += ch
        margins.append(marg)
        last_x1 = bx + bw
    if not text or "?" in text:
        return None if not text else BalanceReading(float("nan"), cfg.unit, text, 0.1, flags + ["unreadable_digit"], box)
    try:
        value = float(text)
    except ValueError:
        return None
    conf = float(np.clip(0.55 + 1.2 * min(margins), 0.0, 0.95))
    if len(text.replace(".", "")) < 2:
        flags.append("single_digit_reading")
        conf *= 0.7
    return BalanceReading(value=value, unit=cfg.unit, text=text, confidence=conf, flags=flags, lcd_box=box)
