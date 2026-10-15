"""Ruler calibration, mask geometry, and scale display tests."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from specimetry.calibration import detect_ruler           # noqa: E402
from specimetry.morphometry import measure_mask            # noqa: E402


def make_ruler_image(px_per_mm=12.0, angle=7.0, W=1600, H=1100, inch_subdivision=16, metric_stride=1):
    """Create a rotated dual-scale ruler image."""
    img = np.full((H, W, 3), (70, 80, 90), np.uint8)
    x0, y0, x1, y1 = 120, 650, 1480, 950
    cv2.rectangle(img, (x0, y0), (x1, y1), (150, 200, 235), -1)
    for i in range(0, int((x1 - x0) / px_per_mm)):                      # Millimeter edge.
        if i % metric_stride:
            continue
        x = int(round(x0 + 20 + i * px_per_mm))
        ln = 70 if i % 10 == 0 else 50 if i % 5 == 0 else 32
        if x < x1 - 10:
            cv2.line(img, (x, y1), (x, y1 - ln), (20, 20, 20), 2)
    step = px_per_mm * 25.4 / inch_subdivision
    for i in range(int((x1 - x0) / step)):                              # Inch edge.
        x = int(round(x0 + 20 + i * step))
        ln = (70 if i % inch_subdivision == 0 else 55 if i % (inch_subdivision // 2) == 0
              else 42 if i % (inch_subdivision // 4) == 0 else 30 if i % (inch_subdivision // 8) == 0 else 20)
        if x < x1 - 10:
            cv2.line(img, (x, y0), (x, y0 + ln), (20, 20, 20), 2)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (W, H), borderValue=(70, 80, 90))


def test_ruler_scale_recovered_within_two_percent():
    for scale, ang in [(12.0, 0.0), (12.0, 7.0), (9.0, -12.0), (15.0, 20.0)]:
        cal = detect_ruler(make_ruler_image(scale, ang))
        assert cal is not None
        assert abs(cal.px_per_mm / scale - 1) < 0.02, (scale, ang, cal.px_per_mm)
        assert cal.units == "mm+in"


def test_no_ruler_returns_none():
    rng = np.random.RandomState(0)
    assert detect_ruler(rng.randint(60, 90, (700, 900, 3)).astype(np.uint8)) is None


def test_ruler_one_thirty_second_inch_edge_is_identified():
    from specimetry.config import RulerConfig
    cal = detect_ruler(make_ruler_image(12.0, 0.0, inch_subdivision=32), RulerConfig(inch_subdivision=32))
    assert cal is not None and cal.units == "mm+in"
    assert abs(cal.px_per_mm / 12 - 1) < .02
    assert "inch_scale_subdivision_32_assumed" in cal.flags


def test_ruler_one_eighth_inch_edge_is_identified():
    from specimetry.config import RulerConfig
    cal = detect_ruler(make_ruler_image(12.0, 0.0, inch_subdivision=8), RulerConfig(inch_subdivision=8))
    assert cal is not None and cal.units == "mm+in"
    assert abs(cal.px_per_mm / 12 - 1) < .02


def test_missing_alternate_metric_ticks_does_not_double_the_scale():
    cal = detect_ruler(make_ruler_image(12.0, 0.0, metric_stride=2))
    assert cal is not None and abs(cal.px_per_mm / 12 - 1) < .02
    assert "metric_ticks_every_2_mm_inferred" in cal.flags


def test_measure_mask_ellipse():
    m = np.zeros((600, 800), np.uint8)
    cv2.ellipse(m, (400, 300), (240, 80), 0, 0, 360, 1, -1)            # 480 x 160 pixels.
    r = measure_mask(m > 0, px_per_mm=10.0, rel_unc=0.01)
    assert abs(r["major_mm"] - 48.0) < 0.6 and abs(r["minor_mm"] - 16.0) < 0.6
    assert abs(r["feret_max_mm"] - 48.0) < 0.6 and abs(r["aspect_ratio"] - 3.0) < 0.15
    assert r["shape_class"] == "elongate"


@pytest.mark.parametrize("edge_artifact", [False, True])
def test_balance_reader_roundtrip(edge_artifact):
    from specimetry.balance import read_balance
    seg = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd",
           "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg"}
    def draw(img, ch, x, y, w, h, t=24):
        r = {"a": (x + t, y, w - 2 * t, t), "b": (x + w - t, y + t, t, h // 2 - t), "c": (x + w - t, y + h // 2, t, h // 2 - t),
             "d": (x + t, y + h - t, w - 2 * t, t), "e": (x, y + h // 2, t, h // 2 - t), "f": (x, y + t, t, h // 2 - t),
             "g": (x + t, y + h // 2 - t // 2, w - 2 * t, t)}
        for s in seg[ch]:
            a, b, c, d = r[s]
            cv2.rectangle(img, (a, b), (a + c, b + d), (255, 255, 255), -1)
    img = np.full((900, 1400, 3), (90, 90, 95), np.uint8)
    cv2.rectangle(img, (300, 300), (1100, 650), (230, 170, 20), -1)    # Cyan display backlight in BGR.
    for i, ch in enumerate("273"):
        draw(img, ch, 520 + i * 160, 350, 110, 240)
    if edge_artifact:
        cv2.line(img, (1096, 300), (1096, 530), (255, 255, 255), 3)
    r = read_balance(img)
    assert r is not None and r.text == "273", r
