"""Single-photo specimen analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .balance import BalanceReading, read_balance
from .calibration import RulerCalibration, detect_ruler
from .config import Config
from .descriptors import Descriptor, compute_descriptor
from .imaging import PhotoMeta, load_bgr, read_meta, rotate_bound
from .morphometry import measure_mask
from .segmentation import Instance, mask_quality, pile_metrics, split_instances


@dataclass
class PhotoRecord:
    path: Path
    meta: PhotoMeta
    role: str = "other"                    # Photo role: ruler, scale, or other.
    calib: RulerCalibration | None = None
    balance: BalanceReading | None = None
    scene: str = "unknown"                 # Scene type: single, pile, or unknown.
    seg_quality: float = 0.0
    seg_flags: list[str] = field(default_factory=list)
    seg_stats: dict = field(default_factory=dict)
    measure: dict | None = None
    descriptor: Descriptor | None = None
    flags: list[str] = field(default_factory=list)
    n_instances: int = 0
    camera_distance_mm: float | None = None
    overlay: np.ndarray | None = None      # Overlay image for report export.
    working_dimensions: tuple[int, int] | None = None
    source_sha256: str | None = None

    @property
    def measured(self) -> bool:
        return self.measure is not None and self.seg_quality >= 0.55 and "touches_frame_edge" not in self.seg_flags


def _camera_distance_mm(meta: PhotoMeta, px_per_mm: float, image_scale: float = 1.0) -> float | None:
    """Estimate camera distance in millimeters from focal length and image scale."""
    if not meta.focal_35mm:
        return None
    diag = float(np.hypot(meta.width, meta.height))
    f_px = meta.focal_35mm / 43.27 * diag
    return f_px * image_scale / px_per_mm


def _draw_overlay(rect: np.ndarray, inst: list[Instance], calib: RulerCalibration | None, label: str) -> np.ndarray:
    vis = rect.copy()
    rng = np.random.RandomState(3)
    for i in inst:
        vis[i.mask] = (0.5 * vis[i.mask] + 0.5 * rng.randint(70, 255, 3)).astype(np.uint8)
        cnts, _ = cv2.findContours(i.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 255), max(2, rect.shape[1] // 600))
    if calib is not None:
        x0, y0, x1, y1 = calib.box
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 120, 0), max(2, rect.shape[1] // 600))
    sc = 900 / max(vis.shape[:2])
    vis = cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


def process_photo(path: Path, cfg: Config, backend, embed_fn=None) -> PhotoRecord:
    img = load_bgr(path)
    rec = PhotoRecord(path=path, meta=read_meta(path))
    image_scale = min(1.0, cfg.max_image_side / max(img.shape[:2]))
    if image_scale < 1:
        img = cv2.resize(img, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA)
    rec.working_dimensions = (img.shape[1], img.shape[0])
    if cfg.balance.enabled:
        rec.balance = read_balance(img, cfg.balance)
    if rec.balance is None:
        rec.calib = detect_ruler(img, cfg.ruler)
    if rec.calib is not None:
        rec.role = "ruler"
        rect, M = rotate_bound(img, rec.calib.angle_deg)
        valid = cv2.warpAffine(np.full(img.shape[:2], 255, np.uint8), M, (rect.shape[1], rect.shape[0])) > 200
        x0, y0, x1, y1 = rec.calib.box
        pad = int(1.5 * rec.calib.px_per_mm)
        exclude = np.zeros(rect.shape[:2], bool)
        exclude[max(0, y0 - pad): y1 + pad, max(0, x0 - pad): x1 + pad] = True
        rec.flags += rec.calib.flags
    else:
        rect, valid = img, np.ones(img.shape[:2], bool)
        exclude = np.zeros(img.shape[:2], bool)
        if rec.balance is not None:
            rec.role = "scale"
            x, y, w, h = rec.balance.lcd_box
            m = int(0.15 * h)
            exclude[max(0, y - m): y + h + m, max(0, x - m): x + w + m] = True
            rec.flags += rec.balance.flags
    pile = None
    if rec.calib is not None:
        pm = pile_metrics(rect, valid, rec.calib.box, rec.calib.px_per_mm)
        if pm and pm["edge_density"] > cfg.seg.pile_edge_density and pm["fg_fraction"] > cfg.seg.pile_fg_fraction:
            pile = pm
    if pile is not None:
        rec.scene = "pile"
        rec.flags.append("pile_of_small_specimens")
        rec.descriptor = compute_descriptor(rect, pile["roi"], "scene", embed_fn)
        fov = rect.shape[1] / rec.calib.px_per_mm
        rec.seg_stats = {"edge_density": round(pile["edge_density"], 4), "fg_fraction": round(pile["fg_fraction"], 3),
                         "field_of_view_mm": round(float(fov), 1)}
        if cfg.debug_overlays:
            ov = rect.copy(); ov[~pile["roi"]] = (ov[~pile["roi"]] * 0.6).astype(np.uint8)
            rec.overlay = _draw_overlay(ov, [], rec.calib, f"{path.name} [pile] {rec.calib.px_per_mm:.1f}px/mm")
        return rec
    prob = backend.probability(rect, exclude, valid)
    inst_all = split_instances(prob, cfg.seg, rec.calib.px_per_mm if rec.calib else None, multi=False)
    rec.n_instances = len(inst_all)
    chosen: list[Instance] = []
    if inst_all:
        px = rec.calib.px_per_mm if rec.calib else None
        box = rec.calib.box if rec.calib else None
        best, best_q = None, -1.0
        for ins in inst_all[:4]:
            q, fl, st = mask_quality(ins.mask, valid, box, px)
            sc = q * np.sqrt(ins.mask.sum())
            if sc > best_q:
                best, best_q, best_pack = ins, sc, (q, fl, st)
        if best is not None:
            rec.seg_quality, rec.seg_flags, rec.seg_stats = best_pack
            chosen = [best]
    # Flag multiple components of comparable size.
    if len(inst_all) >= 3:
        a = sorted([i.mask.sum() for i in inst_all], reverse=True)
        if a[1] > 0.3 * a[0]:
            rec.seg_flags.append("multiple_comparable_objects")
            rec.seg_quality = min(rec.seg_quality, 0.45)
    if (rec.calib is not None and chosen and rec.seg_quality >= 0.55 and 'tiny_mask' not in rec.seg_flags
            and 'implausible_scale' not in rec.calib.flags):
        rec.scene = "single"
        if backend.name == "classical":
            rec.flags.append("classical_segmentation_requires_review")
        scale, unc = rec.calib.scale_near(float(np.mean(np.where(chosen[0].mask.any(axis=1))[0])))
        rec.measure = measure_mask(chosen[0].mask, scale, unc)
        rec.measure["scale_px_per_mm"] = scale
        if any(f.startswith("single_scale_unit_assumed") for f in rec.calib.flags):
            rec.flags.append("measurement_provisional_unit_unconfirmed")
            # Retain the alternative inch-edge scale factor.
            rec.measure["unit_hypothesis"] = cfg.ruler.default_units
            rec.measure["alt_if_inch_scale"] = 25.4 / cfg.ruler.inch_subdivision
        if "touches_frame_edge" in rec.seg_flags:
            rec.flags.append("dimensions_are_lower_bounds_truncated_by_frame")
        rec.camera_distance_mm = _camera_distance_mm(rec.meta, scale, image_scale)
    elif rec.calib is not None:
        rec.flags.append("specimen_segmentation_unreliable")
    if chosen:
        # Select mask or scene pixels for grouping descriptors.
        if rec.seg_quality >= 0.55 or rec.role == "scale":
            rec.descriptor = compute_descriptor(rect, chosen[0].mask, "mask", embed_fn)
        else:
            scene_mask = valid & ~exclude
            rec.descriptor = compute_descriptor(rect, scene_mask, "scene", embed_fn)
            rec.flags.append("grouping_uses_scene_features")
    elif rec.role != "scale":
        rec.descriptor = compute_descriptor(rect, valid & ~exclude, "scene", embed_fn)
    if cfg.debug_overlays:
        tag = f"{path.name} [{rec.role}]"
        if rec.calib:
            tag += f" {rec.calib.px_per_mm:.1f}px/mm"
        if rec.balance and rec.balance.text:
            tag += f" {rec.balance.text}{rec.balance.unit}"
        rec.overlay = _draw_overlay(rect, chosen, rec.calib, tag)
    return rec
