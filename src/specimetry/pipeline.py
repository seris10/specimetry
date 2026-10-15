"""Photo analysis, grouping, and report exports."""
from __future__ import annotations

import csv
import json
import shutil
import hashlib
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import UnidentifiedImageError

from . import __version__
from .config import Config
from .descriptors import make_dinov2_embedder
from .grouping import Group, group_photos
from .imaging import list_images
from .photo import PhotoRecord, process_photo
from .segmentation import make_backend

MM_PER_IN = 25.4
_LENGTHS = ("major_mm", "minor_mm", "feret_max_mm", "feret_min_mm")


def _clean(o):
    """Convert NumPy values to JSON scalars and replace nonfinite floats."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _resolve_units(g: Group) -> None:
    """Resolve single-edge ruler units from cross-view dimension agreement."""
    eligible = [m.rec for m in g.members if m.confidence >= 0.55 and "ambiguous_between_specimens" not in m.flags
                and "touches_frame_edge" not in m.rec.seg_flags]
    sure = [r.measure for r in eligible if r.measure and "unit_hypothesis" not in r.measure]
    unsure = [r for r in eligible if r.measure and "unit_hypothesis" in r.measure]
    if not sure or not unsure:
        return
    ref = [(x["feret_max_mm"], x["feret_min_mm"]) for x in sure]
    for rec in unsure:
        m = rec.measure
        factor = m.get("alt_if_inch_scale", 25.4 / 16)
        current = m["unit_hypothesis"]
        if current not in {"mm", "in"}:
            continue
        # Compare each dimension against the reference views.
        def best_gap(vals):
            return min(abs(np.log(max(a, 1e-6) / max(b, 1e-6))) for a in vals for r in ref for b in r)
        # Apply the declared inch subdivision to the alternative scale.
        mm_vals = tuple(m[k] / (factor if current == "in" else 1.0) for k in ("feret_max_mm", "feret_min_mm"))
        in_vals = tuple(v * factor for v in mm_vals)
        gap_mm, gap_in = best_gap(mm_vals), best_gap(in_vals)
        if min(gap_mm, gap_in) > np.log(1.10) or abs(gap_mm - gap_in) < 0.08:
            continue
        selected = "in" if gap_in < gap_mm else "mm"
        correction = (factor if selected == "in" else 1.0) / (factor if current == "in" else 1.0)
        if correction != 1.0:
            for k in _LENGTHS + ("perimeter_mm",):
                m[k] *= correction
            m["area_mm2"] *= correction ** 2
            m["scale_px_per_mm"] /= correction
        m["unit_hypothesis"] = f"{selected} (resolved by cross-view consistency)"
        m["unit_resolution"] = {"assumed": current, "selected": selected,
                                "method": "cross_view_dimension_agreement", "length_correction": correction}
        rec.flags.append(f"unit_resolved_{selected}_by_cross_view")
        m.pop("alt_if_inch_scale", None)
        rec.flags = [f for f in rec.flags if f != "measurement_provisional_unit_unconfirmed"]


def _specimen_summary(g: Group, min_weight_confidence: float = 0.65) -> dict:
    meas = [(m.rec, m.rec.measure) for m in g.members if m.rec.measure]
    full = [(r, x) for r, x in meas if "touches_frame_edge" not in r.seg_flags]
    out: dict = {}
    if g.kind == "single":
        pool = full or meas
        if pool:
            r, x = max(pool, key=lambda t: (t[0].seg_quality, -t[1]["feret_max_mm"]))
            fm = [t[1]["feret_max_mm"] for t in pool]
            disagree = len(pool) > 1 and max(fm) / min(fm) > 1.8
            if disagree:
                out["views_disagree"] = [round(v, 1) for v in fm]
                out["measurement_provisional"] = True
                out["measurement_candidates"] = [
                    {"photo": rr.path.name, "length_mm": mm["feret_max_mm"], "width_mm": mm["feret_min_mm"],
                     "length_rel_unc": mm["major_rel_unc"]} for rr, mm in pool]
                out["note"] = "view dimensions disagree; aggregate dimensions withheld pending mask and identity review"
            else:
                r, x = max(pool, key=lambda t: t[1]["feret_max_mm"])
                length, width = x["feret_max_mm"], x["feret_min_mm"]
                unc = float(x["major_rel_unc"])
                out["length_mm"] = length
                out["length_rel_unc"] = unc
                out["width_mm"] = width
                out["best_view"] = r.path.name
                out["length_is_lower_bound"] = not bool(full)
                out["measurement_provisional"] = bool({"measurement_provisional_unit_unconfirmed", "classical_segmentation_requires_review"} & set(r.flags))
                out["shape_class"] = x["shape_class"]
                out["aspect_ratio"] = x["aspect_ratio"]
                out["length_in"] = length / MM_PER_IN
                out["width_in"] = out["width_mm"] / MM_PER_IN
    else:
        sizes = [m.rec.seg_stats.get("field_of_view_mm") for m in g.members if m.rec.seg_stats]
        out["note"] = "collection of many small specimens; per-piece measurement is not implemented"
        out["field_of_view_mm"] = [s for s in sizes if s]
    weights = [(m.rec.balance.weight_g, m.rec.balance.confidence) for m in g.members
               if m.rec.balance and m.rec.balance.weight_g is not None and m.rec.balance.confidence >= min_weight_confidence
               and "ambiguous_between_specimens" not in m.flags]
    if weights:
        out["weight_g"] = max(weights, key=lambda t: t[1])[0]
        out["weight_readings_g"] = [w for w, _ in weights]
        out["weight_unit_source"] = "operator"
    return out


def _group_confidence(g: Group) -> float:
    if len(g.members) == 1:
        return 1.0 if g.members[0].how == "anchor" else g.members[0].confidence
    return float(min(m.confidence for m in g.members))


@dataclass
class RunResult:
    groups: list[Group]
    unassigned: list[PhotoRecord]
    specimens: list[dict]
    photos: list[dict]
    out_dir: Path
    errors: list[dict]


def _photo_error(path: Path, exc: Exception) -> dict:
    """Return a path-free photo failure record."""
    if isinstance(exc, UnidentifiedImageError):
        message = "Unsupported or corrupt image."
    elif isinstance(exc, FileNotFoundError):
        message = "Source image is missing."
    elif isinstance(exc, PermissionError):
        message = "Source image is not readable."
    elif isinstance(exc, ImportError):
        message = "An image codec or model dependency is missing."
    elif isinstance(exc, MemoryError):
        message = "Insufficient memory for image analysis."
    elif isinstance(exc, ValueError):
        message = "Invalid image data or analysis parameters."
    else:
        message = "Photo analysis failed."
    return {"original_file": path.name, "error_type": type(exc).__name__, "message": message}


@contextmanager
def _export_directory(out_dir: Path):
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{out_dir.name}-", dir=out_dir.parent) as tmp:
            stage = Path(tmp)
            yield stage
            os.replace(stage, out_dir)
    except BaseException:
        if out_dir.is_dir() and not any(out_dir.iterdir()):
            out_dir.rmdir()
        raise


def run(folder: Path, out_dir: Path, cfg: Config, rename: str = "copy", log=print) -> RunResult:
    cfg.validate()
    if rename not in {"copy", "none"}:
        raise ValueError("rename must be copy or none; move mode is disabled to preserve originals")
    folder, out_dir = Path(folder).resolve(), Path(out_dir).resolve()
    if not folder.is_dir():
        raise ValueError(f"input folder does not exist: {folder}")
    if folder == out_dir or folder in out_dir.parents or out_dir in folder.parents:
        raise ValueError("input and output directories must not overlap")
    if out_dir.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {out_dir}")
    paths = list_images(folder)
    if not paths:
        raise ValueError(f"no images found in {folder}")
    backend = make_backend(cfg.seg)
    embed = make_dinov2_embedder(cfg.grouping.embed_model) if cfg.grouping.use_embeddings else None
    recs: list[PhotoRecord] = []
    errors: list[dict] = []
    for i, p in enumerate(paths, 1):
        log(f"[{i}/{len(paths)}] {p.name}")
        try:
            before = _sha256(p)
            rec = process_photo(p, cfg, backend, embed)
            if before != _sha256(p):
                raise OSError("source changed during analysis; retry with a stable input folder")
            rec.source_sha256 = before
            recs.append(rec)
        except Exception as exc:
            errors.append(_photo_error(p, exc))
            log(f"  ERROR {type(exc).__name__}: {exc}")
    groups, unassigned = group_photos(recs, cfg.grouping)
    for g in groups:
        if g.kind == "single":
            _resolve_units(g)
    with _export_directory(out_dir) as stage:
        return _export_run(stage, out_dir, folder, cfg, rename, groups, unassigned, errors, backend.name)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _copy_original(rec: PhotoRecord, dst: Path) -> None:
    shutil.copy2(rec.path, dst)
    if rec.source_sha256 != _sha256(dst):
        raise OSError(f"source changed after analysis: {rec.path.name}")


def _photo_row(rec: PhotoRecord, filename: str, specimen_id: str | None, assignment: str,
               confidence: float | None = None, margin: float | None = None, flags: tuple[str, ...] | list[str] = ()) -> dict:
    row = {"file": filename, "original_file": rec.path.name, "specimen_id": specimen_id, "role": rec.role,
           "assignment": assignment, "group_confidence": confidence, "margin_over_runner_up": margin,
           "flags": sorted(set(rec.flags + rec.seg_flags + list(flags))), "source_sha256": rec.source_sha256,
           "source_path": rec.path.name,
           "source_dimensions": [rec.meta.width, rec.meta.height], "working_dimensions": rec.working_dimensions,
           "capture_time": rec.meta.capture_time.isoformat() if rec.meta.capture_time else None,
           "scene": rec.scene}
    if rec.calib:
        row["calibration"] = rec.calib.to_dict()
    if rec.balance:
        row["balance"] = rec.balance.to_dict()
    if rec.measure:
        row["measurement"] = {k: v for k, v in rec.measure.items() if k != "centroid_px"}
    if rec.seg_stats:
        row["segmentation"] = {**rec.seg_stats, "quality": rec.seg_quality}
    return row


def _csv_safe(row: dict) -> dict:
    return {k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
            for k, v in row.items()}


def _export_run(stage: Path, final_dir: Path, folder: Path, cfg: Config, rename: str,
                groups: list[Group], unassigned: list[PhotoRecord], errors: list[dict], backend_name: str) -> RunResult:
    out_dir = stage
    groups.sort(key=lambda g: min(m.rec.path.name for m in g.members))
    n_s = n_c = 0
    for g in groups:
        if g.kind == "single":
            n_s += 1; g.id = f"{cfg.prefix}-{n_s:03d}"
        else:
            n_c += 1; g.id = f"COL-{n_c:03d}"
    ov_dir = out_dir / "overlays"
    ren_dir = out_dir / "renamed"
    if cfg.debug_overlays:
        ov_dir.mkdir(exist_ok=True)
    if rename != "none":
        ren_dir.mkdir(exist_ok=True)
    photos: list[dict] = []
    rename_rows: list[dict] = []
    role_tag = {"ruler": "ruler", "scale": "weight", "other": "view"}

    def new_name(gid: str, k: int, rec: PhotoRecord, uncertain: bool) -> str:
        tag = role_tag.get(rec.role, "view")
        flag = "_unsure" if uncertain else ""
        return f"{gid}_{k:02d}_{tag}{flag}{rec.path.suffix.lower()}"

    specimens: list[dict] = []
    for g in groups:
        mem_rows = []
        for k, m in enumerate(sorted(g.members, key=lambda m: m.rec.path.name), 1):
            uncertain = m.confidence < 0.55 or "ambiguous_between_specimens" in m.flags
            nn = new_name(g.id, k, m.rec, uncertain)
            if rename != "none":
                dst = ren_dir / nn
                _copy_original(m.rec, dst)
            rename_rows.append({"original": m.rec.path.name, "new": nn, "group": g.id})
            row = _photo_row(m.rec, nn, g.id, m.how, round(m.confidence, 3),
                             None if m.margin is None else round(m.margin, 3), m.flags)
            photos.append(row)
            mem_rows.append(row)
            if cfg.debug_overlays and m.rec.overlay is not None:
                if not cv2.imwrite(str(ov_dir / (Path(nn).stem + ".jpg")), m.rec.overlay):
                    raise OSError(f"could not write overlay for {m.rec.path.name}")
        summary = _specimen_summary(g, cfg.balance.min_confidence)
        sp = {"specimen_id": g.id, "kind": g.kind, "n_photos": len(g.members),
              "group_confidence": round(_group_confidence(g), 3), **summary,
              "flags": sorted(set(g.flags + [f for r in mem_rows for f in r["flags"]]
                                  + (["view_measurements_disagree"] if "views_disagree" in summary else []))), "photos": [r["file"] for r in mem_rows]}
        specimens.append(sp)
    for i, r in enumerate(unassigned, 1):
        nn = f"UNASSIGNED_{i:04d}_{r.path.stem}{r.path.suffix.lower()}"
        if rename != "none":
            _copy_original(r, ren_dir / nn)
        rename_rows.append({"original": r.path.name, "new": nn, "group": ""})
        photos.append(_photo_row(r, nn, None, "unassigned", flags=["could_not_assign"]))
        if cfg.debug_overlays and r.overlay is not None:
            if not cv2.imwrite(str(ov_dir / (Path(nn).stem + ".jpg")), r.overlay):
                raise OSError(f"could not write overlay for {r.path.name}")
    for err in errors:
        photos.append({"file": None, "original_file": err["original_file"], "specimen_id": None,
                       "role": "unknown", "assignment": "error", "flags": ["analysis_failed"], "error": err})
    doc = {"tool": "specimetry", "version": __version__, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "status": "completed_with_errors" if errors else "complete", "source_path_base": "input_folder",
           "backend": backend_name, "rename_mode": rename, "coordinate_frame": "EXIF-oriented, resized, then ruler-rectified pixels",
           "confidence_kind": "heuristic scores, not calibrated probabilities", "errors": errors,
           "config": cfg.to_dict(), "specimens": _clean(specimens), "photos": _clean(photos)}
    (out_dir / "specimens.json").write_text(json.dumps(_clean(doc), indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    cols = ["specimen_id", "kind", "n_photos", "length_mm", "width_mm", "length_in", "width_in", "weight_g",
            "shape_class", "aspect_ratio", "length_rel_unc", "length_is_lower_bound", "measurement_provisional", "group_confidence",
            "best_view", "flags", "photos"]
    with open(out_dir / "specimens.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for sp in _clean(specimens):
            row = dict(sp)
            row["flags"] = ";".join(sp["flags"])
            row["photos"] = ";".join(sp["photos"])
            for c in ("length_mm", "width_mm", "length_in", "width_in", "aspect_ratio"):
                if row.get(c) is not None:
                    row[c] = round(row[c], 2)
            w.writerow(_csv_safe(row))
    with open(out_dir / "photos.csv", "w", newline="", encoding="utf-8") as f:
        cols_p = ["file", "original_file", "specimen_id", "role", "assignment", "group_confidence",
                  "margin_over_runner_up", "px_per_mm", "units", "ruler_confidence", "scale_value", "scale_unit", "weight_g",
                  "major_mm", "minor_mm", "source_sha256", "error", "flags"]
        w = csv.DictWriter(f, fieldnames=cols_p, extrasaction="ignore")
        w.writeheader()
        for p in _clean(photos):
            row = dict(p)
            row["flags"] = ";".join(p.get("flags", []))
            c = p.get("calibration") or {}
            bal = p.get("balance") or {}
            row["scale_value"], row["scale_unit"], row["weight_g"] = bal.get("value"), bal.get("unit"), bal.get("weight_g")
            ms = p.get("measurement") or {}
            row["px_per_mm"] = ms.get("scale_px_per_mm", c.get("px_per_mm"))
            row["units"], row["ruler_confidence"] = ms.get("unit_hypothesis", c.get("units")), c.get("confidence")
            row["error"] = (p.get("error") or {}).get("message")
            row["major_mm"], row["minor_mm"] = (round(ms["major_mm"], 2) if ms else None), (round(ms["minor_mm"], 2) if ms else None)
            w.writerow(_csv_safe(row))
    with open(out_dir / "rename_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["original", "new", "group"])
        w.writeheader(); w.writerows(_csv_safe(row) for row in rename_rows)
    return RunResult(groups, unassigned, _clean(specimens), _clean(photos), final_dir, errors)
