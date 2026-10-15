from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="specimetry", description="Measure, name and group specimen photos.")
    ap.add_argument("folder", type=Path, help="folder of photos")
    ap.add_argument("-o", "--out", type=Path, default=Path("specimetry_out"))
    ap.add_argument("--rename", choices=["copy", "none"], default="copy",
                    help="copy (default, originals untouched), or none (reports only)")
    ap.add_argument("--backend", choices=["classical", "rembg", "sam3"], default="classical")
    ap.add_argument("--prefix", default="SPC")
    ap.add_argument("--default-units", choices=["mm", "in"], default="mm",
                    help="unit assumed when only one ruler edge is visible")
    ap.add_argument("--inch-subdivision", type=int, choices=[8, 16, 32], default=16,
                    help="inch-edge graduation, e.g. 16 for 1/16 inch ticks (default: 16)")
    ap.add_argument("--embeddings", action="store_true", help="add DINOv2 embeddings to grouping (needs torch)")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--scale-unit", choices=["unknown", "g", "oz", "lb", "ct"], default="unknown",
                    help="unit selected on the scale; unknown preserves raw readings without claiming grams")
    ap.add_argument("--max-image-side", type=int, default=2000, help="processing longest edge in pixels (default: 2000)")
    a = ap.parse_args(argv)
    cfg = Config(prefix=a.prefix, debug_overlays=not a.no_overlays)
    cfg.seg.backend = a.backend
    cfg.ruler.default_units = a.default_units
    cfg.ruler.inch_subdivision = a.inch_subdivision
    cfg.grouping.use_embeddings = a.embeddings
    cfg.balance.unit = a.scale_unit
    cfg.max_image_side = a.max_image_side
    try:
        cfg.validate()
        from .pipeline import run
        res = run(a.folder, a.out, cfg, rename=a.rename)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"specimetry: {exc}", file=sys.stderr)
        return 2
    print(f"\n{len(res.specimens)} groups, {len(res.unassigned)} unassigned photos -> {res.out_dir}")
    for s in res.specimens:
        dims = f"{s['length_mm']:.1f} x {s['width_mm']:.1f} mm" if s.get("length_mm") else "no measurement"
        w = f", {s['weight_g']:g} g" if s.get("weight_g") is not None else ""
        print(f"  {s['specimen_id']}  {s['kind']:10s} {s['n_photos']} photos  {dims}{w}  conf {s['group_confidence']}")
    if res.errors:
        print(f"{len(res.errors)} photos failed; see errors.json", file=sys.stderr)
    return 1 if res.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
