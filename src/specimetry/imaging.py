"""Image loading, EXIF metadata, and affine transforms."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif"}


@dataclass
class PhotoMeta:
    path: Path
    width: int
    height: int
    capture_time: datetime | None
    seq_number: int | None            # Trailing filename sequence number.
    focal_35mm: float | None
    exif_present: bool


def list_images(folder: Path) -> list[Path]:
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                  key=lambda p: (p.name.casefold(), p.name))


def _enable_heif(path: Path) -> None:
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as exc:
            raise ValueError("HEIC/HEIF requires the optional heic extra: pip install '.[heic]'") from exc
        register_heif_opener()


def load_bgr(path: Path) -> np.ndarray:
    """Load an EXIF-oriented image as uint8 BGR pixels."""
    _enable_heif(path)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def read_meta(path: Path) -> PhotoMeta:
    _enable_heif(path)
    with Image.open(path) as im:
        im2 = ImageOps.exif_transpose(im)
        w, h = im2.size
        exif = im.getexif()
        try:
            sub = exif.get_ifd(0x8769) if exif else {}
        except (ValueError, TypeError, KeyError, OSError):
            sub = {}
    tag = {v: k for k, v in ExifTags.TAGS.items()}
    raw = sub.get(tag["DateTimeOriginal"]) or exif.get(tag["DateTime"]) if exif else None
    when = None
    if raw:
        try:
            when = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            when = None
    f35 = sub.get(tag["FocalLengthIn35mmFilm"]) if sub else None
    m = re.search(r"(\d+)(?!.*\d)", path.stem)
    try:
        f35 = float(f35) if f35 else None
        if f35 is not None and (not np.isfinite(f35) or f35 <= 0):
            f35 = None
    except (ValueError, TypeError, ZeroDivisionError):
        f35 = None
    return PhotoMeta(path=path, width=w, height=h, capture_time=when,
                     seq_number=int(m.group(1)) if m else None,
                     focal_35mm=f35,
                     exif_present=bool(exif))


def rotate_bound(img: np.ndarray, angle_deg: float, border=cv2.BORDER_REPLICATE):
    """Rotate the image onto an expanded canvas and return its affine transform."""
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=border), M


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def invert_affine(M: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(M)
