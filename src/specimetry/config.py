"""Image analysis configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
import re


@dataclass
class RulerConfig:
    angle_search_px: int = 700          # Coarse orientation image size in pixels.
    angle_step_coarse: float = 3.0
    angle_step_fine: float = 0.5
    min_ticks: int = 12                 # Minimum accepted tick count.
    min_periodicity: float = 0.35       # Minimum row autocorrelation strength.
    lag_range_px: tuple[int, int] = (6, 90)
    ratio_tolerance: float = 0.045      # Relative dual-edge pitch tolerance.
    default_units: str = "mm"           # Assumed single-edge unit.
    inch_subdivision: int = 16          # Inch-edge ticks per inch.
    px_per_mm_min: float = 1.0
    px_per_mm_max: float = 80.0


@dataclass
class SegmentationConfig:
    backend: str = "classical"          # Segmentation backend name.
    min_area_frac: float = 0.002        # Fraction of image area.
    max_area_frac: float = 0.60
    sam3_prompt: str = "rock"
    sam3_model: str = "facebook/sam3"
    pile_edge_density: float = 0.009
    pile_fg_fraction: float = 0.50


@dataclass
class BalanceConfig:
    enabled: bool = True
    min_digit_height_px: int = 40
    unit: str = "unknown"               # Declared scale display unit.
    min_confidence: float = 0.65


@dataclass
class GroupingConfig:
    # Available-cue affinity weights.
    w_sift: float = 0.40
    w_color: float = 0.45
    w_texture: float = 0.10
    w_sequence: float = 0.08
    w_embed: float = 0.30               # Embedding cue weight.
    sequence_window: int = 12           # Filename sequence distance scale.
    assign_min_affinity: float = 0.40   # Minimum photo assignment score.
    assign_min_margin: float = 0.08     # Minimum best-to-runner-up margin.
    cluster_threshold: float = 0.50     # Measured-photo cluster threshold.
    sift_min_inliers: int = 12
    use_embeddings: bool = False
    embed_model: str = "facebook/dinov2-small"


@dataclass
class Config:
    ruler: RulerConfig = field(default_factory=RulerConfig)
    seg: SegmentationConfig = field(default_factory=SegmentationConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    prefix: str = "SPC"
    debug_overlays: bool = True
    max_image_side: int = 2000

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", self.prefix):
            raise ValueError("prefix must be 1-32 letters, digits, underscores or dashes, starting with a letter or digit")
        if self.prefix.upper() in {"COL", "UNASSIGNED"}:
            raise ValueError("COL and UNASSIGNED are reserved prefixes")
        if not 128 <= self.max_image_side <= 8192:
            raise ValueError("max_image_side must be between 128 and 8192")
        if self.seg.backend not in {"classical", "rembg", "sam3"}:
            raise ValueError("unknown segmentation backend")
        if self.ruler.default_units not in {"mm", "in"}:
            raise ValueError("default ruler units must be mm or in")
        if self.ruler.inch_subdivision not in {8, 16, 32}:
            raise ValueError("inch subdivision must be 8, 16 or 32")
        if self.balance.unit not in {"unknown", "g", "oz", "lb", "ct"}:
            raise ValueError("scale unit must be unknown, g, oz, lb or ct")
        if not 0 < self.seg.min_area_frac < self.seg.max_area_frac <= 1:
            raise ValueError("invalid segmentation area limits")
        if self.ruler.min_ticks < 4 or self.ruler.angle_step_coarse <= 0 or self.ruler.angle_step_fine <= 0:
            raise ValueError("invalid ruler fitting configuration")
        lo, hi = self.ruler.lag_range_px
        if not 2 <= lo < hi:
            raise ValueError("invalid ruler lag range")
        if self.grouping.sequence_window <= 0 or self.grouping.sift_min_inliers < 3:
            raise ValueError("invalid grouping configuration")
        for name in ("w_sift", "w_color", "w_texture", "w_sequence", "w_embed"):
            v = getattr(self.grouping, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for value in (self.grouping.assign_min_affinity, self.grouping.assign_min_margin,
                      self.grouping.cluster_threshold, self.balance.min_confidence):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("confidence and affinity thresholds must be between zero and one")

    def to_dict(self) -> dict:
        return asdict(self)
