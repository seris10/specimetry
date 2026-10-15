"""Photo grouping, unit conversion, and export tests."""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from specimetry.balance import BalanceReading
from specimetry.calibration import _row_periodicity, detect_ruler
from specimetry.config import Config, GroupingConfig
from specimetry.grouping import Group, Member, group_photos
from specimetry.imaging import PhotoMeta, list_images
from specimetry.morphometry import measure_mask
from specimetry.photo import PhotoRecord
from specimetry import pipeline


def record(name, length=None, width=10, quality=1.0, weight=None, unit="g"):
    p = Path(name)
    r = PhotoRecord(p, PhotoMeta(p, 100, 100, None, None, None, False))
    r.seg_quality = quality
    if length is not None:
        r.measure = dict(major_mm=length, minor_mm=width, feret_max_mm=length, feret_min_mm=width,
                         major_rel_unc=0.01, minor_rel_unc=0.01, area_mm2=length * width,
                         perimeter_mm=2 * (length + width), scale_px_per_mm=10.0,
                         shape_class="elongate", aspect_ratio=length / width)
    if weight is not None:
        r.balance = BalanceReading(weight, unit, str(weight), 0.9, [], (0, 0, 20, 10))
    return r


@pytest.mark.parametrize("width", [8, 60, 120])
def test_short_scanlines_do_not_crash(width):
    strength, lag = _row_periodicity(np.zeros((3, width), np.float32), (6, 90))
    assert np.isfinite(strength).all() and not strength.any()


def test_small_photo_has_no_ruler():
    assert detect_ruler(np.zeros((20, 30, 3), np.uint8)) is None


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        measure_mask(np.ones((10, 10), bool), scale, 0.01)


def test_empty_mask_is_rejected():
    with pytest.raises(ValueError):
        measure_mask(np.zeros((20, 20), bool), 10, 0.01)


def test_mask_area_uses_only_largest_component_and_normalizes_255():
    mask = np.zeros((200, 200), np.uint8)
    mask[20:80, 20:100] = 255
    mask[150:155, 150:155] = 255
    m = measure_mask(mask, 10, 0.01)
    assert m["area_mm2"] == pytest.approx(48)
    assert 0 < m["solidity"] <= 1 and 0 < m["circularity"] <= 1


@pytest.mark.parametrize("unit,factor", [("g", 1), ("oz", 28.349523125), ("lb", 453.59237), ("ct", .2)])
def test_scale_unit_conversion(unit, factor):
    r = record("a.jpg", weight=2, unit=unit)
    assert pipeline._specimen_summary(Group("single", [Member(r, .9, None, "assigned")]))["weight_g"] == pytest.approx(2 * factor)


def test_unknown_unit_retains_reading_but_claims_no_grams():
    r = record("a.jpg", weight=101, unit="unknown")
    s = pipeline._specimen_summary(Group("single", [Member(r, .9, None, "assigned")]))
    assert "weight_g" not in s
    assert r.balance.to_dict()["unit_source"] == "unknown"
    assert r.balance.to_dict()["value"] == 101


def test_low_confidence_scale_reading_is_not_a_specimen_weight():
    r = record("a.jpg", weight=101)
    r.balance.confidence = .1
    assert "weight_g" not in pipeline._specimen_summary(Group("single", [Member(r, .9, None, "assigned")]))


def test_summary_uses_dimensions_and_uncertainty_from_one_view():
    a = record("a.jpg", length=50, width=10, quality=1)
    b = record("b.jpg", length=70, width=30, quality=.7)
    b.measure["major_rel_unc"] = .2
    s = pipeline._specimen_summary(Group("single", [Member(r, 1, None, "anchor") for r in (a, b)]))
    assert (s["length_mm"], s["width_mm"], s["best_view"], s["length_rel_unc"]) == (70, 30, "b.jpg", .2)


def test_severely_disagreeing_views_do_not_publish_an_arbitrary_group_length():
    a, b = record("a.jpg", length=45), record("b.jpg", length=91)
    s = pipeline._specimen_summary(Group("single", [Member(r, 1, None, "anchor") for r in (a, b)]))
    assert "length_mm" not in s and "width_mm" not in s and s["measurement_provisional"]
    assert [m["length_mm"] for m in s["measurement_candidates"]] == [45, 91]


def test_unrelated_cross_view_does_not_confirm_units():
    sure, unsure = record("a.jpg", length=50, width=20), record("b.jpg", length=200, width=100)
    unsure.measure.update(unit_hypothesis="mm", alt_if_inch_scale=25.4 / 16)
    unsure.flags.append("measurement_provisional_unit_unconfirmed")
    pipeline._resolve_units(Group("single", [Member(r, 1, None, "anchor") for r in (sure, unsure)]))
    assert "measurement_provisional_unit_unconfirmed" in unsure.flags


def test_inch_assumption_can_be_resolved_back_to_mm():
    sure = record("a.jpg", length=50, width=20)
    unsure = record("b.jpg", length=50 * 1.5875, width=20 * 1.5875)
    unsure.measure.update(unit_hypothesis="in", alt_if_inch_scale=1.5875)
    unsure.flags.append("measurement_provisional_unit_unconfirmed")
    pipeline._resolve_units(Group("single", [Member(r, 1, None, "anchor") for r in (sure, unsure)]))
    assert unsure.measure["feret_max_mm"] == pytest.approx(50)
    assert unsure.measure["unit_resolution"]["selected"] == "mm"
    assert "measurement_provisional_unit_unconfirmed" not in unsure.flags


def test_weight_conflicts_apply_to_whole_group(monkeypatch):
    import specimetry.grouping as grouping
    a = record("a.jpg", length=50)
    b, c = record("b.jpg", weight=100), record("c.jpg", weight=200)
    monkeypatch.setattr(grouping, "pair_affinity", lambda *args: (.95, {}))
    groups, unassigned = group_photos([a, b, c], GroupingConfig())
    assert not any(b in [m.rec for m in g.members] and c in [m.rec for m in g.members] for g in groups)


def test_anchor_bridge_cannot_merge_disagreeing_endpoints(monkeypatch):
    import specimetry.grouping as grouping
    recs = [record(f"{n}.jpg", length=50) for n in "abc"]
    def affinity(a, b, *args):
        return (.25 if {a.path.stem, b.path.stem} == {"a", "c"} else .9), {}
    monkeypatch.setattr(grouping, "pair_affinity", affinity)
    groups, _ = group_photos(recs, GroupingConfig())
    assert max(len(g.members) for g in groups) == 2


def test_dependent_photos_cannot_extend_an_assignment_chain(monkeypatch):
    import specimetry.grouping as grouping
    a, b, c = record("a.jpg", length=50), record("b.jpg"), record("c.jpg")
    def affinity(x, y, *args):
        return {frozenset("ab"): .6, frozenset("ac"): .3, frozenset("bc"): 1}[frozenset((x.path.stem, y.path.stem))], {}
    monkeypatch.setattr(grouping, "pair_affinity", affinity)
    groups, unassigned = group_photos([a, b, c], GroupingConfig())
    assert c in unassigned
    assert [m.rec.path.name for m in groups[0].members] == ["a.jpg", "b.jpg"]


def test_truncated_measurement_is_not_an_identity_anchor():
    r = record("a.jpg", length=50)
    r.seg_flags.append("touches_frame_edge")
    assert not r.measured


def test_scene_penalty_preserves_ranking_instead_of_creating_ties(monkeypatch):
    import specimetry.grouping as grouping
    a, b, c = (record(f"{n}.jpg") for n in "abc")
    for r in (a, b, c):
        r.descriptor = SimpleNamespace(source="scene", embedding=None)
    monkeypatch.setattr(grouping, "color_affinity", lambda x, y: .9 if y is b.descriptor else .6)
    monkeypatch.setattr(grouping, "texture_affinity", lambda *args: 0)
    monkeypatch.setattr(grouping, "sift_match", lambda *args: (0, 0))
    cfg = GroupingConfig(w_color=1, w_texture=0, w_sequence=0, w_embed=0, w_sift=0)
    strong, _ = grouping.pair_affinity(a, b, cfg, {})
    weak, _ = grouping.pair_affinity(a, c, cfg, {})
    assert 0 <= weak < strong <= .5


@pytest.fixture
def batch(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    def process(p, cfg, backend, embed):
        with Image.open(p) as im:
            im.load()
        r = record(p, weight=101, unit=cfg.balance.unit)
        r.overlay = np.zeros((20, 20, 3), np.uint8)
        return r
    monkeypatch.setattr(pipeline, "make_backend", lambda cfg: SimpleNamespace(name="fixture"))
    monkeypatch.setattr(pipeline, "process_photo", process)
    return source, tmp_path / "out"


def save_image(path):
    Image.new("RGB", (80, 80), "gray").save(path)


def test_unassigned_photos_keep_weight_and_metadata(batch):
    source, out = batch
    save_image(source / "stone.jpg")
    cfg = Config()
    cfg.balance.unit = "g"
    r = pipeline.run(source, out, cfg, rename="none", log=lambda _: None)
    p = r.photos[0]
    assert p["assignment"] == "unassigned" and p["balance"]["weight_g"] == 101
    assert p["source_sha256"] == hashlib.sha256((source / "stone.jpg").read_bytes()).hexdigest()
    assert (out / "overlays").is_dir() and not (out / "renamed").exists()
    doc = json.loads((out / "specimens.json").read_text())
    assert doc["status"] == "complete" and doc["backend"] == "fixture"
    row = next(csv.DictReader((out / "photos.csv").open()))
    assert row["weight_g"] == "101.0"


def test_copy_preserves_source_and_output_bytes(batch):
    source, out = batch
    path = source / "stone.png"
    save_image(path)
    before = path.read_bytes()
    result = pipeline.run(source, out, Config(), log=lambda _: None)
    assert path.read_bytes() == before
    assert (out / "renamed" / result.photos[0]["file"]).read_bytes() == before
    assert result.out_dir == out


def test_report_paths_are_relative_to_the_input_folder(batch):
    source, out = batch
    save_image(source / "stone.png")
    result = pipeline.run(source, out, Config(), rename="none", log=lambda _: None)
    doc = json.loads((out / "specimens.json").read_text())
    assert result.photos[0]["source_path"] == "stone.png"
    assert doc["photos"][0]["source_path"] == "stone.png"
    assert doc["source_path_base"] == "input_folder"
    assert "input_directory" not in doc
    for path in out.iterdir():
        if path.suffix in {".json", ".csv"}:
            assert str(source.parent.resolve()) not in path.read_text()


def test_photo_errors_keep_private_paths_in_terminal_diagnostics(batch, monkeypatch):
    source, out = batch
    save_image(source / "stone.png")
    detail = f"Failed to load model from {source.parent.resolve() / 'model-cache'}"
    def fail(*args):
        raise RuntimeError(detail)
    monkeypatch.setattr(pipeline, "process_photo", fail)
    messages = []
    result = pipeline.run(source, out, Config(), rename="none", log=messages.append)
    expected = {"original_file": "stone.png", "error_type": "RuntimeError", "message": "Photo analysis failed."}
    assert result.errors == [expected]
    assert json.loads((out / "errors.json").read_text()) == [expected]
    assert any(detail in message for message in messages)
    for path in out.iterdir():
        if path.suffix in {".json", ".csv"}:
            assert str(source.parent.resolve()) not in path.read_text()


def test_corrupt_photo_is_reported_without_losing_valid_results(batch):
    source, out = batch
    save_image(source / "good.png")
    (source / "bad.jpg").write_bytes(b"corrupt image")
    r = pipeline.run(source, out, Config(), rename="none", log=lambda _: None)
    assert len(r.errors) == 1 and len(r.photos) == 2
    doc = json.loads((out / "specimens.json").read_text())
    assert doc["status"] == "completed_with_errors"
    error = json.loads((out / "errors.json").read_text())[0]
    assert error == {"original_file": "bad.jpg", "error_type": "UnidentifiedImageError",
                     "message": "Unsupported or corrupt image."}
    assert str(source.resolve()) not in json.dumps(doc)


def test_existing_output_is_refused_without_changes(batch):
    source, out = batch
    save_image(source / "stone.jpg")
    out.mkdir()
    (out / "keep.txt").write_text("previous results")
    with pytest.raises(FileExistsError):
        pipeline.run(source, out, Config())
    assert (out / "keep.txt").read_text() == "previous results"


@pytest.mark.parametrize("prefix", ["../../escape", "", "COL", "UNASSIGNED"])
def test_unsafe_prefix_is_refused_before_output(batch, prefix):
    source, out = batch
    save_image(source / "stone.jpg")
    with pytest.raises(ValueError):
        pipeline.run(source, out, Config(prefix=prefix))
    assert not out.exists()


def test_move_mode_is_refused_before_touching_sources(batch):
    source, out = batch
    p = source / "stone.jpg"
    save_image(p)
    with pytest.raises(ValueError):
        pipeline.run(source, out, Config(), rename="move")
    assert p.is_file() and not out.exists()


def test_nested_output_is_refused(batch):
    source, _ = batch
    save_image(source / "stone.jpg")
    with pytest.raises(ValueError):
        pipeline.run(source, source / "out", Config())


def test_failed_export_leaves_no_partial_results_or_source_changes(batch, monkeypatch):
    source, out = batch
    p = source / "stone.jpg"
    save_image(p)
    before = p.read_bytes()
    monkeypatch.setattr(pipeline.cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError):
        pipeline.run(source, out, Config(), log=lambda _: None)
    assert p.read_bytes() == before and not out.exists()
    assert not list(out.parent.glob(".out-*"))


def test_same_stem_has_distinct_overlays(batch):
    source, out = batch
    save_image(source / "stone.jpg")
    save_image(source / "stone.png")
    pipeline.run(source, out, Config(), log=lambda _: None)
    assert len(list((out / "overlays").iterdir())) == 2


def test_csv_formula_filenames_are_escaped_but_json_preserves_original(batch):
    source, out = batch
    save_image(source / "=stone.png")
    r = pipeline.run(source, out, Config(), rename="none", log=lambda _: None)
    assert r.photos[0]["original_file"] == "=stone.png"
    row = next(csv.DictReader((out / "photos.csv").open()))
    assert row["original_file"] == "'=stone.png"


def test_image_named_directory_is_not_an_input(batch):
    source, _ = batch
    (source / "not-a-photo.jpg").mkdir()
    assert list_images(source) == []
