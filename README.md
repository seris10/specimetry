# specimetry

Work in progress. Specimetry estimates dimensions and balance readings from specimen photographs and proposes groups for human review. Measurement accuracy and grouping reliability have not been validated across representative specimens and camera setups.

Photos next to rulers or on scales produce dimension estimates, scale readings, and JSON/CSV
reports. Appearance matching groups photos by specimen. Photos below the assignment threshold
remain unassigned.

## Pipeline

| Stage | Module | Method |
|---|---|---|
| Ruler detection and calibration | `calibration.py` | Black-hat tick enhancement, Hough orientation histogram, row-wise harmonic autocorrelation, sub-pixel tick fitting with outlier rejection. |
| Unit identification | `calibration.py` | Declared inch graduation and dual-edge pitch agreement, including skipped metric ticks. Unit assumptions and inferred metric steps stay flagged. Single-edge photos may be resolved by cross-view consistency. |
| Perspective | `calibration.py` | Scale is taken from the ruler edge nearest the specimen; half the edge-to-edge scale gradient is added to the uncertainty. |
| Balance reading | `balance.py` | LCD panel detection, per-band thresholding, slant removal, seven-segment decoding. |
| Segmentation | `segmentation.py` | GMM + GrabCut with `classical`; optional `rembg` and `sam3` backends. |
| Morphometry | `morphometry.py` | Min-area rectangle, exact Feret diameters, area, perimeter, solidity, circularity, shape class, per-dimension uncertainty. |
| Descriptors | `descriptors.py` | SIFT + RANSAC, CIELAB chroma histogram, uniform LBP, optional DINOv2 embedding. |
| Grouping | `grouping.py` | Anchor clustering, then assignment of dependent photos with confidence and margin over the runner-up. Piles are linked as collections. |
| Export | `pipeline.py` | `specimens.json`, `specimens.csv`, `photos.csv`, `rename_map.csv`, renamed copies, debug overlays. |

## Usage

From this directory, using Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,heic]'
./run.sh PHOTO_FOLDER -o NEW_OUT_FOLDER
./run.sh PHOTO_FOLDER -o NEW_OUT_FOLDER --rename none
./run.sh PHOTO_FOLDER -o NEW_OUT_FOLDER --scale-unit g
.venv/bin/python -m pytest -q
```

The launcher works from any current directory. `specimetry` is also installed in `.venv/bin/`.
The output directory must be new and separate from the input directory.
`--rename copy` creates renamed copies. `--rename none` writes reports and overlays.
Exports are staged and published only after the entire export succeeds. Source hashes are checked
during analysis and again when copying.

The default scale unit is **unknown**: raw display values remain in JSON and `photos.csv`, while
`weight_g` is left empty. Specify `--scale-unit g`, `oz`, `lb`, or `ct` after checking the scale;
units are recorded as operator supplied. No unit OCR is implemented. Do not mix scale units in one
batch. Group weights require a reading confidence of at least 0.65 and an unambiguous assignment.

Processing uses at most 2000 pixels on the longest side. Adjust with `--max-image-side 3000`
for finer ticks; source pixels and metadata are preserved. Calibration pixels refer to this working
image after rotation, not the original image. JSON records source dimensions, working dimensions,
original SHA-256 hashes, the actual backend, configuration, UTC run time, and per-photo errors.
Source paths in reports are filenames relative to the supplied input folder. Serialized error
messages contain the error type and a short description. Full diagnostics are printed in the terminal.
Set `--inch-subdivision 8`, `16`, or `32` to match the ruler's inch ticks; the default is `16`.
Calibration compares 1, 2, 5, and 10 mm metric tick spacings against that inch graduation.
Assumed units and inferred skipped-tick steps remain flagged.

Bad photos do not discard the rest of a batch. `errors.json` and error rows in `photos.csv` record
failures. Exit codes: `0` completed, `1` completed with photo errors, `2` invalid input, setup, or export
failure. An empty input folder is refused. Input discovery is nonrecursive.

Optional backends require separate installation and may download model weights when explicitly
selected: `pip install '.[rembg]'` for `--backend rembg`, or `pip install '.[models]'` for
`--backend sam3` / `--embeddings`. Missing backend dependencies produce a setup error. HEIC/HEIF
requires the `heic` extra. The offline default uses no remote inference or model download.

## Naming

`SPC-001_02_weight.jpg`: specimen 001, second photo, role (`ruler`, `weight`, `view`).
`_unsure` marks assignments below 0.55 confidence or within 0.08 of a competing specimen.
`COL-001_*` is a collection of many small pieces. `UNASSIGNED_*` could not be placed.
`rename_map.csv` records every original name.

## Output fields (specimens.csv)

`length_mm`, `width_mm` (and inches), `weight_g`, `shape_class`, `aspect_ratio`, `length_rel_unc`
(relative uncertainty), `length_is_lower_bound` (true when every view is cut by the frame),
`group_confidence`, `flags`.

`measurement_provisional` marks an unresolved single-edge unit assumption or an unreviewed classical mask. A single ruler edge
does not identify its unit; cross-view resolution requires agreement with a measured view and a
clear advantage for one unit hypothesis. `best_view`, dimensions, and uncertainty refer to the same
photo. Unassigned photos keep their measurements, balance readings, and overlays.
Classical masks are explicitly flagged for review: dark inclusions can merge into the background
and leave only part of a stone measured. Truncated masks do not seed identity groups.
When full-view length estimates disagree by more than a factor of 1.8, grouped dimensions are
withheld. JSON keeps the candidate readings and every per-photo measurement remains available in
`photos.csv`; review the masks and identities before selecting a dimension.

Group scores and segmentation scores are heuristics, not calibrated probabilities. Identity groups
are proposals to review against the overlays and original photos. The exporter does not identify
minerals or establish measurement accuracy. Approximate uncertainties omit height above the ruler,
shadow contamination, and other systematic errors. CSV text that could execute as a spreadsheet
formula is prefixed with an apostrophe; JSON preserves the original names exactly.

## Known limitations

* Dimensions assume the specimen lies in the ruler plane. A specimen raised above the ruler is
  closer to the camera and is overestimated (a few percent for 2 cm of height at 30 cm).
* Cast shadows can enlarge the specimen mask. The overlays show each proposed mask.
* Dark stones on dark backgrounds and stones with strong cast shadows can defeat classical
  segmentation. Comparative performance of the optional model backends has not been verified.
* Piles of chips are detected and grouped but not measured piece by piece. That needs
  a separate instance measurement pipeline; choosing SAM 3 does not implement it.
* Pile detection and the "large mask touching the frame" rule are heuristics tuned on a small
  sample; thresholds live in `config.py`.
* Color-based grouping cannot separate two stones of the same material. SIFT only helps when
  views overlap. Enable `--embeddings` for a learned cue.
* The balance reader is tuned for one backlit seven-segment display style.
* Tests cover generated rulers, masks, scale displays, grouping, and report exports.
  Thresholds require evaluation on each camera and ruler setup.

## License

No reuse license is granted for this source code. Rights to user-supplied photographs remain with their owners. No source photographs are distributed in this repository.
