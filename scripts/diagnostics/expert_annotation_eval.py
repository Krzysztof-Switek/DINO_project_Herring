"""Expert-annotation evaluation — the project's FIRST real ground truth for increment
localization (12.08, "ZEGAR" dataset).

Context (full write-up: plans and summaries/12.08_ZEGAR_ANOTACJE_TO_DO.md): every localization
number this project has ever reported (`mean_dist_final_to_classical_px`,
`localization_quality.json`, cited in every training report since 21.07) compares the model's
predicted increment positions against a NON-HUMAN, algorithmic proxy — `src/candidates.py::
find_candidate_peaks` run on raw image intensity along a single axis, never real ring positions.
`data/ZEGAR/` now holds 42 otoliths (Z01-Z43, Z28 missing) with real (x, y) ring positions from
two independent, experienced readers ("KK", "SS" — a third reader, "DD", was a trainee at the
time and is DELIBERATELY EXCLUDED per the user's explicit instruction, 12.08).

This script does NOT touch training or `src/` — it is purely additive evaluation, fully aligned
with this project's weak-supervision-first stance (see memory `weak_supervision_attention_first`):
using real annotations to MEASURE the existing model is a different act from using them to TRAIN
it.

What it computes, per image and in aggregate:
  - model vs. consensus ground truth (mean t of matched KK/SS pairs) — the main new result.
  - KK vs. SS ("noise floor") — how much two experienced human readers disagree with each other,
    the natural bar the model doesn't need to beat, only approach.
  - classical (OpenCV single-axis peak) vs. consensus ground truth — tests whether the proxy this
    project has relied on for over a month actually tracks real ring positions.
  - Beamish & Fournier (1981) Average Percent Error on ring COUNTS (not positions) between KK/SS —
    canonical fisheries-science precision metric, independent of the position metrics above.

Usage:
    python scripts/diagnostics/expert_annotation_eval.py

Needs the real DINOv2 backbone (network access or a locally cached copy) and
`outputs/22.07_reg/checkpoints/embedded/best.pt` — the last confirmed-good production checkpoint,
deliberately NOT the most recent run (06.08/08.08/09.08 all have a known-dead density head, see
`plans and summaries/9.08_uwaga_plan_TO.DO.md`). No GPU required but much faster with one.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Constants — edit here, not inline below
# ---------------------------------------------------------------------------

ZEGAR_DIR = PROJECT_ROOT / "data" / "ZEGAR"
IMAGE_DIR = ZEGAR_DIR / "Zdjęcia"
ANNOTATION_FILES = {
    "KK": IMAGE_DIR / "ZEGAR_LINIA_KK_ODCZYT.csv",
    "SS": IMAGE_DIR / "ZEGAR_LINIA_SS_ODCZYT.csv",
}
# "DD" (ZEGAR_LINIA_DD_ODCZYT.csv) intentionally NOT listed — trainee reader at the time, user's
# explicit instruction (12.08) to exclude entirely, not even as a secondary comparison.

REFERENCE_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
REFERENCE_CKPT = (PROJECT_ROOT / "outputs" / "22.07_reg" / "checkpoints" / "embedded" / "best.pt")
# 22.07_reg, NOT the most recent run — see module docstring. Verified (08.08/09.08 sessions) to be
# the last checkpoint whose density head is genuinely trained (not dead/degenerate).

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "12.08_zegar_annotation_eval"

N_SAMPLES_AXIS = 50   # mirrors scripts/run_pipeline.py's own n_samples_axis for the classical
                       # profile — matching it keeps our derived max_gap_t (below) on the same
                       # footing as production's own axis sampling resolution.


# ---------------------------------------------------------------------------
# Step 1 — inspect (kept as a standalone function; already run once manually on
# the real files, 12.08 — see plans and summaries/12.08_ZEGAR_ANOTACJE_TO_DO.md §2.1 for the
# schema this produced. Left here so a future, DIFFERENT annotation export can be re-inspected
# the same way rather than assuming today's schema forever.)
# ---------------------------------------------------------------------------

def inspect_annotation_file(path: Path) -> None:
    print(f"--- {path} ---")
    df = pd.read_csv(path)
    print("columns:", list(df.columns))
    print("dtypes:\n", df.dtypes)
    print(df.head(10).to_string())


# ---------------------------------------------------------------------------
# Step 2 — load + normalise annotations (real, known schema — no guessing)
# ---------------------------------------------------------------------------

def load_expert_annotations() -> pd.DataFrame:
    """Long-format DataFrame: [image_id, annotator, Sample, x, y, age, quality].

    One row per annotated INCREMENT. Two non-increment landmark rows are dropped for every
    sample: "core" (the nucleus) and "edge" (the otolith's outer physical boundary, present in
    all 42 samples in both files) — neither is a growth ring, so keeping them would inflate ring
    counts and bias position agreement (the boundary is a fixed anatomical point both readers
    click almost identically, unlike an actual annual ring judgement call).
    """
    frames = []
    for reader, path in ANNOTATION_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing annotation file for reader {reader}: {path}")
        raw = pd.read_csv(path)
        raw = raw[~raw["I"].isin(["core", "edge"])].copy()
        raw["annotator"] = reader
        raw["image_id"] = raw["Sample"].astype(str) + ".jpg"
        frames.append(raw[["image_id", "Sample", "annotator", "X", "Y", "Age", "Quality"]])
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"X": "x", "Y": "y", "Age": "age", "Quality": "quality"})
    return df


def validate_annotation_bounds(df: pd.DataFrame, image_dir: Path) -> None:
    """Loud warning (not a silent drop) for any point outside its own image's real pixel bounds —
    the most likely real-world failure mode (annotations made on a resized/cropped export)."""
    bad = 0
    for image_id, group in df.groupby("image_id"):
        path = image_dir / image_id
        if not path.exists():
            print(f"  [validate] BRAK PLIKU: {image_id}")
            bad += 1
            continue
        w, h = PILImage.open(path).size
        oob = group[(group.x < 0) | (group.x >= w) | (group.y < 0) | (group.y >= h)]
        if len(oob):
            print(f"  [validate] {image_id}: {len(oob)} punkt(y) POZA granicami obrazu ({w}x{h})")
            bad += 1
    if bad == 0:
        print("  [validate] OK — wszystkie punkty mieszczą się w granicach swoich obrazów")


# ---------------------------------------------------------------------------
# Step 2.5 — disambiguate + crop: every ZEGAR photo holds a FISH'S PAIR of otoliths
# side by side (confirmed 12.08 by direct visual inspection — a raw, unprocessed capture,
# unlike production `Z:/.../Processed/*_Single1_Right.jpg` / `*_Single2_Left.jpg`, which are
# already single-otolith crops). `segment_otolith` always returns the LARGEST connected blob
# (`max(contours, key=cv2.contourArea)`), which is only sometimes the one the experts actually
# annotated — empirically the annotated nucleus falls in the left half of the frame for 35/42
# samples, right half for 7/42, so "largest blob" and "annotated blob" disagree unpredictably.
# Feeding the model the raw two-otolith photo is ALSO an out-of-distribution input it was never
# trained on. Fix: segment once, and if the mean annotated point doesn't fall inside that blob,
# erase it and re-segment to recover the second (actually-annotated) blob, then crop to it.
# ---------------------------------------------------------------------------

def resolve_and_crop_target_otolith(
    raw_rgb: np.ndarray, mean_ann_xy: tuple[float, float], seg_params: dict, pad_frac: float = 0.12,
) -> tuple[Optional[np.ndarray], int, int, bool]:
    """Returns (cropped_rgb, crop_x0, crop_y0, used_second_blob); cropped_rgb is None if
    segmentation fails entirely (neither blob found) — caller falls back to the uncropped photo."""
    from src.otolith_axis import segment_otolith, mask_bbox

    mask1 = segment_otolith(raw_rgb, **seg_params)
    if mask1 is None:
        return None, 0, 0, False
    mx, my = int(round(mean_ann_xy[0])), int(round(mean_ann_xy[1]))
    H, W = mask1.shape
    inside1 = 0 <= my < H and 0 <= mx < W and mask1[my, mx] > 0
    target_mask, used_second = mask1, False
    if not inside1:
        erased = raw_rgb.copy()
        erased[mask1 > 0] = 0
        mask2 = segment_otolith(erased, **seg_params)
        if mask2 is not None:
            target_mask, used_second = mask2, True
        # else: mask2 failed (e.g. annotation actually falls outside BOTH blobs) — keep mask1
        # as the best available guess rather than crash; validate_annotation_bounds already
        # confirmed the raw click coordinates themselves are sane.
    x0, y0, w, h = mask_bbox(target_mask, pad_frac=pad_frac)
    return raw_rgb[y0:y0 + h, x0:x0 + w], x0, y0, used_second


# ---------------------------------------------------------------------------
# Step 3 — geometry: (x, y) -> t (inverse of src/ring_extraction.py::_project_to_axis)
# ---------------------------------------------------------------------------

def point_to_axis_t(x: float, y: float, axis_info: dict) -> float:
    """Scalar projection of (x, y) onto the (centroid -> far_edge) measurement axis.

    t=0 at the nucleus, t=1 at the far edge — same convention as the rest of this project
    (otolith_axis.compute_polar_grid, ring_extraction._project_to_axis). NOT clamped to [0, 1] —
    a raw annotator click can legitimately fall slightly outside that range; clamping would hide
    a real (if small) geometry mismatch instead of surfacing it.
    """
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    dx, dy = fx - cx, fy - cy
    denom = dx * dx + dy * dy
    if denom < 1e-9:
        return 0.0
    return float(((x - cx) * dx + (y - cy) * dy) / denom)


def raw_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x1 - x2, y1 - y2))


# ---------------------------------------------------------------------------
# Step 4 — matching (Hungarian assignment with a rejection threshold)
# ---------------------------------------------------------------------------

def match_t_values(t_a: list[float], t_b: list[float], max_gap: float):
    """Match two lists of axis positions (Hungarian algorithm), rejecting any pair further apart
    than ``max_gap``. Returns (pairs, unmatched_a_idx, unmatched_b_idx); pairs are (i, j) indices
    into t_a/t_b respectively.
    """
    n, m = len(t_a), len(t_b)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    cost = np.abs(np.asarray(t_a)[:, None] - np.asarray(t_b)[None, :])
    size = n + m
    padded = np.full((size, size), max_gap, dtype=float)
    padded[:n, :m] = cost
    row_ind, col_ind = linear_sum_assignment(padded)
    pairs, used_a, used_b = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m and cost[r, c] <= max_gap:
            pairs.append((r, c))
            used_a.add(r)
            used_b.add(c)
    unmatched_a = [i for i in range(n) if i not in used_a]
    unmatched_b = [j for j in range(m) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


def mean_nn_distance(pts_a: list[tuple[float, float]], pts_b: list[tuple[float, float]]) -> Optional[float]:
    """Mean nearest-neighbour distance from each point in ``pts_a`` to its closest point in
    ``pts_b`` — same definition as run_pipeline.py's ``_mean_dist`` closure (unimportable, it's
    nested), reimplemented here to the same semantics for a directly comparable number."""
    if not pts_a or not pts_b:
        return None
    a = np.asarray(pts_a, dtype=float)
    b = np.asarray(pts_b, dtype=float)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


# ---------------------------------------------------------------------------
# Step 5 — Beamish & Fournier (1981) Average Percent Error, on RING COUNTS
# ---------------------------------------------------------------------------

def beamish_fournier_ape(counts: pd.DataFrame) -> dict:
    """``counts``: DataFrame indexed by Sample, one column per reader, ring COUNT values.

    APE_j = mean_i |X_ij - Xbar_j| / Xbar_j ; reported as a percentage, averaged over samples.
    CV = APE * sqrt(R) / sqrt(R-1) for R readers (here R=2 -> factor sqrt(2)), the more commonly
    cited modern variant of the same statistic.
    """
    xbar = counts.mean(axis=1)
    valid = xbar > 0
    per_fish_ape = (counts[valid].sub(xbar[valid], axis=0).abs().div(xbar[valid], axis=0)).mean(axis=1)
    ape_pct = float(per_fish_ape.mean() * 100.0)
    cv_pct = ape_pct * float(np.sqrt(2.0))
    return {"APE_percent": ape_pct, "CV_percent": cv_pct, "n_fish": int(valid.sum())}


# ---------------------------------------------------------------------------
# Step 6 — visual overlays (PRIMARY deliverable per this project's own house rule: visual
# interpretation is not optional next to the numbers — see memory `feedback`).
# ---------------------------------------------------------------------------

_KK_COLOR = (255, 0, 255)      # magenta, filled — annotator KK
_SS_COLOR = (0, 128, 255)      # orange-blue, filled — annotator SS
_MODEL_COLOR = (255, 0, 0)     # red, filled+border — model final points (matches project convention)
_CLASSICAL_COLOR = (0, 200, 0)  # green, hollow — classical (OpenCV) reference


def render_expert_annotation_overlay(
    image: np.ndarray, axis_info: dict,
    kk_pts: list, ss_pts: list, model_pts: list, classical_pts: list,
) -> np.ndarray:
    """Otolith photo + contour + axis + all four point sets, distinctly coloured — the
    per-image visual check for this evaluation. Reuses this project's own established
    drawing primitives (src/visualization.py) rather than re-inventing them."""
    from src.visualization import _CONTOUR_COLOR, _AXIS_COLOR, _draw_small_points, _draw_hollow_points

    img = np.ascontiguousarray(image[..., :3]).copy()
    H = img.shape[0]
    lt = max(2, H // 300)
    contour = axis_info.get("contour")
    if contour is not None:
        cv2.drawContours(img, [contour], -1, _CONTOUR_COLOR, lt)
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    cv2.line(img, (int(cx), int(cy)), (int(fx), int(fy)), _AXIS_COLOR, lt)

    r = max(3, H // 250)
    _draw_small_points(img, kk_pts, _KK_COLOR, r)
    _draw_small_points(img, ss_pts, _SS_COLOR, r)
    _draw_small_points(img, model_pts, _MODEL_COLOR, r, border=True)
    _draw_hollow_points(img, classical_pts, _CLASSICAL_COLOR, r + 2)
    return img


# ---------------------------------------------------------------------------
# Step 7 — self-contained HTML report (display:inline-block, NEVER flex — house rule)
# ---------------------------------------------------------------------------

def build_html_report(results_df: pd.DataFrame, aggregate: dict, cards: list, output_path: Path) -> None:
    from src.report_common import fig_to_b64, img_tag
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parts = ["<html><head><meta charset='utf-8'><title>ZEGAR — ewaluacja na adnotacjach eksperckich</title>",
             "<style>body{font-family:sans-serif;max-width:1400px;margin:0 auto;padding:20px;}",
             "table{border-collapse:collapse;margin:12px 0;} td,th{border:1px solid #ccc;padding:6px 10px;text-align:right;}",
             "th{background:#eee;} .card{display:inline-block;vertical-align:top;width:420px;margin:0 12px 24px 0;",
             "border:1px solid #ddd;padding:8px;} .legend span{display:inline-block;width:12px;height:12px;",
             "border-radius:50%;margin-right:4px;vertical-align:middle;}</style></head><body>"]

    parts.append("<h1>ZEGAR — ewaluacja lokalizacji na prawdziwych adnotacjach eksperckich</h1>")
    parts.append("<p>Ground truth = konsensus dwóch doświadczonych czytelników (<b>KK</b>, <b>SS</b>); "
                  "trzeci czytelnik (DD, w trakcie nauki) pominięty. Referencyjny checkpoint: "
                  "<code>outputs/22.07_reg</code> (ostatni potwierdzony dobry punkt odniesienia).</p>")

    parts.append("<h2>Wyniki zbiorcze</h2><table><tr><th>Metryka</th><th>Wartość (px)</th></tr>")
    for label, key in [
        ("Model vs. konsensus GT (na osi)", "model_vs_gt_axis_px"),
        ("Klasyka (OpenCV) vs. konsensus GT (na osi)", "classical_vs_gt_axis_px"),
        ("KK vs. SS — podłoga szumu (na osi)", "kk_vs_ss_axis_px"),
        ("Model vs. GT (surowa odległość 2D)", "model_vs_gt_raw_px"),
        ("KK vs. SS (surowa odległość 2D)", "kk_vs_ss_raw_px"),
    ]:
        v = aggregate.get(key)
        parts.append(f"<tr><td style='text-align:left'>{label}</td><td>{v:.1f}</td></tr>" if v is not None
                     else f"<tr><td style='text-align:left'>{label}</td><td>brak danych</td></tr>")
    parts.append(f"<tr><td style='text-align:left'>Liczba ocenionych zdjęć</td>"
                 f"<td>{aggregate['n_images_scored']}/{aggregate['n_images_total']}</td></tr>")
    parts.append("</table>")

    bf = aggregate.get("beamish_fournier", {})
    parts.append("<h2>Beamish &amp; Fournier (1981) — precyzja liczby przyrostów (KK vs SS)</h2>")
    parts.append(f"<table><tr><th>APE</th><th>CV</th><th>n ryb</th></tr>"
                 f"<tr><td>{bf.get('APE_percent', float('nan')):.1f}%</td>"
                 f"<td>{bf.get('CV_percent', float('nan')):.1f}%</td>"
                 f"<td>{bf.get('n_fish', 0)}</td></tr></table>")

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Model vs GT", "Klasyka vs GT", "KK vs SS\n(podłoga szumu)"]
    values = [aggregate.get("model_vs_gt_axis_px"), aggregate.get("classical_vs_gt_axis_px"),
             aggregate.get("kk_vs_ss_axis_px")]
    colors = ["#d62728", "#2ca02c", "#7f7f7f"]
    valid = [(l, v, c) for l, v, c in zip(labels, values, colors) if v is not None]
    if valid:
        ax.bar([v[0] for v in valid], [v[1] for v in valid], color=[v[2] for v in valid])
        ax.set_ylabel("mean_dist do GT (px, na osi)")
        ax.set_title("Prawdziwy ground truth — model vs klasyka vs podłoga szumu ludzi")
        parts.append(f'<div>{img_tag(fig_to_b64(fig), alt="wykres zbiorczy")}</div>')

    parts.append("<h2>Karty per-obraz</h2>")
    parts.append("<p class='legend'><span style='background:rgb(255,0,255)'></span>KK &nbsp;"
                 "<span style='background:rgb(0,128,255)'></span>SS &nbsp;"
                 "<span style='background:rgb(255,0,0)'></span>model (finalne) &nbsp;"
                 "<span style='background:rgb(0,200,0);border-radius:50%;border:2px solid green'></span>klasyka (obrys)</p>")
    for card in cards:
        b64 = fig_to_b64_from_array(card["overlay"]) if "overlay" in card else None
        row = card["row"]
        parts.append("<div class='card'>")
        parts.append(f"<b>{row['Sample']}</b> — wiek modelu: {row['predicted_age']}, "
                     f"KK: {row.get('kk_age')}, SS: {row.get('ss_age')}<br>")
        if b64:
            parts.append(img_tag(b64, alt=row["Sample"]))
        mv = row.get("model_vs_gt_axis_px")
        cv_ = row.get("classical_vs_gt_axis_px")
        nv = row.get("kk_vs_ss_axis_px")
        parts.append(f"<br>model-vs-GT: {mv:.0f}px | klasyka-vs-GT: {cv_:.0f}px | KK-vs-SS: {nv:.0f}px"
                     if all(v is not None for v in (mv, cv_, nv)) else "<br>(niepełne dane)")
        parts.append("</div>")

    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def fig_to_b64_from_array(arr: np.ndarray) -> str:
    """Encode an RGB numpy array (cv2-drawn overlay) directly to base64 PNG, no matplotlib."""
    import base64, io
    img = PILImage.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Step 8 — main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ZEGAR — ewaluacja na prawdziwych adnotacjach eksperckich (KK + SS, DD pominięty)")
    print("=" * 70)

    print("\n[1/7] Wczytywanie adnotacji...")
    ann = load_expert_annotations()
    samples = sorted(ann["Sample"].unique())
    print(f"  {len(samples)} próbek, {len(ann)} zaadnotowanych przyrostów (KK+SS łącznie)")
    validate_annotation_bounds(ann, IMAGE_DIR)

    print("\n[2/7] Wybór właściwego otolitu z pary + przycinanie...")
    from scripts.run_pipeline import load_merged_config, _compute_axis_data_for_samples
    cfg = load_merged_config(REFERENCE_CONFIG, None)
    # Private mask cache for THIS eval only. The default (data/masks_cache/, shared/global,
    # keyed only by image_id stem — see src/dataset.py's own comment: "a mask depends solely
    # on the raw image") breaks here on purpose-violated assumption: our cropped single-otolith
    # "Z01.jpg" reuses the SAME image_id as the raw two-otolith "Z01.jpg" we already segmented
    # once before applying this fix, so the shared cache would silently serve the stale,
    # wrong-sized mask (IndexError, caught 12.08 on the first post-fix run) instead of
    # recomputing for the new, differently-sized cropped image.
    cfg.data.mask_cache_dir = str(OUTPUT_DIR / "dataset_masks_cache")
    # Same collision hits _compute_axis_data_for_samples's OWN mask cache (cond_dir/"masks/",
    # also keyed only by image_id stem, no override point) — any prior run of this script left
    # masks computed on the uncropped two-otolith photo there. Clear it so every run starts clean.
    import shutil as _shutil
    _shutil.rmtree(OUTPUT_DIR / "masks", ignore_errors=True)
    seg_params = cfg.segmentation.as_params()
    CROPPED_DIR = OUTPUT_DIR / "cropped_singles"
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    crop_offsets: dict[str, tuple[int, int]] = {}
    n_second_blob = n_seg_failed = 0
    for sample in samples:
        image_id = f"{sample}.jpg"
        sub = ann[ann.Sample == sample]
        mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
        raw_rgb = np.array(PILImage.open(IMAGE_DIR / image_id).convert("RGB"), dtype=np.uint8)
        cropped, x0, y0, used_second = resolve_and_crop_target_otolith(raw_rgb, mean_xy, seg_params)
        if cropped is None:
            print(f"  [crop] {image_id}: segmentacja nieudana, używam pełnego zdjęcia")
            cropped, x0, y0 = raw_rgb, 0, 0
            n_seg_failed += 1
        elif used_second:
            n_second_blob += 1
        crop_offsets[sample] = (x0, y0)
        PILImage.fromarray(cropped).save(CROPPED_DIR / image_id)
    print(f"  Przycięto {len(samples)} zdjęć do pojedynczego otolitu -> {CROPPED_DIR}")
    print(f"  Adnotowany otolit = NIE największa plama na zdjęciu w {n_second_blob}/{len(samples)} "
          f"przypadkach (segmentacja nieudana: {n_seg_failed})")

    # Rebase annotation (x, y) into crop-local coordinates — everything downstream (model axis,
    # candidate points, overlays) is computed in the CROPPED frame, so ground truth must match it.
    ann["x"] = ann.apply(lambda r: r.x - crop_offsets[r.Sample][0], axis=1)
    ann["y"] = ann.apply(lambda r: r.y - crop_offsets[r.Sample][1], axis=1)

    print("\n[3/7] Wczytanie modelu (22.07_reg)...")
    from src.dataset import OtolithDataset
    from src.inference import load_model_from_checkpoint, run_inference
    from torch.utils.data import DataLoader

    if not REFERENCE_CKPT.exists():
        sys.exit(f"Brak checkpointu: {REFERENCE_CKPT}")
    model = load_model_from_checkpoint(cfg, REFERENCE_CKPT)

    # Scratch labels CSV (required by OtolithDataset: image_id, age, split) — "age" here is
    # KK's read age (an experienced reader), used ONLY to satisfy the dataset's schema / to get a
    # target_age/abs_error bonus column in predictions.csv; the model's OWN predicted_age (what we
    # actually need) does not depend on this value at all.
    age_by_sample = (ann[ann.annotator == "KK"].drop_duplicates("Sample").set_index("Sample")["age"])
    scratch_csv = OUTPUT_DIR / "scratch_labels.csv"
    pd.DataFrame([
        {"image_id": f"{s}.jpg", "age": int(age_by_sample.get(s, 0)), "split": "test"}
        for s in samples
    ]).to_csv(scratch_csv, index=False)

    print("\n[4/7] Inferencja wieku (na przyciętych, pojedynczych otolitach)...")
    ds = OtolithDataset(cfg, split="test", labels_csv=str(scratch_csv), image_dir=str(CROPPED_DIR))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    run_inference(cfg, model, loader, OUTPUT_DIR)
    preds = pd.read_csv(OUTPUT_DIR / "predictions.csv")
    preds["age"] = preds["target_age"]   # for _compute_axis_data_for_samples's row.get("age")

    print("\n[5/7] Lokalizacja modelu (density/klasyka) na 42 zdjęciach...")
    sample_rows = preds.to_dict("records")
    grids, axis_data, _walkthrough = _compute_axis_data_for_samples(
        sample_rows, CROPPED_DIR, cfg, REFERENCE_CKPT, OUTPUT_DIR,
    )

    print("\n[6/7] Dopasowanie punktów, metryki i nakładki wizualne...")
    from src.otolith_axis import apply_background_mask
    max_gap_t = cfg.candidates.min_peak_distance / N_SAMPLES_AXIS
    per_image_results = []
    ring_counts = {"KK": {}, "SS": {}}
    cards = []

    for sample in samples:
        image_id = f"{sample}.jpg"
        adata = axis_data.get(image_id)
        if adata is None or adata.get("axis_info") is None:
            print(f"  [skip] {image_id}: segmentacja/oś nie powiodła się")
            continue
        axis_info = adata["axis_info"]

        # Annotator points -> t
        sub = ann[ann.image_id == image_id]
        t_kk = sub[sub.annotator == "KK"].apply(lambda r: point_to_axis_t(r.x, r.y, axis_info), axis=1).tolist()
        t_ss = sub[sub.annotator == "SS"].apply(lambda r: point_to_axis_t(r.x, r.y, axis_info), axis=1).tolist()
        xy_kk = list(zip(sub[sub.annotator == "KK"].x, sub[sub.annotator == "KK"].y))
        xy_ss = list(zip(sub[sub.annotator == "SS"].x, sub[sub.annotator == "SS"].y))
        ring_counts["KK"][sample] = len(t_kk)
        ring_counts["SS"][sample] = len(t_ss)

        # Consensus GT = mean t of KK/SS matched pairs (Beamish & Fournier philosophy: compare to
        # the mean across readers, not reader-to-reader directly).
        pairs, unmatched_kk, unmatched_ss = match_t_values(t_kk, t_ss, max_gap_t)
        gt_t = [(t_kk[i] + t_ss[j]) / 2.0 for i, j in pairs]
        n_singletons = len(unmatched_kk) + len(unmatched_ss)

        # Model / classical points -> t (already lie ON the axis by construction)
        final_pts = adata.get("final_axis_pts") or []
        classical_pts = adata.get("classical_pts") or []
        t_model = [point_to_axis_t(x, y, axis_info) for x, y in final_pts]
        t_classical = [point_to_axis_t(x, y, axis_info) for x, y in classical_pts]

        def _axis_dist(t_x, t_gt):
            if not t_x or not t_gt:
                return None
            length = axis_info["length_px"]
            d = np.abs(np.asarray(t_x)[:, None] - np.asarray(t_gt)[None, :]) * length
            return float(d.min(axis=1).mean())

        model_vs_gt_axis = _axis_dist(t_model, gt_t)
        classical_vs_gt_axis = _axis_dist(t_classical, gt_t)
        kk_vs_ss_axis = _axis_dist(t_kk, t_ss)

        model_vs_gt_raw = mean_nn_distance(final_pts, xy_kk + xy_ss) if final_pts else None
        kk_vs_ss_raw = mean_nn_distance(xy_kk, xy_ss)

        row_result = {
            "Sample": sample, "image_id": image_id,
            "n_gt_rings": len(gt_t), "n_kk": len(t_kk), "n_ss": len(t_ss),
            "n_singletons": n_singletons,
            "n_model_final": len(t_model), "n_classical": len(t_classical),
            "predicted_age": int(preds.loc[preds.image_id == image_id, "predicted_age"].iloc[0]),
            "kk_age": int(sub[sub.annotator == "KK"].age.iloc[0]) if len(sub[sub.annotator == "KK"]) else None,
            "ss_age": int(sub[sub.annotator == "SS"].age.iloc[0]) if len(sub[sub.annotator == "SS"]) else None,
            "model_vs_gt_axis_px": model_vs_gt_axis,
            "classical_vs_gt_axis_px": classical_vs_gt_axis,
            "kk_vs_ss_axis_px": kk_vs_ss_axis,
            "model_vs_gt_raw_px": model_vs_gt_raw,
            "kk_vs_ss_raw_px": kk_vs_ss_raw,
        }
        per_image_results.append(row_result)

        # Visual card — reload the real photo (axis_data doesn't retain orig_rgb) and, when the
        # model trains on masked-background input, mask it the same way so the overlay matches
        # what the model actually saw (same rule run_pipeline.py itself follows for its cards).
        try:
            photo = np.array(PILImage.open(CROPPED_DIR / image_id).convert("RGB"), dtype=np.uint8)
            mask_arr = axis_info.get("mask")
            if cfg.data.mask_background and mask_arr is not None:
                photo = apply_background_mask(photo, mask_arr)
            overlay = render_expert_annotation_overlay(
                photo, axis_info, xy_kk, xy_ss, final_pts, classical_pts,
            )
            cards.append({"overlay": overlay, "row": row_result})
        except Exception as e:
            print(f"  [overlay] błąd dla {image_id}: {e}")

    results_df = pd.DataFrame(per_image_results)
    results_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    def _agg(col):
        v = results_df[col].dropna()
        return float(v.mean()) if len(v) else None

    aggregate = {
        "n_images_scored": len(results_df),
        "n_images_total": len(samples),
        "model_vs_gt_axis_px": _agg("model_vs_gt_axis_px"),
        "classical_vs_gt_axis_px": _agg("classical_vs_gt_axis_px"),
        "kk_vs_ss_axis_px": _agg("kk_vs_ss_axis_px"),
        "model_vs_gt_raw_px": _agg("model_vs_gt_raw_px"),
        "kk_vs_ss_raw_px": _agg("kk_vs_ss_raw_px"),
    }

    counts_df = pd.DataFrame(ring_counts)
    ape = beamish_fournier_ape(counts_df)
    aggregate["beamish_fournier"] = ape

    import json
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps({"aggregate": aggregate, "per_image": per_image_results}, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n[7/7] Wyniki zbiorcze:")
    for k, v in aggregate.items():
        print(f"   {k}: {v}")

    print(f"\nZapisano: {OUTPUT_DIR / 'metrics.csv'}, {OUTPUT_DIR / 'metrics.json'}")

    report_path = OUTPUT_DIR / "report.html"
    build_html_report(results_df, aggregate, cards, report_path)
    print(f"Raport HTML z nakładkami: {report_path} ({len(cards)}/{len(samples)} kart)")


if __name__ == "__main__":
    main()
