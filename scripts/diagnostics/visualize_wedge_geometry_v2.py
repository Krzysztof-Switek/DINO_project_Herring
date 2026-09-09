"""09.09 (follow-up to prototype_polar_wedge.py) — regenerates the wedge-geometry report figures
using the PRODUCTION `src/wedge_extraction.py` module instead of the Faza-0 prototype's own
duplicate code, for two reasons:

1. `prototype_polar_wedge.py` predates the non-uniform radial-resolution fix (Faza 0b) — its own
   `extract_polar_wedge`/`ts = row_idx / (canvas_h-1)` is still LINEAR and still uses the old
   95-patch resolution. The report figures built from it (`outputs/09.09_masked_wizualizacje/
   wedge_geometry/*_wedge_prototype.png`, embedded in report4.html §4) are stale relative to what
   is actually implemented and tested in `src/wedge_extraction.py` today (44 patches, non-uniform
   warp). This script uses the real module so the figures show what the code actually does.
2. User feedback (09.09, mid-review of report4.html): the unwrapped wedge canvas reads
   nucleus-at-top / edge-at-bottom (row 0 = t=0 by construction, and image row 0 renders at the
   top) — the OPPOSITE of the direction a reader naturally scans, and opposite of where the
   information actually is (dense near the edge, per the r=-0.53 finding). Flipped here for
   display only (`np.flip(..., axis=0)`) — does NOT touch `src/wedge_extraction.py`'s row/t
   convention, which stays as-is (already wired through the model and validated by round-trip
   tests; this is a report-readability fix, not a geometry change).

Also draws faint patch-row gridlines on the flipped canvas so the non-uniform resolution itself is
directly VISIBLE, not just claimed in the caption — answers, on the image itself, "is the
increasing-resolution-toward-the-edge mechanism actually implemented": if it is, gridlines pack
tightly near the top (edge) and spread out near the bottom (nucleus).

Usage:
    python scripts/diagnostics/visualize_wedge_geometry_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image as PILImage

from scripts.diagnostics.expert_annotation_eval import (
    IMAGE_DIR, REFERENCE_CONFIG, load_expert_annotations, resolve_and_crop_target_otolith,
)
from scripts.run_pipeline import load_merged_config
from src.otolith_axis import detect_axis, apply_background_mask
from src.visualization import _CONTOUR_COLOR, _AXIS_COLOR
from src.wedge_extraction import (
    extract_polar_wedge, point_to_wedge_xy, wedge_geometry_from_axis_info, wedge_xy_to_point,
)

OUT = PROJECT_ROOT / "outputs" / "09.09_masked_wizualizacje" / "wedge_geometry"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = ["Z20", "Z38", "Z37", "Z15"]  # same four as the Faza 0 prototype, for direct comparison

cfg = load_merged_config(REFERENCE_CONFIG, None)
DELTA_THETA_DEG = cfg.data.wedge_delta_theta_deg
N_ANGLE_PATCHES = cfg.data.wedge_n_angle_patches
N_RADIUS_PATCHES = cfg.data.wedge_n_radius_patches
PATCH_SIZE = cfg.data.patch_size
CANVAS_W = N_ANGLE_PATCHES * PATCH_SIZE
CANVAS_H = N_RADIUS_PATCHES * PATCH_SIZE

print(f"Delta-theta={DELTA_THETA_DEG}deg, kolumny={N_ANGLE_PATCHES}, "
      f"wiersze(patche)={N_RADIUS_PATCHES}, canvas={CANVAS_W}x{CANVAS_H} "
      f"(src.wedge_extraction, nierownomierny warp)")

ann = load_expert_annotations()
seg_params = cfg.segmentation.as_params()

for sample in SAMPLES:
    image_id = f"{sample}.jpg"
    sub = ann[ann.Sample == sample]
    mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
    raw_rgb = np.array(PILImage.open(IMAGE_DIR / image_id).convert("RGB"), dtype=np.uint8)
    cropped, x0, y0, _ = resolve_and_crop_target_otolith(raw_rgb, mean_xy, seg_params)
    axis_info = detect_axis(cropped, seg_params=seg_params,
                             nucleus_method=cfg.segmentation.nucleus_method,
                             axis_method=cfg.segmentation.axis_method)
    mask = axis_info["mask"]
    photo = apply_background_mask(cropped, mask)
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]

    geom = wedge_geometry_from_axis_info(mask, axis_info, DELTA_THETA_DEG, CANVAS_W, CANVAS_H)
    wedge = extract_polar_wedge(photo, mask, geom)

    # round-trip check with the PRODUCTION functions (same check as tests/test_wedge_extraction.py,
    # here on a real photo instead of a synthetic circle)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(200):
        col = rng.uniform(0, CANVAS_W - 1)
        row = rng.uniform(1, CANVAS_H - 1)  # row=0 (nucleus) is the documented angle-undefined case
        x, y = wedge_xy_to_point(col, row, geom)
        back = point_to_wedge_xy(x, y, geom)
        if back is None:
            continue
        c2, r2, _ = back
        errs.append(np.hypot(col - c2, row - r2))
    errs = np.array(errs)
    print(f"{sample}: round-trip error px (canvas units) mean={errs.mean():.4f} max={errs.max():.4f}")

    # --- left panel: original photo, wedge boundary rays, axis, KK/SS points (unchanged layout) ---
    H, W = photo.shape[:2]
    overview = photo.copy()
    lt = max(2, H // 300)
    cv2.drawContours(overview, [axis_info["contour"]], -1, _CONTOUR_COLOR, lt)
    cv2.line(overview, (int(cx), int(cy)), (int(fx), int(fy)), _AXIS_COLOR, lt)
    for sign in (-1, 1):
        edge_col = CANVAS_W - 1 if sign > 0 else 0
        x, y = wedge_xy_to_point(edge_col, CANVAS_H - 1, geom)
        cv2.line(overview, (int(cx), int(cy)), (int(x), int(y)), (255, 140, 0), lt)

    kk_xy = list(zip(sub[sub.annotator == "KK"].x - x0, sub[sub.annotator == "KK"].y - y0))
    ss_xy = list(zip(sub[sub.annotator == "SS"].x - x0, sub[sub.annotator == "SS"].y - y0))
    for x, y in kk_xy:
        cv2.circle(overview, (int(x), int(y)), 6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(overview, (int(x), int(y)), 6, (255, 60, 220), 2, cv2.LINE_AA)
    for x, y in ss_xy:
        cv2.circle(overview, (int(x), int(y)), 5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(overview, (int(x), int(y)), 5, (60, 160, 255), 2, cv2.LINE_AA)

    # --- right panel: unwrapped wedge, points drawn, THEN flipped so edge is at the top ---
    wedge_vis = wedge.copy()
    n_outside = 0
    for x, y in kk_xy + ss_xy:
        proj = point_to_wedge_xy(x, y, geom)
        if proj is None:
            n_outside += 1
            continue
        col, row, _t = proj
        color = (255, 60, 220) if (x, y) in kk_xy else (60, 160, 255)
        cv2.circle(wedge_vis, (int(round(col)), int(round(row))), 5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(wedge_vis, (int(round(col)), int(round(row))), 5, color, 2, cv2.LINE_AA)
    total_pts = len(kk_xy) + len(ss_xy)
    print(f"{sample}: {total_pts - n_outside}/{total_pts} prawdziwych przyrostów mieści się "
          f"w wycinku (Delta-theta={DELTA_THETA_DEG:.0f}deg)")

    # patch-row gridlines BEFORE flipping (row index space is unambiguous here) — makes the
    # non-uniform resolution directly visible: dense lines near row=canvas_h (edge), sparse near
    # row=0 (nucleus).
    for p in range(N_RADIUS_PATCHES + 1):
        row_px = min(p * PATCH_SIZE, CANVAS_H - 1)
        cv2.line(wedge_vis, (0, row_px), (CANVAS_W - 1, row_px), (255, 255, 255), 1, cv2.LINE_AA)

    wedge_vis = np.flip(wedge_vis, axis=0).copy()  # row 0 (nucleus) now renders at the BOTTOM

    wedge_vis_big = np.array(
        PILImage.fromarray(wedge_vis).resize((CANVAS_W * 3, CANVAS_H * 2), PILImage.NEAREST))
    # side labels: "BRZEG" at top, "JĄDRO" at bottom, unambiguous even without reading the caption
    label_pad = 34
    labeled = np.full((wedge_vis_big.shape[0] + 2 * label_pad, wedge_vis_big.shape[1], 3),
                       255, dtype=np.uint8)
    labeled[label_pad:label_pad + wedge_vis_big.shape[0]] = wedge_vis_big
    cv2.putText(labeled, "BRZEG (t=1)", (8, label_pad - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(labeled, "JADRO (t=0)", (8, labeled.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (20, 20, 20), 1, cv2.LINE_AA)

    ov_resized = np.array(PILImage.fromarray(overview).resize(
        (int(overview.shape[1] * labeled.shape[0] / overview.shape[0]), labeled.shape[0]),
        PILImage.LANCZOS))
    combo = np.concatenate([ov_resized, labeled], axis=1)
    out_path = OUT / f"{sample}_wedge_v2.png"
    PILImage.fromarray(combo).save(out_path)
    print(f"  zapisano {out_path.name}")

print("\nZapisano wizualizacje do", OUT)
