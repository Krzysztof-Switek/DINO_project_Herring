"""09.09 follow-up to report4.html's §5 ("Zaobserwowany artefakt transformacji") — quantifies, on
real data, the angular-resolution claim: with a FIXED number of angular columns (13,
`data.wedge_n_angle_patches`), one column always spans the SAME real angular width Delta-theta/13,
but the REAL ARC LENGTH (in source-image pixels) that angular width covers grows linearly with
radius. Near the nucleus this means each canvas column is built from very few real source pixels
(harmless oversampling/interpolation); near the otolith edge, each canvas column may need to
represent far MORE real source pixels than its own width (14px, one DINOv2 patch) can hold --
information genuinely lost, not just visually "blurred".

This is a DIFFERENT axis from the non-uniform RADIAL warp already implemented
(`src/wedge_extraction.py::_RADIAL_WARP_T`, Faza 0b) -- that fix reparametrises which real radius
maps to which ROW; it cannot and does not touch how many REAL PIXELS one COLUMN represents, since
every row uses the identical, radius-independent set of column->angle assignments.

Method (same style as `analyze_zegar_ring_spacing.py` / `analyze_wedge_radial_resolution_profile.py`
-- real per-image geometry, not a single assumed otolith size): for each of the 42 ZEGAR images,
crop + `detect_axis` (reused, unchanged) gives the real `length_px` (nucleus-to-edge, in source
pixels). One column's real arc length at radius r is `r * (Delta_theta_rad / n_angle_patches)`;
comparing that to the canvas's own per-column pixel width (`patch_size`=14, the same native
resolution unit every other geometry in this project already uses -- square image, strip, wedge
radial axis) gives a direct, unitless "compression ratio": >1 means information loss (more real
pixels asked to fit into fewer canvas pixels), <1 means harmless oversampling.

Usage:
    python scripts/diagnostics/analyze_wedge_angular_resolution.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image as PILImage

from scripts.diagnostics.expert_annotation_eval import (
    IMAGE_DIR, REFERENCE_CONFIG, load_expert_annotations, resolve_and_crop_target_otolith,
)
from scripts.run_pipeline import load_merged_config
from src.otolith_axis import detect_axis

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "09.09_masked_wizualizacje" / "wedge_angular_resolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = load_merged_config(REFERENCE_CONFIG, None)
seg_params = cfg.segmentation.as_params()
PATCH_SIZE = cfg.data.patch_size                       # 14, native resolution unit everywhere else
DELTA_THETA_DEG = cfg.data.wedge_delta_theta_deg        # 98.7
N_ANGLE_PATCHES = cfg.data.wedge_n_angle_patches        # 13
DELTA_THETA_RAD = np.radians(DELTA_THETA_DEG)
COLUMN_ANGLE_RAD = DELTA_THETA_RAD / N_ANGLE_PATCHES    # real angular width of ONE column


def compression_ratio(r_px: float) -> float:
    """Real arc-length (px) spanned by one column at radius r_px, divided by that column's own
    canvas pixel width (PATCH_SIZE). >1 = information loss (real pixels skipped by point-sample
    remap); <1 = harmless oversampling/interpolation."""
    arc_len_px = r_px * COLUMN_ANGLE_RAD
    return arc_len_px / PATCH_SIZE


def main() -> None:
    print("=" * 78)
    print("Rozdzielczość kątowa wycinka — realna kompresja przy stałej liczbie kolumn (13)")
    print("=" * 78)
    print(f"Delta-theta={DELTA_THETA_DEG}deg, n_angle_patches={N_ANGLE_PATCHES}, "
          f"column_angle={np.degrees(COLUMN_ANGLE_RAD):.3f}deg, patch_size={PATCH_SIZE}px")

    ann = load_expert_annotations()
    samples = sorted(ann["Sample"].unique())

    rows = []
    n_seg_failed = 0
    for sample in samples:
        image_id = f"{sample}.jpg"
        sub = ann[ann.Sample == sample]
        mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
        raw_rgb = np.array(PILImage.open(IMAGE_DIR / image_id).convert("RGB"), dtype=np.uint8)
        cropped, x0, y0, _ = resolve_and_crop_target_otolith(raw_rgb, mean_xy, seg_params)
        if cropped is None:
            n_seg_failed += 1
            continue
        axis_info = detect_axis(cropped, seg_params=seg_params,
                                 nucleus_method=cfg.segmentation.nucleus_method,
                                 axis_method=cfg.segmentation.axis_method)
        if axis_info is None:
            n_seg_failed += 1
            continue
        length_px = axis_info["length_px"]
        rows.append({
            "Sample": sample,
            "length_px": length_px,
            "ratio_at_edge_t1_0": compression_ratio(length_px),
            "ratio_at_t0_9": compression_ratio(0.9 * length_px),
            "ratio_at_t0_7": compression_ratio(0.7 * length_px),
            "ratio_at_t0_5": compression_ratio(0.5 * length_px),
            "ratio_at_t0_3": compression_ratio(0.3 * length_px),
        })

    print(f"\n{len(rows)}/{len(samples)} obrazów z policzoną osią ({n_seg_failed} pominiętych)")
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "angular_compression.csv", index=False)

    print("\nWspółczynnik kompresji (realne px łuku / 14px canvas) — statystyki po 42 obrazach:")
    summary = {}
    for col in ["ratio_at_t0_3", "ratio_at_t0_5", "ratio_at_t0_7", "ratio_at_t0_9",
                "ratio_at_edge_t1_0"]:
        v = df[col]
        summary[col] = {
            "mean": float(v.mean()), "median": float(v.median()),
            "min": float(v.min()), "max": float(v.max()),
            "n_above_1": int((v > 1.0).sum()), "n_above_2": int((v > 2.0).sum()),
        }
        print(f"  {col:20s}: mean={v.mean():6.2f}  median={v.median():6.2f}  "
              f"min={v.min():6.2f}  max={v.max():6.2f}  "
              f"(>1x: {int((v>1.0).sum())}/{len(v)}, >2x: {int((v>2.0).sum())}/{len(v)})")

    print(f"\nNajgorsze 5 przypadków (t=1.0, brzeg):")
    print(df.nlargest(5, "ratio_at_edge_t1_0")[["Sample", "length_px", "ratio_at_edge_t1_0"]]
          .to_string(index=False))

    # for direct comparison: what Delta-theta / n_angle_patches WOULD keep ratio<=1 at the edge,
    # given the MEDIAN real length_px -- not a recommendation, just a concrete "what it would take"
    median_length = float(df["length_px"].median())
    needed_columns_median = median_length * DELTA_THETA_RAD / PATCH_SIZE
    max_length = float(df["length_px"].max())
    needed_columns_max = max_length * DELTA_THETA_RAD / PATCH_SIZE
    print(f"\nDla ratio<=1 przy brzegu (zero utraty względem 14px) trzeba by:")
    print(f"  {needed_columns_median:.0f} kolumn przy medianowym length_px={median_length:.0f}px "
          f"(dziś: {N_ANGLE_PATCHES})")
    print(f"  {needed_columns_max:.0f} kolumn przy NAJDŁUŻSZYM realnym length_px={max_length:.0f}px")

    (OUTPUT_DIR / "angular_compression_summary.json").write_text(
        json.dumps({
            "delta_theta_deg": DELTA_THETA_DEG, "n_angle_patches": N_ANGLE_PATCHES,
            "patch_size": PATCH_SIZE, "n_images": len(rows),
            "summary": summary,
            "needed_columns_for_ratio_1_at_median_length": needed_columns_median,
            "needed_columns_for_ratio_1_at_max_length": needed_columns_max,
        }, indent=2), encoding="utf-8")
    print(f"\nZapisano: {OUTPUT_DIR / 'angular_compression.csv'}, "
          f"{OUTPUT_DIR / 'angular_compression_summary.json'}")


if __name__ == "__main__":
    main()
