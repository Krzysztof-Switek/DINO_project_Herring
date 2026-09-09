"""09.09 — visual explainer for the angular-resolution-bands plan (`plans and summaries/
09.09_wycinek_pasma_katowe_plan.md`): renders the real problem (angular compression growing with
radius), the real 4-band extraction on an actual ZEGAR photo (using the just-implemented
`src.wedge_extraction` band functions, Faza 9), and the cost/target trade-off that led to the
user's decision (target=1.0x, 5852 patches/sample, 4.27x Run N).

Produces three PNGs (embedded into report4.html §6):
  1. angular_compression_schematic_<sample>.png — real photo with widening angular sectors at
     increasing radius, showing WHY a fixed column count loses resolution near the edge.
  2. wedge_bands_demo_<sample>.png — the four REAL bands extracted from one real photo, at their
     true (measured) column counts, labelled.
  3. wedge_bands_cost_tradeoff.png — bar chart of patches/sample across target ratios + Run N.

Usage:
    python scripts/diagnostics/visualize_wedge_angular_bands.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

from scripts.diagnostics.expert_annotation_eval import (
    IMAGE_DIR, REFERENCE_CONFIG, load_expert_annotations, resolve_and_crop_target_otolith,
)
from scripts.run_pipeline import load_merged_config
from src.otolith_axis import apply_background_mask, detect_axis
from src.visualization import _AXIS_COLOR, _CONTOUR_COLOR
from src.wedge_extraction import extract_polar_wedge_band, wedge_band_geometries_from_axis_info

OUT = PROJECT_ROOT / "outputs" / "09.09_masked_wizualizacje" / "wedge_angular_resolution"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE = "Z38"
PATCH_SIZE = 14
DELTA_THETA_DEG = 98.7
BAND_EDGES_T = [0.0, 0.6, 0.8, 0.9, 1.0]
N_ANGLE_PATCHES = [97, 129, 145, 161]   # Faza 8 result, target=1.0x @ true max 1305.1px
N_RADIUS_PATCHES = [11, 12, 9, 12]      # from the existing non-uniform radial warp
BAND_CANVAS_W = [n * PATCH_SIZE for n in N_ANGLE_PATCHES]
BAND_CANVAS_H = [n * PATCH_SIZE for n in N_RADIUS_PATCHES]
BAND_COLORS = ["#2E6F6A", "#4A8C7A", "#C97F2E", "#C0392B"]  # nucleus->edge, cool->warm

cfg = load_merged_config(REFERENCE_CONFIG, None)
seg_params = cfg.segmentation.as_params()
ann = load_expert_annotations()


def _load_sample(sample: str):
    image_id = f"{sample}.jpg"
    sub = ann[ann.Sample == sample]
    mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
    raw_rgb = np.array(PILImage.open(IMAGE_DIR / image_id).convert("RGB"), dtype=np.uint8)
    cropped, x0, y0, _ = resolve_and_crop_target_otolith(raw_rgb, mean_xy, seg_params)
    axis_info = detect_axis(cropped, seg_params=seg_params,
                             nucleus_method=cfg.segmentation.nucleus_method,
                             axis_method=cfg.segmentation.axis_method)
    return cropped, axis_info


def make_compression_schematic(sample: str) -> Path:
    cropped, axis_info = _load_sample(sample)
    mask = axis_info["mask"]
    photo = apply_background_mask(cropped, mask)
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    axis_angle = float(np.arctan2(fy - cy, fx - cx))
    delta_theta_rad = np.radians(DELTA_THETA_DEG)
    length_px = axis_info["length_px"]

    overview = photo.copy()
    lt = max(2, photo.shape[0] // 300)
    cv2.drawContours(overview, [axis_info["contour"]], -1, _CONTOUR_COLOR, lt)
    cv2.line(overview, (int(cx), int(cy)), (int(fx), int(fy)), _AXIS_COLOR, lt)

    # ONE representative column's angular slice (Delta-theta/13, today's column count) drawn at
    # four increasing radii -- the two bounding rays of that single slice, redrawn wider each time
    # because it's the SAME angular width, but the arc it spans grows with radius.
    col_angle = delta_theta_rad / 13
    t_marks = [0.3, 0.5, 0.7, 1.0]
    mark_colors = [(90, 140, 90), (60, 140, 190), (230, 150, 40), (200, 50, 50)]
    for t, color in zip(t_marks, mark_colors):
        r = t * length_px
        n_arc = 40
        arc_pts = []
        for k in range(n_arc + 1):
            a = axis_angle - col_angle / 2 + col_angle * k / n_arc
            arc_pts.append((int(cx + r * np.cos(a)), int(cy + r * np.sin(a))))
        for p0, p1 in zip(arc_pts[:-1], arc_pts[1:]):
            cv2.line(overview, p0, p1, color, lt + 1, cv2.LINE_AA)
        for a_sign in (-1, 1):
            a = axis_angle + a_sign * col_angle / 2
            cv2.line(overview, (int(cx), int(cy)),
                     (int(cx + r * np.cos(a)), int(cy + r * np.sin(a))), color, 1, cv2.LINE_AA)
        arc_len_px = r * col_angle
        ratio = arc_len_px / PATCH_SIZE
        label = f"t={t:.1f}: luk={arc_len_px:.0f}px ({ratio:.1f}x)"
        lx, ly = arc_pts[n_arc // 2]
        cv2.putText(overview, label, (lx + 8, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overview, label, (lx + 8, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)

    out_path = OUT / f"angular_compression_schematic_{sample}.png"
    PILImage.fromarray(overview).save(out_path)
    print(f"Zapisano {out_path.name}")
    return out_path


def make_bands_demo(sample: str) -> Path:
    cropped, axis_info = _load_sample(sample)
    mask = axis_info["mask"]
    bands = wedge_band_geometries_from_axis_info(
        mask, axis_info, DELTA_THETA_DEG, BAND_EDGES_T, BAND_CANVAS_W, BAND_CANVAS_H)

    display_w = 900
    panels = []
    for i, band in enumerate(bands):
        wedge = extract_polar_wedge_band(cropped, mask, band)
        h, w = wedge.shape[:2]
        disp_h = max(1, int(h * display_w / w))
        disp = np.array(PILImage.fromarray(wedge).resize((display_w, disp_h), PILImage.LANCZOS))
        disp = np.flip(disp, axis=0).copy()  # edge-of-band at top, matching report4.html §4 convention
        # label strip
        label_h = 34
        panel = np.full((disp_h + label_h, display_w, 3), 255, dtype=np.uint8)
        panel[label_h:] = disp
        color_bgr = tuple(int(BAND_COLORS[i].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
        cv2.rectangle(panel, (0, 0), (display_w, label_h), color_bgr, -1)
        label = (f"Pasmo {i+1}/4  t=[{BAND_EDGES_T[i]:.2f},{BAND_EDGES_T[i+1]:.2f})   "
                 f"{N_ANGLE_PATCHES[i]} kolumn x {N_RADIUS_PATCHES[i]} wierszy = "
                 f"{N_ANGLE_PATCHES[i]*N_RADIUS_PATCHES[i]} patchy   "
                 f"(kanwa {BAND_CANVAS_W[i]}x{BAND_CANVAS_H[i]}px)")
        cv2.putText(panel, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
        gap = np.full((6, display_w, 3), 235, dtype=np.uint8)
        panels.append(gap)

    combo = np.concatenate(panels[:-1], axis=0)
    out_path = OUT / f"wedge_bands_demo_{sample}.png"
    PILImage.fromarray(combo).save(out_path)
    print(f"Zapisano {out_path.name} (calkowity koszt: {sum(a*b for a,b in zip(N_ANGLE_PATCHES, N_RADIUS_PATCHES))} patchy)")
    return out_path


def make_cost_tradeoff_chart() -> Path:
    labels = ["dzis\n(1 pasmo,\n13 kolumn)", "Run N\n(kwadrat)", "cel 3,0x", "cel 2,0x",
              "cel 1,5x", "cel 1,0x\n(WYBRANE)"]
    values = [572, 1369, 1968, 2948, 3916, 5852]
    colors = ["#9AA5A0", "#7A8B90", "#8FBFB5", "#5FA394", "#C97F2E", "#C0392B"]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 60, f"{v:,}", ha="center",
                fontsize=10, fontweight="bold" if v == 5852 else "normal")
    ax.axhline(1369, color="#7A8B90", lw=1, ls="--", alpha=0.6)
    ax.set_ylabel("patchy DINOv2 / próbkę (gałąź density)")
    ax.set_title("Koszt vs cel kompresji kątowej — realne liczby (Faza 8, maks. otolit=1305px)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 6500)
    fig.tight_layout()
    out_path = OUT / "wedge_bands_cost_tradeoff.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Zapisano {out_path.name}")
    return out_path


if __name__ == "__main__":
    make_compression_schematic(SAMPLE)
    make_bands_demo(SAMPLE)
    make_cost_tradeoff_chart()
