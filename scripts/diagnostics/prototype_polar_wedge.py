"""Faza 0, krok 3-4 planu wycinka katowego: prototypowa funkcja `extract_polar_wedge`
(transformacja Daugmana ograniczona do wycinka Delta-theta wokol kierunku odczytu, z per-katowa
normalizacja promienia -- ten sam R(theta) co `otolith_axis.compute_polar_grid` liczy dla E9),
weryfikacja round-trip, i wizualizacja na prawdziwych zdjeciach ZEGAR.

Parametry geometrii wziete WPROST z `analyze_zegar_wedge_geometry.py` (Delta-theta, kolumny) i
`analyze_zegar_ring_spacing.py` (wiersze/promien) -- nic tu nie jest zgadywane.

Prototyp, NIE kod produkcyjny -- promocja do `src/` dopiero po potwierdzeniu geometrii (Faza 1).
"""
from __future__ import annotations

import json
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
from src.otolith_axis import detect_axis, apply_background_mask, MASK_FILL_RGB
from src.visualization import _CONTOUR_COLOR, _AXIS_COLOR

OUT = PROJECT_ROOT / "outputs" / "09.09_masked_wizualizacje" / "wedge_geometry"
GEOM = json.loads((OUT / "wedge_geometry_summary.json").read_text(encoding="utf-8"))

DELTA_THETA_DEG = GEOM["delta_theta_p95_deg"]
N_ANGLE_COLS = GEOM["n_angle_columns_p95_recommended"]
N_RADIUS_PATCHES = GEOM["radius_patches_recommended"]
PATCH_SIZE = 14
CANVAS_W = N_ANGLE_COLS * PATCH_SIZE
CANVAS_H = N_RADIUS_PATCHES * PATCH_SIZE
N_ANGLE_RAYCAST = 720  # R(theta) ray-cast resolution, independent of N_ANGLE_COLS (mirrors
                        # compute_polar_grid's own n_angle_bins default of 360, doubled for a
                        # smoother R(theta) since our wedge needs finer angular precision)

print(f"Delta-theta={DELTA_THETA_DEG:.1f}deg, N_ANGLE_COLS={N_ANGLE_COLS}, "
      f"N_RADIUS_PATCHES={N_RADIUS_PATCHES}, canvas={CANVAS_W}x{CANVAS_H}")


def compute_R_theta(mask: np.ndarray, centroid: tuple[float, float], n_bins: int) -> np.ndarray:
    """R(theta): farthest in-mask point per angle bin, ray-cast from centroid.
    Same technique `otolith_axis.compute_polar_grid` already uses internally (cited there,
    replicated here as a prototype -- Faza 1 factors this into one shared helper, no
    duplication in production code)."""
    mask_bin = np.asarray(mask) > 0
    H, W = mask_bin.shape[:2]
    cx, cy = float(centroid[0]), float(centroid[1])
    angles = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    r_max = float(np.hypot(max(cx, W - cx), max(cy, H - cy))) + 1.0
    radii = np.arange(1.0, max(r_max, 2.0))
    R = np.zeros(n_bins, dtype=np.float32)
    for i, a in enumerate(angles):
        xs = np.clip((cx + radii * np.cos(a)).astype(np.int64), 0, W - 1)
        ys = np.clip((cy + radii * np.sin(a)).astype(np.int64), 0, H - 1)
        inside = np.nonzero(mask_bin[ys, xs])[0]
        R[i] = float(radii[inside.max()]) if inside.size else 0.0
    return R


def R_at_angle(R_theta: np.ndarray, theta_rad: float) -> float:
    n = len(R_theta)
    idx = int(round((theta_rad + np.pi) / (2 * np.pi) * n)) % n
    return float(R_theta[idx])


def extract_polar_wedge(rgb_masked: np.ndarray, mask: np.ndarray, centroid, axis_angle_rad: float,
                         delta_theta_rad: float, canvas_w: int, canvas_h: int):
    """Daugman-style rubber-sheet unwrap, restricted to a wedge of angular width
    `delta_theta_rad` centred on `axis_angle_rad`. Returns (wedge_rgb, R_theta, meta) where
    meta carries everything needed for the inverse mapping (round-trip check / point projection).
    """
    R_theta = compute_R_theta(mask, centroid, N_ANGLE_RAYCAST)
    cx, cy = float(centroid[0]), float(centroid[1])

    # canvas column j (0..canvas_w-1) -> angle theta_j ; canvas row i (0..canvas_h-1) -> t_i
    col_idx = np.arange(canvas_w, dtype=np.float64)
    thetas = axis_angle_rad + (col_idx / max(canvas_w - 1, 1) - 0.5) * delta_theta_rad  # (W,)
    row_idx = np.arange(canvas_h, dtype=np.float64)
    ts = row_idx / max(canvas_h - 1, 1)  # (H,), 0=nucleus .. 1=edge

    theta_grid = np.tile(thetas[None, :], (canvas_h, 1))  # (H, W)
    t_grid = np.tile(ts[:, None], (1, canvas_w))           # (H, W)

    # R(theta) per column, broadcast over rows
    bin_idx = np.clip(((thetas + np.pi) / (2 * np.pi) * N_ANGLE_RAYCAST).astype(np.int64),
                       0, N_ANGLE_RAYCAST - 1)
    R_cols = R_theta[bin_idx]  # (W,)
    R_grid = np.tile(R_cols[None, :], (canvas_h, 1))

    map_x = (cx + t_grid * R_grid * np.cos(theta_grid)).astype(np.float32)
    map_y = (cy + t_grid * R_grid * np.sin(theta_grid)).astype(np.float32)

    wedge = cv2.remap(rgb_masked, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=MASK_FILL_RGB)
    meta = dict(R_theta=R_theta, centroid=centroid, axis_angle_rad=axis_angle_rad,
                delta_theta_rad=delta_theta_rad, canvas_w=canvas_w, canvas_h=canvas_h)
    return wedge, meta


def point_to_wedge_xy(x: float, y: float, meta: dict):
    """Inverse mapping: real (x,y) in crop space -> (col,row) in wedge canvas, or None if the
    point's angle falls outside the wedge's Delta-theta."""
    cx, cy = meta["centroid"]
    dx, dy = x - cx, y - cy
    r = float(np.hypot(dx, dy))
    theta = float(np.arctan2(dy, dx))
    rel = theta - meta["axis_angle_rad"]
    rel = (rel + np.pi) % (2 * np.pi) - np.pi  # wrap
    half = meta["delta_theta_rad"] / 2
    if abs(rel) > half:
        return None
    col = (rel / meta["delta_theta_rad"] + 0.5) * (meta["canvas_w"] - 1)
    Rth = R_at_angle(meta["R_theta"], theta)
    t = r / Rth if Rth > 1e-6 else 0.0
    row = t * (meta["canvas_h"] - 1)
    return col, row, t


def wedge_xy_to_point(col: float, row: float, meta: dict):
    """Forward-inverse (canvas -> crop space), for round-trip verification."""
    cx, cy = meta["centroid"]
    theta = meta["axis_angle_rad"] + (col / max(meta["canvas_w"] - 1, 1) - 0.5) * meta["delta_theta_rad"]
    t = row / max(meta["canvas_h"] - 1, 1)
    Rth = R_at_angle(meta["R_theta"], theta)
    x = cx + t * Rth * np.cos(theta)
    y = cy + t * Rth * np.sin(theta)
    return x, y


# ---------------------------------------------------------------------------
# Round-trip verification + real-image visualization
# ---------------------------------------------------------------------------
SAMPLES = ["Z20", "Z38", "Z37", "Z15"]  # continuity (Z20/Z38/Z37) + worst angular case (Z15)

ann = load_expert_annotations()
cfg = load_merged_config(REFERENCE_CONFIG, None)
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
    cx, cy = axis_info["centroid"]; fx, fy = axis_info["far_edge"]
    axis_angle = float(np.arctan2(fy - cy, fx - cx))
    delta_theta = np.radians(DELTA_THETA_DEG)

    wedge, meta = extract_polar_wedge(photo, mask, (cx, cy), axis_angle, delta_theta, CANVAS_W, CANVAS_H)

    # round-trip check on a grid of sample points
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(200):
        col = rng.uniform(0, CANVAS_W - 1)
        row = rng.uniform(0, CANVAS_H - 1)
        x, y = wedge_xy_to_point(col, row, meta)
        back = point_to_wedge_xy(x, y, meta)
        if back is None:
            continue
        c2, r2, _ = back
        errs.append(np.hypot(col - c2, row - r2))
    errs = np.array(errs)
    print(f"{sample}: round-trip error px (canvas units) mean={errs.mean():.4f} max={errs.max():.4f} n={len(errs)}")

    # --- visualization: original photo with wedge boundary + KK/SS, next to unwrapped wedge ---
    H, W = photo.shape[:2]
    overview = photo.copy()
    lt = max(2, H // 300)
    cv2.drawContours(overview, [axis_info["contour"]], -1, _CONTOUR_COLOR, lt)
    cv2.line(overview, (int(cx), int(cy)), (int(fx), int(fy)), _AXIS_COLOR, lt)
    # draw wedge boundary rays (both edges of Delta-theta) out to the R(theta) edge
    for sign in (-1, 1):
        th = axis_angle + sign * delta_theta / 2
        Rth = R_at_angle(meta["R_theta"], th)
        ex, ey = cx + Rth * np.cos(th), cy + Rth * np.sin(th)
        cv2.line(overview, (int(cx), int(cy)), (int(ex), int(ey)), (255, 140, 0), lt)

    kk_xy = list(zip(sub[sub.annotator == "KK"].x - x0, sub[sub.annotator == "KK"].y - y0))
    ss_xy = list(zip(sub[sub.annotator == "SS"].x - x0, sub[sub.annotator == "SS"].y - y0))
    for x, y in kk_xy:
        cv2.circle(overview, (int(x), int(y)), 6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(overview, (int(x), int(y)), 6, (255, 60, 220), 2, cv2.LINE_AA)
    for x, y in ss_xy:
        cv2.circle(overview, (int(x), int(y)), 5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(overview, (int(x), int(y)), 5, (60, 160, 255), 2, cv2.LINE_AA)

    wedge_vis = wedge.copy()
    n_outside = 0
    for x, y in kk_xy + ss_xy:
        proj = point_to_wedge_xy(x, y, meta)
        if proj is None:
            n_outside += 1
            continue
        col, row, _ = proj
        color = (255, 60, 220) if (x, y) in kk_xy else (60, 160, 255)
        cv2.circle(wedge_vis, (int(round(col)), int(round(row))), 5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(wedge_vis, (int(round(col)), int(round(row))), 5, color, 2, cv2.LINE_AA)
    total_pts = len(kk_xy) + len(ss_xy)
    print(f"{sample}: {total_pts - n_outside}/{total_pts} prawdziwych przyrostow miesci sie w wycinku "
          f"(Delta-theta={DELTA_THETA_DEG:.0f} deg)")

    wedge_vis_big = np.array(PILImage.fromarray(wedge_vis).resize((CANVAS_W * 3, CANVAS_H * 3), PILImage.NEAREST))
    ov_resized = np.array(PILImage.fromarray(overview).resize(
        (int(overview.shape[1] * wedge_vis_big.shape[0] / overview.shape[0]), wedge_vis_big.shape[0]),
        PILImage.LANCZOS))
    combo = np.concatenate([ov_resized, wedge_vis_big], axis=1)
    PILImage.fromarray(combo).save(OUT / f"{sample}_wedge_prototype.png")

print("\nZapisano wizualizacje do", OUT)
