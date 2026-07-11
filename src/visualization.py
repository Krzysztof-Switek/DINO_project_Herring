"""Reasoning cards for otolith age prediction.

Each card shows the model's path from raw image to age verdict in 6 panels:

  1. Original photo                — model input
  2. Otolith segmentation           — silhouette + nucleus (classical CV)
  3. Attention heatmap              — inferno colormap, masked to otolith
  4. Measurement axis + 1D profile  — biological axis with importance along it
  5. Annual rings                   — concentric contours at detected peak radii
  6. Final verdict                  — numbered dots on axis + predicted age

If segmentation or axis detection fails for an image, panels 2 / 4 / 5 are
replaced with a "data unavailable" placeholder and the card is still produced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from src.interpretation import (
    DEFAULT_COLORMAP,
    apply_colormap_with_mask,
    importance_to_heatmap_2d,
)

# Colors (RGB)
_AXIS_COLOR     = (255, 220, 0)     # yellow — measurement axis
_CENTROID_COLOR = (40, 120, 255)    # blue — nucleus marker
_CONTOUR_COLOR  = (0, 230, 230)     # cyan — otolith contour
_DOT_FILL       = (255, 230, 0)     # yellow — increment dot fill
_DOT_BORDER     = (0, 0, 0)         # black — increment dot border
_TITLE_BG       = (28, 28, 32)      # dark — title bar background
_TITLE_FG       = (240, 240, 240)   # light — title text
_OK_FRAME       = (0, 200, 0)       # green — correct verdict
_BAD_FRAME      = (220, 0, 0)       # red — incorrect verdict
_PLACEHOLDER_BG = (20, 20, 20)
_PLACEHOLDER_FG = (180, 60, 60)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_original_image(image_id: str, image_dir: Path) -> np.ndarray:
    """Load original image from disk as RGB uint8 ``(H, W, 3)``. Raises if missing."""
    path = Path(image_dir) / image_id
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = PILImage.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_top_k_samples(
    predictions_csv: Path,
    k_best: int = 10,
    k_worst: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Sort predictions by |predicted_age - age| and return best/worst k samples.

    Accepts CSVs where the true label is stored as ``age`` *or* as
    ``target_age`` (the column name written by ``run_inference``).
    """
    df = pd.read_csv(predictions_csv)
    if "age" not in df.columns and "target_age" in df.columns:
        df = df.rename(columns={"target_age": "age"})
    df["abs_error"] = (df["predicted_age"] - df["age"]).abs()
    df_sorted = df.sort_values("abs_error").reset_index(drop=True)
    best = df_sorted.head(k_best).to_dict(orient="records")
    worst = df_sorted.tail(k_worst).sort_values("abs_error", ascending=False).to_dict(orient="records")
    return best, worst


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _title_height(H: int) -> int:
    return max(28, H // 22)


def _fit_font_scale(text: str, max_w: int, start_scale: float,
                    thickness_ratio: float = 2.0, min_scale: float = 0.3,
                    step: float = 0.05) -> float:
    """Largest FONT_HERSHEY_SIMPLEX scale (≤ start_scale) whose text fits ``max_w`` px.

    Single source of truth for "shrink text until it fits the panel" — used by the
    verdict label and the panel title bars so nothing spills off the print-out.
    """
    scale = start_scale
    while scale > min_scale:
        thickness = max(1, int(scale * thickness_ratio))
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw <= max_w:
            break
        scale = max(min_scale, scale - step)
    return scale


def _add_title(img: np.ndarray, title: str) -> np.ndarray:
    """Stack a dark title bar (with white text) above the image."""
    H, W = img.shape[:2]
    th = _title_height(H)
    bar = np.full((th, W, 3), _TITLE_BG, dtype=np.uint8)
    font_scale = _fit_font_scale(title, W - 16, max(0.45, th / 55.0), thickness_ratio=1.8)
    thickness = max(1, int(font_scale * 1.8))
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = max(8, (W - tw) // 2)
    ty = int(th * 0.72)
    cv2.putText(bar, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, _TITLE_FG, thickness, lineType=cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def _placeholder_panel(H: int, W: int, message: str) -> np.ndarray:
    """Solid dark panel with red error text — used when a step's data is unavailable."""
    panel = np.full((H, W, 3), _PLACEHOLDER_BG, dtype=np.uint8)
    font_scale = max(0.5, H / 480.0)
    thickness = max(1, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = max(8, (W - tw) // 2)
    ty = (H + th) // 2
    cv2.putText(panel, message, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, _PLACEHOLDER_FG, thickness, lineType=cv2.LINE_AA)
    return panel


def _draw_colorbar(panel: np.ndarray, label_low: str = "niski", label_high: str = "wysoki",
                   colormap: int = DEFAULT_COLORMAP) -> np.ndarray:
    """Render an inferno colorbar inside the right edge of ``panel`` (in place).

    Labels are drawn to the LEFT of the bar (right-aligned) on a dark background,
    so neither the bar nor the text ever spills outside the panel on print-outs.
    """
    H, W = panel.shape[:2]
    bar_w = max(10, W // 22)
    margin = max(6, W // 60)
    bar_h = max(40, int(H * 0.50))
    x0 = W - bar_w - margin
    y0 = (H - bar_h) // 2

    ramp = np.repeat(np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1), bar_w, axis=1)
    panel[y0:y0 + bar_h, x0:x0 + bar_w] = cv2.cvtColor(
        cv2.applyColorMap(ramp, colormap), cv2.COLOR_BGR2RGB)
    cv2.rectangle(panel, (x0 - 1, y0 - 1), (x0 + bar_w, y0 + bar_h), _TITLE_FG, 1)

    font_scale = max(0.3, H / 1300.0)
    thickness = max(1, int(font_scale * 2))

    def _label_left(text: str, y_baseline: int) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = max(2, x0 - 4 - tw)                       # right-align to the LEFT of the bar
        ty = int(np.clip(y_baseline, th + 2, H - 2))
        cv2.rectangle(panel, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), _TITLE_BG, -1)
        cv2.putText(panel, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, _TITLE_FG, thickness, cv2.LINE_AA)

    _label_left(label_high, y0 + 14)                   # near top of bar (high)
    _label_left(label_low, y0 + bar_h - 2)             # near bottom of bar (low)
    return panel


def _draw_profile_inset(
    panel: np.ndarray,
    profile_1d: np.ndarray,
    peak_indices: np.ndarray,
) -> np.ndarray:
    """Draw a small 1D profile plot in the right ~25% of ``panel`` (in place).

    Y axis = position along measurement axis (0 = nucleus, top of plot).
    X axis = importance value.
    Horizontal dashed lines mark detected peaks.
    """
    H, W = panel.shape[:2]
    plot_w = int(W * 0.30)
    plot_h = int(H * 0.85)
    margin_x = max(6, W // 80)
    margin_y = (H - plot_h) // 2
    x0 = W - plot_w - margin_x
    y0 = margin_y

    # Background
    cv2.rectangle(panel, (x0 - 1, y0 - 1), (x0 + plot_w, y0 + plot_h),
                  (245, 245, 245), -1)
    cv2.rectangle(panel, (x0 - 1, y0 - 1), (x0 + plot_w, y0 + plot_h),
                  (80, 80, 80), 1)

    if profile_1d is None or len(profile_1d) < 2:
        return panel

    n = len(profile_1d)
    pmin = float(np.nanmin(profile_1d))
    pmax = float(np.nanmax(profile_1d))
    pr = max(pmax - pmin, 1e-6)
    y_positions = np.linspace(y0 + 2, y0 + plot_h - 2, n).astype(np.int32)
    x_positions = (x0 + plot_w - 4 - (profile_1d - pmin) / pr * (plot_w - 8)).astype(np.int32)

    # Line
    pts = np.stack([x_positions, y_positions], axis=1).reshape(-1, 1, 2)
    cv2.polylines(panel, [pts], isClosed=False,
                  color=(60, 100, 180), thickness=max(1, plot_h // 200))

    # Peak markers (horizontal dashed lines + small red dot)
    for k in peak_indices:
        k = int(k)
        if 0 <= k < n:
            yk = int(y_positions[k])
            for xd in range(x0 + 2, x0 + plot_w - 2, 6):
                cv2.line(panel, (xd, yk), (min(xd + 3, x0 + plot_w - 2), yk),
                         (200, 50, 50), 1)
            cv2.circle(panel, (int(x_positions[k]), yk),
                       max(2, plot_h // 80), (200, 30, 30), -1)

    # Axis labels
    font_scale = max(0.3, H / 1100.0)
    cv2.putText(panel, "profil 1D", (x0 + 4, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, _TITLE_FG, 1, cv2.LINE_AA)
    return panel


def _draw_axis_overlay(
    panel: np.ndarray,
    axis_info: dict,
    line_thickness: int,
    cross_size: int,
    draw_axis: bool = True,
) -> None:
    """Draw contour + centroid cross (+ measurement axis if ``draw_axis``) in place.

    Panel 2 (segmentation) sets ``draw_axis=False`` — the measurement axis belongs
    to panels 4/5/6, not to the pure segmentation view.
    """
    contour = axis_info.get("contour")
    if contour is not None:
        cv2.drawContours(panel, [contour], -1, _CONTOUR_COLOR, line_thickness)
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    cv2.line(panel, (cx - cross_size, cy), (cx + cross_size, cy),
             _CENTROID_COLOR, line_thickness)
    cv2.line(panel, (cx, cy - cross_size), (cx, cy + cross_size),
             _CENTROID_COLOR, line_thickness)
    if draw_axis:
        cv2.line(panel, (cx, cy), (fx, fy), _AXIS_COLOR, line_thickness)


def _draw_numbered_dots(
    panel: np.ndarray,
    peak_indices: np.ndarray,
    line_xy: np.ndarray,
    dot_radius: int,
) -> None:
    """Draw numbered yellow dots at peak positions on the axis (in place)."""
    font_scale = max(0.4, dot_radius / 14.0)
    thickness = max(1, int(font_scale * 2))
    for i, idx in enumerate(peak_indices):
        idx = int(idx)
        if 0 <= idx < len(line_xy):
            x, y = int(line_xy[idx][0]), int(line_xy[idx][1])
            cv2.circle(panel, (x, y), dot_radius + 1, _DOT_BORDER, -1)
            cv2.circle(panel, (x, y), dot_radius, _DOT_FILL, -1)
            label = str(i + 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           font_scale, thickness)
            tx = x - tw // 2
            ty = y + th // 2
            cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, _DOT_BORDER, thickness, cv2.LINE_AA)


# Colours for the count=age increment overlays (11.07 Punkt 7)
_CAND_COLOR = (255, 210, 40)     # yellow — increment CANDIDATES (all axes)
_FINAL_COLOR = (230, 30, 30)     # red   — FINAL increments (count = predicted age)


def _draw_small_points(panel: np.ndarray, points, color, radius: int,
                       border: bool = False) -> None:
    """Draw small filled circles (no numbers) at (x, y) points (in place).

    ``border=True`` adds a thin dark outline so bright dots stay visible on the
    light otolith body (used for the final increments).
    """
    if not points:
        return
    r = max(2, int(radius))
    for (x, y) in points:
        xi, yi = int(x), int(y)
        if border:
            cv2.circle(panel, (xi, yi), r + 1, (30, 30, 30), 1)
        cv2.circle(panel, (xi, yi), r, color, -1)


# ---------------------------------------------------------------------------
# Reasoning card
# ---------------------------------------------------------------------------

def draw_reasoning_card(
    original_rgb: np.ndarray,
    importance_grid: np.ndarray,
    predicted_age: int,
    true_age: int,
    *,
    mask: Optional[np.ndarray] = None,
    axis_info: Optional[dict] = None,
    peak_indices: Optional[np.ndarray] = None,
    line_xy: Optional[np.ndarray] = None,
    profile_1d: Optional[np.ndarray] = None,
    final_axis_pts: Optional[list] = None,
    candidate_pts: Optional[list] = None,
    final_t: Optional[list] = None,
) -> np.ndarray:
    """Compose a 6-panel reasoning card (3 columns × 2 rows).

    Panels:
      1. Raw photo                    — model input
      2. Otolith segmentation         — contour + nucleus cross
      3. Attention heatmap            — inferno colormap, masked to otolith
      4. Measurement axis + 1D profile
      5. Annual rings                 — ring curves from the prob map (dots = axis crossings)
      6. Final verdict                — numbered dots + age label + frame

    Each panel keeps the original-image dimensions ``(H, W, 3)`` and is topped
    with a dark title bar. When ``mask``/``axis_info`` are ``None``, panels 2,
    4 and 5 are replaced with a "data unavailable" placeholder.
    """
    H, W = original_rgb.shape[:2]
    line_thickness = max(2, H // 250)
    cross_size = max(8, H // 80)
    dot_radius = max(8, H // 60)

    # --- Model-derived rings (shared by panels 4/5/6 so they stay consistent) ---
    # Ring curves come from the 2-D probability map; the numbered dots are placed
    # where each curve crosses the measurement axis (ring t → axis index), so the
    # dot count always equals the curve count. Fall back to the incoming axis peaks
    # only when no curves are found (e.g. an untrained model with a flat map).
    # NEW (Punkt 7): increments come from multi-axis consensus with count = age
    # (final_axis_pts / candidate_pts precomputed by ring_extraction.select_increments).
    # Fall back to the old extract_rings ring drawing when they are not provided.
    use_new = final_axis_pts is not None or candidate_pts is not None
    ring_curves: list = []
    dots_idx = (np.asarray(peak_indices, dtype=int)
                if peak_indices is not None else np.array([], dtype=int))
    if use_new:
        if final_t and line_xy is not None and len(line_xy) > 0:
            n = len(line_xy)
            dots_idx = np.clip(
                np.array([int(round(t * (n - 1))) for t in final_t], dtype=int), 0, n - 1)
    elif axis_info is not None:
        from src.ring_extraction import extract_rings
        rings = extract_rings(importance_grid, axis_info, H, W)
        ring_curves = [c for (_t, c) in rings]
        if rings and line_xy is not None and len(line_xy) > 0:
            n = len(line_xy)
            dots_idx = np.clip(
                np.array([int(round(t * (n - 1))) for (t, _c) in rings], dtype=int),
                0, n - 1)
    n_rings = len(ring_curves)
    n_cand = len(candidate_pts) if candidate_pts else 0

    # --- Panel 1: raw ---
    panel1 = original_rgb.copy()

    # --- Panel 2: segmentation (contour + nucleus only, NO measurement axis) ---
    if axis_info is not None and mask is not None:
        panel2 = original_rgb.copy()
        _draw_axis_overlay(panel2, axis_info, line_thickness, cross_size, draw_axis=False)
    else:
        panel2 = _placeholder_panel(H, W, "Segmentacja nieudana")

    # --- Panel 3: attention heatmap ---
    heatmap = importance_to_heatmap_2d(importance_grid, H, W)
    panel3 = apply_colormap_with_mask(
        heatmap, original_rgb, mask=mask, alpha=0.55, colormap=DEFAULT_COLORMAP
    )
    _draw_colorbar(panel3)

    # --- Panel 4: axis + 1D profile (peaks = ring axis-crossings) ---
    if axis_info is not None and line_xy is not None and profile_1d is not None:
        panel4 = original_rgb.copy()
        _draw_axis_overlay(panel4, axis_info, line_thickness, cross_size)
        _draw_profile_inset(panel4, profile_1d, dots_idx)
    else:
        panel4 = _placeholder_panel(H, W, "Oś niedostępna")

    # --- Panel 5: increments — candidates (yellow) in new mode; ring curves otherwise ---
    if axis_info is not None:
        panel5 = original_rgb.copy()
        if use_new:
            # Panel 5 = ALL candidate increments (yellow) — "where we see possible
            # increments" across every axis. Final increments live on panel 6.
            _draw_axis_overlay(panel5, axis_info, line_thickness, cross_size)
            _draw_small_points(panel5, candidate_pts or [], _CAND_COLOR, max(2, H // 300))
        else:
            from src.ring_extraction import draw_ring_curves
            draw_ring_curves(panel5, ring_curves, thickness=max(2, line_thickness + 1))
            _draw_axis_overlay(panel5, axis_info, line_thickness, cross_size)
            if line_xy is not None and len(dots_idx) > 0:
                _draw_numbered_dots(panel5, dots_idx, line_xy, dot_radius)
    else:
        panel5 = _placeholder_panel(H, W, "Pierscienie niedostepne")

    # --- Panel 6: final verdict — red FINAL increments on the axis (count = age) ---
    panel6 = original_rgb.copy()
    if use_new and axis_info is not None:
        cx, cy = axis_info["centroid"]
        fx, fy = axis_info["far_edge"]
        cv2.line(panel6, (cx, cy), (fx, fy), _AXIS_COLOR, line_thickness)
        _draw_small_points(panel6, final_axis_pts or [], _FINAL_COLOR, max(3, H // 110),
                           border=True)
    elif axis_info is not None and line_xy is not None and len(dots_idx) > 0:
        cx, cy = axis_info["centroid"]
        fx, fy = axis_info["far_edge"]
        cv2.line(panel6, (cx, cy), (fx, fy), _AXIS_COLOR, line_thickness)
        _draw_numbered_dots(panel6, dots_idx, line_xy, dot_radius)
    # Age label (always rendered) — shrink font until it fits the panel width
    label = f"Wiek: {int(predicted_age)} (true: {int(true_age)})"
    pad = max(6, H // 80)
    max_w = W - 2 * pad
    font_scale = _fit_font_scale(label, max_w, max(0.5, H / 480.0))
    thickness = max(1, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(panel6, (pad - 4, H - th - 3 * pad),
                   (min(pad + tw + 8, W - 2), H - pad), (0, 0, 0), -1)
    cv2.putText(panel6, label, (pad + 2, H - 2 * pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, _DOT_FILL, thickness, cv2.LINE_AA)
    frame_color = _OK_FRAME if int(predicted_age) == int(true_age) else _BAD_FRAME
    cv2.rectangle(panel6, (0, 0), (W - 1, H - 1), frame_color, max(3, line_thickness * 2))

    # --- Compose 3×2 grid ---
    titles = [
        "1. Surowe zdjecie",
        "2. Segmentacja otolitu",
        "3. Mapa uwagi (inferno)",
        "4. Os pomiaru + profil 1D",
        (f"5. Kandydaci przyrostow (N={n_cand})" if use_new
         else f"5. Pierscienie roczne (N={n_rings})"),
        f"6. Werdykt: wiek = {int(predicted_age)}",
    ]
    panels = [panel1, panel2, panel3, panel4, panel5, panel6]
    panels_titled = [_add_title(p, t) for p, t in zip(panels, titles)]
    row1 = np.concatenate(panels_titled[:3], axis=1)
    row2 = np.concatenate(panels_titled[3:], axis=1)
    card = np.concatenate([row1, row2], axis=0)
    return card


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_reasoning_cards(
    samples: list[dict],
    image_dir: Path,
    importance_grids: dict,
    axis_data: dict,
    output_dir: Path,
    label: str,
) -> list[Path]:
    """Generate and save reasoning cards for a list of prediction samples.

    Parameters
    ----------
    samples         : list of row-dicts from predictions CSV (image_id, age, predicted_age)
    image_dir       : directory containing original images
    importance_grids: mapping image_id → (H_p, W_p) ndarray
    axis_data       : mapping image_id → dict with keys
                        {mask, axis_info, peak_indices, line_xy, profile_1d}
                      (any of these may be None when segmentation/axis failed)
    output_dir      : directory to write PNG cards
    label           : 'best' or 'worst' — used in filename

    Returns
    -------
    List of saved PNG paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for sample in samples:
        image_id = sample["image_id"]
        try:
            original = load_original_image(image_id, image_dir)
        except FileNotFoundError:
            continue

        grid = importance_grids.get(image_id)
        if grid is None:
            continue

        axis = axis_data.get(image_id, {})
        card = draw_reasoning_card(
            original_rgb=original,
            importance_grid=grid,
            predicted_age=int(sample["predicted_age"]),
            true_age=int(sample["age"]),
            mask=axis.get("mask"),
            axis_info=axis.get("axis_info"),
            peak_indices=axis.get("peak_indices"),
            line_xy=axis.get("line_xy"),
            profile_1d=axis.get("profile_1d"),
            final_axis_pts=axis.get("final_axis_pts"),
            candidate_pts=axis.get("candidate_pts"),
            final_t=axis.get("final_t"),
        )

        stem = Path(image_id).stem
        out_path = output_dir / f"{label}_{stem}_card.png"
        PILImage.fromarray(card, mode="RGB").save(out_path)
        saved.append(out_path)

    return saved

