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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from src.interpretation import (
    DEFAULT_COLORMAP,
    apply_colormap_with_mask,
    importance_to_heatmap_2d,
)
from src.otolith_axis import project_distance_to_axis

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

# Qualitative palette for ring zones (RGB) — cycled if more zones than colors
_ZONE_PALETTE = [
    (228,  26,  28),   # red
    ( 55, 126, 184),   # blue
    ( 77, 175,  74),   # green
    (152,  78, 163),   # purple
    (255, 127,   0),   # orange
    (255, 255,  51),   # yellow
    (166,  86,  40),   # brown
    (247, 129, 191),   # pink
    (153, 153, 153),   # gray
    ( 23, 190, 207),   # cyan
]


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
# Ring zones
# ---------------------------------------------------------------------------

def compute_ring_zones(
    mask: np.ndarray,
    centroid: tuple[int, int],
    far_edge: tuple[int, int],
    peak_t_values: list[float],
) -> np.ndarray:
    """Label each mask pixel with a ring-zone index.

    A zone is the band between two consecutive peaks on the measurement axis.
    Zone 0 covers pixels from the centroid up to the first peak; zone k spans
    between peak ``k-1`` and peak ``k``; the final zone reaches the far edge.

    Args:
        mask:           (H, W) uint8 — otolith silhouette
        centroid:       (cx, cy) — nucleus
        far_edge:       (fx, fy) — terminus of the measurement axis
        peak_t_values:  list of axis positions t ∈ [0, 1] (0 = nucleus, 1 = far edge)

    Returns:
        ``(H, W)`` uint8. Inside-mask pixels carry their zone index (0..N).
        Outside-mask pixels are set to 255 (sentinel).
    """
    distance = project_distance_to_axis(mask, centroid, far_edge)
    out = np.full(mask.shape[:2], 255, dtype=np.uint8)
    inside = mask > 0
    if not inside.any():
        return out
    if not peak_t_values:
        out[inside] = 0
        return out
    thresholds = sorted(float(t) for t in peak_t_values)
    t = distance.copy()
    t[~inside] = np.nan
    zone = np.zeros(mask.shape[:2], dtype=np.int32)
    for thr in thresholds:
        zone += (t > thr).astype(np.int32)
    out[inside] = np.clip(zone[inside], 0, 254).astype(np.uint8)
    return out


def _colorize_zones(
    original_rgb: np.ndarray,
    zones: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a per-zone categorical colour overlay onto the original image."""
    out = original_rgb.copy().astype(np.float32)
    inside = zones != 255
    if not inside.any():
        return original_rgb.copy()
    overlay = np.zeros_like(original_rgb, dtype=np.float32)
    for z in np.unique(zones[inside]):
        color = _ZONE_PALETTE[int(z) % len(_ZONE_PALETTE)]
        overlay[zones == z] = color
    blended = alpha * overlay + (1.0 - alpha) * out
    out[inside] = blended[inside]
    return out.clip(0, 255).astype(np.uint8)


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


def _draw_concentric_rings(
    panel: np.ndarray,
    axis_info: dict,
    peak_indices: np.ndarray,
    line_xy: np.ndarray,
    thickness: int,
) -> None:
    """Draw one concentric ring per detected peak (in place).

    Each ring is the otolith contour scaled about the nucleus (centroid) by the
    peak's fractional distance along the measurement axis, so the ring passes
    exactly through the numbered dot on that axis. This approximates the roughly
    self-similar annual growth rings — far more faithful than the old
    perpendicular "zone" bands (which sliced the otolith into stripes).
    """
    contour = axis_info.get("contour")
    if contour is None or line_xy is None or len(line_xy) < 2:
        return
    centroid = np.asarray(axis_info["centroid"], dtype=np.float32)
    cpts = contour.reshape(-1, 2).astype(np.float32)
    n = len(line_xy)
    for i, idx in enumerate(peak_indices):
        idx = int(idx)
        if not (0 <= idx < n):
            continue
        t = idx / max(1, n - 1)                        # fraction of axis = scale factor
        if t <= 0.0:
            continue
        ring = (centroid + t * (cpts - centroid)).astype(np.int32).reshape(-1, 1, 2)
        color = _ZONE_PALETTE[i % len(_ZONE_PALETTE)]
        cv2.polylines(panel, [ring], isClosed=True, color=color, thickness=thickness)


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
) -> np.ndarray:
    """Compose a 6-panel reasoning card (3 columns × 2 rows).

    Panels:
      1. Raw photo                    — model input
      2. Otolith segmentation         — contour + nucleus cross
      3. Attention heatmap            — inferno colormap, masked to otolith
      4. Measurement axis + 1D profile
      5. Annual rings                 — concentric contours at detected peak radii
      6. Final verdict                — numbered dots + age label + frame

    Each panel keeps the original-image dimensions ``(H, W, 3)`` and is topped
    with a dark title bar. When ``mask``/``axis_info`` are ``None``, panels 2,
    4 and 5 are replaced with a "data unavailable" placeholder.
    """
    H, W = original_rgb.shape[:2]
    line_thickness = max(2, H // 250)
    cross_size = max(8, H // 80)
    dot_radius = max(8, H // 60)

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

    # --- Panel 4: axis + 1D profile ---
    if axis_info is not None and line_xy is not None and profile_1d is not None:
        panel4 = original_rgb.copy()
        _draw_axis_overlay(panel4, axis_info, line_thickness, cross_size)
        _draw_profile_inset(panel4, profile_1d,
                            peak_indices if peak_indices is not None else np.array([], dtype=int))
    else:
        panel4 = _placeholder_panel(H, W, "Oś niedostępna")

    # --- Panel 5: model-derived ring CURVES (locus of high MIL probability) ---
    # Extract real ring lines from the 2-D probability map (radial peaks per
    # direction → clustered → connected), instead of the old scaled-contour
    # approximation. Empty on an untrained model (flat map) — expected.
    if axis_info is not None:
        from src.ring_extraction import extract_ring_curves, draw_ring_curves
        panel5 = original_rgb.copy()
        ring_curves = extract_ring_curves(importance_grid, axis_info, H, W)
        draw_ring_curves(panel5, ring_curves, thickness=max(2, line_thickness + 1))
        _draw_axis_overlay(panel5, axis_info, line_thickness, cross_size)
        if line_xy is not None and peak_indices is not None:
            _draw_numbered_dots(panel5, peak_indices, line_xy, dot_radius)
        n_rings = len(ring_curves)
    else:
        panel5 = _placeholder_panel(H, W, "Pierscienie niedostepne")
        n_rings = 0

    # --- Panel 6: final verdict ---
    panel6 = original_rgb.copy()
    if axis_info is not None and line_xy is not None and peak_indices is not None:
        cx, cy = axis_info["centroid"]
        fx, fy = axis_info["far_edge"]
        cv2.line(panel6, (cx, cy), (fx, fy), _AXIS_COLOR, line_thickness)
        _draw_numbered_dots(panel6, peak_indices, line_xy, dot_radius)
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
        f"5. Pierscienie roczne (N={n_rings})",
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
        )

        stem = Path(image_id).stem
        out_path = output_dir / f"{label}_{stem}_card.png"
        PILImage.fromarray(card, mode="RGB").save(out_path)
        saved.append(out_path)

    return saved


# ---------------------------------------------------------------------------
# Backwards-compatible shims (legacy 3-panel card — kept for old tests)
# ---------------------------------------------------------------------------

def compute_dot_positions(
    importance_grid: np.ndarray,
    predicted_age: int,
    image_height: int,
    profile: np.ndarray | None = None,
) -> list[int]:
    """Legacy helper: detect N peaks in the vertical importance profile.

    Kept for the old `draw_increment_card` path used by `save_increment_cards`.
    New code should use the biological-axis peaks from `src/candidates.py`.
    """
    from scipy.signal import find_peaks

    if profile is None:
        profile = importance_grid.mean(axis=1).astype(np.float32)

    H_p = len(profile)
    n = max(1, int(predicted_age))

    peaks, props = find_peaks(profile, distance=max(1, H_p // (n + 1)))
    if len(peaks) >= n:
        prom = props.get("prominences", profile[peaks])
        top_n_idx = np.argsort(prom)[-n:]
        selected = np.sort(peaks[top_n_idx])
    else:
        selected = np.linspace(0, H_p - 1, n).astype(int)

    patch_h = image_height / H_p
    y_pixels = [int((idx + 0.5) * patch_h) for idx in selected]
    return sorted(y_pixels)


def draw_increment_card(
    original_rgb: np.ndarray,
    dot_y_positions: list[int],
    importance_grid: np.ndarray,
    predicted_age: int,
    true_age: int,
    last_sigmoid: float = 0.0,
) -> np.ndarray:
    """Legacy 3-panel card (original+dots | heatmap+dots | profile).

    Retained for backwards compatibility with older tests. New pipeline uses
    :func:`draw_reasoning_card` (6 panels).
    """
    H_orig, W_orig = original_rgb.shape[:2]
    H_p, W_p = importance_grid.shape

    panel_a = original_rgb.copy()
    x_mid = W_orig // 2
    cv2.line(panel_a, (x_mid, 0), (x_mid, H_orig - 1), (220, 30, 30), 1)

    n_dots = len(dot_y_positions)
    for i, y in enumerate(dot_y_positions):
        is_last = (i == n_dots - 1)
        is_partial = is_last and last_sigmoid > 0.3
        number = str(i + 1)
        if is_partial:
            cv2.circle(panel_a, (x_mid, y), 8, _DOT_FILL, 1)
            cv2.putText(panel_a, f"{number}+", (x_mid + 10, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        else:
            cv2.circle(panel_a, (x_mid, y), 9, _DOT_BORDER, -1)
            cv2.circle(panel_a, (x_mid, y), 8, _DOT_FILL, -1)
            cv2.putText(panel_a, number, (x_mid - 3, y + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    error = abs(predicted_age - true_age)
    frame_color = (0, 200, 0) if error == 0 else (200, 0, 0)
    cv2.putText(panel_a, f"Pred:{predicted_age} True:{true_age}", (4, H_orig - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(panel_a, (0, 0), (W_orig - 1, H_orig - 1), frame_color, 2)

    norm_grid = importance_grid.astype(np.float32).copy()
    if norm_grid.max() > norm_grid.min():
        norm_grid = (norm_grid - norm_grid.min()) / (norm_grid.max() - norm_grid.min())
    heatmap_full = cv2.resize((norm_grid * 255).astype(np.uint8), (W_orig, H_orig),
                              interpolation=cv2.INTER_LINEAR)
    bgr = cv2.applyColorMap(heatmap_full, DEFAULT_COLORMAP)
    rgb_map = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    panel_b = (0.5 * rgb_map + 0.5 * original_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    for i, y in enumerate(dot_y_positions):
        is_partial = (i == n_dots - 1) and last_sigmoid > 0.3
        if is_partial:
            cv2.circle(panel_b, (x_mid, y), 8, _DOT_FILL, 1)
        else:
            cv2.circle(panel_b, (x_mid, y), 9, _DOT_BORDER, -1)
            cv2.circle(panel_b, (x_mid, y), 8, _DOT_FILL, -1)

    profile = importance_grid.mean(axis=1).astype(np.float32)
    fig_h = H_orig / 100
    fig, ax = plt.subplots(figsize=(2.0, fig_h))
    y_positions = np.linspace(0, H_orig, len(profile))
    ax.plot(profile, y_positions, color="steelblue", linewidth=1)
    for y in dot_y_positions:
        ax.axhline(y, color="gold", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.set_ylim(H_orig, 0)
    ax.set_xlabel("Importance", fontsize=6)
    ax.set_ylabel("y-pixel", fontsize=6)
    ax.tick_params(labelsize=5)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    w_fig, h_fig = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    panel_c_raw = buf.reshape(h_fig, w_fig, 4)[:, :, :3]
    plt.close(fig)
    panel_c = cv2.resize(panel_c_raw, (panel_c_raw.shape[1], H_orig))

    return np.concatenate([panel_a, panel_b, panel_c], axis=1)


def save_increment_cards(
    samples: list[dict],
    image_dir: Path,
    importance_grids: dict,
    last_sigmoids: dict,
    output_dir: Path,
    label: str,
) -> list[Path]:
    """Legacy 3-panel card saver. Use :func:`save_reasoning_cards` for the new pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for sample in samples:
        image_id = sample["image_id"]
        predicted_age = int(sample["predicted_age"])
        true_age = int(sample["age"])
        try:
            original = load_original_image(image_id, image_dir)
        except FileNotFoundError:
            continue

        grid = importance_grids.get(image_id)
        if grid is None:
            continue

        profile = grid.mean(axis=1).astype(np.float32)
        dot_positions = compute_dot_positions(grid, predicted_age, original.shape[0],
                                               profile=profile)
        last_sig = last_sigmoids.get(image_id, 0.0)
        card = draw_increment_card(original, dot_positions, grid,
                                    predicted_age, true_age, last_sig)
        stem = Path(image_id).stem
        out_path = output_dir / f"{label}_{stem}_card.png"
        PILImage.fromarray(card, mode="RGB").save(out_path)
        saved.append(out_path)

    return saved
