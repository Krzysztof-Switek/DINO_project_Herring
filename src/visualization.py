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
from src.ring_extraction import draw_ring_curves

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


def _add_title(img: np.ndarray, title: str, bar_color=None) -> np.ndarray:
    """Stack a title bar (with white text) above the image.

    ``bar_color`` overrides the default dark bar — used to colour-code which head a
    panel belongs to (granatowy = CORAL/wiek, pomarańcz = MIL/lokalizacja).
    """
    H, W = img.shape[:2]
    th = _title_height(H)
    bar = np.full((th, W, 3), _TITLE_BG if bar_color is None else bar_color, dtype=np.uint8)
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
    # Shrink to fit the panel width so the text never spills into the neighbouring panel.
    font_scale = _fit_font_scale(message, W - 16, max(0.5, H / 480.0))
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

# Title-bar colours = which head a panel belongs to (11.07 Punkt 7 redesign)
_HEAD_CORAL_BAR = (30, 60, 130)  # granatowy — GŁOWICA WIEKU (CORAL)
_HEAD_MIL_BAR = (150, 70, 20)    # ciemny pomarańcz — GŁOWICA LOKALIZACJI (MIL)


_CLASSICAL_COLOR = (0, 255, 180)   # spring-green — classical (OpenCV) cross-check peaks


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


def _draw_hollow_points(panel: np.ndarray, points, color, radius: int) -> None:
    """Draw hollow rings at (x, y) points (in place) — classical cross-check markers.

    Hollow so they read as a distinct overlay next to the filled model increments.
    """
    if not points:
        return
    r = max(3, int(radius))
    for (x, y) in points:
        cv2.circle(panel, (int(x), int(y)), r, color, max(1, r // 3))


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
    coral_gradcam: Optional[np.ndarray] = None,
    cls_attention: Optional[np.ndarray] = None,
    cls_is_fallback: bool = False,
    ring_curves: Optional[list] = None,
    classical_pts: Optional[list] = None,
) -> np.ndarray:
    """Compose a 6-panel reasoning card (3 columns × 2 rows) — two rows = two heads.

    Rząd 1 — GŁOWICA WIEKU (CORAL):
      1. Grad-CAM werdyktu   — które rejony wpływają na przewidziany wiek
      2. Uwaga CLS           — z których patchy CLS złożył streszczenie (albo placeholder)
      3. Werdykt             — wiek = X (true: Y) + ramka OK/błąd
    Rząd 2 — GŁOWICA LOKALIZACJI (MIL):
      4. Mapa MIL            — prawdopodobieństwo przyrostu per patch (inferno)
      5. Kandydaci           — żółte kropki ze wszystkich osi + oś pomiaru
      6. Finalne (N=wiek)    — czerwone finalne przyrosty na osi

    Paski tytułów są kolorowane wg głowicy (granatowy = CORAL, pomarańcz = MIL).
    ``coral_gradcam`` / ``cls_attention`` to siatki (H_p, W_p) lub None (→ placeholder).
    Stare parametry (``peak_indices``/``profile_1d``) przyjmowane dla zgodności wstecz;
    panele 5/6 spadają do numerowanych kropek, gdy brak nowych punktów przyrostów.
    """
    H, W = original_rgb.shape[:2]
    line_thickness = max(2, H // 250)
    cross_size = max(8, H // 80)
    dot_radius = max(8, H // 60)

    use_new = final_axis_pts is not None or candidate_pts is not None
    n_cand = len(candidate_pts) if candidate_pts else 0
    n_final = len(final_axis_pts) if final_axis_pts else 0

    # Fallback dots (old mode / non-MIL): from incoming peak_indices or final radii.
    dots_idx = (np.asarray(peak_indices, dtype=int)
                if peak_indices is not None else np.array([], dtype=int))
    if use_new and final_t and line_xy is not None and len(line_xy) > 0:
        n = len(line_xy)
        dots_idx = np.clip(
            np.array([int(round(t * (n - 1))) for t in final_t], dtype=int), 0, n - 1)

    def _heat_panel(grid, robust: bool = False) -> np.ndarray:
        """Nakładka mapy ciepła z (a) robust-normalizacją percentylową (2–98) — środkowe
        tony widoczne, outliery przycięte — i (b) ALFĄ PER-PIKSEL: zimne piksele przezroczyste
        (widać otolit), gorące nieprzezroczyste i jaskrawe. Naprawia „ciemne kolory giną"
        i „density nic nie widać" (20.07): jednorodna alfa robiła muł, w którym niski sygnał
        (ciemny koniec inferno) zlewał się z ciemnym otolitem."""
        if grid is None:
            return _placeholder_panel(H, W, "niedostepne")
        arr = np.asarray(grid, dtype=np.float32)
        lo, hi = float(np.percentile(arr, 2.0)), float(np.percentile(arr, 98.0))
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        norm = np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        hm = cv2.resize(norm, (W, H), interpolation=cv2.INTER_LINEAR).clip(0.0, 1.0)
        bgr = cv2.applyColorMap((hm * 255).astype(np.uint8), DEFAULT_COLORMAP)
        rgb_map = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        alpha = np.power(hm, 1.5) * 0.9                          # gamma>1 → tylko piki mocne
        if mask is not None:
            alpha = alpha * (np.asarray(mask) > 0)
        a = alpha[..., None]
        panel = (a * rgb_map + (1.0 - a) * original_rgb.astype(np.float32)
                 ).clip(0, 255).astype(np.uint8)
        _draw_colorbar(panel)
        _draw_contour(panel)                 # obrys otolitu — widać, gdzie kończy się overlay
        return panel

    def _letterbox_to(img: np.ndarray) -> np.ndarray:
        """Wpasuj obraz w płótno (H, W) zachowując proporcje (czarne pasy)."""
        ih, iw = img.shape[:2]
        s = min(W / iw, H / ih)
        nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        y0, x0 = (H - nh) // 2, (W - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh))
        return canvas

    def _attn_outside_frac(grid) -> float | None:
        """Udział masy uwagi CLS POZA maską (diagnostyka: czy model patrzy na tło)."""
        if grid is None or mask is None:
            return None
        hm = cv2.resize(np.asarray(grid, dtype=np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        hm = np.clip(hm, 0.0, None)
        total = float(hm.sum())
        if total <= 1e-9:
            return None
        return float(hm[np.asarray(mask) == 0].sum() / total)

    def _draw_contour(panel) -> None:
        """Narysuj cyjanowy kontur otolitu (jeśli dostępny) — na KAŻDYM panelu ze zdjęciem."""
        if axis_info is not None and axis_info.get("contour") is not None:
            cv2.drawContours(panel, [axis_info["contour"]], -1, _CONTOUR_COLOR, line_thickness)

    # ===== Rząd 1 — GŁOWICA WIEKU (CORAL) =====
    # Panel 1 — WEJŚCIE MODELU (zastąpiło Grad-CAM, 20.07): model dostaje CAŁE zdjęcie
    # ściśnięte do 518×518 (z tłem!), NIE maskę. Pokazujemy ten realny, kwadratowy input,
    # żeby było widać squash proporcji i obecność tła (patrz „% uwagi poza maską" w panelu 2).
    panel1 = _letterbox_to(cv2.resize(original_rgb, (518, 518)))
    attn_bg_frac = _attn_outside_frac(cls_attention)
    panel2 = (_heat_panel(cls_attention) if cls_attention is not None
              else _placeholder_panel(H, W, "Uwaga CLS niedostepna"))
    # Panel 3 — werdykt: etykieta wieku + ramka OK/błąd
    panel3 = original_rgb.copy()
    _draw_contour(panel3)
    label = f"Wiek: {int(predicted_age)} (true: {int(true_age)})"
    pad = max(6, H // 80)
    font_scale = _fit_font_scale(label, W - 2 * pad, max(0.5, H / 480.0))
    thickness = max(1, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(panel3, (pad - 4, H - th - 3 * pad),
                   (min(pad + tw + 8, W - 2), H - pad), (0, 0, 0), -1)
    cv2.putText(panel3, label, (pad + 2, H - 2 * pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, _DOT_FILL, thickness, cv2.LINE_AA)
    frame_color = _OK_FRAME if int(predicted_age) == int(true_age) else _BAD_FRAME
    cv2.rectangle(panel3, (0, 0), (W - 1, H - 1), frame_color, max(3, line_thickness * 2))

    # ===== Rząd 2 — GŁOWICA LOKALIZACJI (density) =====
    has_rings = bool(ring_curves) and len(ring_curves) >= 2   # 1 „pierścień" = zwykle artefakt konturu
    panel4 = _heat_panel(importance_grid, robust=True)   # mapa density (odporna na dominujący outlier)
    if has_rings:                            # krzywe pierścieni tylko gdy jest ich sensowny zestaw
        draw_ring_curves(panel4, ring_curves, thickness=max(2, H // 300))
    # Panel 5 — kandydaci
    if axis_info is not None:
        panel5 = original_rgb.copy()
        # Dorysuj 48 osi jądro→kontur (20.07) — POKAZUJE, skąd biorą się kandydaci: piki
        # sygnału wzdłuż każdego z 48 promieni (per-promień normalizacja + find_peaks).
        _contour = axis_info.get("contour")
        _cent = axis_info.get("centroid")
        if _contour is not None and _cent is not None:
            _cx, _cy = int(_cent[0]), int(_cent[1])
            _cpts = np.asarray(_contour).reshape(-1, 2)
            for _ci in np.linspace(0, len(_cpts) - 1, min(48, len(_cpts)), dtype=int):
                cv2.line(panel5, (_cx, _cy), (int(_cpts[_ci][0]), int(_cpts[_ci][1])),
                         _RAY_COLOR, 1, cv2.LINE_AA)
        _draw_axis_overlay(panel5, axis_info, line_thickness, cross_size)
        if use_new:
            _draw_small_points(panel5, candidate_pts or [], _CAND_COLOR, max(2, H // 300))
        elif line_xy is not None and len(dots_idx) > 0:
            _draw_numbered_dots(panel5, dots_idx, line_xy, dot_radius)
    else:
        panel5 = _placeholder_panel(H, W, "Os niedostepna")
    # Panel 6 — finalne przyrosty (N = wiek)
    if axis_info is not None:
        panel6 = original_rgb.copy()
        _draw_contour(panel6)
        cx, cy = axis_info["centroid"]
        fx, fy = axis_info["far_edge"]
        cv2.line(panel6, (cx, cy), (fx, fy), _AXIS_COLOR, line_thickness)
        if use_new:
            _draw_small_points(panel6, final_axis_pts or [], _FINAL_COLOR,
                               max(3, H // 110), border=True)
        elif line_xy is not None and len(dots_idx) > 0:
            _draw_numbered_dots(panel6, dots_idx, line_xy, dot_radius)
        # Classical (OpenCV) cross-check peaks as hollow green rings — model vs classic.
        if classical_pts:
            _draw_hollow_points(panel6, classical_pts, _CLASSICAL_COLOR, max(3, H // 110))
        # Legenda znaczników (20.07) — żeby czerwone/zielone/krzyż były jednoznaczne.
        _legend_items = [(_FINAL_COLOR, True, "finalne (N=wiek)"),
                         (_CLASSICAL_COLOR, False, "klasyka (OpenCV)"),
                         (_AXIS_COLOR, None, "os pomiaru")]
        _ly, _fs = max(10, H // 40), max(0.4, H / 900.0)
        for _col, _filled, _txt in _legend_items:
            if _filled is None:
                cv2.line(panel6, (8, _ly - 4), (22, _ly - 4), _col, max(2, H // 300))
            elif _filled:
                cv2.circle(panel6, (15, _ly - 4), max(4, H // 150), _col, -1)
            else:
                cv2.circle(panel6, (15, _ly - 4), max(4, H // 150), _col, max(2, H // 400))
            cv2.putText(panel6, _txt, (30, _ly), cv2.FONT_HERSHEY_SIMPLEX, _fs,
                        (255, 255, 255), max(1, int(_fs * 2)), cv2.LINE_AA)
            _ly += max(16, H // 28)
    else:
        panel6 = _placeholder_panel(H, W, "Os niedostepna")

    # ===== Kompozycja 3×2: rząd 1 = GŁOWICA WIEKU (CORAL), rząd 2 = GŁOWICA LOKALIZACJI (density) =====
    # ASCII "-" jako separator — cv2 (Hershey) NIE renderuje "·" i pokazuje "??".
    # Nazwa głowicy w każdym tytule (CORAL / density) — jasne, która głowica daje który obraz.
    input_title = "WIEK (CORAL) - wejscie modelu (518x518, tlo+squash)"
    if cls_is_fallback:
        cls_title = "WIEK (CORAL) - proxy L2 (CLS niedost.)"
    elif attn_bg_frac is not None:
        cls_title = f"WIEK (CORAL) - uwaga CLS (tlo {attn_bg_frac * 100:.0f}%)"
    else:
        cls_title = "WIEK (CORAL) - uwaga CLS"
    mil_title = ("PRZYROSTY (density) - mapa + pierscienie" if has_rings
                 else "PRZYROSTY (density) - mapa")
    final_title = (f"PRZYROSTY (density) - finalne (N={n_final}) vs klasyka" if classical_pts
                   else f"PRZYROSTY (density) - finalne (N={n_final})")
    row1_titles = [
        input_title,
        cls_title,
        f"WIEK (CORAL) - werdykt = {int(predicted_age)}",
    ]
    row2_titles = [
        mil_title,
        f"PRZYROSTY (density) - kandydaci (N={n_cand})",
        final_title,
    ]
    row1 = np.concatenate(
        [_add_title(p, t, _HEAD_CORAL_BAR)
         for p, t in zip([panel1, panel2, panel3], row1_titles)], axis=1)
    row2 = np.concatenate(
        [_add_title(p, t, _HEAD_MIL_BAR)
         for p, t in zip([panel4, panel5, panel6], row2_titles)], axis=1)
    card = np.concatenate([row1, row2], axis=0)
    return card


# ---------------------------------------------------------------------------
# Localisation-method overlay (report sections I / J / K)
# ---------------------------------------------------------------------------

def render_localization_overlay(
    image: np.ndarray,
    axis_info: Optional[dict],
    final_pts: Optional[list],
    candidate_pts: Optional[list] = None,
) -> np.ndarray:
    """RGB overlay for one localisation method: contour + axis + candidates + finals.

    Used by the I/J/K bake-off sections (density / classical / fusion): each renders
    the SAME otolith with that method's ``final_pts`` (red, count = age) over faint
    ``candidate_pts`` (yellow), plus the otolith contour and measurement axis.
    """
    img = np.ascontiguousarray(image[..., :3]).copy()
    H = img.shape[0]
    lt = max(2, H // 300)
    if axis_info is not None:
        contour = axis_info.get("contour")
        if contour is not None:
            cv2.drawContours(img, [contour], -1, _CONTOUR_COLOR, lt)
        if axis_info.get("centroid") and axis_info.get("far_edge"):
            cx, cy = axis_info["centroid"]
            fx, fy = axis_info["far_edge"]
            cv2.line(img, (cx, cy), (fx, fy), _AXIS_COLOR, lt)
    if candidate_pts:
        _draw_small_points(img, candidate_pts, _CAND_COLOR, max(1, H // 500))
    if final_pts:
        _draw_small_points(img, final_pts, _FINAL_COLOR, max(3, H // 130), border=True)
    return img


_RAY_COLOR = (110, 110, 110)     # faint gray — the n_dirs measurement rays (walkthrough panel 1)


def render_rays_and_candidates(
    image: np.ndarray,
    axis_info: Optional[dict],
    density_pts: Optional[list],
    classical_pts: Optional[list],
    n_dirs: int = 48,
) -> np.ndarray:
    """Panel 1 of the DP walkthrough: otolith + ``n_dirs`` faint rays nucleus→contour,
    with density candidates (yellow) and classical candidates (green) at their pixels.

    Shows where candidates come from — every local maximum along every ray, in all
    directions — before they are reduced to the 1D radius (kolejne panele).
    """
    img = np.ascontiguousarray(image[..., :3]).copy()
    H = img.shape[0]
    lt = max(2, H // 300)
    if axis_info is not None:
        contour = axis_info.get("contour")
        centroid = axis_info.get("centroid")
        if contour is not None and centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])
            cpts = contour.reshape(-1, 2)
            idx_sel = np.linspace(0, len(cpts) - 1, min(n_dirs, len(cpts)), dtype=int)
            for ci in idx_sel:
                cv2.line(img, (cx, cy), (int(cpts[ci][0]), int(cpts[ci][1])),
                         _RAY_COLOR, 1, cv2.LINE_AA)
            cv2.drawContours(img, [contour], -1, _CONTOUR_COLOR, lt)
            if axis_info.get("far_edge"):
                fx, fy = axis_info["far_edge"]
                cv2.line(img, (cx, cy), (int(fx), int(fy)), _AXIS_COLOR, lt)
    if density_pts:
        _draw_small_points(img, density_pts, _CAND_COLOR, max(1, H // 500))
    if classical_pts:
        _draw_small_points(img, classical_pts, _CLASSICAL_COLOR, max(1, H // 500))
    return img


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
            coral_gradcam=axis.get("coral_gradcam"),
            cls_attention=axis.get("cls_attention"),
            cls_is_fallback=axis.get("cls_is_fallback", False),
            ring_curves=axis.get("ring_curves"),
            classical_pts=axis.get("classical_pts"),
        )

        stem = Path(image_id).stem
        out_path = output_dir / f"{label}_{stem}_card.png"
        PILImage.fromarray(card, mode="RGB").save(out_path)
        saved.append(out_path)

    return saved

