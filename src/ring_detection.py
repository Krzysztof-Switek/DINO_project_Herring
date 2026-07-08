"""Image-based annual-ring detection (experimental, complements the model).

Otolith annual increments show as alternating light/dark concentric bands. This
module reads those bands directly from the photo — independently of the model —
by sampling intensity along **scaled copies of the otolith contour** (the same
parametrisation used for the reasoning-card rings), so a "ring at scale s" always
follows the otolith's real (elliptical) shape rather than a circle.

Pipeline:
    radial_band_profile()  — mean intensity vs. scale s∈(0,1] along scaled contours
    detect_ring_scales()   — peaks of that profile = candidate ring radii (scales)
    draw_scaled_rings()    — overlay those rings on the photo
    detect_and_draw_rings()— convenience wrapper returning (scales, overlay)

Status: EXPERIMENTAL / DIAGNOSTIC ONLY. Evaluated on the current photos (whole
otoliths, reflected light) via scaled-contour, axis-transect+CLAHE and polar-unwrap
methods — annual rings are NOT reliably recoverable (signal too weak; irregular
otolith boundary produces false peaks). See "wynik negatywny" in
``plans and summaries/7.07_TO_DO.md``. Kept as a tool for future, better imaging
(sections / transmitted light / higher magnification); OFF by default
(``candidates.detect_image_rings``). The trained model is the ring signal.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks


def radial_band_profile(
    gray: np.ndarray,
    mask: Optional[np.ndarray],
    centroid: Tuple[int, int],
    contour: np.ndarray,
    n_scales: int = 120,
    min_scale: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean intensity along scaled-contour rings, from nucleus (0) to edge (1).

    Args:
        gray:     (H, W) uint8/float grayscale otolith image
        mask:     (H, W) otolith mask (values > 0 inside); None = use whole image
        centroid: (cx, cy) nucleus
        contour:  otolith outline as returned by cv2.findContours
        n_scales: number of concentric rings sampled between ``min_scale`` and 1.0

    Returns:
        (scales, profile) — both length ``n_scales`` float32 arrays.
    """
    H, W = gray.shape[:2]
    c = np.asarray(centroid, dtype=np.float32)
    cpts = contour.reshape(-1, 2).astype(np.float32)
    scales = np.linspace(min_scale, 1.0, n_scales).astype(np.float32)
    profile = np.zeros(n_scales, dtype=np.float32)

    for i, s in enumerate(scales):
        pts = c + s * (cpts - c)
        xs = np.clip(pts[:, 0].astype(np.int32), 0, W - 1)
        ys = np.clip(pts[:, 1].astype(np.int32), 0, H - 1)
        vals = gray[ys, xs].astype(np.float32)
        if mask is not None:
            inside = mask[ys, xs] > 0
            if inside.any():
                vals = vals[inside]
        profile[i] = float(vals.mean()) if vals.size else 0.0

    return scales, profile


def detect_ring_scales(
    scales: np.ndarray,
    profile: np.ndarray,
    min_distance_frac: float = 0.04,
    prominence: float = 0.05,
    polarity: str = "bright",
) -> np.ndarray:
    """Return the scales (∈(0,1]) at which the profile forms a band.

    ``polarity='bright'`` detects light bands (maxima), ``'dark'`` the dark bands
    (minima). The profile is min-max normalised so ``prominence`` is in [0, 1].
    """
    p = np.asarray(profile, dtype=np.float32)
    rng = float(p.max() - p.min())
    if rng <= 1e-6:
        return np.array([], dtype=np.float32)
    p = (p - p.min()) / rng
    if polarity == "dark":
        p = 1.0 - p
    distance = max(1, int(min_distance_frac * len(scales)))
    peaks, _ = find_peaks(p, distance=distance, prominence=float(prominence))
    return scales[peaks].astype(np.float32)


def draw_scaled_rings(
    panel: np.ndarray,
    centroid: Tuple[int, int],
    contour: np.ndarray,
    scales: np.ndarray,
    color: Tuple[int, int, int] = (0, 220, 220),
    thickness: int = 2,
) -> None:
    """Draw concentric rings (scaled contours) at the given scales, in place."""
    c = np.asarray(centroid, dtype=np.float32)
    cpts = contour.reshape(-1, 2).astype(np.float32)
    for s in np.asarray(scales, dtype=np.float32):
        if s <= 0.0:
            continue
        ring = (c + float(s) * (cpts - c)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [ring], isClosed=True, color=color, thickness=thickness)


def detect_and_draw_rings(
    image_rgb: np.ndarray,
    axis_info: dict,
    n_scales: int = 120,
    prominence: float = 0.05,
    polarity: str = "bright",
    color: Tuple[int, int, int] = (0, 220, 220),
    thickness: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect image rings and return ``(ring_scales, overlay_rgb)``.

    Requires ``axis_info`` with ``centroid`` and ``contour`` (as from
    ``otolith_axis.detect_axis``); ``mask`` is used when present.
    """
    contour = axis_info.get("contour")
    if contour is None:
        return np.array([], dtype=np.float32), image_rgb.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    scales, profile = radial_band_profile(
        gray, axis_info.get("mask"), axis_info["centroid"], contour, n_scales)
    ring_scales = detect_ring_scales(scales, profile, prominence=prominence, polarity=polarity)
    overlay = image_rgb.copy()
    draw_scaled_rings(overlay, axis_info["centroid"], contour, ring_scales, color, thickness)
    return ring_scales, overlay
