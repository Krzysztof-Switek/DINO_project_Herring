"""Ring-CURVE extraction from the model's MIL probability map (post-training).

A ring is a continuous/partially-continuous contrast line from the nucleus to the
edge — not an artifact, crack or the preparation edge. The model's native ring
signal is the 2-D per-patch increment-probability map (MIL head; weakly supervised
so ``sum(patch_probs) ≈ age``). We derive the actual ring CURVES from it:

  1. sample the prob map radially in many directions (centroid → contour points),
  2. per direction, find the peak radius (dropping the outer ``edge_margin`` so the
     preparation edge / boundary is never counted, and the inner ``inner_margin``
     so the nucleus is not counted),
  3. cluster the peak radii across directions into rings,
  4. connect each cluster's per-direction points, ordered by angle, into a closed
     curve = the locus of high MIL probability = the ring line.

Only meaningful on a TRAINED model; on an untrained/demo model the prob map is
noise → few/no curves (expected). Whether the map localises at all is measured by
``scripts/validate_f5_f11.py`` (F11: #active patches ≈ age).

Reuses ``otolith_axis.sample_profile_along_axis`` for the radial sampling.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks

from src.otolith_axis import sample_profile_along_axis

# distinct ring colours (RGB), cycled if more rings than colours
_RING_PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (0, 206, 209), (166, 86, 40), (247, 129, 191),
]


def _to_numpy(grid) -> np.ndarray:
    if hasattr(grid, "cpu"):
        grid = grid.cpu().numpy()
    return np.asarray(grid, dtype=np.float32)


def _radial_peaks(prob_grid, centroid, contour_pt, image_h, image_w,
                  n_samples, min_distance, prominence, inner_margin, edge_margin
                  ) -> List[Tuple[float, int, int]]:
    """Peaks along one radial transect → list of (t, x, y) in image pixels."""
    profile, line_xy = sample_profile_along_axis(
        prob_grid, centroid, contour_pt, image_h, image_w, n_samples=n_samples)
    p = profile.astype(np.float32)
    rng = float(p.max() - p.min())
    if rng <= 1e-6:
        return []
    pn = (p - p.min()) / rng
    idxs, _ = find_peaks(pn, distance=max(1, int(min_distance)), prominence=float(prominence))
    out: List[Tuple[float, int, int]] = []
    for idx in idxs:
        t = idx / max(1, n_samples - 1)
        if t < inner_margin or t > 1.0 - edge_margin:
            continue
        out.append((t, int(line_xy[idx][0]), int(line_xy[idx][1])))
    return out


def extract_ring_curves(
    prob_grid,
    axis_info: dict,
    image_h: int,
    image_w: int,
    *,
    n_dirs: int = 48,
    n_samples: int = 64,
    min_distance: int = 3,
    prominence: float = 0.1,
    inner_margin: float = 0.05,
    edge_margin: float = 0.08,
    t_tol: float = 0.06,
    min_dir_frac: float = 0.5,
) -> List[np.ndarray]:
    """Extract ring curves from a per-patch probability map.

    Returns a list of curves ordered inner→outer; each curve is an ``(M, 2)`` int
    array of image-pixel points (one per direction that saw the ring), ordered by
    angle — draw as a closed polygon. Empty list when no rings are found.
    """
    contour = axis_info.get("contour")
    centroid = axis_info.get("centroid")
    if contour is None or centroid is None:
        return []
    prob_grid = _to_numpy(prob_grid)
    cpts = contour.reshape(-1, 2)
    if len(cpts) < 3:
        return []
    cx, cy = centroid

    idx_sel = np.linspace(0, len(cpts) - 1, min(n_dirs, len(cpts)), dtype=int)
    n_used = len(idx_sel)

    peaks = []  # (dir_j, angle, t, x, y)
    for j, ci in enumerate(idx_sel):
        cpt = (int(cpts[ci][0]), int(cpts[ci][1]))
        angle = float(np.arctan2(cpt[1] - cy, cpt[0] - cx))
        for (t, x, y) in _radial_peaks(prob_grid, centroid, cpt, image_h, image_w,
                                       n_samples, min_distance, prominence,
                                       inner_margin, edge_margin):
            peaks.append((j, angle, t, x, y))
    if not peaks:
        return []

    # cluster peaks by normalised radius t (greedy on sorted t)
    peaks.sort(key=lambda r: r[2])
    clusters = [[peaks[0]]]
    for r in peaks[1:]:
        if r[2] - clusters[-1][-1][2] <= t_tol:
            clusters[-1].append(r)
        else:
            clusters.append([r])

    curves: List[Tuple[float, np.ndarray]] = []
    min_dirs = max(3, int(min_dir_frac * n_used))
    for cl in clusters:
        mean_t = float(np.mean([r[2] for r in cl]))
        by_dir: dict[int, tuple] = {}     # one point per direction (closest to mean_t)
        for (j, angle, t, x, y) in cl:
            if j not in by_dir or abs(t - mean_t) < abs(by_dir[j][0] - mean_t):
                by_dir[j] = (t, angle, x, y)
        if len(by_dir) < min_dirs:         # too few directions → not a real ring
            continue
        pts = sorted(by_dir.values(), key=lambda v: v[1])          # order by angle
        curve = np.array([[x, y] for (_, _, x, y) in pts], dtype=np.int32)
        curves.append((mean_t, curve))

    curves.sort(key=lambda c: c[0])        # inner → outer
    return [c for (_, c) in curves]


def draw_ring_curves(panel: np.ndarray, curves: List[np.ndarray],
                     thickness: int = 2, colors=None) -> None:
    """Draw each ring curve as a coloured closed polyline (in place)."""
    for i, curve in enumerate(curves):
        if len(curve) < 3:
            continue
        color = colors[i % len(colors)] if colors else _RING_PALETTE[i % len(_RING_PALETTE)]
        cv2.polylines(panel, [curve.reshape(-1, 1, 2).astype(np.int32)],
                      isClosed=True, color=color, thickness=thickness)
