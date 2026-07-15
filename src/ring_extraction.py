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


def extract_rings(
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
) -> List[Tuple[float, np.ndarray]]:
    """Extract rings from a per-patch probability map.

    Returns a list of ``(mean_t, curve)`` ordered inner→outer, where ``mean_t`` is
    the ring's normalised radius along the measurement axis (0 = nucleus, 1 = edge)
    and ``curve`` is an ``(M, 2)`` int array of image-pixel points (one per direction
    that saw the ring), ordered by angle — draw as a closed polygon. ``mean_t`` gives
    the axis crossing (``centroid + t·(far_edge − centroid)``) so the numbered dots
    and the ring curves stay consistent. Empty list when no rings are found.
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
    return curves


def extract_ring_curves(prob_grid, axis_info: dict, image_h: int, image_w: int,
                        **kwargs) -> List[np.ndarray]:
    """Convenience wrapper of :func:`extract_rings` returning only the curves."""
    return [curve for (_t, curve) in
            extract_rings(prob_grid, axis_info, image_h, image_w, **kwargs)]


def draw_ring_curves(panel: np.ndarray, curves: List[np.ndarray],
                     thickness: int = 2, colors=None) -> None:
    """Draw each ring curve as a coloured closed polyline (in place)."""
    for i, curve in enumerate(curves):
        if len(curve) < 3:
            continue
        color = colors[i % len(colors)] if colors else _RING_PALETTE[i % len(_RING_PALETTE)]
        cv2.polylines(panel, [curve.reshape(-1, 1, 2).astype(np.int32)],
                      isClosed=True, color=color, thickness=thickness)


# ---------------------------------------------------------------------------
# Shared radial-peak / clustering primitives (used by select_increments,
# classical_increments and fuse_increments — one implementation, no duplication)
# ---------------------------------------------------------------------------

def _all_ray_peaks(
    signal_grid, axis_info: dict, image_h: int, image_w: int,
    *, n_dirs: int = 48, n_samples: int = 64, min_distance: int = 3,
    prominence: float = 0.1, inner_margin: float = 0.05, edge_margin: float = 0.08,
    smooth_sigma: float = 0.0,
) -> Tuple[List[Tuple[float, float, int, int]], List[Tuple[int, int]]]:
    """Cast ``n_dirs`` rays nucleus→contour, per-ray normalise + find peaks.

    ``signal_grid`` may be the model density map ``(H_p, W_p)`` or a full-res
    grayscale image ``(H_img, W_img)`` — ``sample_profile_along_axis`` maps pixel
    positions to grid indices either way. Returns ``(peaks, candidate_pts)`` where
    ``peaks = [(t, strength, x, y)]`` (t = normalised radius, 0=nucleus, 1=edge).
    """
    contour = axis_info.get("contour") if axis_info else None
    centroid = axis_info.get("centroid") if axis_info else None
    if contour is None or centroid is None:
        return [], []
    grid = _to_numpy(signal_grid)
    if grid.ndim == 3:
        grid = grid.mean(axis=2).astype(np.float32)
    cpts = contour.reshape(-1, 2)
    if len(cpts) < 3:
        return [], []
    idx_sel = np.linspace(0, len(cpts) - 1, min(n_dirs, len(cpts)), dtype=int)

    smoother = None
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        smoother = lambda a: gaussian_filter1d(a, float(smooth_sigma))

    peaks: List[Tuple[float, float, int, int]] = []
    candidate_pts: List[Tuple[int, int]] = []
    for ci in idx_sel:
        cpt = (int(cpts[ci][0]), int(cpts[ci][1]))
        profile, line_xy = sample_profile_along_axis(
            grid, centroid, cpt, image_h, image_w, n_samples=n_samples)
        p = profile.astype(np.float32)
        if smoother is not None:
            p = smoother(p)
        rng = float(p.max() - p.min())
        if rng <= 1e-6:
            continue
        pn = (p - p.min()) / rng
        idxs, _ = find_peaks(pn, distance=max(1, int(min_distance)),
                             prominence=float(prominence))
        for idx in idxs:
            t = idx / max(1, n_samples - 1)
            if t < inner_margin or t > 1.0 - edge_margin:
                continue
            x, y = int(line_xy[idx][0]), int(line_xy[idx][1])
            peaks.append((t, float(pn[idx]), x, y))
            candidate_pts.append((x, y))
    return peaks, candidate_pts


def _cluster_by_radius(peaks, t_tol: float = 0.06) -> List[Tuple[float, int, float]]:
    """peaks ``[(t, strength, ...)]`` → clusters ``[(mean_t, support, mean_strength)]``.

    Greedy grouping on sorted normalised radius ``t`` (a real ring shows up at the
    same radius across many rays). ``support`` = #peaks (rays) in the cluster.
    """
    if not peaks:
        return []
    ps = sorted(peaks, key=lambda r: r[0])
    groups = [[ps[0]]]
    for r in ps[1:]:
        if r[0] - groups[-1][-1][0] <= t_tol:
            groups[-1].append(r)
        else:
            groups.append([r])
    out: List[Tuple[float, int, float]] = []
    for g in groups:
        mean_t = float(np.mean([r[0] for r in g]))
        out.append((mean_t, len(g), float(np.mean([r[1] for r in g]))))
    return out


def _topk_cluster_t(clusters, k: int) -> List[float]:
    """Keep the ``k`` highest-scoring clusters (score = support × mean_strength),
    return their radii sorted inner→outer."""
    if k <= 0 or not clusters:
        return []
    scored = sorted(clusters, key=lambda c: c[1] * c[2], reverse=True)
    return sorted(mt for (mt, _s, _st) in scored[:k])


def _project_to_axis(chosen_t, axis_info: dict) -> List[Tuple[int, int]]:
    """Project normalised radii onto the measurement axis (nucleus → far edge)."""
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    return [(int(round(cx + t * (fx - cx))), int(round(cy + t * (fy - cy))))
            for t in chosen_t]


def select_increments(
    prob_grid,
    axis_info: dict,
    predicted_age: int,
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
) -> dict:
    """Multi-axis increment localisation with count = ``predicted_age`` (11.07 Punkt 7).

    Casts ``n_dirs`` rays from the nucleus, finds probability peaks along each
    (candidates), clusters them by normalised radius across rays, ranks clusters by
    (support × mean strength), and keeps the top ``predicted_age`` as the FINAL
    increments — projected onto the measurement axis (nucleus → far edge).

    Returns dict:
      final_t        : list[float]   normalised radii of chosen increments (≤ age), inner→outer
      final_axis_pts : list[(x, y)]  those radii projected onto the measurement axis (image px)
      candidate_pts  : list[(x, y)]  every per-ray peak (image px)
    """
    empty = {"final_t": [], "final_axis_pts": [], "candidate_pts": []}
    if not axis_info or axis_info.get("far_edge") is None:
        return empty
    peaks, candidate_pts = _all_ray_peaks(
        prob_grid, axis_info, image_h, image_w,
        n_dirs=n_dirs, n_samples=n_samples, min_distance=min_distance,
        prominence=prominence, inner_margin=inner_margin, edge_margin=edge_margin,
    )
    chosen_t = _topk_cluster_t(_cluster_by_radius(peaks, t_tol), max(0, int(predicted_age)))
    return {"final_t": chosen_t,
            "final_axis_pts": _project_to_axis(chosen_t, axis_info),
            "candidate_pts": candidate_pts}


def classical_increments(
    gray_image,
    axis_info: dict,
    *,
    n_dirs: int = 48,
    n_samples: int = 64,
    smooth_sigma: float = 0.0,
    min_distance: int = 1,
    prominence: float = 0.02,
    inner_margin: float = 0.05,
    edge_margin: float = 0.08,
    t_tol: float = 0.06,
) -> dict:
    """Classical (image-intensity) increment candidates along the SAME rays as the model.

    Mirrors :func:`select_increments` but samples the raw **grayscale image** instead of
    the model's density map: casts ``n_dirs`` rays from the nucleus, optionally smooths
    each ray's intensity profile (Gaussian ``smooth_sigma``), finds peaks and clusters
    them by normalised radius. NO count is imposed here — increment SELECTION (fusion with
    the model + CORAL count) is a separate step. Defaults follow the values that gave a
    usable classical signal (sigma 0, prominence 0.02, min-distance 1).

    Returns dict:
      candidate_pts : list[(x, y)]                every per-ray intensity peak (image px)
      peaks         : list[(t, strength, x, y)]   same, with normalised radius + strength
      clusters      : list[(mean_t, support, mean_strength)]  rings-by-radius (consensus)
    """
    if not axis_info:
        return {"candidate_pts": [], "peaks": [], "clusters": []}
    gray = _to_numpy(gray_image)
    if gray.ndim == 3:                                  # RGB → luminancja
        gray = gray.mean(axis=2).astype(np.float32)
    image_h, image_w = gray.shape[:2]
    peaks, candidate_pts = _all_ray_peaks(
        gray, axis_info, image_h, image_w,
        n_dirs=n_dirs, n_samples=n_samples, min_distance=min_distance,
        prominence=prominence, inner_margin=inner_margin, edge_margin=edge_margin,
        smooth_sigma=smooth_sigma,
    )
    return {"candidate_pts": candidate_pts, "peaks": peaks,
            "clusters": _cluster_by_radius(peaks, t_tol)}


def density_peaks(
    prob_grid, axis_info: dict, image_h: int, image_w: int,
    *, n_dirs: int = 48, n_samples: int = 64, min_distance: int = 3,
    prominence: float = 0.1, inner_margin: float = 0.05, edge_margin: float = 0.08,
) -> Tuple[List[Tuple[float, float, int, int]], List[Tuple[int, int]]]:
    """Per-ray peaks of the model DENSITY map (for fusion). ``(peaks, candidate_pts)``.

    Same ray-casting as :func:`select_increments`, but exposes the raw
    ``[(t, strength, x, y)]`` peaks so :func:`fuse_increments` can combine them with
    the classical peaks before choosing the final increments.
    """
    return _all_ray_peaks(
        prob_grid, axis_info, image_h, image_w,
        n_dirs=n_dirs, n_samples=n_samples, min_distance=min_distance,
        prominence=prominence, inner_margin=inner_margin, edge_margin=edge_margin,
    )


def fuse_increments(
    density_pks, classical_pks, predicted_age: int, axis_info: dict,
    *, method: str = "consensus", t_tol: float = 0.06,
) -> dict:
    """Choose the final ``predicted_age`` increments on the axis from peak sources.

    ``density_pks`` / ``classical_pks`` are ``[(t, strength, x, y)]`` lists
    (from :func:`density_peaks` / :func:`classical_increments`). ``method``:
      * ``"density"``   — top-`age` clusters of density peaks only (model localisation).
      * ``"classical"`` — top-`age` clusters of classical (image-intensity) peaks only.
      * ``"consensus"`` — clusters where density AND classical agree on radius ``t``
        (combined support), top-`age`; falls back to top density clusters if fewer than
        ``age`` agree.
    Returns ``{final_t, final_axis_pts, candidate_pts}`` (candidates = the source(s) used).
    """
    empty = {"final_t": [], "final_axis_pts": [], "candidate_pts": []}
    if not axis_info or axis_info.get("far_edge") is None:
        return empty
    k = max(0, int(predicted_age))

    if method == "density":
        chosen = _topk_cluster_t(_cluster_by_radius(density_pks, t_tol), k)
        cand = [(p[2], p[3]) for p in density_pks]
    elif method == "classical":
        chosen = _topk_cluster_t(_cluster_by_radius(classical_pks, t_tol), k)
        cand = [(p[2], p[3]) for p in classical_pks]
    else:  # consensus
        dclust = _cluster_by_radius(density_pks, t_tol)
        cclust = _cluster_by_radius(classical_pks, t_tol)
        agreed = []                                     # (combined_support, mean_t)
        for (dt, ds, _dstr) in dclust:
            near = [c for c in cclust if abs(c[0] - dt) <= t_tol]
            if near:
                c = min(near, key=lambda cc: abs(cc[0] - dt))
                agreed.append((ds + c[1], 0.5 * (dt + c[0])))
        agreed.sort(key=lambda s: s[0], reverse=True)
        chosen = sorted(mt for _s, mt in agreed[:k])
        if len(chosen) < k:                             # fallback: fill from top density clusters
            for mt in _topk_cluster_t(dclust, len(dclust)):
                if len(chosen) >= k:
                    break
                if all(abs(mt - c) > t_tol for c in chosen):
                    chosen.append(mt)
            chosen = sorted(chosen)
        cand = [(p[2], p[3]) for p in density_pks] + [(p[2], p[3]) for p in classical_pks]

    return {"final_t": chosen,
            "final_axis_pts": _project_to_axis(chosen, axis_info),
            "candidate_pts": cand}
