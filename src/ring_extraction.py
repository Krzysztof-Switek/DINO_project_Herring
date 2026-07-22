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

from typing import List, Optional, Tuple

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

def _all_ray_profiles(
    signal_grid, axis_info: dict, image_h: int, image_w: int,
    *, n_dirs: int = 48, n_samples: int = 64, smooth_sigma: float = 0.0,
) -> Tuple[List[Optional[np.ndarray]], List[np.ndarray], List[Tuple[int, int]]]:
    """Sample + per-ray normalise ALL ``n_dirs`` rays (jądro→kontur); NO peak-finding.

    Split out of :func:`_all_ray_peaks` so the interactive Krok-4 widget (JS sliders)
    can redo peak-finding client-side at an arbitrary prominence from the SAME raw
    profiles the server uses — see :func:`dp_interactive_data`.

    Returns ``(profiles, line_xys, contour_pts)``: ``profiles[i]`` is the ray's
    ``(n_samples,)`` float32 profile normalised to ``[0, 1]``, or ``None`` for a
    degenerate (flat) ray; ``line_xys[i]`` is that ray's ``(n_samples, 2)`` pixel path;
    ``contour_pts[i]`` is the ray's contour endpoint.
    """
    contour = axis_info.get("contour") if axis_info else None
    centroid = axis_info.get("centroid") if axis_info else None
    if contour is None or centroid is None:
        return [], [], []
    grid = _to_numpy(signal_grid)
    if grid.ndim == 3:
        grid = grid.mean(axis=2).astype(np.float32)
    cpts = contour.reshape(-1, 2)
    if len(cpts) < 3:
        return [], [], []
    idx_sel = np.linspace(0, len(cpts) - 1, min(n_dirs, len(cpts)), dtype=int)

    smoother = None
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        smoother = lambda a: gaussian_filter1d(a, float(smooth_sigma))

    profiles: List[Optional[np.ndarray]] = []
    line_xys: List[np.ndarray] = []
    contour_pts: List[Tuple[int, int]] = []
    for ci in idx_sel:
        cpt = (int(cpts[ci][0]), int(cpts[ci][1]))
        contour_pts.append(cpt)
        profile, line_xy = sample_profile_along_axis(
            grid, centroid, cpt, image_h, image_w, n_samples=n_samples)
        line_xys.append(line_xy)
        p = profile.astype(np.float32)
        if smoother is not None:
            p = smoother(p)
        rng = float(p.max() - p.min())
        profiles.append((p - p.min()) / rng if rng > 1e-6 else None)
    return profiles, line_xys, contour_pts


def _shift_peak_to_falling_edge(p: np.ndarray, idx: int) -> int:
    """Move a detected peak index to the boundary where it biologically belongs.

    Under transmitted light (this project's images) the translucent, fast-growth zone
    is BRIGHT and the opaque, slow-growth "winter" zone — the annulus itself — is DARK;
    standard otolith age-reading marks the increment at that light→dark boundary, not
    at the brightness peak (Campana lab / DFO / NOAA otolith-ageing references — see
    ``plans and summaries/20.07_session_summary.md``). This applies the same falling-
    edge convention to any per-ray signal (density or classical): walk forward from the
    peak to the next trough, then return the index where the signal crosses HALFWAY
    down between them (a standard, noise-robust edge/half-max-crossing location).
    Falls back to ``idx`` unchanged when the signal doesn't descend after the peak
    (e.g. a peak sitting at the very end of the profile).
    """
    n = len(p)
    end = idx
    while end + 1 < n and p[end + 1] <= p[end]:
        end += 1
    if end == idx:
        return idx
    half = 0.5 * (float(p[idx]) + float(p[end]))
    for i in range(idx, end):
        if p[i] >= half > p[i + 1]:
            return i + 1
    return end


def _all_ray_peaks(
    signal_grid, axis_info: dict, image_h: int, image_w: int,
    *, n_dirs: int = 48, n_samples: int = 64, min_distance: int = 3,
    prominence: float = 0.1, inner_margin: float = 0.05, edge_margin: float = 0.08,
    smooth_sigma: float = 0.0,
) -> Tuple[List[Tuple[float, float, int, int, int]], List[Tuple[int, int]]]:
    """Cast ``n_dirs`` rays nucleus→contour, per-ray normalise + find peaks.

    ``signal_grid`` may be the model density map ``(H_p, W_p)`` or a full-res
    grayscale image ``(H_img, W_img)`` — ``sample_profile_along_axis`` maps pixel
    positions to grid indices either way. Returns ``(peaks, candidate_pts)`` where
    ``peaks = [(t, strength, x, y, ray_idx)]`` (t = normalised radius, 0=nucleus,
    1=edge; ``ray_idx`` in ``[0, n_dirs)`` identifies WHICH of the ``n_dirs`` rays
    this peak came from — used to tell a ring seen along a compact angular arc apart
    from one seen at scattered, unrelated directions; see
    :func:`_cluster_by_radius_with_arcs`). Each peak's reported POSITION is shifted to
    its falling edge (:func:`_shift_peak_to_falling_edge`) — ``strength`` still
    reflects the peak's own height (for scoring/clustering), only WHERE we say the
    ring sits moves. ``inner_margin``/``edge_margin`` are checked against the RAW
    (pre-shift) peak position — a genuine peak found safely inside the valid window
    must not be dropped just because shifting it toward its biological (falling-edge)
    marker pushes the reported ``t`` past the edge cutoff.
    """
    profiles, line_xys, _contour_pts = _all_ray_profiles(
        signal_grid, axis_info, image_h, image_w,
        n_dirs=n_dirs, n_samples=n_samples, smooth_sigma=smooth_sigma)

    peaks: List[Tuple[float, float, int, int, int]] = []
    candidate_pts: List[Tuple[int, int]] = []
    for ray_idx, (pn, line_xy) in enumerate(zip(profiles, line_xys)):
        if pn is None:
            continue
        idxs, _ = find_peaks(pn, distance=max(1, int(min_distance)),
                             prominence=float(prominence))
        for idx in idxs:
            t_orig = idx / max(1, n_samples - 1)
            if t_orig < inner_margin or t_orig > 1.0 - edge_margin:
                continue
            edge_idx = _shift_peak_to_falling_edge(pn, int(idx))
            t = edge_idx / max(1, n_samples - 1)
            x, y = int(line_xy[edge_idx][0]), int(line_xy[edge_idx][1])
            peaks.append((t, float(pn[idx]), x, y, ray_idx))
            candidate_pts.append((x, y))
    return peaks, candidate_pts


def _cluster_by_radius(peaks, t_tol: float = 0.06) -> List[Tuple[float, int, float]]:
    """peaks ``[(t, strength, ...)]`` → clusters ``[(mean_t, support, mean_strength)]``.

    Finds MODES of the radius-vote density (a real ring shows up at the same radius across
    many rays), then assigns each peak to its nearest mode. ``support`` = #peaks (rays) in
    the cluster. Modes are kept ≥ ``t_tol`` apart, so clusters have bounded width.

    (16.07 E1 fix.) The old greedy chaining — "extend the current group while the next peak
    is within ``t_tol`` of the previous" — collapsed DENSE peak sets: ~273 near-uniform
    classical peaks (spacing ~0.003 ≪ t_tol) chained into 1–2 mega-clusters, so dp/consensus
    got far fewer rings than the age. Voting on a smoothed histogram + non-max suppression
    spreads such peaks into distinct radii instead (dendro-style radial vote peak-finding).
    """
    if not peaks:
        return []
    ts = np.asarray([r[0] for r in peaks], dtype=np.float64)
    ss = np.asarray([r[1] for r in peaks], dtype=np.float64)
    if len(ts) == 1:
        return [(float(ts[0]), 1, float(ss[0]))]

    # Smoothed vote histogram over [0, 1] (bin ≈ t_tol/3, moving-average window ≈ t_tol).
    nbins = max(4, int(round(1.0 / max(t_tol / 3.0, 1e-3))))
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(np.clip(ts, 0.0, 1.0), bins=edges)
    win = max(1, int(round(t_tol * nbins)))
    smooth = (np.convolve(counts.astype(np.float64), np.ones(win) / win, mode="same")
              if win > 1 else counts.astype(np.float64))

    # Non-max suppression: greedily take the highest-vote bin, claim ±t_tol around it.
    # Ties (equal smoothed vote, common with small integer counts) broken by ascending
    # bin index — np.argsort's default quicksort is NOT stable, so without an explicit
    # tie-break this order (and thus which bin claims a contested region) is unspecified
    # and can differ between equivalent implementations (caught by cross-checking this
    # exact function against a JS port for the Krok-4 live widget, 20.07).
    modes: List[float] = []
    claimed = np.zeros(nbins, dtype=bool)
    order = np.lexsort((np.arange(nbins), -smooth))
    for bi in order:
        if smooth[bi] <= 0 or claimed[bi]:
            continue
        c = float(centers[bi])
        claimed |= np.abs(centers - c) <= t_tol
        modes.append(c)
    if not modes:
        return []
    modes_arr = np.asarray(sorted(modes))

    # Assign each peak to its NEAREST mode (within t_tol) → no double counting.
    nearest = np.abs(ts[:, None] - modes_arr[None, :]).argmin(axis=1)
    out: List[Tuple[float, int, float]] = []
    for mi in range(len(modes_arr)):
        sel = (nearest == mi) & (np.abs(ts - modes_arr[mi]) <= t_tol)
        if not sel.any():
            continue
        out.append((float(ts[sel].mean()), int(sel.sum()), float(ss[sel].mean())))
    out.sort(key=lambda c: c[0])
    return out


def _best_arc(ray_idxs: np.ndarray, strengths: np.ndarray, n_dirs: int,
             max_gap: int) -> Tuple[int, float]:
    """Longest run of angularly-CONSECUTIVE ray indices among a cluster's members.

    The ray circle wraps (ray ``n_dirs-1`` is adjacent to ray ``0``); a run tolerates
    gaps of up to ``max_gap`` missing rays (real bands are rarely unbroken for their
    whole visible stretch). Returns ``(run_len, run_strength)``: ``run_len`` is the
    SPAN in ray-slots the run covers (inclusive of tolerated gaps — one missing ray in
    an otherwise solid arc still counts its full width), ``run_strength`` is the mean
    peak strength of the members actually inside that run.
    """
    uniq = sorted(set(int(r) for r in ray_idxs))
    if not uniq:
        return 0, 0.0
    if len(uniq) == 1:
        return 1, float(strengths[ray_idxs == uniq[0]].mean())

    # Unroll the circle by duplicating the sequence shifted by n_dirs, so a run that
    # wraps past n_dirs-1 -> 0 is just a normal contiguous slice — no special-casing.
    doubled = uniq + [r + n_dirs for r in uniq]
    n = len(doubled)
    best_len, best_strength = 0, 0.0
    for start in range(len(uniq)):
        end = start
        while (end + 1 < n and doubled[end + 1] - doubled[end] - 1 <= max_gap
               and doubled[end + 1] - doubled[start] < n_dirs):
            end += 1
        span = min(doubled[end] - doubled[start] + 1, n_dirs)
        members = {x % n_dirs for x in doubled[start:end + 1]}
        mask = np.isin(ray_idxs, list(members))
        strength = float(strengths[mask].mean()) if mask.any() else 0.0
        if span > best_len or (span == best_len and strength > best_strength):
            best_len, best_strength = span, strength
    return best_len, best_strength


def _cluster_by_radius_with_arcs(
    peaks, t_tol: float = 0.06, n_dirs: int = 48, max_gap: int = 2,
) -> List[Tuple[float, int, float, int, float]]:
    """Like :func:`_cluster_by_radius`, but each cluster also reports its strongest
    CONTIGUOUS angular arc: ``(mean_t, support, mean_strength, arc_len, arc_strength)``.

    A ring genuinely visible along a compact stretch of the circumference (e.g. only
    the upper-left quadrant of an otolith) is stronger evidence than the same total
    ray count scattered randomly around all ``n_dirs`` directions — ``_cluster_by_radius``
    gives both the same ``support`` and can't tell them apart. This is a SEPARATE
    function (not a change to ``_cluster_by_radius``) so its many existing consumers
    (``_topk_cluster_t``, ``select_increments``, the JS ``clusterByRadius``) are
    untouched; only :func:`_merge_clusters` (the ``dp`` fusion path) uses this one.
    Peaks must be ``_all_ray_peaks``-shaped tuples with a ray index at position 4.
    """
    if not peaks:
        return []
    ts = np.asarray([r[0] for r in peaks], dtype=np.float64)
    ss = np.asarray([r[1] for r in peaks], dtype=np.float64)
    rays = np.asarray([r[4] for r in peaks], dtype=np.int64)
    if len(ts) == 1:
        return [(float(ts[0]), 1, float(ss[0]), 1, float(ss[0]))]

    nbins = max(4, int(round(1.0 / max(t_tol / 3.0, 1e-3))))
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(np.clip(ts, 0.0, 1.0), bins=edges)
    win = max(1, int(round(t_tol * nbins)))
    smooth = (np.convolve(counts.astype(np.float64), np.ones(win) / win, mode="same")
              if win > 1 else counts.astype(np.float64))

    modes: List[float] = []
    claimed = np.zeros(nbins, dtype=bool)
    order = np.lexsort((np.arange(nbins), -smooth))
    for bi in order:
        if smooth[bi] <= 0 or claimed[bi]:
            continue
        c = float(centers[bi])
        claimed |= np.abs(centers - c) <= t_tol
        modes.append(c)
    if not modes:
        return []
    modes_arr = np.asarray(sorted(modes))

    nearest = np.abs(ts[:, None] - modes_arr[None, :]).argmin(axis=1)
    out: List[Tuple[float, int, float, int, float]] = []
    for mi in range(len(modes_arr)):
        sel = (nearest == mi) & (np.abs(ts - modes_arr[mi]) <= t_tol)
        if not sel.any():
            continue
        arc_len, arc_strength = _best_arc(rays[sel], ss[sel], n_dirs, max_gap)
        out.append((float(ts[sel].mean()), int(sel.sum()), float(ss[sel].mean()),
                   arc_len, arc_strength))
    out.sort(key=lambda c: c[0])
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


def _dp_select_t(cands, k: int, min_gap: float, spread_weight: float = 1.5) -> List[float]:
    """Pick exactly ``k`` radii from ``cands=[(t, score)]`` maximising total score (PLUS a
    spread bonus) with a minimum spacing ``min_gap`` between chosen radii (DP peak selection,
    monotone in ``t``).

    This is the ``method="dp"`` selector: unlike top-k (which can bunch several picks at
    almost the same radius), the spacing constraint spreads the ``k`` increments along the
    axis — the classic dynamic-programming ring/peak selection used in tree-ring counting.

    ``spread_weight`` (20.07): a flat minimum-gap ALONE doesn't stop DP from bunching all
    ``k`` picks in whichever single sub-region happens to have the highest raw score — real
    otoliths often have one very strong, near-universally-supported band (e.g. close to the
    nucleus) that would otherwise swallow every pick, even when decent candidates exist all
    the way to the edge (20.07, user report: 3/3 picks landed in the inner third of the axis
    despite good-scoring candidates out to t=0.98). Each DP transition adds
    ``spread_weight * gap * mean_score`` (``mean_score`` = mean of all candidate scores, so the
    bonus is on the same scale as the scores themselves regardless of their absolute units) —
    rewarding WIDER gaps between consecutive picks, not just enforcing the ``min_gap`` floor.
    ``spread_weight=0`` reproduces the exact previous (score-only) behaviour. Default 1.5 —
    calibrated against the actual reported case (real classical+density scores, inner cluster
    ~2-4x the outer candidates' score): weights below ~1.0 left the bunched selection
    unchanged; 1.0-5.0 all reliably swap in the distant candidate. Still a first pass, not
    exhaustively validated across many otoliths — worth revisiting once more real cards are
    available.

    Falls back to the top-``k`` by score (spacing ignored) when spacing makes ``k`` picks
    infeasible, so it always returns as many as possible up to ``k``.
    """
    if k <= 0 or not cands:
        return []
    cs = sorted(cands, key=lambda c: c[0])                    # by radius (inner→outer)
    ts = [float(c[0]) for c in cs]
    ss = [float(c[1]) for c in cs]
    M = len(cs)
    k = min(k, M)
    mean_score = sum(ss) / M
    NEG = float("-inf")
    dp = [[NEG] * M for _ in range(k + 1)]                    # dp[j][i]: best score, j picks, last at i
    par = [[-1] * M for _ in range(k + 1)]
    for i in range(M):
        dp[1][i] = ss[i]
    for j in range(2, k + 1):
        for i in range(M):
            best, bp = NEG, -1
            for p in range(i):
                if ts[i] - ts[p] >= min_gap:
                    cand_val = dp[j - 1][p] + spread_weight * (ts[i] - ts[p]) * mean_score
                    if dp[j - 1][p] > NEG and cand_val > best:
                        best, bp = cand_val, p
            if bp != -1:
                dp[j][i] = best + ss[i]
                par[j][i] = bp
    end, best_val = -1, NEG
    for i in range(M):
        if dp[k][i] > best_val:
            best_val, end = dp[k][i], i
    if end == -1:                                            # spacing infeasible for k → top-k by score
        top = sorted(range(M), key=lambda i: ss[i], reverse=True)[:k]
        return sorted(ts[i] for i in top)
    chosen, j = [], k
    while end != -1 and j >= 1:
        chosen.append(ts[end])
        end = par[j][end]
        j -= 1
    return sorted(chosen)


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


def polar_averaged_increments(
    gray_image, axis_info: dict, image_h: int, image_w: int,
    *, n_dirs: int = 48, n_samples: int = 64, smooth_sigma: float = 0.0,
    min_distance: int = 1, prominence: float = 0.02,
    inner_margin: float = 0.05, edge_margin: float = 0.08,
) -> dict:
    """E3 (``16.07_lokalizacja_przyrostów.md`` Tier 1): a SINGLE angularly-averaged
    radial profile — the polar-transform idea (pierścienie → poziome linie w obrazie
    polarnym) — as an alternative classical candidate source, to compare against the
    default per-ray ``classical_increments`` (many independent per-ray peaks).

    Deliberately NOT a literal ``cv2.warpPolar`` at a fixed physical radius: that warps
    well for roughly circular shapes, but averaging raw pixels at a fixed physical
    radius across all angles would mix real otolith signal (long-radius directions)
    with background (short-radius directions) for a strongly non-circular, scalloped,
    "tailed" contour like ours — exactly the otolith shape seen throughout this
    project's cards. Instead this averages the SAME per-ray profiles
    ``classical_increments``/``_all_ray_peaks`` already sample, each already normalised
    to ITS OWN [0,1] jądro→kontur range — equivalent to a polar warp where every row
    (ray) is stretched to the same length before averaging, so only real interior
    otolith pixels ever contribute at a given normalised radius ``t``, regardless of
    that ray's physical length.

    Returns dict: ``profile`` (n_samples,) averaged+renormalised signal, ``peak_t``
    (peak positions, t=0..1, falling-edge shifted like every other candidate source),
    ``clusters`` — kept as a single ``(mean_t, n_dirs, mean_strength)`` per peak (there
    is only ONE profile, so "support" isn't meaningful the way it is for per-ray
    clustering; n_dirs is reported as a reminder how many rays informed the average).
    """
    profiles, _line_xys, _contour_pts = _all_ray_profiles(
        gray_image, axis_info, image_h, image_w,
        n_dirs=n_dirs, n_samples=n_samples, smooth_sigma=smooth_sigma)
    valid = [p for p in profiles if p is not None]
    if not valid:
        return {"profile": [], "peak_t": [], "clusters": []}
    avg = np.mean(np.stack(valid), axis=0)
    rng = float(avg.max() - avg.min())
    norm = (avg - avg.min()) / rng if rng > 1e-6 else np.zeros_like(avg)

    peak_t: List[float] = []
    if rng > 1e-6:
        idxs, _ = find_peaks(norm, distance=max(1, int(min_distance)), prominence=float(prominence))
        for idx in idxs:
            t_orig = idx / max(1, n_samples - 1)
            if t_orig < inner_margin or t_orig > 1.0 - edge_margin:
                continue
            edge_idx = _shift_peak_to_falling_edge(norm, int(idx))
            peak_t.append(float(edge_idx / max(1, n_samples - 1)))
    clusters = [(t, len(valid), float(norm[int(round(t * (n_samples - 1)))])) for t in peak_t]
    return {"profile": norm.tolist(), "peak_t": peak_t, "clusters": clusters}


def density_peaks(
    prob_grid, axis_info: dict, image_h: int, image_w: int,
    *, n_dirs: int = 48, n_samples: int = 64, min_distance: int = 3,
    prominence: float = 0.1, inner_margin: float = 0.05, edge_margin: float = 0.08,
) -> Tuple[List[Tuple[float, float, int, int, int]], List[Tuple[int, int]]]:
    """Per-ray peaks of the model DENSITY map (for fusion). ``(peaks, candidate_pts)``.

    Same ray-casting as :func:`select_increments`, but exposes the raw
    ``[(t, strength, x, y, ray_idx)]`` peaks so :func:`fuse_increments` can combine them
    with the classical peaks before choosing the final increments.
    """
    return _all_ray_peaks(
        prob_grid, axis_info, image_h, image_w,
        n_dirs=n_dirs, n_samples=n_samples, min_distance=min_distance,
        prominence=prominence, inner_margin=inner_margin, edge_margin=edge_margin,
    )


def _merge_clusters(density_pks, classical_pks, t_tol: float = 0.06, n_dirs: int = 48):
    """Merge density + classical radius-clusters into candidate rings ``[(t, score, source)]``.

    A density ring corroborated by a classical ring at a similar radius (``t_tol``) becomes
    one **consensus** ring whose score is the SUM of both sources' scores — this is how
    ``fuse_increments(method="dp")`` (and the step-by-step walkthrough) reward agreement
    between the two sources. Rings seen by only one source stay in with their own score.
    ``source`` ∈ {``"consensus"``, ``"density"``, ``"classical"``}.

    Each source's clusters come from :func:`_cluster_by_radius_with_arcs`, so a cluster's
    score blends its overall support with its strongest CONTIGUOUS angular arc (20.07):
    a ring genuinely visible along a compact stretch of the circumference outscores one
    with the same total support scattered randomly around all ``n_dirs`` directions —
    rings on real otoliths are very often clearly banded only in PART of the image (e.g.
    one otolith's user-reported example: sharp bands visible in the top/left arcs only,
    faint elsewhere). Weighted 0.4 total-support + 0.6 best-arc — a reasonable first-pass
    split, not yet tuned/validated against real cards.
    """
    dclust = _cluster_by_radius_with_arcs(density_pks, t_tol, n_dirs)
    cclust = _cluster_by_radius_with_arcs(classical_pks, t_tol, n_dirs)

    def _score(c) -> float:
        _t, support, mean_strength, arc_len, arc_strength = c
        return support * mean_strength * 0.4 + arc_len * arc_strength * 0.6

    merged: List[Tuple[float, float, str]] = []
    used = [False] * len(cclust)
    for dc in dclust:
        dt = dc[0]
        score = _score(dc)
        t, source = dt, "density"
        best_i, best_d = -1, t_tol
        for i, c in enumerate(cclust):
            if not used[i] and abs(c[0] - dt) <= best_d:
                best_i, best_d = i, abs(c[0] - dt)
        if best_i >= 0:                                  # density ring corroborated by classical
            c = cclust[best_i]
            used[best_i] = True
            score += _score(c)
            t, source = 0.5 * (dt + c[0]), "consensus"
        merged.append((t, score, source))
    for i, c in enumerate(cclust):                       # classical-only rings still eligible
        if not used[i]:
            merged.append((c[0], _score(c), "classical"))
    return merged


def fuse_increments(
    density_pks, classical_pks, predicted_age: int, axis_info: dict,
    *, method: str = "consensus", t_tol: float = 0.06, dp_min_gap: float = 0.04,
    n_dirs: int = 48, dp_spread_weight: float = 1.5,
) -> dict:
    """Choose the final ``predicted_age`` increments on the axis from peak sources.

    ``density_pks`` / ``classical_pks`` are ``[(t, strength, x, y, ray_idx)]`` lists
    (from :func:`density_peaks` / :func:`classical_increments`); ``n_dirs`` must match
    the ray count they were cast with (only used by ``method="dp"``, for the arc-aware
    scoring in :func:`_merge_clusters`). ``method``:
      * ``"density"``   — top-`age` clusters of density peaks only (model localisation).
      * ``"classical"`` — top-`age` clusters of classical (image-intensity) peaks only.
      * ``"consensus"`` — clusters where density AND classical agree on radius ``t``
        (combined support), top-`age`; falls back to top density clusters if fewer than
        ``age`` agree.
      * ``"dp"``        — merge density+classical clusters (consensus rings scored higher),
        then dynamic-programming select exactly `age` radii maximising total score (plus a
        spread bonus, ``dp_spread_weight`` — see :func:`_dp_select_t`) with a minimum
        spacing ``dp_min_gap`` (spreads increments along the axis instead of bunching all
        picks in whichever single sub-region scores highest).
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
    elif method == "dp":
        merged = _merge_clusters(density_pks, classical_pks, t_tol, n_dirs)
        chosen = _dp_select_t([(t, s) for (t, s, _src) in merged], k, dp_min_gap, dp_spread_weight)
        cand = [(p[2], p[3]) for p in density_pks] + [(p[2], p[3]) for p in classical_pks]
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


def _example_ray_profiles(grid, axis_info: dict, image_h: int, image_w: int, *,
                          n_dirs: int = 48, n_samples: int = 64, min_distance: int = 3,
                          prominence: float = 0.1, inner_margin: float = 0.05,
                          edge_margin: float = 0.08, n_example: int = 3) -> List[dict]:
    """A few representative per-ray profiles (for the walkthrough panel „normalizacja").

    Mirrors one ray of :func:`_all_ray_peaks`: samples the signal jądro→kontur, normalises
    per-ray to [0,1], finds peaks. Returns ``[{t, raw, norm, peak_t, contour_pt}]`` for
    ``n_example`` rays evenly spaced among the ``n_dirs`` directions.
    """
    contour = axis_info.get("contour") if axis_info else None
    centroid = axis_info.get("centroid") if axis_info else None
    if contour is None or centroid is None:
        return []
    g = _to_numpy(grid)
    if g.ndim == 3:
        g = g.mean(axis=2).astype(np.float32)
    cpts = contour.reshape(-1, 2)
    if len(cpts) < 3:
        return []
    idx_sel = np.linspace(0, len(cpts) - 1, min(n_dirs, len(cpts)), dtype=int)
    pick = np.linspace(0, len(idx_sel) - 1, min(n_example, len(idx_sel)), dtype=int)
    out: List[dict] = []
    for pi in pick:
        ci = idx_sel[int(pi)]
        cpt = (int(cpts[ci][0]), int(cpts[ci][1]))
        profile, _ = sample_profile_along_axis(g, centroid, cpt, image_h, image_w, n_samples=n_samples)
        p = profile.astype(np.float32)
        rng = float(p.max() - p.min())
        norm = (p - p.min()) / rng if rng > 1e-6 else np.zeros_like(p)
        peak_t: List[float] = []
        if rng > 1e-6:
            idxs, _ = find_peaks(norm, distance=max(1, int(min_distance)), prominence=float(prominence))
            # Margin checked against the RAW peak index, not the falling-edge-shifted
            # one — see _all_ray_peaks: shifting toward the biological marker must not
            # cause a genuinely valid peak to be dropped for landing near the edge.
            valid_idxs = [int(i) for i in idxs
                          if inner_margin <= i / max(1, n_samples - 1) <= 1.0 - edge_margin]
            edge_idxs = [_shift_peak_to_falling_edge(norm, i) for i in valid_idxs]
            peak_t = [float(i / max(1, n_samples - 1)) for i in edge_idxs]
        out.append({
            "t": np.linspace(0.0, 1.0, n_samples).tolist(),
            "raw": p.tolist(),
            "norm": norm.tolist(),
            "peak_t": peak_t,
            "contour_pt": cpt,
        })
    return out


def dp_walkthrough_data(density_grid, gray_image, axis_info: dict, image_h: int, image_w: int,
                        predicted_age: int, *, n_dirs: int = 48, n_samples: int = 64,
                        t_tol: float = 0.06, dp_min_gap: float = 0.04, dp_spread_weight: float = 1.5,
                        density_min_distance: int = 3, density_prominence: float = 0.1,
                        classical_smooth_sigma: float = 0.0, classical_min_distance: int = 1,
                        classical_prominence: float = 0.02, inner_margin: float = 0.05,
                        edge_margin: float = 0.08, n_example_rays: int = 3) -> dict:
    """All intermediate artifacts of ``fuse_increments(method="dp")`` for ONE otolith.

    Feeds the step-by-step report section: candidates from 48 rays (density + classical),
    per-ray profiles, radius-clusters, merged candidate rings with scores, and the DP
    selection. Reuses the SAME helpers as the real fusion, so what it shows == what runs.
    ``sample_profiles`` (the Krok-2 per-ray example charts) use the CLASSICAL (image-intensity)
    signal, not density — density is the model's own learned map and can be weak or effectively
    empty on a given otolith (esp. an undertrained/early-checkpoint density head), while classical
    intensity is what a human eye — and, downstream, most of the fused score — actually responds
    to (20.07, user report: a chart showing near-zero density looked like "no signal" even though
    classical clearly found peaks at the same visible bands).
    """
    dpk, dpts = density_peaks(density_grid, axis_info, image_h, image_w,
                              n_dirs=n_dirs, n_samples=n_samples, min_distance=density_min_distance,
                              prominence=density_prominence, inner_margin=inner_margin,
                              edge_margin=edge_margin)
    cinc = classical_increments(gray_image, axis_info, n_dirs=n_dirs, n_samples=n_samples,
                                smooth_sigma=classical_smooth_sigma, min_distance=classical_min_distance,
                                prominence=classical_prominence, inner_margin=inner_margin,
                                edge_margin=edge_margin, t_tol=t_tol)
    cpk, cpts = cinc["peaks"], cinc["candidate_pts"]
    merged = _merge_clusters(dpk, cpk, t_tol, n_dirs)
    k = max(0, int(predicted_age))
    chosen = _dp_select_t([(t, s) for (t, s, _src) in merged], k, dp_min_gap, dp_spread_weight)
    return {
        "predicted_age": k,
        "n_dirs": n_dirs,
        "dp_min_gap": dp_min_gap,
        "density_peaks": dpk,
        "classical_peaks": cpk,
        "density_pts": dpts,
        "classical_pts": cpts,
        "density_clusters": _cluster_by_radius(dpk, t_tol),
        "classical_clusters": _cluster_by_radius(cpk, t_tol),
        "merged": merged,
        "chosen_t": chosen,
        "final_axis_pts": _project_to_axis(chosen, axis_info),
        "sample_profiles": _example_ray_profiles(
            gray_image, axis_info, image_h, image_w, n_dirs=n_dirs, n_samples=n_samples,
            min_distance=classical_min_distance, prominence=classical_prominence,
            inner_margin=inner_margin, edge_margin=edge_margin, n_example=n_example_rays),
    }


def dp_interactive_data(density_grid, gray_image, axis_info: dict, image_h: int, image_w: int,
                        predicted_age: int, *, n_dirs: int = 48, n_samples: int = 64,
                        inner_margin: float = 0.05, edge_margin: float = 0.08,
                        density_min_distance: int = 3, classical_min_distance: int = 1,
                        classical_smooth_sigma: float = 0.0) -> dict:
    """RAW per-ray profiles (density + classical) + geometry for the Krok-4 LIVE slider
    widget (prominencja / min-rozstaw DP / tolerancja klastra).

    Unlike :func:`dp_walkthrough_data` (which bakes in ONE prominence/t_tol/min-gap and
    returns already-detected peaks/clusters), this exposes the un-peaked normalised
    profiles for ALL ``n_dirs`` rays so the browser can rerun peak-finding → clustering →
    DP selection at ANY prominence/tolerance/gap the user picks — using the SAME math as
    ``_all_ray_peaks`` / ``_cluster_by_radius`` / ``_dp_select_t`` (reimplemented in JS;
    see ``comparison_report._KROK4_JS``), so the widget's output matches what a real
    server run would produce at those settings. ``density_min_distance`` /
    ``classical_min_distance`` / ``classical_smooth_sigma`` stay fixed (not sliders) —
    only the 3 requested parameters are interactive.
    """
    density_profiles, _dl, contour_pts = _all_ray_profiles(
        density_grid, axis_info, image_h, image_w, n_dirs=n_dirs, n_samples=n_samples)
    gray = _to_numpy(gray_image)
    if gray.ndim == 3:
        gray = gray.mean(axis=2).astype(np.float32)
    classical_profiles, _cl, _ = _all_ray_profiles(
        gray, axis_info, image_h, image_w, n_dirs=n_dirs, n_samples=n_samples,
        smooth_sigma=classical_smooth_sigma)
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    return {
        "predicted_age": max(0, int(predicted_age)),
        "n_samples": n_samples,
        "inner_margin": inner_margin,
        "edge_margin": edge_margin,
        "density_min_distance": density_min_distance,
        "classical_min_distance": classical_min_distance,
        "centroid": [int(cx), int(cy)],
        "far_edge": [int(fx), int(fy)],
        "contour_pts": [[int(x), int(y)] for (x, y) in contour_pts],
        "density_profiles": [(p.tolist() if p is not None else None) for p in density_profiles],
        "classical_profiles": [(p.tolist() if p is not None else None) for p in classical_profiles],
    }
