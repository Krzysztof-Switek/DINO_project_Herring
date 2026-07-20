"""Tests for src/ring_extraction.py — ring CURVE extraction from a prob map."""
from __future__ import annotations

import cv2
import numpy as np


def _synthetic(a: int = 50, b: int = 80, bands=(0.4, 0.75)):
    """Elliptical otolith + a patch prob-map with concentric bands at ``bands`` (in t)."""
    H, W = 220, 180
    cx, cy = W // 2, H // 2
    fill = np.zeros((H, W), np.uint8)
    cv2.ellipse(fill, (cx, cy), (a, b), 0, 0, 360, 255, -1)
    cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(cnts, key=cv2.contourArea)
    axis_info = {"centroid": (cx, cy), "contour": contour, "mask": fill,
                 "far_edge": (cx, cy - b), "length_px": float(b)}

    Hp, Wp = 55, 45
    cgx, cgy = cx / W * Wp, cy / H * Hp
    ga, gb = a / W * Wp, b / H * Hp
    yy, xx = np.mgrid[0:Hp, 0:Wp].astype(np.float32)
    rho = np.sqrt(((xx - cgx) / ga) ** 2 + ((yy - cgy) / gb) ** 2)   # 0 centre, 1 edge
    prob = np.zeros((Hp, Wp), np.float32)
    for bnd in bands:
        prob += np.exp(-((rho - bnd) ** 2) / (2 * 0.04 ** 2))
    return np.clip(prob, 0, 1).astype(np.float32), axis_info, H, W


def test_extract_ring_curves_finds_bands():
    from src.ring_extraction import extract_ring_curves
    prob, axis_info, H, W = _synthetic(bands=(0.4, 0.75))
    curves = extract_ring_curves(prob, axis_info, H, W, n_dirs=36, prominence=0.15)
    assert len(curves) >= 2                         # both bands recovered as rings
    for c in curves:
        assert c.ndim == 2 and c.shape[1] == 2
        assert len(c) >= 6                          # spans many directions
        assert (c[:, 0] >= 0).all() and (c[:, 0] < W).all()
        assert (c[:, 1] >= 0).all() and (c[:, 1] < H).all()


def test_extract_ring_curves_excludes_edge():
    """A band at the very edge (t≈0.97) must be dropped by edge_margin."""
    from src.ring_extraction import extract_ring_curves
    prob, axis_info, H, W = _synthetic(bands=(0.5, 0.97))
    curves = extract_ring_curves(prob, axis_info, H, W, n_dirs=36,
                                 prominence=0.15, edge_margin=0.08)
    assert len(curves) == 1                         # only the t=0.5 band survives


def test_extract_ring_curves_flat_is_empty():
    from src.ring_extraction import extract_ring_curves
    prob, axis_info, H, W = _synthetic()
    flat = np.full_like(prob, 0.5)
    assert extract_ring_curves(flat, axis_info, H, W) == []


def test_draw_ring_curves_modifies_panel():
    from src.ring_extraction import extract_ring_curves, draw_ring_curves
    prob, axis_info, H, W = _synthetic(bands=(0.4, 0.75))
    curves = extract_ring_curves(prob, axis_info, H, W, n_dirs=36, prominence=0.15)
    panel = np.full((H, W, 3), 200, np.uint8)
    before = panel.copy()
    draw_ring_curves(panel, curves, thickness=2)
    assert not np.array_equal(before, panel)


# ---------------------------------------------------------------------------
# _shift_peak_to_falling_edge (biological annulus-boundary convention, 20.07)
# ---------------------------------------------------------------------------

def test_shift_peak_to_falling_edge_finds_half_max_crossing():
    from src.ring_extraction import _shift_peak_to_falling_edge
    p = np.array([0.0, 0.2, 0.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    # peak at idx=3 (value 1.0); descends monotonically to idx=8 (value 0.1).
    # half = (1.0+0.1)/2 = 0.55 → crossing between idx 5 (0.6) and idx 6 (0.4).
    assert _shift_peak_to_falling_edge(p, 3) == 6


def test_shift_peak_to_falling_edge_moves_toward_edge_not_before_it():
    """The shifted index must lie AFTER the peak (moving jądro→brzeg), never before —
    the annulus boundary is where the signal falls FROM the peak, not on its way up."""
    from src.ring_extraction import _shift_peak_to_falling_edge
    p = np.array([0.0, 0.3, 0.6, 1.0, 0.7, 0.3, 0.05])
    idx = 3
    edge = _shift_peak_to_falling_edge(p, idx)
    assert edge >= idx


def test_shift_peak_to_falling_edge_noop_when_peak_at_tail():
    """Peak sitting at the profile's last index has nothing to descend into — no shift."""
    from src.ring_extraction import _shift_peak_to_falling_edge
    p = np.array([0.0, 0.2, 0.5, 1.0])
    assert _shift_peak_to_falling_edge(p, 3) == 3


def test_shift_peak_to_falling_edge_noop_when_signal_rises_immediately_after():
    """Signal keeps RISING right after ``idx`` (idx isn't a real local peak from this
    point's perspective) → no descent found → index returned unchanged."""
    from src.ring_extraction import _shift_peak_to_falling_edge
    p = np.array([0.0, 0.5, 1.0, 1.5])
    assert _shift_peak_to_falling_edge(p, 2) == 2


def _synthetic_asym(radial_fn, a: int = 50, b: int = 80):
    """Elliptical otolith + patch-grid whose radial profile follows ``radial_fn(rho)``
    (``rho`` = elliptical-normalised radius, 0=centroid, 1=contour — matches the ``t``
    used by ray sampling exactly, since it is evaluated along straight rays through
    the centroid)."""
    H, W = 220, 180
    cx, cy = W // 2, H // 2
    fill = np.zeros((H, W), np.uint8)
    cv2.ellipse(fill, (cx, cy), (a, b), 0, 0, 360, 255, -1)
    cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(cnts, key=cv2.contourArea)
    axis_info = {"centroid": (cx, cy), "contour": contour, "mask": fill,
                 "far_edge": (cx, cy - b), "length_px": float(b)}
    Hp, Wp = 55, 45
    cgx, cgy = cx / W * Wp, cy / H * Hp
    ga, gb = a / W * Wp, b / H * Hp
    yy, xx = np.mgrid[0:Hp, 0:Wp].astype(np.float32)
    rho = np.sqrt(((xx - cgx) / ga) ** 2 + ((yy - cgy) / gb) ** 2)
    prob = radial_fn(rho).astype(np.float32)
    return np.clip(prob, 0, 1).astype(np.float32), axis_info, H, W


def test_all_ray_peaks_keeps_valid_peak_shifted_past_edge_margin():
    """Regression (20.07): a peak whose RAW position is safely inside the valid
    window must not be dropped just because falling-edge-shifting it (toward the
    biological light->dark boundary) pushes the reported t past the edge cutoff.

    Sharp rise to a peak at t=0.80, then a long, gentle linear decay all the way to
    the contour (t=1.0) — the falling-edge shift lands around t~0.90, well past
    ``1 - edge_margin`` (0.85 here), but the RAW peak at t=0.80 is comfortably valid.
    """
    from src.ring_extraction import density_peaks

    def radial_fn(rho):
        out = np.zeros_like(rho)
        rise = (rho >= 0.75) & (rho < 0.80)
        out[rise] = (rho[rise] - 0.75) / 0.05
        fall = rho >= 0.80
        out[fall] = np.clip(1.0 - (rho[fall] - 0.80) / 0.6, 0.0, 1.0)
        return out

    prob, axis_info, H, W = _synthetic_asym(radial_fn)
    peaks, _ = density_peaks(prob, axis_info, H, W, n_dirs=24, prominence=0.15,
                             edge_margin=0.15)
    assert len(peaks) > 0
    # reported (shifted) position must sit clearly past the raw peak, near the cutoff
    mean_t = sum(p[0] for p in peaks) / len(peaks)
    assert mean_t > 0.85


def test_all_ray_peaks_still_drops_genuinely_out_of_margin_peak(monkeypatch):
    """Regression guard: a peak whose RAW position is already outside the valid
    window must still be rejected — the fix only changes WHAT is checked against the
    margin (raw vs. shifted), not that the margin is enforced. Forces the shift to
    always land safely mid-profile (t=0.5) so any peak surviving the filter can only
    have done so via its raw (pre-shift) position — isolating the exact mechanism."""
    import src.ring_extraction as re
    from src.ring_extraction import density_peaks
    monkeypatch.setattr(re, "_shift_peak_to_falling_edge", lambda p, idx: len(p) // 2)
    prob, axis_info, H, W = _synthetic(bands=(0.99,))
    peaks, _ = density_peaks(prob, axis_info, H, W, n_dirs=24, prominence=0.15,
                             edge_margin=0.08)
    assert peaks == []


def test_all_ray_peaks_reports_edge_not_peak_position():
    """End-to-end: density_peaks() candidate positions must sit AT or AFTER their ray's
    brightness peak (falling-edge convention), not exactly at the peak itself when a
    real descent follows."""
    from src.ring_extraction import density_peaks
    prob, axis_info, H, W = _synthetic(bands=(0.5,))
    peaks, _ = density_peaks(prob, axis_info, H, W, n_dirs=24, prominence=0.15)
    assert len(peaks) > 0
    # the synthetic band is a symmetric Gaussian bump centred at t=0.5 — the falling
    # edge must land strictly past the centre (closer to the otolith margin).
    mean_t = sum(p[0] for p in peaks) / len(peaks)
    assert mean_t > 0.5


# ---------------------------------------------------------------------------
# _cluster_by_radius_with_arcs / _best_arc (arc-aware ring scoring, 20.07)
# ---------------------------------------------------------------------------

def test_cluster_by_radius_with_arcs_prefers_contiguous_over_scattered():
    """A cluster whose peaks come from 8 ANGULARLY-CONSECUTIVE rays (a real, localised
    band, like the top/left-only bands a user reported seeing on a real otolith) must
    report a longer arc than one with the SAME total support (8 rays) scattered evenly
    around the full 48-ray circumference — support alone can't tell them apart
    (that's exactly the gap this fixes)."""
    from src.ring_extraction import _cluster_by_radius_with_arcs
    contiguous = [(0.5, 1.0, 0, 0, i) for i in range(8)]              # rays 0..7
    scattered = [(0.5, 1.0, 0, 0, i) for i in range(0, 48, 6)]        # rays 0,6,...,42
    assert len(scattered) == 8
    c_clusters = _cluster_by_radius_with_arcs(contiguous, t_tol=0.06, n_dirs=48)
    s_clusters = _cluster_by_radius_with_arcs(scattered, t_tol=0.06, n_dirs=48)
    assert len(c_clusters) == 1 and len(s_clusters) == 1
    assert c_clusters[0][1] == s_clusters[0][1] == 8      # same support either way
    assert c_clusters[0][3] == 8                           # fully contiguous run
    assert s_clusters[0][3] == 1                           # no two rays within max_gap
    assert c_clusters[0][3] > s_clusters[0][3]


def test_cluster_by_radius_with_arcs_wraps_around_zero():
    """A run spanning the 0 / n_dirs-1 boundary (rays 46,47,0,1) is ONE contiguous arc,
    not two separate fragments — the ray circle wraps all the way around."""
    from src.ring_extraction import _cluster_by_radius_with_arcs
    peaks = [(0.5, 1.0, 0, 0, r) for r in (46, 47, 0, 1)]
    clusters = _cluster_by_radius_with_arcs(peaks, t_tol=0.06, n_dirs=48)
    assert len(clusters) == 1
    assert clusters[0][3] == 4


def test_cluster_by_radius_with_arcs_tolerates_small_gaps():
    """A single missing ray inside an otherwise solid arc doesn't break it into two
    runs — the reported span still covers the full width, gap included."""
    from src.ring_extraction import _cluster_by_radius_with_arcs
    peaks = [(0.5, 1.0, 0, 0, r) for r in (0, 1, 3, 4)]     # ray 2 missing (gap=1)
    clusters = _cluster_by_radius_with_arcs(peaks, t_tol=0.06, n_dirs=48, max_gap=2)
    assert len(clusters) == 1
    assert clusters[0][3] == 5                              # span 0..4 inclusive


def test_merge_clusters_scores_contiguous_ring_higher():
    """End-to-end through _merge_clusters: a ring seen along a compact arc must
    outscore one with identical support scattered across the circumference — this is
    what actually drives the DP selection in fuse_increments(method="dp")."""
    from src.ring_extraction import _merge_clusters
    contiguous = [(0.5, 1.0, 0, 0, i) for i in range(8)]
    scattered = [(0.5, 1.0, 0, 0, i) for i in range(0, 48, 6)]
    c_merged = _merge_clusters(contiguous, [], t_tol=0.06, n_dirs=48)
    s_merged = _merge_clusters(scattered, [], t_tol=0.06, n_dirs=48)
    assert len(c_merged) == 1 and len(s_merged) == 1
    assert c_merged[0][1] > s_merged[0][1]                  # contiguous scores higher
