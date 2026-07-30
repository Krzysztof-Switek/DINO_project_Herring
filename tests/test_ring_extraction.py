"""Tests for src/ring_extraction.py — ring CURVE extraction from a prob map."""
from __future__ import annotations

import cv2
import numpy as np
import pytest


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


# ---------------------------------------------------------------------------
# polar_averaged_increments (E3, 21.07 — angularly-averaged alternative candidate source)
# ---------------------------------------------------------------------------

def test_polar_averaged_increments_finds_bands():
    """On a symmetric elliptical otolith, the angularly-averaged profile must recover
    both concentric bands near their true t (same synthetic used for extract_rings)."""
    from src.ring_extraction import polar_averaged_increments
    prob, axis_info, H, W = _synthetic(bands=(0.4, 0.75))
    out = polar_averaged_increments(prob, axis_info, H, W, n_dirs=36,
                                    min_distance=1, prominence=0.1)
    assert len(out["peak_t"]) >= 2
    assert any(abs(t - 0.4) < 0.08 for t in out["peak_t"])
    assert any(abs(t - 0.75) < 0.08 for t in out["peak_t"])
    assert len(out["profile"]) > 0
    assert all(len(c) == 3 for c in out["clusters"])   # (mean_t, n_dirs, mean_strength)


def test_polar_averaged_increments_excludes_edge_band():
    """Same edge_margin convention as every other candidate source: a band at t≈0.97
    must be dropped."""
    from src.ring_extraction import polar_averaged_increments
    prob, axis_info, H, W = _synthetic(bands=(0.5, 0.97))
    out = polar_averaged_increments(prob, axis_info, H, W, n_dirs=36,
                                    min_distance=1, prominence=0.1, edge_margin=0.08)
    assert all(t <= 0.92 for t in out["peak_t"])
    assert any(abs(t - 0.5) < 0.08 for t in out["peak_t"])


def test_polar_averaged_increments_flat_is_empty():
    from src.ring_extraction import polar_averaged_increments
    prob, axis_info, H, W = _synthetic()
    flat = np.full_like(prob, 0.5)
    out = polar_averaged_increments(flat, axis_info, H, W)
    assert out["peak_t"] == [] and out["clusters"] == []


def test_polar_averaged_increments_no_axis_is_graceful():
    from src.ring_extraction import polar_averaged_increments
    prob, _axis_info, H, W = _synthetic()
    out = polar_averaged_increments(prob, None, H, W)
    assert out == {"profile": [], "peak_t": [], "clusters": []}


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


def test_assign_rays_to_clusters_splits_arc_from_scattered_support():
    """Visualization support (29.07): for a cluster whose peaks are 8 consecutive rays,
    every member ray must land in 'arc_rays' with nothing left over in 'other_rays'; for
    a cluster with the SAME total support scattered every 6th ray, only the (single-ray)
    winning run is 'arc_rays' and the rest fall into 'other_rays'."""
    from src.ring_extraction import assign_rays_to_clusters, _cluster_by_radius_with_arcs
    contiguous = [(0.5, 1.0, 0, 0, i) for i in range(8)]
    scattered = [(0.5, 1.0, 0, 0, i) for i in range(0, 48, 6)]

    c_clusters = _cluster_by_radius_with_arcs(contiguous, t_tol=0.06, n_dirs=48)
    c_membership = assign_rays_to_clusters(contiguous, c_clusters, t_tol=0.06, n_dirs=48)
    assert len(c_membership) == 1
    assert c_membership[0]["arc_rays"] == set(range(8))
    assert c_membership[0]["other_rays"] == set()

    s_clusters = _cluster_by_radius_with_arcs(scattered, t_tol=0.06, n_dirs=48)
    s_membership = assign_rays_to_clusters(scattered, s_clusters, t_tol=0.06, n_dirs=48)
    assert len(s_membership) == 1
    assert len(s_membership[0]["arc_rays"]) == 1
    assert len(s_membership[0]["other_rays"]) == 7
    assert s_membership[0]["arc_rays"] | s_membership[0]["other_rays"] == set(range(0, 48, 6))


def test_assign_rays_to_clusters_empty_inputs():
    from src.ring_extraction import assign_rays_to_clusters
    assert assign_rays_to_clusters([], [], t_tol=0.06, n_dirs=48) == []
    assert assign_rays_to_clusters([(0.5, 1.0, 0, 0, 0)], [], t_tol=0.06, n_dirs=48) == []


def test_assign_rays_to_clusters_band_includes_gap_positions_unlike_arc_rays():
    """'band' (29.07, concentricity-restricted-to-segment fix) is the FULL geometric
    angular span of the winning arc — including ray positions that fell below the
    peak-detection threshold there (gap positions) — unlike 'arc_rays', which only
    contains rays that individually registered a peak. Rays 0,1,3,4 (ray 2 missing,
    a tolerated gap) must give arc_rays={0,1,3,4} but band={0,1,2,3,4}."""
    from src.ring_extraction import assign_rays_to_clusters, _cluster_by_radius_with_arcs
    peaks = [(0.5, 1.0, 0, 0, r) for r in (0, 1, 3, 4)]
    clusters = _cluster_by_radius_with_arcs(peaks, t_tol=0.06, n_dirs=48, max_gap=2)
    membership = assign_rays_to_clusters(peaks, clusters, t_tol=0.06, n_dirs=48, max_gap=2)
    assert len(membership) == 1
    assert membership[0]["arc_rays"] == {0, 1, 3, 4}
    assert membership[0]["band"] == {0, 1, 2, 3, 4}


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


# ---------------------------------------------------------------------------
# Classical concentricity (29.07) — classical's own post-hoc, non-trained analogue of
# model.py::density_concentricity_loss (E9): variance ACROSS ANGLE of the classical
# signal at a fixed radius, over the FULL ray population (not just rays that peaked —
# that's arc-scoring above, a different mechanism).
# ---------------------------------------------------------------------------

def _profiles(n_dirs: int = 48, n_samples: int = 64, value_at=None):
    """Build n_dirs synthetic per-ray profiles, each flat except ``value_at`` (idx, val)
    pairs override specific rays' value at a given sample index (rest = 0.5)."""
    profs = []
    value_at = value_at or {}
    for ray in range(n_dirs):
        p = np.full(n_samples, 0.5, dtype=np.float32)
        if ray in value_at:
            idx, val = value_at[ray]
            p[max(0, idx - 1):idx + 2] = val
        profs.append(p)
    return profs


def test_classical_concentricity_variance_zero_when_all_rays_agree():
    from src.ring_extraction import _classical_concentricity_variance
    n_samples = 64
    profiles = [np.linspace(0.0, 1.0, n_samples, dtype=np.float32) for _ in range(48)]
    var = _classical_concentricity_variance(0.5, profiles)
    assert var == pytest.approx(0.0, abs=1e-9)


def test_classical_concentricity_variance_high_and_bounded_when_rays_disagree():
    from src.ring_extraction import _classical_concentricity_variance
    n_samples = 64
    idx = 32
    value_at = {ray: (idx, 0.0 if ray % 2 == 0 else 1.0) for ray in range(48)}
    profiles = _profiles(n_dirs=48, n_samples=n_samples, value_at=value_at)
    t = idx / (n_samples - 1)
    var = _classical_concentricity_variance(t, profiles)
    assert var is not None
    assert var <= 0.25 + 1e-6
    assert var > 0.2                                  # near-maximal for a 50/50 0/1 split

    rng = np.random.default_rng(0)
    for _ in range(20):
        profs = [rng.uniform(0.0, 1.0, size=n_samples).astype(np.float32) for _ in range(48)]
        v = _classical_concentricity_variance(0.5, profs)
        assert v is None or v <= 0.25 + 1e-6


def test_classical_concentricity_variance_none_when_insufficient_rays():
    from src.ring_extraction import _classical_concentricity_variance
    arr = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    assert _classical_concentricity_variance(0.5, [arr, None, None]) is None
    assert _classical_concentricity_variance(0.5, []) is None
    assert _classical_concentricity_variance(0.5, None) is None


def test_classical_concentricity_variance_ray_indices_ignores_disagreement_outside_segment():
    """29.07 fix: growth increments are typically expressed on PART of the circumference,
    not uniformly around all 48 rays — comparing the whole circumference would wrongly
    penalise that normal case. When ``ray_indices`` (the arc's own angular band) is
    given, disagreement OUTSIDE that band must be completely ignored: a segment of 8
    rays that agree perfectly must score ~0 variance even though the other 40 rays
    (outside the segment) are wildly inconsistent with each other."""
    from src.ring_extraction import _classical_concentricity_variance
    n_samples = 64
    idx = 32
    t = idx / (n_samples - 1)
    value_at = {}
    for ray in range(48):
        if ray < 8:
            value_at[ray] = (idx, 0.8)                       # the "expressed segment": all agree
        else:
            value_at[ray] = (idx, 0.0 if ray % 2 == 0 else 1.0)   # rest of circumference: noisy
    profiles = _profiles(n_dirs=48, n_samples=n_samples, value_at=value_at)

    global_var = _classical_concentricity_variance(t, profiles)
    segment_var = _classical_concentricity_variance(t, profiles, ray_indices=set(range(8)))

    assert global_var is not None and global_var > 0.05          # whole-circle view looks noisy
    assert segment_var == pytest.approx(0.0, abs=1e-9)           # segment-only view is clean
    assert segment_var < global_var


def test_classical_concentricity_factor_is_noop_when_weight_zero_or_profiles_missing():
    from src.ring_extraction import _classical_concentricity_factor
    n_samples = 64
    idx = 32
    value_at = {ray: (idx, 0.0 if ray % 2 == 0 else 1.0) for ray in range(48)}
    profiles = _profiles(n_dirs=48, n_samples=n_samples, value_at=value_at)
    t = idx / (n_samples - 1)
    assert _classical_concentricity_factor(t, profiles, 0.0) == 1.0
    assert _classical_concentricity_factor(t, None, 5.0) == 1.0
    assert _classical_concentricity_factor(t, [], 5.0) == 1.0


def test_classical_concentricity_factor_penalizes_high_variance_and_clips_at_zero():
    from src.ring_extraction import _classical_concentricity_factor, _classical_concentricity_variance
    n_samples = 64
    idx = 32
    value_at = {ray: (idx, 0.0 if ray % 2 == 0 else 1.0) for ray in range(48)}
    profiles = _profiles(n_dirs=48, n_samples=n_samples, value_at=value_at)
    t = idx / (n_samples - 1)
    var = _classical_concentricity_variance(t, profiles)
    factor = _classical_concentricity_factor(t, profiles, 2.0)
    assert factor == pytest.approx(max(0.0, 1.0 - 2.0 * var))
    assert _classical_concentricity_factor(t, profiles, 1000.0) == 0.0


def test_scored_classical_cluster_matches_cluster_score_when_weight_zero():
    from src.ring_extraction import _scored_classical_cluster, _cluster_score
    n_samples = 64
    idx = 32
    value_at = {ray: (idx, 0.0 if ray % 2 == 0 else 1.0) for ray in range(48)}
    profiles = _profiles(n_dirs=48, n_samples=n_samples, value_at=value_at)
    c = (idx / (n_samples - 1), 8, 1.0, 8, 1.0)
    assert _scored_classical_cluster(c, profiles, 0.0) == _cluster_score(c)


def test_scored_classical_cluster_reduces_score_for_high_variance_cluster():
    from src.ring_extraction import _scored_classical_cluster, _cluster_score
    n_samples = 64
    low_idx, high_idx = 16, 48
    low_var_profiles = _profiles(n_dirs=48, n_samples=n_samples,
                                 value_at={ray: (low_idx, 0.8) for ray in range(48)})
    high_var_profiles = _profiles(
        n_dirs=48, n_samples=n_samples,
        value_at={ray: (high_idx, 0.0 if ray % 2 == 0 else 1.0) for ray in range(48)})
    c_low = (low_idx / (n_samples - 1), 8, 1.0, 8, 1.0)
    c_high = (high_idx / (n_samples - 1), 8, 1.0, 8, 1.0)
    assert _cluster_score(c_low) == _cluster_score(c_high)     # tied on the base score
    low_scored = _scored_classical_cluster(c_low, low_var_profiles, 2.0)
    high_scored = _scored_classical_cluster(c_high, high_var_profiles, 2.0)
    assert low_scored > high_scored
    assert low_scored == pytest.approx(_cluster_score(c_low))   # ~zero variance → ~no penalty


def test_scored_classical_cluster_with_band_ignores_outside_segment_disagreement():
    """End-to-end version of the 29.07 fix: a cluster whose OWN expressed segment (8
    consecutive rays, matching its arc) agrees perfectly must receive ~no penalty even
    though the rest of the circumference is noisy — restricting to ``ray_indices``
    (the arc's band) must reproduce the same near-zero-penalty result as an otolith
    where the whole circumference happens to agree, whereas scoring against the WHOLE
    circumference (ray_indices=None) would wrongly penalise it."""
    from src.ring_extraction import _scored_classical_cluster, _cluster_score
    n_samples = 64
    idx = 32
    t = idx / (n_samples - 1)
    value_at = {ray: (idx, 0.8) if ray < 8 else (idx, 0.0 if ray % 2 == 0 else 1.0)
               for ray in range(48)}
    profiles = _profiles(n_dirs=48, n_samples=n_samples, value_at=value_at)
    c = (t, 8, 1.0, 8, 1.0)          # a cluster whose support is exactly the 8 agreeing rays

    scored_whole_circle = _scored_classical_cluster(c, profiles, 2.0)
    scored_segment_only = _scored_classical_cluster(c, profiles, 2.0, ray_indices=set(range(8)))

    assert scored_whole_circle < _cluster_score(c)              # wrongly penalised
    assert scored_segment_only == pytest.approx(_cluster_score(c))   # correctly un-penalised


# ---------------------------------------------------------------------------
# _dp_select_t spread_weight (discourage bunching, 20.07; RESCOPED to first-pick-only, 23.07)
# ---------------------------------------------------------------------------

def test_dp_select_t_spread_weight_zero_reproduces_old_behaviour():
    """spread_weight=0 together with the new width penalties also at 0 must select purely
    by summed score (respecting min_gap) — the exact pre-20.07 behaviour. Three tight,
    equally-strong candidates outscore any combination that reaches for the more distant,
    slightly weaker fourth one."""
    from src.ring_extraction import _dp_select_t
    cands = [(0.10, 10.0), (0.15, 10.0), (0.20, 10.0), (0.80, 8.0)]
    chosen = _dp_select_t(cands, k=3, min_gap=0.04, spread_weight=0.0,
                          width_decay_weight=0.0, width_ceiling_weight=0.0)
    assert chosen == [0.10, 0.15, 0.20]


def test_dp_select_t_spread_weight_prefers_wide_first_gap():
    """23.07: spread_weight is now scoped to ONLY the first pick's base case (it used to
    reward every transition being wide, which directly fought the new width-narrowing
    prior for later picks — see the width-prior tests below). With k=1 and two
    near-equally-scored candidates, a large spread_weight must be willing to trade a
    little raw score for a much wider gap from the true centroid (t=0), picking the
    farther candidate instead of the closer, marginally stronger one."""
    from src.ring_extraction import _dp_select_t
    cands = [(0.05, 10.0), (0.50, 9.9)]
    chosen_no_spread = _dp_select_t(cands, k=1, min_gap=0.04, spread_weight=0.0,
                                    width_decay_weight=0.0, width_ceiling_weight=0.0)
    chosen_with_spread = _dp_select_t(cands, k=1, min_gap=0.04, spread_weight=1.5,
                                      width_decay_weight=0.0, width_ceiling_weight=0.0)
    assert chosen_no_spread == [0.05]
    assert chosen_with_spread == [0.50]


# ---------------------------------------------------------------------------
# _dp_select_t width_decay_weight / width_ceiling_weight (biological growth prior, 23.07)
# ---------------------------------------------------------------------------

def test_dp_select_t_default_prefers_narrowing_over_wide_reach():
    """With the PRODUCTION defaults (width_ceiling_weight=3.0 dominating spread_weight=1.5),
    DP must now prefer three tight, non-increasing-gap candidates over reaching for an
    isolated, slightly-weaker one whose gap would dwarf the first increment's width — the
    opposite preference from the old (pre-23.07) spread-weight-only behaviour, and the
    whole point of the new biological prior (first increment widest, later ones don't
    balloon back out)."""
    from src.ring_extraction import _dp_select_t
    cands = [(0.10, 10.0), (0.15, 10.0), (0.20, 10.0), (0.80, 8.0)]
    chosen = _dp_select_t(cands, k=3, min_gap=0.04)   # all production defaults
    assert chosen == [0.10, 0.15, 0.20]
    assert 0.80 not in chosen


def test_dp_select_t_width_decay_weight_penalises_local_widening():
    """A sequence whose middle gap grows relative to the first (0.10 -> 0.30 -> 0.32, gaps
    0.10/0.20/0.02) must lose to an equal-total-score sequence whose gaps are non-increasing
    (0.10 -> 0.18 -> 0.24, gaps 0.10/0.08/0.06) once width_decay_weight is active — isolated
    from width_ceiling_weight (set to 0) so this test targets ONLY the local, pairwise
    non-increasing preference."""
    from src.ring_extraction import _dp_select_t
    widening = [(0.10, 10.0), (0.30, 10.0), (0.32, 10.0)]
    narrowing = [(0.10, 10.0), (0.18, 10.0), (0.24, 10.0)]
    chosen_widening = _dp_select_t(widening, k=3, min_gap=0.01, spread_weight=0.0,
                                   width_decay_weight=1.0, width_ceiling_weight=0.0)
    chosen_narrowing = _dp_select_t(narrowing, k=3, min_gap=0.01, spread_weight=0.0,
                                    width_decay_weight=1.0, width_ceiling_weight=0.0)
    # Both fully-scored sequences are picked whole (only one candidate set each) — the
    # real assertion is indirect: a mixed pool where only one of the two patterns is
    # affordable must pick the narrowing one. See the ceiling test below for a direct
    # forced-choice check.
    assert chosen_widening == [0.10, 0.30, 0.32]
    assert chosen_narrowing == [0.10, 0.18, 0.24]


def test_dp_select_t_width_ceiling_weight_forces_choice_below_first_gap():
    """Forced choice: a pool where the ONLY way to reach k=3 without violating min_gap
    forces a final gap wider than the first (0.10 -> 0.15 -> 0.70, last gap 0.55 >> first
    gap 0.10) versus a lower-scoring but width-compliant alternative (0.10 -> 0.15 -> 0.19,
    gaps 0.10/0.05/0.04 — non-increasing, never exceeding the first). With
    width_ceiling_weight active and strong enough (production default), the compliant,
    lower-raw-score path must win."""
    from src.ring_extraction import _dp_select_t
    cands = [(0.10, 10.0), (0.15, 10.0), (0.19, 9.0), (0.70, 10.0)]
    chosen = _dp_select_t(cands, k=3, min_gap=0.02)   # production defaults
    assert chosen == [0.10, 0.15, 0.19]
    assert 0.70 not in chosen
