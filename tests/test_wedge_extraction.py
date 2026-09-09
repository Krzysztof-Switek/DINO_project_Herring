"""Unit tests for src/wedge_extraction.py — synthetic images, no Z: drive access."""
from __future__ import annotations

import numpy as np
import pytest

from src.otolith_axis import MASK_FILL_RGB
from src.wedge_extraction import (
    WedgeBandGeometry,
    WedgeGeometry,
    extract_polar_wedge,
    extract_polar_wedge_band,
    extract_polar_wedge_band_validity,
    extract_polar_wedge_validity,
    get_or_compute_wedge,
    get_or_compute_wedge_band,
    load_wedge,
    point_to_wedge_band_xy,
    point_to_wedge_xy,
    save_wedge,
    wedge_band_geometries_from_axis_info,
    wedge_band_polar_coords,
    wedge_band_xy_to_point,
    wedge_geometry_from_axis_info,
    wedge_polar_coords,
    wedge_xy_to_point,
)


def _synthetic_circle_rgb_and_mask(h: int = 400, w: int = 400, cx: float = 200, cy: float = 200,
                                    radius: float = 150) -> tuple[np.ndarray, np.ndarray]:
    """A filled circle — unlike a rectangle, gives a smooth, well-defined R(theta) at every
    angle, matching a real (roughly star-shaped-from-nucleus) otolith mask better than a box."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2).astype(np.uint8) * 255
    return rgb, mask


def _axis_info(cx: float, cy: float, fx: float, fy: float, mask: np.ndarray) -> dict:
    return {"centroid": (cx, cy), "far_edge": (fx, fy), "mask": mask,
            "length_px": float(np.hypot(fx - cx, fy - cy))}


# ---------------------------------------------------------------------------
# wedge_geometry_from_axis_info
# ---------------------------------------------------------------------------

def test_wedge_geometry_axis_angle_matches_far_edge_direction():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)  # far_edge straight along +x
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg=90.0, canvas_w=13, canvas_h=95)
    assert geom.axis_angle_rad == pytest.approx(0.0, abs=1e-6)
    assert geom.delta_theta_rad == pytest.approx(np.radians(90.0))
    assert geom.canvas_w == 13 and geom.canvas_h == 95
    assert geom.R_theta.shape == (720,)


def test_wedge_geometry_R_theta_matches_true_circle_radius():
    rgb, mask = _synthetic_circle_rgb_and_mask(cx=200, cy=200, radius=150)
    axis_info = _axis_info(200, 200, 350, 200, mask)
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg=60.0, canvas_w=8, canvas_h=40)
    # A circle has (near-)constant R(theta) at every angle — real segmentation never is this
    # clean, but this pins the ray-casting itself is correct before any wedge-specific logic runs.
    assert geom.R_theta.mean() == pytest.approx(150.0, rel=0.05)
    assert geom.R_theta.std() < 5.0


# ---------------------------------------------------------------------------
# Round-trip: canvas -> real point -> canvas recovers the same position
# ---------------------------------------------------------------------------

def test_wedge_round_trip_grid_of_points():
    """row=0 (t=0, the nucleus itself) is deliberately EXCLUDED: at r=0 every column maps to the
    exact same point (the centroid) -- angle is mathematically undefined there, so the inverse
    cannot (and should not be expected to) recover which column it came from. Same degeneracy
    Daugman's own polar model has at the pupil centre -- not a bug in this module."""
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg=80.0, canvas_w=13, canvas_h=95)

    max_err = 0.0
    for col in np.linspace(0, geom.canvas_w - 1, 7):
        for row in np.linspace(1, geom.canvas_h - 1, 7):
            x, y = wedge_xy_to_point(col, row, geom)
            back = point_to_wedge_xy(x, y, geom)
            assert back is not None
            c2, r2, _t = back
            max_err = max(max_err, abs(col - c2), abs(row - r2))
    assert max_err < 1e-6


def test_point_to_wedge_xy_none_outside_delta_theta():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)  # axis along +x (angle 0)
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg=20.0, canvas_w=5, canvas_h=20)
    # a point straight "up" from the centroid (angle ~90 deg) is far outside a 20 deg wedge
    result = point_to_wedge_xy(200.0, 100.0, geom)
    assert result is None


# ---------------------------------------------------------------------------
# extract_polar_wedge — shape/dtype + background-fill regression
# ---------------------------------------------------------------------------

def test_extract_polar_wedge_output_shape_and_dtype():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg=90.0, canvas_w=13, canvas_h=95)
    wedge = extract_polar_wedge(rgb, mask, geom)
    assert wedge.shape == (95, 13, 3)
    assert wedge.dtype == np.uint8


def test_extract_polar_wedge_fill_is_never_black():
    """Nucleus placed OFF-CENTRE, close to the mask edge on one side, so part of the wedge's
    angular span at large t inevitably falls outside the mask on that side — those cells must
    be MASK_FILL_RGB, never black (same register-token-artifact concern as the strip)."""
    rgb, mask = _synthetic_circle_rgb_and_mask(cx=200, cy=200, radius=150)
    axis_info = _axis_info(320, 200, 340, 200, mask)  # centroid near the circle's right edge
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg=170.0, canvas_w=21, canvas_h=40)
    wedge = extract_polar_wedge(rgb, mask, geom)
    # far column, far row (large t, edge of the wide wedge) is the most likely to fall outside
    corner = tuple(int(v) for v in wedge[-1, 0])
    assert corner != (0, 0, 0)


# ---------------------------------------------------------------------------
# extract_polar_wedge_validity
# ---------------------------------------------------------------------------

def test_extract_polar_wedge_validity_range_and_shape():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg=90.0, canvas_w=14, canvas_h=98)
    valid = extract_polar_wedge_validity(mask, geom, patch_size=14)
    assert valid.shape == (7, 1)
    assert np.all(valid >= 0.0) and np.all(valid <= 1.0)
    # near the nucleus (row 0), the wedge is by construction always inside the real boundary
    assert valid[0].min() > 0.9


# ---------------------------------------------------------------------------
# wedge_polar_coords
# ---------------------------------------------------------------------------

def test_wedge_polar_coords_shape_and_monotonic_radius():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg=90.0, canvas_w=14, canvas_h=98)
    t_grid, theta_grid = wedge_polar_coords(geom, patch_size=14)
    assert t_grid.shape == (7, 1) and theta_grid.shape == (7, 1)
    assert np.all(np.diff(t_grid[:, 0]) > 0)          # radius increases row by row
    assert t_grid.min() > 0.0 and t_grid.max() < 1.0  # patch CENTRES, never exactly 0 or 1


def test_wedge_radial_resolution_is_non_uniform_denser_near_edge():
    """09.09 follow-up: rows must be denser near t=1 (otolith edge) than near t=0 (nucleus),
    matching the measured ~7x tightening of real ring spacing toward the edge
    (scripts/diagnostics/analyze_wedge_radial_resolution_profile.py). A regression to a linear
    row->t mapping would make these two spans equal."""
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg=90.0, canvas_w=14, canvas_h=616)
    t_grid, _ = wedge_polar_coords(geom, patch_size=14)
    t_col = t_grid[:, 0]
    # t-span covered by the first 5 patch rows (near the nucleus) vs the last 5 (near the edge).
    span_near_nucleus = t_col[4] - t_col[0]
    span_near_edge = t_col[-1] - t_col[-5]
    assert span_near_nucleus > 3 * span_near_edge


def test_wedge_row_frac_to_t_endpoints_and_monotonic():
    from src.wedge_extraction import _RADIAL_WARP_T, _row_frac_to_t, _t_to_row_frac
    row_fracs = np.linspace(0, 1, 50)
    t_vals = _row_frac_to_t(row_fracs)
    assert t_vals[0] == pytest.approx(0.0)
    assert t_vals[-1] == pytest.approx(1.0)
    assert np.all(np.diff(t_vals) >= 0)
    # round-trips through both directions
    back = _t_to_row_frac(t_vals)
    np.testing.assert_allclose(back, row_fracs, atol=1e-9)
    assert _RADIAL_WARP_T[0] == 0.0 and _RADIAL_WARP_T[-1] == 1.0


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def test_save_load_wedge_round_trip(tmp_path):
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg=90.0, canvas_w=13, canvas_h=95)
    wedge = extract_polar_wedge(rgb, mask, geom)

    path = tmp_path / "z01_wedge.png"
    save_wedge(wedge, geom, path)
    loaded = load_wedge(path)
    assert loaded is not None
    loaded_wedge, loaded_geom = loaded
    assert np.array_equal(loaded_wedge, wedge)
    assert loaded_geom.canvas_w == geom.canvas_w and loaded_geom.canvas_h == geom.canvas_h
    assert loaded_geom.axis_angle_rad == pytest.approx(geom.axis_angle_rad)
    np.testing.assert_allclose(loaded_geom.R_theta, geom.R_theta)


def test_load_wedge_missing_file_returns_none(tmp_path):
    assert load_wedge(tmp_path / "does_not_exist.png") is None


def test_get_or_compute_wedge_cache_hit_matches_shape(tmp_path):
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    path = tmp_path / "z01_wedge.png"

    wedge1, geom1 = get_or_compute_wedge(rgb, mask, axis_info, path, 90.0, 13, 95)
    wedge2, geom2 = get_or_compute_wedge(rgb, mask, axis_info, path, 90.0, 13, 95)
    assert np.array_equal(wedge1, wedge2)
    assert geom1.canvas_w == geom2.canvas_w == 13


def _band_edges_and_canvas():
    """Two bands, deliberately different canvas sizes (wider/taller for the second, edge-side
    band) — mirrors the real design (more columns near the edge) without needing the full
    measured 4-band/5852-patch production numbers for a unit test."""
    band_edges_t = [0.0, 0.5, 1.0]
    band_canvas_w = [8, 14]
    band_canvas_h = [20, 30]
    return band_edges_t, band_canvas_w, band_canvas_h


# ---------------------------------------------------------------------------
# Angular-resolution bands (09.09 follow-up)
# ---------------------------------------------------------------------------

def test_wedge_band_geometries_shares_R_theta_across_bands():
    """R_theta (the expensive ray-cast) must be computed ONCE and shared, not once per band."""
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    band_edges_t, band_canvas_w, band_canvas_h = _band_edges_and_canvas()
    bands = wedge_band_geometries_from_axis_info(
        mask, axis_info, delta_theta_deg=90.0, band_edges_t=band_edges_t,
        band_canvas_w=band_canvas_w, band_canvas_h=band_canvas_h)
    assert len(bands) == 2
    assert bands[0].geom.R_theta is bands[1].geom.R_theta
    assert bands[0].geom.centroid == bands[1].geom.centroid == (200.0, 200.0)
    assert bands[0].geom.axis_angle_rad == pytest.approx(bands[1].geom.axis_angle_rad)


def test_wedge_band_geometries_row_frac_edges_and_own_canvas():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    band_edges_t, band_canvas_w, band_canvas_h = _band_edges_and_canvas()
    bands = wedge_band_geometries_from_axis_info(
        mask, axis_info, delta_theta_deg=90.0, band_edges_t=band_edges_t,
        band_canvas_w=band_canvas_w, band_canvas_h=band_canvas_h)
    # band 0 covers t in [0, 0.5), band 1 covers [0.5, 1.0] -- contiguous, no gap/overlap
    assert bands[0].row_frac_lo == pytest.approx(0.0)
    assert bands[0].row_frac_hi == pytest.approx(bands[1].row_frac_lo)
    assert bands[1].row_frac_hi == pytest.approx(1.0)
    assert bands[0].geom.canvas_w == 8 and bands[0].geom.canvas_h == 20
    assert bands[1].geom.canvas_w == 14 and bands[1].geom.canvas_h == 30


def test_wedge_band_geometries_raises_on_mismatched_lengths():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    with pytest.raises(ValueError):
        wedge_band_geometries_from_axis_info(
            mask, axis_info, delta_theta_deg=90.0, band_edges_t=[0.0, 0.5, 1.0],
            band_canvas_w=[8], band_canvas_h=[20, 30])


def test_extract_polar_wedge_band_shape_and_dtype():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    band_edges_t, band_canvas_w, band_canvas_h = _band_edges_and_canvas()
    bands = wedge_band_geometries_from_axis_info(
        mask, axis_info, delta_theta_deg=90.0, band_edges_t=band_edges_t,
        band_canvas_w=band_canvas_w, band_canvas_h=band_canvas_h)
    for band, w, h in zip(bands, band_canvas_w, band_canvas_h):
        wedge = extract_polar_wedge_band(rgb, mask, band)
        assert wedge.shape == (h, w, 3)
        assert wedge.dtype == np.uint8


def test_wedge_band_polar_coords_t_stays_within_own_band_range():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    band_edges_t, band_canvas_w, band_canvas_h = _band_edges_and_canvas()
    bands = wedge_band_geometries_from_axis_info(
        mask, axis_info, delta_theta_deg=90.0, band_edges_t=band_edges_t,
        band_canvas_w=band_canvas_w, band_canvas_h=band_canvas_h)
    for band in bands:
        t_grid, theta_grid = wedge_band_polar_coords(band, patch_size=5)
        assert np.all(np.diff(t_grid[:, 0]) > 0)  # radius increases row by row within the band
    # first band's patches must all be < second band's minimum patch t (contiguous, non-overlapping)
    t0, _ = wedge_band_polar_coords(bands[0], patch_size=5)
    t1, _ = wedge_band_polar_coords(bands[1], patch_size=5)
    assert t0.max() < t1.min()


def test_wedge_band_round_trip_grid_of_points():
    """Same round-trip discipline as the single-band wedge, per band."""
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    band_edges_t, band_canvas_w, band_canvas_h = _band_edges_and_canvas()
    bands = wedge_band_geometries_from_axis_info(
        mask, axis_info, delta_theta_deg=80.0, band_edges_t=band_edges_t,
        band_canvas_w=band_canvas_w, band_canvas_h=band_canvas_h)
    for band_idx, band in enumerate(bands):
        w, h = band.geom.canvas_w, band.geom.canvas_h
        row_start = 1 if band_idx == 0 else 0  # row=0 of band 0 is the nucleus (t=0), degenerate
        max_err = 0.0
        for col in np.linspace(0, w - 1, 5):
            for row in np.linspace(row_start, h - 1, 5):
                x, y = wedge_band_xy_to_point(col, row, band)
                back = point_to_wedge_band_xy(x, y, band)
                assert back is not None, f"band {band_idx}: round-trip point unexpectedly outside"
                c2, r2, _t = back
                max_err = max(max_err, abs(col - c2), abs(row - r2))
        assert max_err < 1e-6


def test_point_to_wedge_band_xy_none_when_point_belongs_to_other_band():
    rgb, mask = _synthetic_circle_rgb_and_mask(cx=200, cy=200, radius=150)
    axis_info = _axis_info(200, 200, 350, 200, mask)
    band_edges_t, band_canvas_w, band_canvas_h = _band_edges_and_canvas()
    bands = wedge_band_geometries_from_axis_info(
        mask, axis_info, delta_theta_deg=80.0, band_edges_t=band_edges_t,
        band_canvas_w=band_canvas_w, band_canvas_h=band_canvas_h)
    # a point near the far edge (t close to 1) belongs to band 1, not band 0
    x, y = wedge_band_xy_to_point(band_canvas_w[1] / 2, band_canvas_h[1] - 1, bands[1])
    assert point_to_wedge_band_xy(x, y, bands[1]) is not None
    assert point_to_wedge_band_xy(x, y, bands[0]) is None


def test_point_to_wedge_band_xy_none_outside_delta_theta():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)  # axis along +x (angle 0)
    band_edges_t, band_canvas_w, band_canvas_h = _band_edges_and_canvas()
    bands = wedge_band_geometries_from_axis_info(
        mask, axis_info, delta_theta_deg=20.0, band_edges_t=band_edges_t,
        band_canvas_w=band_canvas_w, band_canvas_h=band_canvas_h)
    result = point_to_wedge_band_xy(200.0, 100.0, bands[0])  # ~90 deg away, outside a 20 deg wedge
    assert result is None


def test_extract_polar_wedge_band_validity_range_and_shape():
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    band_edges_t, band_canvas_w, band_canvas_h = _band_edges_and_canvas()
    bands = wedge_band_geometries_from_axis_info(
        mask, axis_info, delta_theta_deg=90.0, band_edges_t=band_edges_t,
        band_canvas_w=band_canvas_w, band_canvas_h=band_canvas_h)
    valid = extract_polar_wedge_band_validity(mask, bands[0], patch_size=4)
    assert valid.shape == (5, 2)
    assert np.all(valid >= 0.0) and np.all(valid <= 1.0)
    # near the nucleus (band 0's own row 0), the wedge is by construction always inside the mask
    assert valid[0].min() > 0.9


def test_get_or_compute_wedge_band_cache_hit_and_stale_shape_recomputes(tmp_path):
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    band_edges_t, band_canvas_w, band_canvas_h = _band_edges_and_canvas()
    bands = wedge_band_geometries_from_axis_info(
        mask, axis_info, delta_theta_deg=90.0, band_edges_t=band_edges_t,
        band_canvas_w=band_canvas_w, band_canvas_h=band_canvas_h)
    path = tmp_path / "z01_wedge_band0.png"

    wedge1 = get_or_compute_wedge_band(rgb, mask, bands[0], path)
    wedge2 = get_or_compute_wedge_band(rgb, mask, bands[0], path)
    assert np.array_equal(wedge1, wedge2)
    assert wedge1.shape == (band_canvas_h[0], band_canvas_w[0], 3)

    # stale shape (built at a different canvas size) must never be served silently
    bigger_band = WedgeBandGeometry(
        geom=bands[0].geom._replace(canvas_w=21, canvas_h=40),
        row_frac_lo=bands[0].row_frac_lo, row_frac_hi=bands[0].row_frac_hi,
    )
    wedge3 = get_or_compute_wedge_band(rgb, mask, bigger_band, path)
    assert wedge3.shape == (40, 21, 3)


def test_get_or_compute_wedge_stale_shape_recomputes(tmp_path):
    """A cached wedge built at DIFFERENT canvas dimensions must never be served silently — same
    guard `get_or_compute_strip`/`get_or_compute_strip_validity` already enforce for the strip."""
    rgb, mask = _synthetic_circle_rgb_and_mask()
    axis_info = _axis_info(200, 200, 350, 200, mask)
    path = tmp_path / "z01_wedge.png"

    wedge_small, _ = get_or_compute_wedge(rgb, mask, axis_info, path, 90.0, 13, 95)
    assert wedge_small.shape == (95, 13, 3)
    wedge_big, geom_big = get_or_compute_wedge(rgb, mask, axis_info, path, 90.0, 21, 140)
    assert wedge_big.shape == (140, 21, 3)
    assert geom_big.canvas_w == 21 and geom_big.canvas_h == 140
