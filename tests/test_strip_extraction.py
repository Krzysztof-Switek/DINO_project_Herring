"""Unit tests for src/strip_extraction.py — synthetic images, no Z: drive access."""
from __future__ import annotations

import numpy as np
import pytest

from src.otolith_axis import MASK_FILL_RGB
from src.strip_extraction import (
    canonicalizing_matrix,
    extract_strip,
    extract_strip_from_axis_info,
    extract_strip_validity,
    extract_strip_validity_from_axis_info,
    get_or_compute_strip,
    get_or_compute_strip_validity,
    load_strip,
    load_strip_validity,
    save_strip,
    save_strip_validity,
    strip_transform_matrix,
)


def _apply(M: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    x, y = pt
    v = M @ np.array([x, y, 1.0])
    return float(v[0]), float(v[1])


# ---------------------------------------------------------------------------
# canonicalizing_matrix
# ---------------------------------------------------------------------------

def test_canonicalizing_matrix_aligns_far_edge_with_centroid_y():
    centroid = (120.0, 80.0)
    far_edge = (300.0, 210.0)
    M = canonicalizing_matrix(centroid, far_edge)
    cx, cy = _apply(M, centroid)
    fx, fy = _apply(M, far_edge)
    assert cx == pytest.approx(centroid[0], abs=1e-6)
    assert cy == pytest.approx(centroid[1], abs=1e-6)
    assert fy == pytest.approx(cy, abs=1e-6)          # far_edge now horizontal from centroid
    assert fx > cx                                     # points along +x, not -x


# ---------------------------------------------------------------------------
# strip_transform_matrix — control points
# ---------------------------------------------------------------------------

def test_strip_transform_matrix_control_points():
    centroid = (100.0, 150.0)
    far_edge = (400.0, 250.0)
    length_px = float(np.hypot(far_edge[0] - centroid[0], far_edge[1] - centroid[1]))
    strip_length_px, strip_width_px = 300, 60

    M = strip_transform_matrix(centroid, far_edge, length_px, strip_length_px, strip_width_px)
    cx, cy = _apply(M, centroid)
    fx, fy = _apply(M, far_edge)

    assert cx == pytest.approx(0.0, abs=1e-6)
    assert cy == pytest.approx(strip_width_px / 2, abs=1e-6)
    assert fx == pytest.approx(strip_length_px, abs=1e-6)
    assert fy == pytest.approx(strip_width_px / 2, abs=1e-6)


def test_strip_transform_matrix_degenerate_axis_does_not_raise():
    # length_px ~ 0 — must floor at 1.0px, not divide by zero / produce inf or nan.
    M = strip_transform_matrix((50.0, 50.0), (50.001, 50.0), 0.0, 200, 40)
    assert np.all(np.isfinite(M))


# ---------------------------------------------------------------------------
# Round-trip: a point warped then inverse-warped returns to itself
# ---------------------------------------------------------------------------

def test_strip_transform_round_trip():
    import cv2

    centroid = (137.0, 92.0)
    far_edge = (410.0, 260.0)
    length_px = float(np.hypot(far_edge[0] - centroid[0], far_edge[1] - centroid[1]))
    strip_length_px, strip_width_px = 500, 80
    M = strip_transform_matrix(centroid, far_edge, length_px, strip_length_px, strip_width_px)
    M_inv = cv2.invertAffineTransform(M)

    probe_points = [
        centroid, far_edge, (200.0, 100.0), (5.0, 400.0), (300.0, 150.0),
    ]
    for pt in probe_points:
        strip_pt = _apply(M, pt)
        recovered = _apply(M_inv, strip_pt)
        assert recovered[0] == pytest.approx(pt[0], abs=1e-6)
        assert recovered[1] == pytest.approx(pt[1], abs=1e-6)


# ---------------------------------------------------------------------------
# extract_strip — shape/dtype + background-fill regression
# ---------------------------------------------------------------------------

def _synthetic_rgb_and_mask(h: int = 400, w: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:350, 50:450] = 255
    return rgb, mask


def test_extract_strip_output_shape_and_dtype():
    rgb, mask = _synthetic_rgb_and_mask()
    centroid, far_edge, length_px = (100.0, 150.0), (400.0, 250.0), 300.0
    strip_length_px, strip_width_px = 300, 60

    strip, M = extract_strip(rgb, mask, centroid, far_edge, length_px, strip_length_px, strip_width_px)
    assert strip.shape == (strip_width_px, strip_length_px, 3)
    assert strip.dtype == np.uint8
    assert M.shape == (2, 3)


def test_extract_strip_border_fill_is_never_black():
    """A strip deliberately much wider/longer than the otolith forces out-of-bounds warp
    corners — those pixels must be MASK_FILL_RGB (the project's established ImageNet-mean
    fill), never black, or the register-token/high-contrast-edge artifact this project
    already fixed once (switch to the `_reg` backbone) would reappear at the strip's edges."""
    h, w = 100, 100
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    centroid, far_edge, length_px = (50.0, 50.0), (85.0, 50.0), 35.0
    strip_length_px, strip_width_px = 400, 200   # far larger than the source image

    strip, _M = extract_strip(rgb, mask, centroid, far_edge, length_px, strip_length_px, strip_width_px)
    corner = tuple(int(v) for v in strip[0, 0])
    assert corner == MASK_FILL_RGB
    assert corner != (0, 0, 0)


def test_extract_strip_degenerate_axis_does_not_raise():
    rgb, mask = _synthetic_rgb_and_mask()
    strip, M = extract_strip(rgb, mask, (200.0, 200.0), (200.001, 200.0), 0.0, 200, 50)
    assert strip.shape == (50, 200, 3)
    assert np.all(np.isfinite(M))


def test_extract_strip_from_axis_info_matches_extract_strip():
    rgb, mask = _synthetic_rgb_and_mask()
    axis_info = {"centroid": (100.0, 150.0), "far_edge": (400.0, 250.0), "length_px": 300.0}
    strip_a, M_a = extract_strip(
        rgb, mask, axis_info["centroid"], axis_info["far_edge"], axis_info["length_px"], 300, 60,
    )
    strip_b, M_b = extract_strip_from_axis_info(rgb, mask, axis_info, 300, 60)
    assert np.array_equal(strip_a, strip_b)
    assert np.array_equal(M_a, M_b)


# ---------------------------------------------------------------------------
# Strip cache I/O
# ---------------------------------------------------------------------------

def test_save_and_load_strip_roundtrip(tmp_path):
    original = np.zeros((40, 100, 3), dtype=np.uint8)
    original[:, :, 0] = 77
    out = tmp_path / "strip.png"
    save_strip(original, out)
    assert out.exists()
    loaded = load_strip(out)
    assert loaded is not None
    assert loaded.shape == original.shape
    assert np.array_equal(loaded, original)


def test_load_strip_missing_returns_none(tmp_path):
    assert load_strip(tmp_path / "missing.png") is None


def test_get_or_compute_strip_computes_and_caches(tmp_path):
    rgb, mask = _synthetic_rgb_and_mask()
    axis_info = {"centroid": (100.0, 150.0), "far_edge": (400.0, 250.0), "length_px": 300.0}
    cache_path = tmp_path / "fish1_strip.png"
    assert not cache_path.exists()
    strip = get_or_compute_strip(rgb, mask, axis_info, cache_path, 300, 60)
    assert strip is not None
    assert strip.shape == (60, 300, 3)
    assert cache_path.exists()


def test_get_or_compute_strip_reuses_cache(tmp_path, monkeypatch):
    rgb, mask = _synthetic_rgb_and_mask()
    axis_info = {"centroid": (100.0, 150.0), "far_edge": (400.0, 250.0), "length_px": 300.0}
    cache_path = tmp_path / "fish1_strip.png"
    first = get_or_compute_strip(rgb, mask, axis_info, cache_path, 300, 60)

    def _boom(*a, **kw):
        raise AssertionError("extract_strip_from_axis_info should NOT be called on a cache hit")
    monkeypatch.setattr("src.strip_extraction.extract_strip_from_axis_info", _boom)

    second = get_or_compute_strip(rgb, mask, axis_info, cache_path, 300, 60)
    assert np.array_equal(first, second)


def test_get_or_compute_strip_recomputes_on_dimension_mismatch(tmp_path):
    """A cached strip built at a DIFFERENT (strip_length_px, strip_width_px) must never be
    served silently — same lesson as the project's documented mask-cache collision bug
    (expert_annotation_eval.py, 12.08)."""
    rgb, mask = _synthetic_rgb_and_mask()
    axis_info = {"centroid": (100.0, 150.0), "far_edge": (400.0, 250.0), "length_px": 300.0}
    cache_path = tmp_path / "fish1_strip.png"
    stale = np.zeros((999, 999, 3), dtype=np.uint8)   # wrong shape, simulates a stale cache entry
    save_strip(stale, cache_path)

    fresh = get_or_compute_strip(rgb, mask, axis_info, cache_path, 300, 60)
    assert fresh.shape == (60, 300, 3)


# ---------------------------------------------------------------------------
# 08.09 (plans and summaries/08.09_metodyka_i_diagnoza_paska.md): per-patch tissue validity
# mask -- the fix for the diagnosed background-fixation failure (density_count_loss's new
# valid_mask param).
# ---------------------------------------------------------------------------

_PATCH = 10  # divides both 300 (strip_length_px) and 60 (strip_width_px) used throughout this file


def test_extract_strip_validity_shape_and_range():
    _, mask = _synthetic_rgb_and_mask()
    centroid, far_edge, length_px = (100.0, 150.0), (400.0, 250.0), 300.0
    valid = extract_strip_validity(mask, centroid, far_edge, length_px, 300, 60, _PATCH)
    assert valid.shape == (60 // _PATCH, 300 // _PATCH)
    assert valid.dtype == np.float32
    assert float(valid.min()) >= 0.0 and float(valid.max()) <= 1.0


def test_extract_strip_validity_all_valid_when_mask_covers_everything():
    """A mask that is 255 everywhere the strip could possibly be warped from -> every patch
    must come back fully valid (1.0)."""
    mask = np.full((400, 500), 255, dtype=np.uint8)
    centroid, far_edge, length_px = (100.0, 150.0), (400.0, 250.0), 300.0
    valid = extract_strip_validity(mask, centroid, far_edge, length_px, 300, 60, _PATCH)
    assert np.allclose(valid, 1.0)


def test_extract_strip_validity_zero_when_mask_empty():
    """An all-background mask -> every patch must come back fully invalid (0.0)."""
    mask = np.zeros((400, 500), dtype=np.uint8)
    centroid, far_edge, length_px = (100.0, 150.0), (400.0, 250.0), 300.0
    valid = extract_strip_validity(mask, centroid, far_edge, length_px, 300, 60, _PATCH)
    assert np.allclose(valid, 0.0)


def test_extract_strip_validity_matches_extract_strip_masking():
    """Where extract_strip's OWN masked-background pixel is exactly MASK_FILL_RGB, the
    validity mask at the corresponding patch must be 0; where extract_strip shows real
    (non-fill) content, validity must be > 0 -- the two must describe the SAME geometry."""
    rgb, mask = _synthetic_rgb_and_mask()
    centroid, far_edge, length_px = (100.0, 150.0), (400.0, 250.0), 300.0
    strip, _M = extract_strip(rgb, mask, centroid, far_edge, length_px, 300, 60)
    valid = extract_strip_validity(mask, centroid, far_edge, length_px, 300, 60, _PATCH)

    h_p, w_p = valid.shape
    is_fill = np.all(strip == np.array(MASK_FILL_RGB, dtype=np.uint8), axis=-1)
    patch_fill_frac = is_fill.reshape(h_p, _PATCH, w_p, _PATCH).mean(axis=(1, 3))
    # A patch that's entirely MASK_FILL_RGB in the real strip must be invalid here, and
    # vice versa -- same corridor, same edges, computed two different ways.
    assert np.allclose(patch_fill_frac > 0.99, valid < 0.01)
    assert np.allclose(patch_fill_frac < 0.01, valid > 0.99)


def test_extract_strip_validity_from_axis_info_matches():
    _, mask = _synthetic_rgb_and_mask()
    axis_info = {"centroid": (100.0, 150.0), "far_edge": (400.0, 250.0), "length_px": 300.0}
    a = extract_strip_validity(
        mask, axis_info["centroid"], axis_info["far_edge"], axis_info["length_px"], 300, 60, _PATCH)
    b = extract_strip_validity_from_axis_info(mask, axis_info, 300, 60, _PATCH)
    assert np.array_equal(a, b)


def test_save_and_load_strip_validity_roundtrip(tmp_path):
    original = np.linspace(0.0, 1.0, 6 * 30, dtype=np.float32).reshape(6, 30)
    out = tmp_path / "fish1_strip_valid.npy"
    save_strip_validity(original, out)
    assert out.exists()
    loaded = load_strip_validity(out)
    assert loaded is not None
    assert np.array_equal(loaded, original)


def test_load_strip_validity_missing_returns_none(tmp_path):
    assert load_strip_validity(tmp_path / "missing.npy") is None


def test_get_or_compute_strip_validity_computes_and_caches(tmp_path):
    _, mask = _synthetic_rgb_and_mask()
    axis_info = {"centroid": (100.0, 150.0), "far_edge": (400.0, 250.0), "length_px": 300.0}
    cache_path = tmp_path / "fish1_strip_valid.npy"
    assert not cache_path.exists()
    valid = get_or_compute_strip_validity(mask, axis_info, cache_path, 300, 60, _PATCH)
    assert valid.shape == (60 // _PATCH, 300 // _PATCH)
    assert cache_path.exists()


def test_get_or_compute_strip_validity_reuses_cache(tmp_path, monkeypatch):
    _, mask = _synthetic_rgb_and_mask()
    axis_info = {"centroid": (100.0, 150.0), "far_edge": (400.0, 250.0), "length_px": 300.0}
    cache_path = tmp_path / "fish1_strip_valid.npy"
    first = get_or_compute_strip_validity(mask, axis_info, cache_path, 300, 60, _PATCH)

    def _boom(*a, **kw):
        raise AssertionError("extract_strip_validity_from_axis_info should NOT run on a cache hit")
    monkeypatch.setattr("src.strip_extraction.extract_strip_validity_from_axis_info", _boom)

    second = get_or_compute_strip_validity(mask, axis_info, cache_path, 300, 60, _PATCH)
    assert np.array_equal(first, second)


def test_get_or_compute_strip_validity_recomputes_on_dimension_mismatch(tmp_path):
    _, mask = _synthetic_rgb_and_mask()
    axis_info = {"centroid": (100.0, 150.0), "far_edge": (400.0, 250.0), "length_px": 300.0}
    cache_path = tmp_path / "fish1_strip_valid.npy"
    stale = np.zeros((999, 999), dtype=np.float32)
    save_strip_validity(stale, cache_path)

    fresh = get_or_compute_strip_validity(mask, axis_info, cache_path, 300, 60, _PATCH)
    assert fresh.shape == (60 // _PATCH, 300 // _PATCH)
