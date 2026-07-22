"""Unit tests for src/otolith_axis.py — synthetic images, no Z: drive access."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.otolith_axis import (
    apply_background_mask,
    detect_axis,
    find_centroid,
    find_farthest_edge,
    find_intensity_centroid,
    get_or_compute_mask,
    load_mask,
    mask_bbox,
    MASK_FILL_RGB,
    resolve_centroid,
    sample_profile_along_axis,
    save_mask,
    segment_otolith,
    shift_axis_info,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic image factories
# ---------------------------------------------------------------------------

def _make_dark_ellipse(
    img_h: int = 600,
    img_w: int = 800,
    center: tuple[int, int] = (400, 300),
    axes: tuple[int, int] = (100, 200),   # (semi-width, semi-height)
) -> np.ndarray:
    """White background (255) with a dark (40) ellipse painted on top."""
    img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    cv2.ellipse(img, center, axes, angle=0, startAngle=0, endAngle=360,
                color=(40, 40, 40), thickness=-1)
    return img


# ---------------------------------------------------------------------------
# segment_otolith
# ---------------------------------------------------------------------------

def test_segment_dark_ellipse_on_white_background():
    img = _make_dark_ellipse(center=(400, 300), axes=(100, 200))
    mask = segment_otolith(img)
    assert mask is not None
    assert mask.shape == (600, 800)
    # Expected pixel count ≈ π·a·b ≈ 62 832, allow ±10% (morphology may inflate)
    area = int((mask > 0).sum())
    assert 56_000 <= area <= 72_000, f"unexpected mask area: {area}"


def test_segment_returns_none_for_uniform_image():
    """Uniformly white image → no foreground → None."""
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    assert segment_otolith(img) is None


# ---------------------------------------------------------------------------
# Radial fade-detection segmentation (bright otolith on dark background)
# ---------------------------------------------------------------------------

def _make_faded_disk(H=400, W=400, center=(200, 200), r_core=80, r_outer=140,
                     bg=10, fg=220) -> np.ndarray:
    """Bright disk that FADES to background between r_core and r_outer.

    Mimics a transilluminated embedded otolith: opaque core → thinning translucent
    rim → dark background. The strong edge sits inside r_outer, the true edge at r_outer.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - center[0], yy - center[1])
    inten = np.full((H, W), float(bg), dtype=np.float32)
    inten[d <= r_core] = fg
    ramp = (d > r_core) & (d <= r_outer)
    inten[ramp] = fg - (fg - bg) * (d[ramp] - r_core) / (r_outer - r_core)
    img = np.clip(inten, 0, 255).astype(np.uint8)
    return np.stack([img] * 3, axis=2)


def test_radial_captures_faded_margin():
    """Radial method must reach INTO the faint fading rim, past the bright core."""
    img = _make_faded_disk(r_core=80, r_outer=140)
    mask = segment_otolith(img, method="radial")
    assert mask is not None
    area = int((mask > 0).sum())
    core_area = np.pi * 80 ** 2       # bright opaque core  ≈ 20 106
    outer_area = np.pi * 140 ** 2     # true faded edge     ≈ 61 575
    assert area > 1.5 * core_area, f"radial didn't reach the fade: {area}"
    assert area < outer_area,      f"radial blew past the true edge: {area}"


def _contour_jaggedness(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    per = cv2.arcLength(c, True)
    return (per * per) / (4.0 * np.pi * area + 1e-9)


def test_radial_smoothing_knob_reduces_jaggedness():
    """Higher smooth_sigma yields a smoother (lower-jaggedness) radial outline.

    smooth_sigma is the "follow the scalloped teeth (low) vs smooth envelope
    (high)" knob — this checks the mechanism directly, independent of the default.
    """
    rng = np.random.default_rng(0)
    H = W = 400
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - 200, yy - 200)
    img = np.where(d <= 120, 210.0, 15.0)
    band = (d > 105) & (d < 135)                       # speckle only near the edge
    img[band] += rng.normal(0, 60, (H, W))[band]
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.stack([img] * 3, axis=2)

    sharp = segment_otolith(img, method="radial", smooth_sigma=2.0)
    smooth = segment_otolith(img, method="radial", smooth_sigma=20.0)
    assert sharp is not None and smooth is not None
    assert _contour_jaggedness(smooth) < _contour_jaggedness(sharp)


def test_threshold_method_still_works():
    """The old method remains available as a fallback via method='threshold'."""
    img = _make_dark_ellipse(center=(400, 300), axes=(100, 200))
    mask = segment_otolith(img, method="threshold")
    assert mask is not None
    assert 56_000 <= int((mask > 0).sum()) <= 72_000


def test_segment_picks_largest_component():
    """Two ellipses (small + large) → mask covers only the large one."""
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.ellipse(img, (200, 150), (30, 40), 0, 0, 360, (40, 40, 40), -1)   # small
    cv2.ellipse(img, (500, 400), (80, 150), 0, 0, 360, (40, 40, 40), -1)  # large
    mask = segment_otolith(img)
    assert mask is not None
    # The small ellipse area is ~π·30·40 ≈ 3 770; large is ~π·80·150 ≈ 37 700.
    # Mask area should be near the large one.
    area = int((mask > 0).sum())
    assert area > 30_000, f"mask too small, likely picked wrong contour: {area}"
    assert area < 45_000, f"mask too large, picked both contours: {area}"


def test_segment_handles_invalid_input():
    assert segment_otolith(None) is None                       # type: ignore[arg-type]
    assert segment_otolith(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert segment_otolith(np.zeros((10, 10),  dtype=np.uint8)) is None   # 2D, not RGB


# ---------------------------------------------------------------------------
# find_centroid / find_farthest_edge
# ---------------------------------------------------------------------------

def test_centroid_of_centered_ellipse():
    img = _make_dark_ellipse(center=(400, 300), axes=(100, 200))
    mask = segment_otolith(img)
    cx, cy = find_centroid(mask)
    assert abs(cx - 400) <= 5
    assert abs(cy - 300) <= 5


def test_centroid_returns_none_for_empty_mask():
    empty = np.zeros((100, 100), dtype=np.uint8)
    assert find_centroid(empty) is None


# ---------------------------------------------------------------------------
# find_intensity_centroid / resolve_centroid (nucleus estimate, 20.07)
# ---------------------------------------------------------------------------

def _make_ellipse_with_dark_core(
    core_center: tuple[int, int] = (360, 260),
    core_radius: int = 25,
) -> np.ndarray:
    """Medium-dark ellipse (uniform body) with an off-centre, much darker "core" patch —
    stands in for an asymmetric primordium the geometric centroid would miss."""
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.ellipse(img, (400, 300), (100, 200), angle=0, startAngle=0, endAngle=360,
                color=(150, 150, 150), thickness=-1)
    cv2.circle(img, core_center, core_radius, (5, 5, 5), -1)
    return img


def test_intensity_centroid_uniform_matches_geometric():
    """No intensity variation inside the mask → intensity-weighted centroid reduces to
    the geometric centroid (constant weight everywhere)."""
    img = _make_dark_ellipse(center=(400, 300), axes=(100, 200))
    mask = segment_otolith(img)
    geo = find_centroid(mask)
    intensity = find_intensity_centroid(img, mask)
    assert intensity is not None
    assert abs(geo[0] - intensity[0]) <= 2
    assert abs(geo[1] - intensity[1]) <= 2


def test_intensity_centroid_pulls_toward_darker_core():
    """An off-centre, more opaque sub-region should pull the intensity centroid toward
    it, further than the plain geometric centroid."""
    core = np.array([360, 260])
    img = _make_ellipse_with_dark_core(core_center=tuple(core))
    mask = segment_otolith(img)
    assert mask is not None
    geo = np.array(find_centroid(mask))
    intensity = np.array(find_intensity_centroid(img, mask))
    d_geo = float(np.hypot(*(geo - core)))
    d_int = float(np.hypot(*(intensity - core)))
    assert d_int < d_geo


def test_intensity_centroid_returns_none_for_empty_mask():
    empty = np.zeros((100, 100), dtype=np.uint8)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert find_intensity_centroid(img, empty) is None


def test_resolve_centroid_geometric_is_default():
    img = _make_dark_ellipse()
    mask = segment_otolith(img)
    assert resolve_centroid(img, mask) == find_centroid(mask)
    assert resolve_centroid(img, mask, "geometric") == find_centroid(mask)


def test_resolve_centroid_intensity_dispatches():
    img = _make_ellipse_with_dark_core()
    mask = segment_otolith(img)
    assert resolve_centroid(img, mask, "intensity") == find_intensity_centroid(img, mask)


def test_farthest_point_along_major_axis():
    """Vertical ellipse (taller than wide) → farthest point near top or bottom pole."""
    img = _make_dark_ellipse(center=(400, 300), axes=(100, 200))
    mask = segment_otolith(img)
    centroid = find_centroid(mask)
    far_x, far_y = find_farthest_edge(mask, centroid)
    # Major axis is vertical → far point should be ≈ (400, 100) or (400, 500)
    assert abs(far_x - 400) <= 15
    assert abs(far_y - 100) <= 25 or abs(far_y - 500) <= 25


# ---------------------------------------------------------------------------
# detect_axis (high-level)
# ---------------------------------------------------------------------------

def test_detect_axis_returns_dict_for_valid_ellipse():
    img = _make_dark_ellipse()
    info = detect_axis(img)
    assert info is not None
    assert set(info.keys()) >= {"mask", "centroid", "far_edge", "contour", "length_px"}
    assert info["length_px"] > 0


def test_detect_axis_returns_none_for_uniform():
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    assert detect_axis(img) is None


def test_detect_axis_intensity_nucleus_method_shifts_centroid():
    """nucleus_method="intensity" must change the centroid picked by detect_axis (and
    nothing else breaks — this exercises the whole segment→centroid→far_edge chain)."""
    img = _make_ellipse_with_dark_core(core_center=(360, 260))
    info_geo = detect_axis(img)
    info_intensity = detect_axis(img, nucleus_method="intensity")
    assert info_geo is not None and info_intensity is not None
    assert info_geo["centroid"] != info_intensity["centroid"]


def test_segmentation_config_as_params_excludes_nucleus_method():
    """nucleus_method is consumed by detect_axis(), NOT a segment_otolith() kwarg —
    as_params() must exclude it or segment_otolith(**params) raises TypeError."""
    from src.config import SegmentationConfig

    cfg = SegmentationConfig()
    assert cfg.nucleus_method == "geometric"      # default = unchanged behaviour
    params = cfg.as_params()
    assert "nucleus_method" not in params
    img = _make_dark_ellipse()
    assert segment_otolith(img, **params) is not None   # would TypeError if leaked


def test_axis_info_json_serializable():
    """centroid + far_edge + length_px should serialise after explicit conversion."""
    img = _make_dark_ellipse()
    info = detect_axis(img)
    payload = {
        "centroid":  list(info["centroid"]),
        "far_edge":  list(info["far_edge"]),
        "length_px": info["length_px"],
    }
    json_str = json.dumps(payload)
    assert "centroid" in json_str


# ---------------------------------------------------------------------------
# sample_profile_along_axis
# ---------------------------------------------------------------------------

def test_sample_profile_length():
    grid = np.random.rand(37, 37).astype(np.float32)
    profile, line_xy = sample_profile_along_axis(
        grid, centroid=(400, 300), far_edge=(400, 500),
        image_h=600, image_w=800, n_samples=20,
    )
    assert profile.shape == (20,)
    assert line_xy.shape == (20, 2)


def test_sample_profile_endpoints_match_pixel_coords():
    grid = np.random.rand(37, 37).astype(np.float32)
    profile, line_xy = sample_profile_along_axis(
        grid, centroid=(100, 200), far_edge=(700, 500),
        image_h=600, image_w=800, n_samples=10,
    )
    # First and last sample should land on the centroid and far_edge respectively
    assert tuple(line_xy[0])  == (100, 200)
    assert tuple(line_xy[-1]) == (700, 500)


# ---------------------------------------------------------------------------
# Mask I/O cache
# ---------------------------------------------------------------------------

def test_save_and_load_mask_roundtrip(tmp_path):
    original = np.zeros((50, 60), dtype=np.uint8)
    original[10:40, 15:45] = 255
    out = tmp_path / "mask.png"
    save_mask(original, out)
    assert out.exists()
    loaded = load_mask(out)
    assert loaded is not None
    assert loaded.shape == original.shape
    assert np.array_equal(loaded, original)


def test_load_mask_missing_returns_none(tmp_path):
    assert load_mask(tmp_path / "missing.png") is None


# ---------------------------------------------------------------------------
# get_or_compute_mask / apply_background_mask (input masking, 20.07)
# ---------------------------------------------------------------------------

def test_get_or_compute_mask_computes_and_caches(tmp_path):
    img = _make_dark_ellipse()
    cache_path = tmp_path / "fish1_mask.png"
    assert not cache_path.exists()
    mask = get_or_compute_mask(img, cache_path)
    assert mask is not None
    assert cache_path.exists()


def test_get_or_compute_mask_reuses_cache(tmp_path, monkeypatch):
    img = _make_dark_ellipse()
    cache_path = tmp_path / "fish1_mask.png"
    first = get_or_compute_mask(img, cache_path)

    def _boom(*a, **kw):
        raise AssertionError("segment_otolith should NOT be called on a cache hit")
    monkeypatch.setattr("src.otolith_axis.segment_otolith", _boom)

    second = get_or_compute_mask(img, cache_path)
    assert np.array_equal(first, second)


def test_get_or_compute_mask_returns_none_without_caching_on_failure(tmp_path):
    uniform = np.full((100, 100, 3), 255, dtype=np.uint8)   # no segmentable foreground
    cache_path = tmp_path / "fail_mask.png"
    assert get_or_compute_mask(uniform, cache_path) is None
    assert not cache_path.exists()


def test_apply_background_mask_fills_outside_only():
    img = np.full((20, 20, 3), 200, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    out = apply_background_mask(img, mask)
    assert tuple(out[0, 0]) == MASK_FILL_RGB               # outside mask → filled
    assert tuple(out[10, 10]) == (200, 200, 200)           # inside mask → untouched


# ---------------------------------------------------------------------------
# mask_bbox / shift_axis_info (22.07 — crop-to-bbox for higher-resolution density)
# ---------------------------------------------------------------------------

def test_mask_bbox_tight_box_no_padding():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:70] = 255      # rows 20..39, cols 30..69 -> box (30, 20, 40, 20)
    x0, y0, w, h = mask_bbox(mask, pad_frac=0.0)
    assert (x0, y0, w, h) == (30, 20, 40, 20)


def test_mask_bbox_pads_proportionally_to_box_size():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255    # 100x100 box at (50, 50)
    x0, y0, w, h = mask_bbox(mask, pad_frac=0.10)   # 10% of 100 = 10px pad each side
    assert (x0, y0) == (40, 40)
    assert (w, h) == (120, 120)


def test_mask_bbox_clamps_to_image_bounds():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[0:10, 0:10] = 255        # box touches the top-left corner
    x0, y0, w, h = mask_bbox(mask, pad_frac=0.5)    # padding would go negative
    assert x0 == 0 and y0 == 0
    assert w <= 50 and h <= 50


def test_mask_bbox_empty_mask_returns_whole_image():
    mask = np.zeros((30, 40), dtype=np.uint8)
    x0, y0, w, h = mask_bbox(mask)
    assert (x0, y0, w, h) == (0, 0, 40, 30)


def test_shift_axis_info_translates_geometry():
    contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32)
    axis_info = {
        "mask": np.zeros((30, 30), dtype=np.uint8),
        "centroid": (15, 15),
        "far_edge": (20, 20),
        "contour": contour,
        "length_px": 7.07,
    }
    shifted = shift_axis_info(axis_info, dx=-5, dy=-5)
    assert shifted["centroid"] == (10, 10)
    assert shifted["far_edge"] == (15, 15)
    assert np.array_equal(shifted["contour"], contour - np.array([5, 5]))
    assert shifted["contour"].dtype == contour.dtype
    assert shifted["length_px"] == 7.07                      # distance is translation-invariant
    # original untouched
    assert axis_info["centroid"] == (15, 15)
