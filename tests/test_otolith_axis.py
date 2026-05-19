"""Unit tests for src/otolith_axis.py — synthetic images, no Z: drive access."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.otolith_axis import (
    detect_axis,
    find_centroid,
    find_farthest_edge,
    load_mask,
    project_distance_to_axis,
    sample_profile_along_axis,
    save_mask,
    segment_otolith,
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
# project_distance_to_axis
# ---------------------------------------------------------------------------

def test_project_distance_to_axis_endpoints():
    """At the centroid t≈0, at the far edge t≈1."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (10, 40), (90, 60), 255, -1)
    centroid = (20, 50)
    far_edge = (80, 50)
    t = project_distance_to_axis(mask, centroid, far_edge)
    # Pixel at centroid → ~0; pixel at far_edge → ~1
    assert abs(float(t[50, 20])) < 1e-3
    assert abs(float(t[50, 80]) - 1.0) < 1e-3


def test_project_distance_to_axis_outside_mask_is_nan():
    mask = np.zeros((50, 60), dtype=np.uint8)
    mask[20:30, 10:50] = 255
    t = project_distance_to_axis(mask, (15, 25), (45, 25))
    # Pixel outside the mask must be NaN
    assert np.isnan(t[0, 0])
    # Pixel inside the mask must be finite
    assert np.isfinite(t[25, 25])


def test_project_distance_to_axis_increases_along_axis():
    """t values grow monotonically as we walk from centroid to far edge."""
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[15:25, 5:75] = 255
    t = project_distance_to_axis(mask, (10, 20), (70, 20))
    row = t[20, 10:71]   # walk along the axis
    assert row[0] < row[10] < row[30] < row[60]


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
