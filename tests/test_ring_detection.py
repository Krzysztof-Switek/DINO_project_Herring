"""Tests for src/ring_detection.py — image-based annual-ring detection."""
from __future__ import annotations

import cv2
import numpy as np


def _synthetic_otolith(n_bands: int = 4):
    """Elliptical otolith with ``n_bands`` concentric intensity bands aligned to scale."""
    H, W = 200, 160
    cx, cy, a, b = W // 2, H // 2, 60, 90
    Y, X = np.mgrid[0:H, 0:W]
    rho = np.sqrt(((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2)   # 0 at centre, 1 at edge
    gray = np.clip(128 + 100 * np.cos(2 * np.pi * n_bands * rho), 0, 255).astype(np.uint8)
    mask = (rho <= 1.0).astype(np.uint8) * 255

    fill = np.zeros((H, W), np.uint8)
    cv2.ellipse(fill, (cx, cy), (a, b), 0, 0, 360, 255, -1)
    cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(cnts, key=cv2.contourArea)
    return gray, mask, (cx, cy), contour


def test_radial_band_profile_oscillates():
    from src.ring_detection import radial_band_profile
    gray, mask, centroid, contour = _synthetic_otolith(n_bands=4)
    scales, profile = radial_band_profile(gray, mask, centroid, contour, n_scales=120)
    assert scales.shape == profile.shape == (120,)
    assert profile.max() - profile.min() > 20    # clear banding, not flat


def test_detect_ring_scales_finds_bands():
    from src.ring_detection import radial_band_profile, detect_ring_scales
    gray, mask, centroid, contour = _synthetic_otolith(n_bands=4)
    scales, profile = radial_band_profile(gray, mask, centroid, contour, n_scales=120)
    rings = detect_ring_scales(scales, profile, prominence=0.1, polarity="bright")
    assert len(rings) >= 2
    assert np.all((rings > 0.0) & (rings <= 1.0))


def test_detect_ring_scales_flat_profile_empty():
    from src.ring_detection import detect_ring_scales
    scales = np.linspace(0.04, 1.0, 50)
    profile = np.full(50, 100.0, dtype=np.float32)
    assert detect_ring_scales(scales, profile).size == 0


def test_draw_scaled_rings_modifies_panel():
    from src.ring_detection import draw_scaled_rings
    _, _, centroid, contour = _synthetic_otolith()
    panel = np.full((200, 160, 3), 200, np.uint8)
    before = panel.copy()
    draw_scaled_rings(panel, centroid, contour, np.array([0.3, 0.6]), thickness=2)
    assert not np.array_equal(before, panel)


def test_detect_and_draw_rings_wrapper():
    from src.ring_detection import detect_and_draw_rings
    gray, mask, centroid, contour = _synthetic_otolith(n_bands=4)
    axis_info = {"centroid": centroid, "contour": contour, "mask": mask}
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    scales, overlay = detect_and_draw_rings(img_rgb, axis_info, prominence=0.1)
    assert overlay.shape == img_rgb.shape
    assert len(scales) >= 1


def test_build_tuning_montage():
    from scripts.tune_image_rings import build_tuning_montage
    gray, mask, centroid, contour = _synthetic_otolith(n_bands=4)
    axis_info = {"centroid": centroid, "contour": contour, "mask": mask}
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    montage, counts = build_tuning_montage(img_rgb, axis_info, [0.1, 0.2], ["bright"])
    assert len(counts) == 2                              # 2 prominence × 1 polarity
    assert montage.shape[1] >= img_rgb.shape[1] * 3      # original + 2 variants side by side
    assert montage.shape[0] >= img_rgb.shape[0]          # caption bar added on top
