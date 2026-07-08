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
