"""E11 (07.10): the row-space population prior uses each image's own row → axis geometry."""
from __future__ import annotations

import numpy as np

from scripts.diagnostics.zegar_band_position_test import row_mapping, row_prior_points


def test_row_mapping_sorts_and_drops_non_tissue_rows():
    dec_t = np.array([0.5, 0.1, 0.9, 0.3])
    valid = np.array([1.0, 1.0, 0.2, 1.0])
    axis_t = np.array([0.4, 0.1, 0.8, 0.35])
    ct, at = row_mapping(dec_t, valid, axis_t)
    assert list(ct) == [0.1, 0.3, 0.5] and list(at) == [0.1, 0.35, 0.4]


def test_row_prior_follows_the_recipient_geometry():
    # every donor has its ring at contour t = 0.8; recipient's axis is stretched (axis = contour / 2)
    ct = np.linspace(0, 1, 21)
    maps = {f"d{i}": (ct, ct.copy()) for i in range(5)}
    gt_axis = {f"d{i}": np.array([0.8]) for i in range(5)}
    maps["r"] = (ct, ct / 2)
    gt_axis["r"] = np.array([0.4])
    pts = row_prior_points(maps, gt_axis, "r", 1)
    assert abs(pts[0] - 0.4) < 0.02


def test_count_conditioned_prior_uses_only_similar_counts():
    from scripts.diagnostics.zegar_band_position_test import row_prior_points_k
    ct = np.linspace(0, 1, 41)
    maps, gt = {}, {}
    for i in range(6):                       # young fish: 2 rings near the edge
        maps[f"y{i}"] = (ct, ct.copy()); gt[f"y{i}"] = np.array([0.7, 0.9])
    for i in range(6):                       # old fish: 6 rings spread out
        maps[f"o{i}"] = (ct, ct.copy()); gt[f"o{i}"] = np.array([0.2, 0.4, 0.55, 0.7, 0.82, 0.92])
    maps["r"] = (ct, ct.copy()); gt["r"] = np.array([0.7, 0.9])
    pts = sorted(row_prior_points_k(maps, gt, "r", 2))
    assert abs(pts[0] - 0.7) < 0.03 and abs(pts[1] - 0.9) < 0.03
