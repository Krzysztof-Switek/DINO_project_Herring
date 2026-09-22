"""Tests for src/wedge_cards.py — the wedge-aware localisation data behind the report's
decision-path section (22.09).

No DINOv2 here: the density head is stubbed, because what these tests are about is the GEOMETRY
(which patch is where, and does a decoded peak land back on the right pixel), not what a trained
head happens to output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import wedge_cards as wc
from src.wedge_extraction import point_to_wedge_xy, wedge_band_xy_to_point, wedge_xy_to_point


# ---------------------------------------------------------------------------
# Fixtures: a real elliptical otolith mask + axis, and two configs (single / bands)
# ---------------------------------------------------------------------------

def _mask_and_axis(h=300, w=260):
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 255, -1)
    centroid = (float(w // 2), float(h // 2))
    far_edge = (float(w // 2 + w // 3 - 1), float(h // 2))
    return mask, {"centroid": centroid, "far_edge": far_edge, "mask": mask,
                  "length_px": float(w // 3)}


class _Cfg:
    class data:
        patch_size = 14
        image_size = 518
        dual_branch_wedge = True
        wedge_delta_theta_deg = 98.7
        wedge_n_angle_patches = 13
        wedge_n_radius_patches = 44
        wedge_band_edges_t = None
        wedge_band_n_angle_patches = None
        wedge_band_n_radius_patches = None


def _cfg_single():
    class C(_Cfg):
        pass
    return C


def _cfg_bands():
    class C(_Cfg):
        class data(_Cfg.data):
            wedge_band_edges_t = [0.0, 0.6, 0.8, 0.9, 1.0]
            wedge_band_n_angle_patches = [9, 11, 12, 13]      # small, same SHAPE as production
            wedge_band_n_radius_patches = [11, 12, 9, 12]
    return C


class _StubModel:
    """Density head stand-in: returns a deterministic, position-dependent map so a test can
    assert WHICH patch the decoder picked, not just that it picked something."""

    def __init__(self, hot=None):
        self.hot = hot          # (band_idx, row, col) to make the single brightest patch
        self.calls = []

    def _maps(self, shapes):
        out = []
        for b, shp in enumerate(shapes):
            a = np.full(shp, 0.01, dtype=np.float32)
            if self.hot is not None and self.hot[0] == b:
                a[self.hot[1], self.hot[2]] = 0.99
            out.append(a)
        return out

    # single-canvas API
    def get_density_probs(self, image, polar_t=None, polar_theta=None, polar_valid=None,
                          patch_grid=None):
        import torch
        self.calls.append(("single", tuple(image.shape)))
        h_p, w_p = patch_grid
        return torch.from_numpy(self._maps([(h_p, w_p)])[0][None, ...])

    # bands API
    def get_density_probs_bands(self, images, polar_t=None, polar_theta=None, polar_valid=None):
        import torch
        self.calls.append(("bands", [tuple(i.shape) for i in images]))
        shapes = [(int(t.shape[1] // self._w[i]), self._w[i]) for i, t in enumerate(polar_t)]
        flat = np.concatenate([m.reshape(-1) for m in self._maps(shapes)])
        return torch.from_numpy(flat[None, ...])


# ---------------------------------------------------------------------------
# build_geometries
# ---------------------------------------------------------------------------

def test_single_canvas_is_expressed_as_one_full_range_band():
    mask, axis = _mask_and_axis()
    bands = wc.build_geometries(mask, axis, _cfg_single())
    assert len(bands) == 1
    assert bands[0].row_frac_lo == 0.0 and bands[0].row_frac_hi == 1.0
    assert bands[0].geom.canvas_w == 13 * 14
    assert bands[0].geom.canvas_h == 44 * 14


def test_bands_config_builds_one_geometry_per_band_with_growing_width():
    mask, axis = _mask_and_axis()
    bands = wc.build_geometries(mask, axis, _cfg_bands())
    assert len(bands) == 4
    widths = [b.geom.canvas_w for b in bands]
    assert widths == sorted(widths) and widths[0] < widths[-1], widths
    # bands tile the radius without gaps or overlap
    for prev, nxt in zip(bands, bands[1:]):
        assert prev.row_frac_hi == pytest.approx(nxt.row_frac_lo)
    assert bands[0].row_frac_lo == pytest.approx(0.0)
    assert bands[-1].row_frac_hi == pytest.approx(1.0)


def test_full_range_band_mapping_equals_plain_wedge_mapping():
    """The one-element-list trick in build_geometries is only legitimate if a full-range band
    maps points EXACTLY like the plain single-canvas functions. Assert that, don't assume it."""
    mask, axis = _mask_and_axis()
    band = wc.build_geometries(mask, axis, _cfg_single())[0]
    for col, row in ((0.0, 0.0), (50.0, 100.0), (band.geom.canvas_w - 1, band.geom.canvas_h - 1)):
        a = wedge_xy_to_point(col, row, band.geom)
        b = wedge_band_xy_to_point(col, row, band)
        assert a == pytest.approx(b, abs=1e-9)


# ---------------------------------------------------------------------------
# Peak decoding round-trip — the property the whole report rests on
# ---------------------------------------------------------------------------

def test_decoded_peak_maps_back_to_the_pixel_it_came_from_single():
    mask, axis = _mask_and_axis()
    cfg = _cfg_single()
    bands = wc.build_geometries(mask, axis, cfg)
    dens = np.full((44, 13), 0.01, dtype=np.float32)
    dens[30, 6] = 0.99
    peaks = wc.decode_wedge_peaks([dens], bands, cfg, k=1)
    assert len(peaks) == 1
    p = peaks[0]
    assert (p["row"], p["col"]) == (30, 6)
    # forward map (already asserted above) then INVERSE map must return the same patch centre
    back = point_to_wedge_xy(p["x"], p["y"], bands[0].geom)
    assert back is not None
    col_back, row_back, _t = back
    assert col_back == pytest.approx((6 + 0.5) * 14, abs=1e-6)
    assert row_back == pytest.approx((30 + 0.5) * 14, abs=1e-6)


def test_decoded_peaks_are_sorted_nucleus_outwards_and_respect_min_gap():
    mask, axis = _mask_and_axis()
    cfg = _cfg_bands()
    bands = wc.build_geometries(mask, axis, cfg)
    dens = []
    for b, band in enumerate(bands):
        h_p = band.geom.canvas_h // 14
        w_p = band.geom.canvas_w // 14
        a = np.full((h_p, w_p), 0.01, dtype=np.float32)
        a[h_p // 2, w_p // 2] = 0.5 + 0.1 * b        # one clear peak per band
        dens.append(a)
    peaks = wc.decode_wedge_peaks(dens, bands, cfg, k=4)
    assert len(peaks) == 4
    ts = [p["t"] for p in peaks]
    assert ts == sorted(ts), ts
    assert len({p["band"] for p in peaks}) == 4      # one from each band, they complement
    for a, b in zip(ts, ts[1:]):
        assert b - a >= wc.DEFAULT_MIN_GAP_T - 1e-9


def test_decode_respects_k_zero():
    mask, axis = _mask_and_axis()
    cfg = _cfg_single()
    bands = wc.build_geometries(mask, axis, cfg)
    assert wc.decode_wedge_peaks([np.ones((44, 13), np.float32)], bands, cfg, k=0) == []


# ---------------------------------------------------------------------------
# Back-projection into image space
# ---------------------------------------------------------------------------

def test_back_projection_is_zero_outside_the_wedge_and_hot_inside():
    mask, axis = _mask_and_axis()
    cfg = _cfg_single()
    bands = wc.build_geometries(mask, axis, cfg)
    dens = np.full((44, 13), 0.5, dtype=np.float32)
    h, w = mask.shape
    grid = wc.wedge_density_to_image_grid([dens], bands, 37, 37, h, w)

    assert grid.shape == (37, 37)
    assert grid.max() == pytest.approx(0.5)
    assert (grid == 0.0).any(), "a 98.7-degree wedge cannot cover the whole image"
    covered = float((grid > 0).mean())
    # 98.7 deg of 360 is ~27% of a full disc, and the disc itself is a fraction of the bbox
    assert 0.02 < covered < 0.35, covered


def test_back_projection_puts_a_hot_patch_on_the_correct_side_of_the_axis():
    """A peak decoded at a known wedge position must light up image cells NEAR that pixel —
    the property that makes the re-projected grid usable by the existing card machinery."""
    mask, axis = _mask_and_axis()
    cfg = _cfg_single()
    bands = wc.build_geometries(mask, axis, cfg)
    dens = np.zeros((44, 13), dtype=np.float32)
    dens[40, 6] = 1.0                                  # near the edge, on the axis column
    h, w = mask.shape
    grid = wc.wedge_density_to_image_grid([dens], bands, 37, 37, h, w)

    assert grid.max() == pytest.approx(1.0)
    hot_i, hot_j = np.unravel_index(int(np.argmax(grid)), grid.shape)
    hot_x = (hot_j + 0.5) * (w / 37)
    hot_y = (hot_i + 0.5) * (h / 37)
    x_true, y_true = wedge_xy_to_point((6 + 0.5) * 14, (40 + 0.5) * 14, bands[0].geom)
    cell_diag = np.hypot(w / 37, h / 37)
    assert np.hypot(hot_x - x_true, hot_y - y_true) <= cell_diag, (hot_x, hot_y, x_true, y_true)


def test_bands_back_projection_covers_more_radius_than_any_single_band():
    mask, axis = _mask_and_axis()
    cfg = _cfg_bands()
    bands = wc.build_geometries(mask, axis, cfg)
    dens = [np.full((b.geom.canvas_h // 14, b.geom.canvas_w // 14), 0.4, np.float32)
            for b in bands]
    h, w = mask.shape
    full = wc.wedge_density_to_image_grid(dens, bands, 37, 37, h, w)

    only_last = [np.zeros_like(d) for d in dens]
    only_last[-1][:] = 0.4
    partial = wc.wedge_density_to_image_grid(only_last, bands, 37, 37, h, w)
    assert (full > 0).sum() > (partial > 0).sum()


# ---------------------------------------------------------------------------
# Canvas extraction + end-to-end shape contract
# ---------------------------------------------------------------------------

def test_extract_canvases_shapes_match_geometry_for_both_modes():
    mask, axis = _mask_and_axis()
    rgb = np.dstack([mask] * 3)
    for cfg in (_cfg_single(), _cfg_bands()):
        bands = wc.build_geometries(mask, axis, cfg)
        canvases, valids = wc.extract_canvases(rgb, mask, bands, cfg)
        assert len(canvases) == len(bands) == len(valids)
        for canvas, valid, band in zip(canvases, valids, bands):
            assert canvas.shape == (band.geom.canvas_h, band.geom.canvas_w, 3)
            assert valid.shape == (band.geom.canvas_h // 14, band.geom.canvas_w // 14)
            assert valid.min() >= 0.0 and valid.max() <= 1.0


def test_wedge_enabled_helpers_read_the_config():
    assert wc.wedge_enabled(_cfg_single()) is True
    assert wc.wedge_bands_enabled(_cfg_single()) is False
    assert wc.wedge_bands_enabled(_cfg_bands()) is True

    class Off:
        class data:
            dual_branch_wedge = False
            wedge_band_edges_t = None
    assert wc.wedge_enabled(Off()) is False
    assert wc.wedge_bands_enabled(Off()) is False


# ---------------------------------------------------------------------------
# Renderers (src/visualization.py) — geometry-only, no model
# ---------------------------------------------------------------------------

def test_render_wedge_canvas_panel_marks_every_peak_and_flips_for_display():
    from src.visualization import render_wedge_canvas_panel

    canvases = [np.full((11 * 14, 9 * 14, 3), 90, np.uint8),
                np.full((12 * 14, 13 * 14, 3), 90, np.uint8)]
    dens = [np.zeros((11, 9), np.float32), np.zeros((12, 13), np.float32)]
    dens[1][10, 6] = 1.0
    peaks = [{"band": 1, "row": 10, "col": 6, "t": 0.95, "theta": 0.0, "score": 1.0,
              "x": 0.0, "y": 0.0}]

    panel = render_wedge_canvas_panel(canvases, dens, peaks, patch_size=14)
    assert panel.ndim == 3 and panel.shape[2] == 3
    assert panel.shape[1] >= 13 * 14                 # widest band drives the width
    red = ((panel[..., 0] > 180) & (panel[..., 1] < 90) & (panel[..., 2] < 90))
    assert red.sum() > 0, "zdekodowany pik nie został narysowany"

    # flip is display-only: the marked peak (row 10 of 12, near the edge) must sit in the
    # UPPER half when flipped and the LOWER half when not.
    ys_flip = np.nonzero(red.any(axis=1))[0]
    panel_noflip = render_wedge_canvas_panel(canvases, dens, peaks, patch_size=14,
                                             flip_display=False)
    red2 = ((panel_noflip[..., 0] > 180) & (panel_noflip[..., 1] < 90) & (panel_noflip[..., 2] < 90))
    ys_noflip = np.nonzero(red2.any(axis=1))[0]
    assert ys_flip.mean() != pytest.approx(ys_noflip.mean(), abs=5)


def test_render_wedge_canvas_panel_survives_empty_input():
    from src.visualization import render_wedge_canvas_panel
    panel = render_wedge_canvas_panel([])
    assert panel.ndim == 3 and panel.shape[0] > 0


def test_render_wedge_sector_on_photo_draws_sector_and_numbered_peaks():
    import cv2
    from src.visualization import render_wedge_sector_on_photo

    mask, axis = _mask_and_axis()
    bands = wc.build_geometries(mask, axis, _cfg_single())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    axis["contour"] = max(contours, key=cv2.contourArea)
    photo = np.full((*mask.shape, 3), 160, np.uint8)

    peaks = [{"band": 0, "row": 20, "col": 6, "t": 0.5, "theta": 0.0, "score": 0.7,
              "x": axis["centroid"][0] + 30, "y": axis["centroid"][1]}]
    out = render_wedge_sector_on_photo(photo, axis, bands[0].geom, peaks=peaks)

    assert out.shape == photo.shape
    magenta = ((out[..., 0] > 200) & (out[..., 1] < 170) & (out[..., 2] > 180))
    assert magenta.sum() > 50, "sektor klina nie został narysowany"
    red = ((out[..., 0] > 180) & (out[..., 1] < 90) & (out[..., 2] < 90))
    assert red.sum() > 0, "pik nie został narysowany na zdjęciu"


def test_render_wedge_sector_handles_missing_geometry():
    from src.visualization import render_wedge_sector_on_photo
    photo = np.full((80, 90, 3), 100, np.uint8)
    out = render_wedge_sector_on_photo(photo, None, None)
    assert np.array_equal(out, photo)
