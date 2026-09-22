"""Wedge-aware localisation data for the report cards (22.09).

Why this module exists
----------------------
A wedge/bands-trained checkpoint learned its density head on the POLAR WEDGE canvas
(``src.wedge_extraction``), not on the square 518px image. Until now the report's card path
(``scripts/run_pipeline.py::_compute_axis_data_for_samples``) only ever asked the model for
``get_density_probs(square_image_tensor, ...)`` — i.e. it fed a wedge-trained head an input from a
geometry it never saw. Both wedge configs say so in their own headers ("karty report.html NIE są
świadome gałęzi wycinka/pasm — ufać tylko predictions.csv"), which is exactly the report the user
asked to stop shipping.

This module closes that gap: it runs the density head on the geometry it was TRAINED on, decodes
the peaks there, maps them back to real pixels, and — crucially — also re-projects the wedge
density into an image-space ``(H_p, W_p)`` grid, so every downstream step that already exists
(axis profile, ``density_peaks``, ``fuse_increments``, the DP walkthrough) keeps working unchanged
and now operates on the real wedge signal instead of a meaningless one.

Nothing here decides anything new about localisation — it only routes the existing decision path
through the correct geometry, and exposes each intermediate stage so the report can DRAW it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.peak_decoding import DEFAULT_MIN_GAP_T, decode_topk_peaks, decode_topk_peaks_real_t
from src.wedge_extraction import (
    _row_frac_to_t,          # private-but-vectorised warp helpers: one implementation, not a copy
    _t_to_row_frac,
    WedgeBandGeometry,
    WedgeGeometry,
    extract_polar_wedge,
    extract_polar_wedge_band,
    extract_polar_wedge_band_validity,
    extract_polar_wedge_validity,
    point_to_wedge_band_xy,
    point_to_wedge_xy,
    wedge_band_geometries_from_axis_info,
    wedge_band_polar_coords,
    wedge_band_xy_to_point,
    wedge_geometry_from_axis_info,
    wedge_polar_coords,
    wedge_xy_to_point,
)


def wedge_enabled(cfg) -> bool:
    """True when this config's density branch is a polar wedge (single canvas or bands)."""
    return bool(getattr(cfg.data, "dual_branch_wedge", False))


def wedge_bands_enabled(cfg) -> bool:
    """True when the wedge is split into angular-resolution bands (09.09 Wariant B)."""
    return wedge_enabled(cfg) and getattr(cfg.data, "wedge_band_edges_t", None) is not None


# ---------------------------------------------------------------------------
# Geometry construction — one entry point for both variants
# ---------------------------------------------------------------------------

def build_geometries(mask: np.ndarray, axis_info: dict, cfg) -> list[WedgeBandGeometry]:
    """Return the wedge geometry/geometries for one sample, ALWAYS as a list of
    :class:`WedgeBandGeometry`.

    The single-canvas variant is expressed as a one-element list spanning the full radial range
    (``row_frac_lo=0.0``, ``row_frac_hi=1.0``) — that is not a hack, it is literally what a
    single canvas is, and it lets every function below have ONE code path instead of two nearly
    identical ones. ``point_to_wedge_band_xy``/``wedge_band_xy_to_point`` reduce exactly to
    ``point_to_wedge_xy``/``wedge_xy_to_point`` for that degenerate range (verified by test).
    """
    patch = cfg.data.patch_size
    if wedge_bands_enabled(cfg):
        return wedge_band_geometries_from_axis_info(
            mask, axis_info, cfg.data.wedge_delta_theta_deg,
            list(cfg.data.wedge_band_edges_t),
            [n * patch for n in cfg.data.wedge_band_n_angle_patches],
            [n * patch for n in cfg.data.wedge_band_n_radius_patches],
        )
    geom = wedge_geometry_from_axis_info(
        mask, axis_info, cfg.data.wedge_delta_theta_deg,
        cfg.data.wedge_n_angle_patches * patch, cfg.data.wedge_n_radius_patches * patch,
    )
    return [WedgeBandGeometry(geom=geom, row_frac_lo=0.0, row_frac_hi=1.0)]


def extract_canvases(
    rgb: np.ndarray, mask: np.ndarray, bands: Sequence[WedgeBandGeometry], cfg,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-band (canvas RGB, per-patch tissue-validity) — the images the backbone actually sees."""
    patch = cfg.data.patch_size
    canvases, validities = [], []
    for band in bands:
        if band.row_frac_lo == 0.0 and band.row_frac_hi == 1.0:
            canvases.append(extract_polar_wedge(rgb, mask, band.geom))
            validities.append(extract_polar_wedge_validity(mask, band.geom, patch))
        else:
            canvases.append(extract_polar_wedge_band(rgb, mask, band))
            validities.append(extract_polar_wedge_band_validity(mask, band, patch))
    return canvases, validities


# ---------------------------------------------------------------------------
# Back-projection: wedge density -> image-space grid
# ---------------------------------------------------------------------------

def _r_at_angles(R_theta: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Vectorised twin of ``wedge_extraction._r_at_angle`` — same index arithmetic, whole arrays
    at once (this runs on every image cell AND every wedge patch of every card)."""
    n = len(R_theta)
    idx = np.round((theta + np.pi) / (2 * np.pi) * n).astype(np.int64) % n
    return R_theta[idx].astype(np.float64)


def wedge_density_to_image_grid(
    band_density: Sequence[np.ndarray], bands: Sequence[WedgeBandGeometry],
    out_h_p: int, out_w_p: int, image_h: int, image_w: int,
) -> np.ndarray:
    """Re-project per-band wedge density into an image-space ``(out_h_p, out_w_p)`` patch grid.

    Two passes, combined with ``max`` — and the second one is not an optimisation, it is a
    correctness requirement discovered by this module's own test:

    * **gather** (image cell -> wedge patch): fills AREA. Every output cell samples its own centre
      pixel in wedge space and takes that patch's density. Alone, this silently drops the density
      of narrow patches — and near the otolith edge the wedge's patches are deliberately much
      NARROWER than an image cell (that is the whole point of the non-uniform radial warp and of
      the angular-resolution bands), so the most informative peaks are exactly the ones a pure
      gather loses.
    * **scatter** (wedge patch -> image cell): preserves PEAKS. Every wedge patch centre is mapped
      forward into the image and written into its covering cell with ``maximum``, so no patch can
      vanish however narrow it is. Alone, this leaves holes wherever one coarse near-nucleus patch
      spans several cells.

    Cells outside the wedge's angular span — most of the otolith, by construction — stay 0.0.
    Nearest-patch throughout, never interpolation: a density value is a property of one patch, and
    smoothing across neighbours would invent structure the model never produced.
    """
    out = np.zeros((out_h_p, out_w_p), dtype=np.float32)
    cell_h, cell_w = image_h / out_h_p, image_w / out_w_p

    ys = (np.arange(out_h_p, dtype=np.float64) + 0.5) * cell_h
    xs = (np.arange(out_w_p, dtype=np.float64) + 0.5) * cell_w
    gx, gy = np.meshgrid(xs, ys)

    geom0 = bands[0].geom
    cx, cy = geom0.centroid
    dtheta = geom0.delta_theta_rad
    half = dtheta / 2.0

    dx, dy = gx - cx, gy - cy
    r_img = np.hypot(dx, dy)
    theta_img = np.arctan2(dy, dx)
    rel_img = (theta_img - geom0.axis_angle_rad + np.pi) % (2 * np.pi) - np.pi
    inside = np.abs(rel_img) <= half + 1e-9
    Rth_img = _r_at_angles(geom0.R_theta, theta_img)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_img = np.where(Rth_img > 1e-6, r_img / Rth_img, 0.0)
    rf_img = _t_to_row_frac(np.clip(t_img, 0.0, 1.0))

    for dens, band in zip(band_density, bands):
        h_p, w_p = dens.shape
        geom = band.geom
        span = max(band.row_frac_hi - band.row_frac_lo, 1e-12)

        # --- gather ---
        in_band = inside & (rf_img >= band.row_frac_lo - 1e-9) & (rf_img <= band.row_frac_hi + 1e-9)
        if in_band.any():
            local = (rf_img[in_band] - band.row_frac_lo) / span
            r_idx = np.clip((local * h_p).astype(np.int64), 0, h_p - 1)
            c_idx = np.clip(((rel_img[in_band] / dtheta + 0.5) * w_p).astype(np.int64), 0, w_p - 1)
            np.maximum.at(out, np.nonzero(in_band), dens[r_idx, c_idx])

        # --- scatter ---
        rows = (np.arange(h_p, dtype=np.float64) + 0.5) / h_p
        cols = (np.arange(w_p, dtype=np.float64) + 0.5) / w_p
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        theta_p = geom.axis_angle_rad + (cc - 0.5) * dtheta
        t_p = _row_frac_to_t(band.row_frac_lo + rr * span)
        Rth_p = _r_at_angles(geom.R_theta, theta_p)
        x_p = cx + t_p * Rth_p * np.cos(theta_p)
        y_p = cy + t_p * Rth_p * np.sin(theta_p)
        j_p = np.clip((x_p / cell_w).astype(np.int64), 0, out_w_p - 1)
        i_p = np.clip((y_p / cell_h).astype(np.int64), 0, out_h_p - 1)
        np.maximum.at(out, (i_p.ravel(), j_p.ravel()), dens.ravel())

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Model forward on the wedge geometry + peak decoding
# ---------------------------------------------------------------------------

def run_density_on_wedge(
    model, canvases: Sequence[np.ndarray], bands: Sequence[WedgeBandGeometry], cfg,
    device=None,
) -> list[np.ndarray]:
    """Run the density head on the geometry it was TRAINED on. Returns one ``(h_p, w_p)`` density
    array per band.

    Bands mode uses ``model.get_density_probs_bands`` (K separate backbone passes, concatenated
    before the head — the 09.09 Wariant B path); the single canvas uses ``get_density_probs`` with
    its own per-patch ``(t, theta)``, which ``radial_attention`` requires to reproduce training
    behaviour (same requirement ``expert_annotation_eval_wedge.py`` already documents).
    """
    import torch
    from PIL import Image as PILImage

    from src.dataset import build_transforms

    transform = build_transforms(1, split="test", wedge=True)
    tensors = [transform(PILImage.fromarray(c)).unsqueeze(0) for c in canvases]
    if device is not None:
        tensors = [t.to(device) for t in tensors]

    patch = cfg.data.patch_size
    t_grids, theta_grids = [], []
    for band in bands:
        if band.row_frac_lo == 0.0 and band.row_frac_hi == 1.0:
            t_g, th_g = wedge_polar_coords(band.geom, patch)
        else:
            t_g, th_g = wedge_band_polar_coords(band, patch)
        t_grids.append(t_g)
        theta_grids.append(th_g)

    def _flat(arr, dtype):
        t = torch.from_numpy(np.ascontiguousarray(arr.reshape(1, -1))).to(dtype)
        return t.to(device) if device is not None else t

    if len(bands) > 1:
        pt = [_flat(g, torch.float32) for g in t_grids]
        pth = [_flat(g, torch.float32) for g in theta_grids]
        pv = [torch.ones_like(x, dtype=torch.bool) for x in pt]
        flat = model.get_density_probs_bands(tensors, polar_t=pt, polar_theta=pth,
                                             polar_valid=pv)[0].detach().cpu().numpy()
        out, off = [], 0
        for g in t_grids:
            n = g.size
            out.append(flat[off:off + n].reshape(g.shape).astype(np.float32))
            off += n
        return out

    g = t_grids[0]
    h_p, w_p = g.shape
    with torch.no_grad():
        grid = model.get_density_probs(
            tensors[0], polar_t=_flat(g, torch.float32),
            polar_theta=_flat(theta_grids[0], torch.float32),
            polar_valid=_flat(np.ones_like(g, dtype=bool), torch.bool),
            patch_grid=(h_p, w_p),
        )[0].detach().cpu().numpy()
    return [np.asarray(grid, dtype=np.float32).reshape(h_p, w_p)]


def decode_wedge_peaks(
    band_density: Sequence[np.ndarray], bands: Sequence[WedgeBandGeometry], cfg, k: int,
    min_gap_t: float = DEFAULT_MIN_GAP_T,
) -> list[dict]:
    """Decode ``k`` increment positions from the wedge density and map each back to real pixels.

    Bands: one greedy top-k over the CONCATENATED sequence in real ``t`` space
    (``decode_topk_peaks_real_t``) — bands complement each other, they are not competing
    candidates, so a peak must not be chosen per-band. Single canvas: the gridded
    ``decode_topk_peaks``, the same decoder ``expert_annotation_eval_wedge.py`` uses.

    Each returned dict carries BOTH coordinate systems, because the report draws in both:
    ``band``/``row``/``col``/``t``/``theta`` (wedge space) and ``x``/``y`` (cropped-image pixels).
    Sorted nucleus-outwards by ``t``.
    """
    patch = cfg.data.patch_size
    peaks: list[dict] = []

    if len(bands) > 1:
        t_flat, th_flat, d_flat, loc = [], [], [], []
        for b_idx, (band, dens) in enumerate(zip(bands, band_density)):
            t_g, th_g = wedge_band_polar_coords(band, patch)
            h_p, w_p = dens.shape
            t_flat.append(t_g.reshape(-1))
            th_flat.append(th_g.reshape(-1))
            d_flat.append(dens.reshape(-1))
            loc.extend((b_idx, r, c) for r in range(h_p) for c in range(w_p))
        t_all = np.concatenate(t_flat)
        d_all = np.concatenate(d_flat)
        th_all = np.concatenate(th_flat)
        for idx in decode_topk_peaks_real_t(d_all, t_all, k, min_gap_t=min_gap_t):
            b_idx, r, c = loc[idx]
            band = bands[b_idx]
            x, y = wedge_band_xy_to_point((c + 0.5) * patch, (r + 0.5) * patch, band)
            peaks.append({"band": b_idx, "row": r, "col": c, "t": float(t_all[idx]),
                          "theta": float(th_all[idx]), "score": float(d_all[idx]),
                          "x": float(x), "y": float(y)})
    else:
        band, dens = bands[0], band_density[0]
        t_g, th_g = wedge_polar_coords(band.geom, patch)
        for r, c in decode_topk_peaks(dens, k):
            x, y = wedge_xy_to_point((c + 0.5) * patch, (r + 0.5) * patch, band.geom)
            peaks.append({"band": 0, "row": int(r), "col": int(c), "t": float(t_g[r, c]),
                          "theta": float(th_g[r, c]), "score": float(dens[r, c]),
                          "x": float(x), "y": float(y)})

    peaks.sort(key=lambda p: p["t"])
    return peaks


# ---------------------------------------------------------------------------
# One call the card path can use
# ---------------------------------------------------------------------------

def build_wedge_card_data(
    model, rgb: np.ndarray, mask: np.ndarray, axis_info: dict, cfg, k: int,
    out_h_p: Optional[int] = None, out_w_p: Optional[int] = None, device=None,
) -> dict:
    """Everything the report needs about the wedge branch for ONE otolith.

    ``rgb``/``mask``/``axis_info`` must all be in the SAME coordinate frame (the cropped single
    otolith, as produced by the card path). ``k`` is the number of increments to decode —
    normally the model's own predicted age, so the wedge branch is asked for exactly as many
    increments as the age head believes exist.

    Returns a dict with the canvases, the per-band density, the decoded peaks (wedge AND pixel
    coordinates), and ``image_density`` — the back-projected grid that lets the rest of the card
    path run unchanged.
    """
    image_h, image_w = rgb.shape[:2]
    if out_h_p is None or out_w_p is None:
        side = cfg.data.image_size // cfg.data.patch_size
        out_h_p = out_h_p or side
        out_w_p = out_w_p or side

    bands = build_geometries(mask, axis_info, cfg)
    canvases, validities = extract_canvases(rgb, mask, bands, cfg)
    band_density = run_density_on_wedge(model, canvases, bands, cfg, device=device)
    peaks = decode_wedge_peaks(band_density, bands, cfg, k)
    image_density = wedge_density_to_image_grid(
        band_density, bands, out_h_p, out_w_p, image_h, image_w)

    return {
        "mode": "bands" if len(bands) > 1 else "single",
        "bands": list(bands),
        "canvases": canvases,
        "validities": validities,
        "band_density": band_density,
        "peaks": peaks,
        "image_density": image_density,
        "delta_theta_deg": float(cfg.data.wedge_delta_theta_deg),
        "k": int(k),
    }
