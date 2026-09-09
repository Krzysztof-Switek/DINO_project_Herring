"""Polar-wedge extraction — Daugman rubber-sheet unwrap restricted to an angular sector around
the otolith's reading axis, instead of the full 360 degrees (09.09, `plans and summaries/
09.09_pasek_maskowanie_wyniki_i_literatura.md` follow-up; geometry measured in
`scripts/diagnostics/analyze_zegar_wedge_geometry.py` / prototyped in
`scripts/diagnostics/prototype_polar_wedge.py`, both promoted here unchanged in behaviour).

Why this exists (not a rebuild of `strip_extraction.py`): the straightened rectangular strip
warps ONE straight line + a fixed pixel margin — no notion of multiple directions. Run N's own
advantage comes partly from casting 48 rays from the nucleus, each ending exactly on the
segmentation contour (so it structurally cannot sample background — verified 0/3072 ray-sampled
points outside the mask, `plans and summaries/` Run N mechanism report). A polar wedge unwrap
gives the SAME property (every column's row range spans nucleus to that column's own true
boundary, per-angle radius R(theta) — never past it) while producing ONE new image per sample
(one backbone forward pass, like the strip — not one per candidate axis, like Track B's rejected
multi-strip approach), because every column of the SAME wedge image is a different real angular
direction.

Literature this borrows from (verified via WebSearch, not from memory — see the 09.09 report
series): Daugman's iris "rubber sheet model" (annulus (x,y) -> rectangular (r,theta)); DarSwin
(Athwale et al., ICCV 2023, arXiv:2304.09691) "polar patch partitioning" — patches defined
natively in (angle, radius) so none fall outside the sector; Gillert et al. (CVPR 2023) polar-grid
sampling for tree-ring instance segmentation, confirming polar representations are an active
direction for exactly this class of concentric-ring problem, not just iris biometrics.

Geometry (`R(theta)` per-angle boundary radius) reuses `otolith_axis.compute_R_theta` — the SAME
ray-casting `compute_polar_grid` already uses for the E9/RadialAttentionDensityHead positional
prior, promoted there (09.09) specifically so this module does not duplicate it.

Radial (row->t) sampling is NON-UNIFORM (09.09 follow-up, see the `_RADIAL_WARP_T` comment below
and `scripts/diagnostics/analyze_wedge_radial_resolution_profile.py`): real ZEGAR annotations show
ring spacing shrinks sharply toward the edge (Pearson r=-0.53, zero measured increment pairs below
t=0.5), so canvas rows are denser near t=1 than near t=0 rather than evenly spaced.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import cv2
import numpy as np
from PIL import Image as PILImage

from src.otolith_axis import MASK_FILL_RGB, apply_background_mask, compute_R_theta


# Non-uniform radial-resolution warp (09.09 follow-up — see `plans and summaries/
# 09.09_wycinek_katowy_plan.md` addendum and
# `scripts/diagnostics/analyze_wedge_radial_resolution_profile.py`). Real ZEGAR ring-spacing
# measurements (335 consecutive-increment gaps, `outputs/02.09_zegar_ring_spacing/ring_gaps.csv`
# — the same pool the strip's own resolution was derived from) show delta_t shrinks sharply with
# radius (Pearson r=-0.53): zero of the 335 measured gaps even fall below t=0.5, while the
# tightest real gaps concentrate in t>0.85. A LINEAR row->t mapping applies ONE global worst-case
# density everywhere, wasting resolution near the nucleus and, for the very tightest real gaps,
# under-resolving the outer edge relative to what a LOCAL threshold would require. These
# breakpoints are a monotonic piecewise-linear warp whose local row-density satisfies the same
# MARGIN_PATCHES=2-patches-of-separation rule `analyze_zegar_ring_spacing.py` used for the strip,
# applied per radial zone (p1 of delta_t within the zone, then a running cumulative-min
# left-to-right to enforce the expected monotonic tightening) instead of once globally.
_RADIAL_WARP_T = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0], dtype=np.float64)
_RADIAL_WARP_ROW_FRAC = np.array(
    [0.0, 0.187395, 0.24838, 0.370305, 0.519843, 0.599422, 0.732948, 0.866474, 1.0],
    dtype=np.float64,
)


def _row_frac_to_t(row_frac: np.ndarray) -> np.ndarray:
    """Canvas row position (0..1, endpoint-inclusive) -> normalised radius t (0..1). Denser rows
    near t=1 than near t=0 — see the module-level warp comment above."""
    return np.interp(row_frac, _RADIAL_WARP_ROW_FRAC, _RADIAL_WARP_T)


def _t_to_row_frac(t: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_row_frac_to_t` (both arrays are monotonic increasing, so swapping the
    interpolation axes is an exact inverse of the piecewise-linear warp)."""
    return np.interp(t, _RADIAL_WARP_T, _RADIAL_WARP_ROW_FRAC)


class WedgeGeometry(NamedTuple):
    """Everything needed to invert a polar-wedge extraction (point <-> canvas position), or to
    derive each canvas patch's own (t, theta) for a density head's positional encoding — kept as
    plain floats/arrays (not the whole ``axis_info`` dict) so it round-trips through caching
    (``.npz``) without pulling in ``mask``/``contour`` objects that don't belong in a cache file.
    """
    R_theta: np.ndarray          # (n_angle_raycast,) float32, otolith boundary radius per angle bin
    centroid: tuple[float, float]
    axis_angle_rad: float        # angle of (far_edge - centroid), the wedge's own centre direction
    delta_theta_rad: float       # full angular width of the wedge
    canvas_w: int                # angle columns
    canvas_h: int                # radius rows


def _r_at_angle(R_theta: np.ndarray, theta_rad: float) -> float:
    n = len(R_theta)
    idx = int(round((theta_rad + np.pi) / (2 * np.pi) * n)) % n
    return float(R_theta[idx])


def wedge_geometry_from_axis_info(
    mask: np.ndarray, axis_info: dict, delta_theta_deg: float,
    canvas_w: int, canvas_h: int, n_angle_raycast: int = 720,
) -> WedgeGeometry:
    """Build a :class:`WedgeGeometry` from ``otolith_axis.detect_axis``'s return dict — the
    wedge is always centred on the EXISTING reading axis (``centroid`` -> ``far_edge``), never a
    separately-chosen direction (unlike Track B's multiple independent candidate axes)."""
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    axis_angle = float(np.arctan2(fy - cy, fx - cx))
    R_theta = compute_R_theta(mask, (cx, cy), n_angle_raycast)
    return WedgeGeometry(
        R_theta=R_theta.astype(np.float32), centroid=(float(cx), float(cy)),
        axis_angle_rad=axis_angle, delta_theta_rad=float(np.radians(delta_theta_deg)),
        canvas_w=int(canvas_w), canvas_h=int(canvas_h),
    )


def _remap_maps(
    geom: WedgeGeometry, row_frac_lo: float = 0.0, row_frac_hi: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(map_x, map_y) float32 arrays for ``cv2.remap``, mapping every wedge canvas cell back to
    its source pixel in the original (cropped) image.

    ``row_frac_lo``/``row_frac_hi`` (09.09, angular-bands follow-up — default ``0.0``/``1.0``,
    byte-identical to every existing single-band caller): restrict this canvas's rows to a
    SUB-RANGE of the global non-uniform radial warp's own ``row_frac`` domain, instead of the
    full ``[0, 1]``. Lets a "band" canvas (e.g. ``WedgeBandGeometry``'s own ``geom``, which has its
    own, band-specific ``canvas_h``) reuse the EXACT SAME ``_row_frac_to_t`` warp function as the
    single-band wedge, just evaluated over the slice of it this band is responsible for.
    """
    cx, cy = geom.centroid
    col_idx = np.arange(geom.canvas_w, dtype=np.float64)
    thetas = geom.axis_angle_rad + (col_idx / max(geom.canvas_w - 1, 1) - 0.5) * geom.delta_theta_rad
    row_idx = np.arange(geom.canvas_h, dtype=np.float64)
    local_frac = row_idx / max(geom.canvas_h - 1, 1)
    global_row_frac = row_frac_lo + local_frac * (row_frac_hi - row_frac_lo)
    ts = _row_frac_to_t(global_row_frac)

    theta_grid = np.tile(thetas[None, :], (geom.canvas_h, 1))
    t_grid = np.tile(ts[:, None], (1, geom.canvas_w))

    n_bins = len(geom.R_theta)
    bin_idx = np.clip(((thetas + np.pi) / (2 * np.pi) * n_bins).astype(np.int64), 0, n_bins - 1)
    R_cols = geom.R_theta[bin_idx]
    R_grid = np.tile(R_cols[None, :], (geom.canvas_h, 1))

    map_x = (cx + t_grid * R_grid * np.cos(theta_grid)).astype(np.float32)
    map_y = (cy + t_grid * R_grid * np.sin(theta_grid)).astype(np.float32)
    return map_x, map_y


def extract_polar_wedge(rgb: np.ndarray, mask: np.ndarray, geom: WedgeGeometry) -> np.ndarray:
    """Daugman-style rubber-sheet unwrap of ``rgb``, restricted to the wedge described by
    ``geom``. Masks the background BEFORE remapping (same convention as ``extract_strip``) so
    any cell whose source falls outside the segmented tissue is filled with ``MASK_FILL_RGB`` —
    never black (register-token/high-contrast-edge artifact this project already fixed once, see
    ``otolith_axis.MASK_FILL_RGB``'s own docstring).

    Returns ``(canvas_h, canvas_w, 3)`` uint8.
    """
    masked = apply_background_mask(rgb, mask)
    map_x, map_y = _remap_maps(geom)
    return cv2.remap(masked, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=MASK_FILL_RGB)


def extract_polar_wedge_validity(mask: np.ndarray, geom: WedgeGeometry, patch_size: int) -> np.ndarray:
    """(h_p, w_p) float32 in [0, 1] — fraction of each wedge patch lying inside ``mask``, mirrors
    ``strip_extraction.extract_strip_validity`` exactly (nearest-neighbour remap of the BINARY
    mask, borderValue=0, then exact patch-average pooling)."""
    mask_bin = (np.asarray(mask) > 0).astype(np.uint8)
    map_x, map_y = _remap_maps(geom)
    warped = cv2.remap(mask_bin, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    h_p, w_p = geom.canvas_h // patch_size, geom.canvas_w // patch_size
    return (
        warped.reshape(h_p, patch_size, w_p, patch_size)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def wedge_polar_coords(geom: WedgeGeometry, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-PATCH ``(t, theta)`` for the wedge's own canvas — trivially known from the extraction's
    own construction (each column IS a specific angle, each row IS a specific normalised radius,
    non-uniformly warped — see the module-level ``_RADIAL_WARP_*`` comment), unlike
    ``compute_polar_grid``'s square-image case which has to ESTIMATE these from segmentation.
    Returns radians, same convention as ``compute_polar_grid``'s ``theta_grid`` (absolute
    image-space angle via ``atan2``, NOT wedge-relative) so ``RadialAttentionDensityHead``'s
    existing Fourier positional encoding consumes it unchanged.

    Returns ``(t_grid, theta_grid)``, both ``(h_p, w_p)`` float32 — no ``valid_grid`` here (that's
    :func:`extract_polar_wedge_validity`, which needs the real mask, not just the wedge's own
    construction).
    """
    h_p, w_p = geom.canvas_h // patch_size, geom.canvas_w // patch_size
    col_centers = (np.arange(w_p, dtype=np.float64) + 0.5) / max(w_p, 1)
    thetas = geom.axis_angle_rad + (col_centers - 0.5) * geom.delta_theta_rad
    row_centers = (np.arange(h_p, dtype=np.float64) + 0.5) / max(h_p, 1)
    t_centers = _row_frac_to_t(row_centers)
    t_grid = np.tile(t_centers[:, None], (1, w_p)).astype(np.float32)
    theta_grid = np.tile(thetas[None, :], (h_p, 1)).astype(np.float32)
    return t_grid, theta_grid


def point_to_wedge_xy(x: float, y: float, geom: WedgeGeometry) -> Optional[tuple[float, float, float]]:
    """Inverse mapping: real ``(x, y)`` (same frame as the ORIGINAL cropped image) -> ``(col, row,
    t)`` in wedge-canvas pixel space, or ``None`` when the point's angle falls outside the wedge's
    ``delta_theta_rad`` (a real, expected outcome — see the 09.09 wedge-geometry report's Z15
    case, not an error)."""
    cx, cy = geom.centroid
    dx, dy = x - cx, y - cy
    r = float(np.hypot(dx, dy))
    theta = float(np.arctan2(dy, dx))
    rel = theta - geom.axis_angle_rad
    rel = (rel + np.pi) % (2 * np.pi) - np.pi
    half = geom.delta_theta_rad / 2
    if abs(rel) > half + 1e-9:  # tolerance for float round-trip error at the exact boundary column
        return None
    col = (rel / geom.delta_theta_rad + 0.5) * (geom.canvas_w - 1)
    Rth = _r_at_angle(geom.R_theta, theta)
    t = r / Rth if Rth > 1e-6 else 0.0
    row = float(_t_to_row_frac(t)) * (geom.canvas_h - 1)
    return col, row, t


def wedge_xy_to_point(col: float, row: float, geom: WedgeGeometry) -> tuple[float, float]:
    """Forward mapping: wedge-canvas ``(col, row)`` -> real ``(x, y)`` in the original cropped
    image's frame. Inverse of :func:`point_to_wedge_xy` (round-trip verified to <1e-6px in
    ``scripts/diagnostics/prototype_polar_wedge.py`` and in this module's own unit tests)."""
    cx, cy = geom.centroid
    theta = geom.axis_angle_rad + (col / max(geom.canvas_w - 1, 1) - 0.5) * geom.delta_theta_rad
    t = float(_row_frac_to_t(row / max(geom.canvas_h - 1, 1)))
    Rth = _r_at_angle(geom.R_theta, theta)
    x = cx + t * Rth * np.cos(theta)
    y = cy + t * Rth * np.sin(theta)
    return x, y


# ---------------------------------------------------------------------------
# Angular-resolution bands (09.09 follow-up, `plans and summaries/
# 09.09_wycinek_pasma_katowe_plan.md`) — a SEPARATE, genuinely new architectural experiment, NOT a
# literature-precedented mechanism (see that plan's literature section: DarSwin/Daugman/Gillert all
# have the identical fixed-angular-column limitation this addresses, none solve it). Measured
# problem: with a fixed column count, one column's real arc length grows linearly with radius
# (`scripts/diagnostics/analyze_wedge_angular_resolution.py`, 42 ZEGAR images) — mean compression
# 5.49x at the edge (max 8.57x), already >1x for ALL 42 images by t=0.3. Fix: split the radius into
# bands (reusing the existing non-uniform radial warp's own zone structure), each band rendered as
# its own small wedge canvas with its OWN, wider-near-the-edge column count — sized from a REAL,
# broad sample of otolith sizes (`analyze_real_otolith_size_distribution.py`, 400 real Embedded
# images, true max=1305.1px — NOT the smaller 42-image ZEGAR sample, which would have understated
# the true population maximum by 44%).
#
# Each band is extracted with the SAME `_remap_maps`/`extract_polar_wedge`/`wedge_polar_coords`
# machinery above, just restricted to that band's own `row_frac` sub-range of the global 44-row
# warp (via `WedgeBandGeometry.row_frac_lo/hi`) and given its own, band-specific `canvas_w`. No
# padding to a common shape across bands (deliberately -- see the plan's Faza 8 revision): each
# band has a FIXED shape across every sample, so a plain `DataLoader` handles 4 separate per-band
# fields with zero collation changes, and the model runs 4 separate (not stacked) backbone forward
# passes, one per band, avoiding the wasted compute a common padded shape would cost the narrower
# bands.
# ---------------------------------------------------------------------------

class WedgeBandGeometry(NamedTuple):
    """One radial band's geometry — its own ``WedgeGeometry`` (shares ``R_theta``/``centroid``/
    ``axis_angle_rad``/``delta_theta_rad`` with every other band of the same sample; has its OWN
    ``canvas_w``/``canvas_h``) plus the ``row_frac`` sub-range of the GLOBAL non-uniform radial warp
    (``_RADIAL_WARP_T``/``_RADIAL_WARP_ROW_FRAC``) this band is responsible for."""
    geom: WedgeGeometry
    row_frac_lo: float
    row_frac_hi: float


def wedge_band_row_frac_edges(band_edges_t: Sequence[float]) -> np.ndarray:
    """``band_edges_t`` (radius, e.g. ``[0.0, 0.6, 0.8, 0.9, 1.0]``) -> the corresponding
    ``row_frac`` breakpoints of the shared global radial warp. Depends ONLY on the (fixed,
    config-derived) band edges — not on any per-image geometry — so a caller can compute this
    ONCE (e.g. in ``OtolithDataset.__init__``) and reuse it on every sample, including a wedge-
    image CACHE HIT where no per-image ``axis_info`` is available/needed at all."""
    return _t_to_row_frac(np.asarray(band_edges_t, dtype=np.float64))


def wedge_band_geometries_from_axis_info(
    mask: np.ndarray, axis_info: dict, delta_theta_deg: float,
    band_edges_t: Sequence[float], band_canvas_w: Sequence[int], band_canvas_h: Sequence[int],
    n_angle_raycast: int = 720,
) -> list[WedgeBandGeometry]:
    """Build all bands for one sample from a single shared ray-cast (`compute_R_theta` run ONCE,
    not once per band) — mirrors :func:`wedge_geometry_from_axis_info`, generalised to N bands.

    ``band_edges_t``: monotonic increasing, e.g. ``[0.0, 0.6, 0.8, 0.9, 1.0]`` — length K+1 for K
    bands. ``band_canvas_w``/``band_canvas_h``: length K, pixel width/height of each band's own
    canvas (``= n_angle_patches_k * patch_size`` / ``= n_radius_patches_k * patch_size``).
    """
    n_bands = len(band_edges_t) - 1
    if len(band_canvas_w) != n_bands or len(band_canvas_h) != n_bands:
        raise ValueError(
            f"band_edges_t implies {n_bands} bands but got "
            f"{len(band_canvas_w)} widths / {len(band_canvas_h)} heights"
        )
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    axis_angle = float(np.arctan2(fy - cy, fx - cx))
    R_theta = compute_R_theta(mask, (cx, cy), n_angle_raycast).astype(np.float32)
    delta_theta_rad = float(np.radians(delta_theta_deg))
    row_frac_edges = wedge_band_row_frac_edges(band_edges_t)

    bands = []
    for i in range(n_bands):
        geom = WedgeGeometry(
            R_theta=R_theta, centroid=(float(cx), float(cy)), axis_angle_rad=axis_angle,
            delta_theta_rad=delta_theta_rad,
            canvas_w=int(band_canvas_w[i]), canvas_h=int(band_canvas_h[i]),
        )
        bands.append(WedgeBandGeometry(
            geom=geom, row_frac_lo=float(row_frac_edges[i]), row_frac_hi=float(row_frac_edges[i + 1]),
        ))
    return bands


def extract_polar_wedge_band(rgb: np.ndarray, mask: np.ndarray, band: WedgeBandGeometry) -> np.ndarray:
    """Same warp as :func:`extract_polar_wedge`, restricted to ``band``'s own row-fraction range
    and column count. Returns ``(band.geom.canvas_h, band.geom.canvas_w, 3)`` uint8."""
    masked = apply_background_mask(rgb, mask)
    map_x, map_y = _remap_maps(band.geom, band.row_frac_lo, band.row_frac_hi)
    return cv2.remap(masked, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=MASK_FILL_RGB)


def extract_polar_wedge_band_validity(
    mask: np.ndarray, band: WedgeBandGeometry, patch_size: int,
) -> np.ndarray:
    """Mirrors :func:`extract_polar_wedge_validity` for one band."""
    mask_bin = (np.asarray(mask) > 0).astype(np.uint8)
    map_x, map_y = _remap_maps(band.geom, band.row_frac_lo, band.row_frac_hi)
    warped = cv2.remap(mask_bin, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    geom = band.geom
    h_p, w_p = geom.canvas_h // patch_size, geom.canvas_w // patch_size
    return (
        warped.reshape(h_p, patch_size, w_p, patch_size)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def wedge_band_polar_coords(band: WedgeBandGeometry, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirrors :func:`wedge_polar_coords` for one band — per-patch ``(t, theta)``, ``t`` resolved
    through the band's own ``row_frac`` sub-range of the shared global radial warp."""
    geom = band.geom
    h_p, w_p = geom.canvas_h // patch_size, geom.canvas_w // patch_size
    col_centers = (np.arange(w_p, dtype=np.float64) + 0.5) / max(w_p, 1)
    thetas = geom.axis_angle_rad + (col_centers - 0.5) * geom.delta_theta_rad
    row_centers = (np.arange(h_p, dtype=np.float64) + 0.5) / max(h_p, 1)
    global_row_frac = band.row_frac_lo + row_centers * (band.row_frac_hi - band.row_frac_lo)
    t_centers = _row_frac_to_t(global_row_frac)
    t_grid = np.tile(t_centers[:, None], (1, w_p)).astype(np.float32)
    theta_grid = np.tile(thetas[None, :], (h_p, 1)).astype(np.float32)
    return t_grid, theta_grid


def point_to_wedge_band_xy(
    x: float, y: float, band: WedgeBandGeometry,
) -> Optional[tuple[float, float, float]]:
    """Inverse mapping restricted to one band: real ``(x, y)`` -> ``(col, row, t)`` in THIS band's
    canvas, or ``None`` when the point falls outside the wedge's angular span OR outside this
    band's own radial range (a real point whose ``t`` belongs to a DIFFERENT band, not an error)."""
    geom = band.geom
    cx, cy = geom.centroid
    dx, dy = x - cx, y - cy
    r = float(np.hypot(dx, dy))
    theta = float(np.arctan2(dy, dx))
    rel = theta - geom.axis_angle_rad
    rel = (rel + np.pi) % (2 * np.pi) - np.pi
    half = geom.delta_theta_rad / 2
    if abs(rel) > half + 1e-9:
        return None
    Rth = _r_at_angle(geom.R_theta, theta)
    t = r / Rth if Rth > 1e-6 else 0.0
    global_row_frac = float(_t_to_row_frac(t))
    if not (band.row_frac_lo - 1e-9 <= global_row_frac <= band.row_frac_hi + 1e-9):
        return None
    span = max(band.row_frac_hi - band.row_frac_lo, 1e-12)
    local_frac = (global_row_frac - band.row_frac_lo) / span
    col = (rel / geom.delta_theta_rad + 0.5) * (geom.canvas_w - 1)
    row = local_frac * (geom.canvas_h - 1)
    return col, row, t


def wedge_band_xy_to_point(col: float, row: float, band: WedgeBandGeometry) -> tuple[float, float]:
    """Forward mapping: this band's canvas ``(col, row)`` -> real ``(x, y)``. Inverse of
    :func:`point_to_wedge_band_xy`."""
    geom = band.geom
    cx, cy = geom.centroid
    theta = geom.axis_angle_rad + (col / max(geom.canvas_w - 1, 1) - 0.5) * geom.delta_theta_rad
    local_frac = row / max(geom.canvas_h - 1, 1)
    global_row_frac = band.row_frac_lo + local_frac * (band.row_frac_hi - band.row_frac_lo)
    t = float(_row_frac_to_t(global_row_frac))
    Rth = _r_at_angle(geom.R_theta, theta)
    x = cx + t * Rth * np.cos(theta)
    y = cy + t * Rth * np.sin(theta)
    return x, y


def get_or_compute_wedge_band(
    rgb: np.ndarray, mask: np.ndarray, band: WedgeBandGeometry, cache_path: str | Path,
) -> np.ndarray:
    """Cache-or-compute for a single band, reusing the existing single-band cache I/O
    (:func:`load_wedge`/:func:`save_wedge`) unchanged — ``row_frac_lo``/``row_frac_hi`` are fixed,
    config-derived values (not per-image), so the caller always knows them again on a cache hit
    without needing to persist them in the cache file itself."""
    cached = load_wedge(cache_path)
    if cached is not None and cached[0].shape[:2] == (band.geom.canvas_h, band.geom.canvas_w):
        return cached[0]
    wedge = extract_polar_wedge_band(rgb, mask, band)
    save_wedge(wedge, band.geom, cache_path)
    return wedge


# ---------------------------------------------------------------------------
# Cache I/O — mirrors strip_extraction.py's save/load/get_or_compute pattern exactly. The wedge
# additionally needs R_theta (not just the pixel array) to support point projection later
# (e.g. a ZEGAR eval script decoding peaks back into axis-t) — cached alongside as .npz.
# ---------------------------------------------------------------------------

def save_wedge(wedge: np.ndarray, geom: WedgeGeometry, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(wedge.astype(np.uint8), mode="RGB").save(path)
    geom_path = path.with_suffix(".geom.npz")
    np.savez(geom_path, R_theta=geom.R_theta, centroid=np.array(geom.centroid, dtype=np.float64),
              axis_angle_rad=geom.axis_angle_rad, delta_theta_rad=geom.delta_theta_rad,
              canvas_w=geom.canvas_w, canvas_h=geom.canvas_h)


def load_wedge(path: str | Path) -> Optional[tuple[np.ndarray, WedgeGeometry]]:
    path = Path(path)
    geom_path = path.with_suffix(".geom.npz")
    if not path.exists() or not geom_path.exists():
        return None
    try:
        wedge = np.array(PILImage.open(path).convert("RGB"), dtype=np.uint8)
        z = np.load(geom_path)
        geom = WedgeGeometry(
            R_theta=z["R_theta"], centroid=tuple(z["centroid"].tolist()),
            axis_angle_rad=float(z["axis_angle_rad"]), delta_theta_rad=float(z["delta_theta_rad"]),
            canvas_w=int(z["canvas_w"]), canvas_h=int(z["canvas_h"]),
        )
    except Exception:
        return None
    return wedge, geom


def get_or_compute_wedge(
    rgb: np.ndarray, mask: np.ndarray, axis_info: dict, cache_path: str | Path,
    delta_theta_deg: float, canvas_w: int, canvas_h: int,
) -> tuple[np.ndarray, WedgeGeometry]:
    """Mirrors ``strip_extraction.get_or_compute_strip``'s "compute once, reuse everywhere"
    pattern, including the same deliberate shape check on a cache hit (a stale cache built at
    different canvas dimensions must never be served silently)."""
    cached = load_wedge(cache_path)
    if cached is not None and cached[0].shape[:2] == (canvas_h, canvas_w):
        return cached
    geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg, canvas_w, canvas_h)
    wedge = extract_polar_wedge(rgb, mask, geom)
    save_wedge(wedge, geom, cache_path)
    return wedge, geom
