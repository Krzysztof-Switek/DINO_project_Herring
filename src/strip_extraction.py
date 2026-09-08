"""Straightened rectangular "strip" extraction along the otolith reading axis (nucleus ->
far_edge) — the dendrochronology-inspired input geometry for the localization (density) branch
of the dual-branch strip experiment (``plans and summaries/02.09_wycinki_plan.md``).

The reading axis (``otolith_axis.detect_axis``'s ``centroid``/``far_edge``) is already
geometrically a RADIUS — the exact analogue of a dendrochronology increment core (a straight
radial drill sample from bark to pith). This module turns that axis into an actual straightened
image crop: rotate around the centroid so the axis points along +x, rescale ONLY that axis
(``length_px`` -> ``strip_length_px``), and crop a fixed NATIVE-pixel-width band around it — the
direct analogue of a physical increment-core drill bit having a constant diameter, only the
core's LENGTH varies with the sample's size. See the plan file for the full design rationale,
including why the density head reads this crop through a SEPARATE backbone forward pass
(``torch.no_grad()``) while the age (CORAL/MIL) heads keep reading the full square image
unchanged — this module only produces the crop, it has no opinion on how it's consumed.

Rotation math (``canonicalizing_matrix``) is the exact formula already validated in
``scripts/diagnostics/train_zegar_localization_head_canonical.py`` — that script now imports it
from here instead of keeping its own private copy.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage

from src.otolith_axis import MASK_FILL_RGB, apply_background_mask


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def canonicalizing_matrix(
    centroid: tuple[float, float], far_edge: tuple[float, float],
) -> np.ndarray:
    """2x3 affine (``cv2.getRotationMatrix2D``) rotating around ``centroid`` so
    ``(far_edge - centroid)`` points along +x.

    Promoted, formula unchanged, from the private ``_canonicalizing_matrix`` in
    ``scripts/diagnostics/train_zegar_localization_head_canonical.py`` (26.08) — that script's
    own runtime sanity check (rotated ``far_edge`` lands within 1px of the centroid's y) already
    validated this on real ZEGAR images; reused here instead of duplicated.
    """
    cx, cy = centroid
    fx, fy = far_edge
    angle_deg = math.degrees(math.atan2(fy - cy, fx - cx))
    return cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)


def strip_transform_matrix(
    centroid: tuple[float, float],
    far_edge: tuple[float, float],
    length_px: float,
    strip_length_px: int,
    strip_width_px: int,
) -> np.ndarray:
    """2x3 affine mapping ORIGINAL ``rgb`` pixel coordinates to strip-local pixel coordinates.

    Composes :func:`canonicalizing_matrix` (rotation around ``centroid`` so the axis points
    along +x) with an x-only anisotropic scale (``length_px`` -> ``strip_length_px``) and a
    recenter (``centroid`` -> ``(0, strip_width_px / 2)``, ``far_edge`` -> ``(strip_length_px,
    strip_width_px / 2)``). The width (y) axis is only translated, never scaled — it stays a
    fixed NATIVE-pixel crop (see module docstring: the constant-diameter drill-bit analogy).

    ``length_px`` is floored at 1.0px to guard against a degenerate (near-zero) axis — the same
    style of denominator guard ``otolith_axis.point_to_axis_t`` already uses for its own (squared)
    denominator.
    """
    cx, cy = centroid
    length_px = max(float(length_px), 1.0)

    rot = canonicalizing_matrix(centroid, far_edge)
    scale_x = strip_length_px / length_px
    scale = np.array([
        [scale_x, 0.0, -cx * scale_x],
        [0.0,     1.0, -cy + strip_width_px / 2.0],
    ], dtype=np.float64)

    rot_3x3 = np.vstack([rot, [0.0, 0.0, 1.0]])
    scale_3x3 = np.vstack([scale, [0.0, 0.0, 1.0]])
    combined = scale_3x3 @ rot_3x3
    return combined[:2, :].astype(np.float64)


def extract_strip(
    rgb: np.ndarray,
    mask: np.ndarray,
    centroid: tuple[float, float],
    far_edge: tuple[float, float],
    length_px: float,
    strip_length_px: int,
    strip_width_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop+warp a straightened rectangular strip along the (``centroid`` -> ``far_edge``) axis.

    Masks the background BEFORE warping (:func:`otolith_axis.apply_background_mask`, the same
    fill colour used by every other masked training path in this project) so that any border
    pixels the warp itself introduces (the width band running past the image edge, a
    near-degenerate axis) are filled with ``MASK_FILL_RGB`` too, via ``borderValue`` — NEVER
    black. A black border here would reintroduce the exact high-contrast register-token artifact
    this project already fixed once by switching to the ``_reg`` DINOv2 backbone (see
    ``otolith_axis.MASK_FILL_RGB``'s own docstring).

    Returns ``(strip_rgb, M)``: ``strip_rgb`` is ``(strip_width_px, strip_length_px, 3)``
    ``uint8``; ``M`` is the 2x3 affine used — invert with ``cv2.invertAffineTransform(M)`` to map
    strip-local points back into ``rgb``'s original pixel space (e.g. for the held-out ZEGAR
    evaluation script, which needs predicted ring positions back in the frame ``point_to_axis_t``
    already operates in).
    """
    masked = apply_background_mask(rgb, mask)
    M = strip_transform_matrix(centroid, far_edge, length_px, strip_length_px, strip_width_px)
    strip = cv2.warpAffine(
        masked, M, (strip_length_px, strip_width_px),
        flags=cv2.INTER_LINEAR, borderValue=MASK_FILL_RGB,
    )
    return strip, M


def extract_strip_from_axis_info(
    rgb: np.ndarray,
    mask: np.ndarray,
    axis_info: dict,
    strip_length_px: int,
    strip_width_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper unpacking ``axis_info`` (``otolith_axis.detect_axis``'s return dict)
    into :func:`extract_strip`'s explicit arguments."""
    return extract_strip(
        rgb, mask, axis_info["centroid"], axis_info["far_edge"], axis_info["length_px"],
        strip_length_px, strip_width_px,
    )


# ---------------------------------------------------------------------------
# Per-patch tissue validity mask (08.09, plans and summaries/
# 08.09_metodyka_i_diagnoza_paska.md) -- diagnosed background-fixation fix: the strip's density
# loss (src/model.py::density_count_loss's new `valid_mask` param) needs to know, per patch,
# whether it lies on real otolith tissue or in the MASK_FILL_RGB-filled corridor beyond the
# real edge. Warps the SEGMENTATION MASK (not the RGB image) with the EXACT SAME
# `strip_transform_matrix` used by `extract_strip`, so the result lines up 1:1 with the strip's
# own pixels/patches -- never re-derive this geometry independently.
# ---------------------------------------------------------------------------

def extract_strip_validity(
    mask: np.ndarray,
    centroid: tuple[float, float],
    far_edge: tuple[float, float],
    length_px: float,
    strip_length_px: int,
    strip_width_px: int,
    patch_size: int,
) -> np.ndarray:
    """(h_p, w_p) float32 in [0, 1] -- fraction of each strip patch lying inside ``mask``.

    ``mask`` is the ORIGINAL image's binary segmentation mask (same one ``extract_strip``
    itself masks the background with before warping) -- NOT the already-warped strip image.
    Nearest-neighbour interpolation (a fractional-mask blend would invent fake partial-tissue
    values at the boundary) and ``borderValue=0`` (anywhere the warp's source footprint falls
    outside the original image is invalid, same as outside the mask).

    ``strip_length_px``/``strip_width_px`` must both be divisible by ``patch_size`` (already
    validated at the config level, same guard as the strip dimensions themselves) so the
    average-pool reshape below is exact, never truncating.
    """
    M = strip_transform_matrix(centroid, far_edge, length_px, strip_length_px, strip_width_px)
    # Binarise FIRST (masks in this project are uint8 0/255, e.g. otolith_axis.segment_otolith's
    # output -- ">0" is the established foreground test everywhere else, see e.g.
    # apply_background_mask) so the warped values are exactly {0, 1} for NEAREST interpolation,
    # never {0, 255}, before the patch-level average below turns them into a real [0, 1] fraction.
    mask_bin = (mask > 0).astype(np.uint8)
    warped = cv2.warpAffine(
        mask_bin, M, (strip_length_px, strip_width_px),
        flags=cv2.INTER_NEAREST, borderValue=0,
    )
    h_p, w_p = strip_width_px // patch_size, strip_length_px // patch_size
    return (
        warped.reshape(h_p, patch_size, w_p, patch_size)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def extract_strip_validity_from_axis_info(
    mask: np.ndarray,
    axis_info: dict,
    strip_length_px: int,
    strip_width_px: int,
    patch_size: int,
) -> np.ndarray:
    """Convenience wrapper unpacking ``axis_info`` into :func:`extract_strip_validity`'s
    explicit arguments -- mirrors :func:`extract_strip_from_axis_info`."""
    return extract_strip_validity(
        mask, axis_info["centroid"], axis_info["far_edge"], axis_info["length_px"],
        strip_length_px, strip_width_px, patch_size,
    )


# ---------------------------------------------------------------------------
# Strip cache I/O — mirrors otolith_axis.save_mask / load_mask / get_or_compute_mask exactly,
# one cached strip per (image, strip_length_px, strip_width_px) combination. Callers are
# responsible for keying ``cache_path`` by BOTH the image stem AND the strip dimensions (e.g. the
# ``data/strips_cache/{strip_length_px}x{strip_width_px}/<stem>_strip.png`` directory convention
# from the plan) — a shared, dimension-oblivious cache is exactly the class of bug already hit
# once for masks (``expert_annotation_eval.py``'s 12.08 stale-cache comment).
# ---------------------------------------------------------------------------

def save_strip(strip: np.ndarray, path: str | Path) -> None:
    """Save a warped strip crop as an RGB PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(strip.astype(np.uint8), mode="RGB").save(path)


def load_strip(path: str | Path) -> Optional[np.ndarray]:
    """Load a previously cached strip; returns ``None`` if the file is missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        arr = np.array(PILImage.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None
    return arr


def get_or_compute_strip(
    rgb: np.ndarray,
    mask: np.ndarray,
    axis_info: dict,
    cache_path: str | Path,
    strip_length_px: int,
    strip_width_px: int,
) -> Optional[np.ndarray]:
    """Load ``cache_path`` if present and the right shape, else build the strip and cache it.

    Mirrors ``otolith_axis.get_or_compute_mask``'s "compute once, reuse everywhere" pattern, used
    by ``OtolithDataset``'s strip-branch loading path. The extra shape check (beyond what
    ``get_or_compute_mask`` does for masks) is a deliberate, cheap safety net specifically for
    this cache: a stale cached PNG built at different ``strip_length_px``/``strip_width_px``
    would otherwise be served silently, corrupting the density branch's patch-grid shape.
    """
    cached = load_strip(cache_path)
    if cached is not None and cached.shape[:2] == (strip_width_px, strip_length_px):
        return cached
    strip, _M = extract_strip_from_axis_info(rgb, mask, axis_info, strip_length_px, strip_width_px)
    save_strip(strip, cache_path)
    return strip


def save_strip_validity(valid: np.ndarray, path: str | Path) -> None:
    """Save a validity mask as ``.npy`` (not PNG -- these are fractional [0,1] values at patch
    resolution, not an 8-bit image)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, valid.astype(np.float32))


def load_strip_validity(path: str | Path) -> Optional[np.ndarray]:
    """Load a previously cached validity mask; returns ``None`` if missing/unreadable.
    ``path`` should already end in ``.npy`` -- ``np.save`` only appends it when absent, and a
    caller that always passes an explicit ``.npy`` name (as ``OtolithDataset`` does) avoids the
    ambiguity of relying on that auto-append behaviour for the matching ``load``."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def get_or_compute_strip_validity(
    mask: np.ndarray,
    axis_info: dict,
    cache_path: str | Path,
    strip_length_px: int,
    strip_width_px: int,
    patch_size: int,
) -> np.ndarray:
    """Load ``cache_path`` if present and the right shape, else compute + cache it.

    Mirrors :func:`get_or_compute_strip`'s pattern exactly, including the same deliberate
    shape check on a cache hit (a stale mask cached at different strip/patch dimensions must
    never be served silently -- the documented mask-cache collision bug class,
    ``expert_annotation_eval.py``, 12.08).
    """
    h_p, w_p = strip_width_px // patch_size, strip_length_px // patch_size
    cached = load_strip_validity(cache_path)
    if cached is not None and cached.shape == (h_p, w_p):
        return cached
    valid = extract_strip_validity_from_axis_info(
        mask, axis_info, strip_length_px, strip_width_px, patch_size)
    save_strip_validity(valid, cache_path)
    return valid
