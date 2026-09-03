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
