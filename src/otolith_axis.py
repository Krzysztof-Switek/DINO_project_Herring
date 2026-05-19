"""Otolith segmentation and biological measurement-axis detection.

The measurement axis follows the ICES protocol convention:
  - origin   : geometric centroid of the segmented otolith (≈ nucleus / primordium)
  - terminus : the farthest point on the otolith contour from the centroid
               (the post-rostral tip in most herring otoliths)

Used by ``src/candidates.py`` to draw biologically meaningful overlays and to
sample the DINOv2 patch-importance grid along the actual otolith axis instead
of an arbitrary vertical line through the image centre.

Robust to: off-centre otoliths, oblique orientation, image-edge artefacts
(rulers, captions). Returns ``None`` for uniform/featureless images so callers
can fall back to the vertical-centre heuristic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _detect_polarity(gray: np.ndarray, sample_size: int = 20) -> str:
    """Decide whether the otolith is brighter or darker than the background.

    Samples the four image corners. If the average corner intensity is low
    (dark frame), the otolith is the bright region → ``"bright"``; otherwise
    the otolith is the dark region on a bright background → ``"dark"``.
    """
    H, W = gray.shape
    s = max(1, min(sample_size, H // 4, W // 4))
    corners = [
        gray[0:s,    0:s   ],
        gray[0:s,    W-s:W ],
        gray[H-s:H,  0:s   ],
        gray[H-s:H,  W-s:W ],
    ]
    avg_corner = float(np.mean([c.mean() for c in corners]))
    return "bright" if avg_corner < 128 else "dark"


def _hysteresis_mask(
    blur: np.ndarray,
    polarity: str,
    weak_offset: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hysteresis thresholding: keep weak pixels only if connected to strong ones.

    For embedded otoliths in transmitted light the boundary is gradual: the
    central part is opaque (high contrast vs background) and the rim is thin
    and translucent (low contrast). A single Otsu threshold catches only the
    central core; hysteresis grows the mask from the core into the rim.

    The weak threshold is set **adaptively** half-way between the Otsu cut and
    the mean of the background class — i.e. the weak band scales with the
    image contrast. If ``weak_offset`` is provided, it overrides the adaptive
    rule with a fixed offset from Otsu (used by tests / for tuning).

    Returns:
        strong : Otsu mask (high-contrast core)
        final  : hysteresis mask (core + connected weak rim)
    """
    if polarity == "bright":
        otsu_t, strong = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if weak_offset is None:
            bg = blur[blur < otsu_t]
            bg_mean = float(bg.mean()) if bg.size > 0 else max(0.0, otsu_t - 30)
            weak_t = (otsu_t + bg_mean) / 2.0
        else:
            weak_t = otsu_t - weak_offset
        weak_t = int(max(0.0, weak_t))
        _, weak = cv2.threshold(blur, weak_t, 255, cv2.THRESH_BINARY)
    else:
        otsu_t, strong = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        if weak_offset is None:
            bg = blur[blur > otsu_t]
            bg_mean = float(bg.mean()) if bg.size > 0 else min(255.0, otsu_t + 30)
            weak_t = (otsu_t + bg_mean) / 2.0
        else:
            weak_t = otsu_t + weak_offset
        weak_t = int(min(255.0, weak_t))
        _, weak = cv2.threshold(blur, weak_t, 255, cv2.THRESH_BINARY_INV)

    # Connected components in weak; keep only those overlapping strong
    n_labels, labels = cv2.connectedComponents(weak, connectivity=8)
    final = np.zeros_like(weak)
    if n_labels <= 1:
        return strong, strong
    overlap_labels = np.unique(labels[strong > 0])
    overlap_labels = overlap_labels[overlap_labels != 0]
    if overlap_labels.size == 0:
        return strong, strong
    final[np.isin(labels, overlap_labels)] = 255
    return strong, final


def segment_otolith(
    rgb: np.ndarray,
    weak_offset: int = 30,
) -> Optional[np.ndarray]:
    """Binary mask of the otolith (largest connected blob).

    Auto-detects polarity from corner pixels:
      - dark background (e.g. NotEmbedded on black) → otolith is the bright region
      - light background (Embedded in transmitted light) → otolith is the dark region

    Uses HYSTERESIS thresholding (two thresholds with region-growing) to capture
    the thin, low-contrast edges typical of transmitted-light embedded otoliths.

    Pipeline:
      1. RGB → grayscale + Gaussian blur (5×5)
      2. Polarity detection from image corners
      3. Otsu (strong) + Otsu±offset (weak) thresholds
      4. Hysteresis: keep weak-mask components overlapping the strong mask
      5. Morphological close (15×15) + light open (5×5) — fills holes,
         preserves thin edges
      6. Largest external contour → filled mask

    Args:
        rgb         : (H, W, 3) uint8 RGB
        weak_offset : how far below/above Otsu to set the weak threshold

    Returns ``None`` for uniform images or when segmentation collapses.
    """
    if rgb is None or rgb.size == 0:
        return None
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    polarity = _detect_polarity(blur)
    _, mask = _hysteresis_mask(blur, polarity, weak_offset=weak_offset)

    # Fill interior holes (e.g. translucent core) but keep thin edges
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    # Light opening — remove tiny salt noise, preserve fine edges
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 100:                                # too small to be an otolith
        return None
    if area > 0.95 * blur.shape[0] * blur.shape[1]:   # segmentation collapsed
        return None

    final = np.zeros_like(mask)
    cv2.drawContours(final, [largest], -1, color=255, thickness=-1)
    return final


# ---------------------------------------------------------------------------
# Centroid / farthest edge
# ---------------------------------------------------------------------------

def find_centroid(mask: np.ndarray) -> Optional[tuple[int, int]]:
    """Return (cx, cy) — geometric centroid of the binary mask, or None if empty."""
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    return int(round(M["m10"] / M["m00"])), int(round(M["m01"] / M["m00"]))


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def find_farthest_edge(
    mask: np.ndarray,
    centroid: tuple[int, int],
    direction: str = "down",
) -> Optional[tuple[int, int]]:
    """Contour point with maximum Euclidean distance from the centroid.

    Args:
        mask      : binary uint8 mask
        centroid  : (cx, cy) origin
        direction : ``"down"`` restricts the search to contour points BELOW the
                    centroid (y > cy) — matches the convention that all herring
                    otolith photos are oriented with the rostrum pointing down.
                    ``"any"`` returns the globally farthest point.

    Returns (fx, fy) or None.
    """
    contour = _largest_contour(mask)
    if contour is None:
        return None
    pts = contour.squeeze(axis=1).astype(np.float32)   # (N, 2)
    cx, cy = centroid

    if direction == "down":
        mask_below = pts[:, 1] > cy
        if mask_below.any():
            pts = pts[mask_below]
        # else: fall through to full-contour search

    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    idx = int(np.argmax(dx * dx + dy * dy))
    fx, fy = pts[idx]
    return int(round(fx)), int(round(fy))


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def detect_axis(rgb: np.ndarray) -> Optional[dict]:
    """Run full pipeline: segment → centroid → farthest edge.

    Returns:
        dict with keys:
          - ``mask``      : (H, W) uint8 mask
          - ``centroid``  : (cx, cy)
          - ``far_edge``  : (fx, fy)
          - ``contour``   : (N, 1, 2) int32 — for drawing
          - ``length_px`` : Euclidean axis length in pixels
        or ``None`` if any stage fails.
    """
    mask = segment_otolith(rgb)
    if mask is None:
        return None
    centroid = find_centroid(mask)
    if centroid is None:
        return None
    contour = _largest_contour(mask)
    if contour is None:
        return None
    far_edge = find_farthest_edge(mask, centroid)
    if far_edge is None:
        return None
    cx, cy = centroid
    fx, fy = far_edge
    length = float(np.hypot(fx - cx, fy - cy))
    return {
        "mask":      mask,
        "centroid":  centroid,
        "far_edge":  far_edge,
        "contour":   contour,
        "length_px": length,
    }


# ---------------------------------------------------------------------------
# Profile sampling
# ---------------------------------------------------------------------------

def sample_profile_along_axis(
    importance_grid: np.ndarray,
    centroid: tuple[int, int],
    far_edge: tuple[int, int],
    image_h: int,
    image_w: int,
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample patch-importance values along the (centroid → far_edge) line.

    Maps pixel positions on the line back to the corresponding patch in the
    importance grid (nearest-neighbour). Useful when the axis is oblique —
    the line cuts through different rows AND columns of the patch grid.

    Args:
        importance_grid : (H_p, W_p) patch importance from compute_patch_importance
        centroid        : (cx, cy) on the original image
        far_edge        : (fx, fy) on the original image
        image_h, image_w: original image dimensions in pixels
        n_samples       : number of points to sample along the axis

    Returns:
        profile : (n_samples,) float32 — importance values
        line_xy : (n_samples, 2) int — pixel coordinates on the original image
    """
    grid = np.asarray(importance_grid, dtype=np.float32)
    H_p, W_p = grid.shape
    cx, cy = centroid
    fx, fy = far_edge

    xs = np.linspace(cx, fx, n_samples)
    ys = np.linspace(cy, fy, n_samples)

    # Map original pixels → patch grid indices (nearest)
    px = np.clip((xs / max(image_w, 1) * W_p).astype(np.int64), 0, W_p - 1)
    py = np.clip((ys / max(image_h, 1) * H_p).astype(np.int64), 0, H_p - 1)

    profile = grid[py, px].astype(np.float32)
    line_xy = np.stack([xs.astype(np.int64), ys.astype(np.int64)], axis=1)
    return profile, line_xy


# ---------------------------------------------------------------------------
# Axis projection (for ring-zone visualization)
# ---------------------------------------------------------------------------

def project_distance_to_axis(
    mask: np.ndarray,
    centroid: tuple[int, int],
    far_edge: tuple[int, int],
) -> np.ndarray:
    """Scalar projection of every mask pixel onto the centroid → far_edge axis.

    For each pixel ``p`` in the mask, returns the parametric position ``t`` on the
    axis where ``t = 0`` at the centroid and ``t = 1`` at the far edge::

        t = ((p - centroid) · (far_edge - centroid)) / |far_edge - centroid|²

    Pixels outside the mask are set to NaN. Values <0 or >1 are kept (the
    projection of mask pixels off the axis endpoints).

    Used by ``compute_ring_zones`` to colour the otolith silhouette with one
    colour per annual zone (band between two consecutive peaks).
    """
    H, W = mask.shape[:2]
    cx, cy = float(centroid[0]), float(centroid[1])
    fx, fy = float(far_edge[0]), float(far_edge[1])
    vx, vy = fx - cx, fy - cy
    norm_sq = vx * vx + vy * vy
    if norm_sq <= 0:
        return np.full((H, W), np.nan, dtype=np.float32)

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    t = ((xs - cx) * vx + (ys - cy) * vy) / norm_sq
    out = t.astype(np.float32)
    out[mask == 0] = np.nan
    return out


# ---------------------------------------------------------------------------
# Mask cache I/O
# ---------------------------------------------------------------------------

def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save a binary mask as an 8-bit grayscale PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(mask.astype(np.uint8), mode="L").save(path)


def load_mask(path: str | Path) -> Optional[np.ndarray]:
    """Load a previously cached mask; returns None if file is missing."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        arr = np.array(PILImage.open(path).convert("L"), dtype=np.uint8)
    except Exception:
        return None
    return arr
