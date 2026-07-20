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


# Default radial-segmentation parameters (overridable via SegmentationConfig).
_RADIAL_DEFAULTS = {
    "frac":          0.18,   # per-ray threshold = this fraction of the ray's body brightness
    "background_k":  3.0,    # background floor = bg_mean + k·bg_std
    "n_angles":      720,    # rays cast from the nucleus
    "smooth_sigma":  4.0,    # Gaussian sigma for periodic r(θ) smoothing (low = follow scalloped teeth)
    "gap_tolerance": 8,      # pixels a ray may dip below threshold before the boundary commits
}


def segment_otolith(
    rgb: np.ndarray,
    method: str = "radial",
    weak_offset: int = 30,
    **radial_params,
) -> Optional[np.ndarray]:
    """Binary mask of the otolith (largest connected blob).

    Two methods (``method``):
      * ``"radial"`` (default) — cast rays from the nucleus and place the boundary
        where each ray fades into the background; produces a SMOOTH outline that
        reaches the faint, thinning rim (see ``_segment_radial``). Best for
        transilluminated embedded otoliths whose edge dissolves into the dark
        background and whose late increments sit in that faint margin.
      * ``"threshold"`` — the original Otsu + hysteresis region-growing
        (``_segment_threshold``). Kept as a fallback and for light-background /
        high-contrast cases.

    The radial method falls back to the threshold method (then ``None``) if it
    cannot find a core. ``**radial_params`` overrides ``_RADIAL_DEFAULTS``.

    Args:
        rgb         : (H, W, 3) uint8 RGB
        method      : "radial" | "threshold"
        weak_offset : threshold-method weak-band offset from Otsu
        radial_params: frac, background_k, n_angles, smooth_sigma, gap_tolerance

    Returns ``None`` for uniform images or when segmentation collapses.
    """
    if rgb is None or rgb.size == 0:
        return None
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None

    if method == "radial":
        params = {**_RADIAL_DEFAULTS, **radial_params}
        mask = _segment_radial(rgb, **params)
        if mask is not None:
            return mask
        # radial could not find a core → fall back to the threshold method
    return _segment_threshold(rgb, weak_offset=weak_offset)


def _segment_threshold(
    rgb: np.ndarray,
    weak_offset: int = 30,
) -> Optional[np.ndarray]:
    """Otsu-strong + adaptive-weak hysteresis segmentation (original method).

    Pipeline:
      1. RGB → grayscale + Gaussian blur (5×5)
      2. Polarity detection from image corners
      3. Otsu (strong) + Otsu±offset (weak) thresholds
      4. Hysteresis: keep weak-mask components overlapping the strong mask
      5. Morphological close (15×15) + light open (5×5)
      6. Largest external contour → filled mask
    """
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


def _segment_radial(
    rgb: np.ndarray,
    frac: float = 0.18,
    background_k: float = 3.0,
    n_angles: int = 720,
    smooth_sigma: float = 4.0,
    gap_tolerance: int = 8,
) -> Optional[np.ndarray]:
    """Smooth otolith mask by casting rays from the nucleus to the fade edge.

    Transilluminated embedded otoliths are opaque at the core and fade toward a
    thin, translucent rim before meeting the dark background. A contrast-based
    threshold stops at the opaque core (outline too small) and wiggles along the
    ring structure (squiggles). Instead:

      1. Estimate the background level (``bg_mean``/``bg_std``) from the four image
         corners; detect polarity and invert so the otolith is the *bright* region.
      2. Find a coarse core (strong Otsu → largest component) and its centroid
         (the nucleus).
      3. Cast ``n_angles`` rays from the centroid. Along each ray, the boundary is
         the OUTERMOST radius whose intensity exceeds a PER-RAY threshold
         ``max(bg_mean + k·bg_std, bg_mean + frac·(ray_peak − bg_mean))`` — a
         fraction of *that ray's* own body brightness, floored at the background.
         This keeps genuinely faint tissue but rejects dim scatter/smoke where the
         body is bright. A short run below threshold (``gap_tolerance`` px) is
         tolerated so a faint bridge isn't cut early.
      4. Smooth ``r(θ)`` periodically (Gaussian) → removes squiggles, keeps lobes.
      5. Rasterise the smooth radial polygon into a filled mask.

    Returns ``None`` if no core is found (caller falls back to the threshold method).
    """
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3).astype(np.float32)

    # Background from corners + polarity (invert so otolith is bright).
    polarity = _detect_polarity(blur.astype(np.uint8))
    if polarity == "dark":                       # otolith darker than background
        blur = 255.0 - blur
    s = max(10, min(H, W) // 20)
    corners = np.concatenate([
        blur[:s, :s].ravel(), blur[:s, -s:].ravel(),
        blur[-s:, :s].ravel(), blur[-s:, -s:].ravel(),
    ])
    bg_mean = float(corners.mean())
    bg_std = float(corners.std()) + 1e-3
    bg_floor = bg_mean + background_k * bg_std

    # Coarse core → centroid (nucleus).
    core_t, _ = cv2.threshold(
        cv2.GaussianBlur(blur.astype(np.uint8), (5, 5), 0),
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    core = (blur > core_t).astype(np.uint8) * 255
    core = cv2.morphologyEx(
        core, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    core_mask = np.zeros_like(core)
    cv2.drawContours(core_mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    centroid = find_centroid(core_mask)
    if centroid is None:
        return None
    cx, cy = centroid

    # Cast rays; find the fade-to-background radius per angle.
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    r_max = int(np.hypot(max(cx, W - cx), max(cy, H - cy)))
    radii = np.arange(3, max(r_max, 4))
    rs = np.full(n_angles, 6.0, dtype=np.float32)
    for i, a in enumerate(angles):
        xs = np.clip((cx + radii * np.cos(a)).astype(np.int64), 0, W - 1)
        ys = np.clip((cy + radii * np.sin(a)).astype(np.int64), 0, H - 1)
        prof = blur[ys, xs]
        thr = max(bg_floor, bg_mean + frac * (float(prof.max()) - bg_mean))
        last = 0
        below = 0
        for j in range(len(radii)):
            if prof[j] > thr:
                last = j
                below = 0
            else:
                below += 1
                if below > gap_tolerance and last > 0:
                    break
        rs[i] = radii[last]

    # Periodic Gaussian smoothing of r(θ) → smooth outline, no squiggles.
    ksize = int(smooth_sigma * 4) | 1
    g = cv2.getGaussianKernel(ksize, smooth_sigma).ravel()
    rs = np.convolve(np.concatenate([rs] * 3), g, mode="same")[n_angles:2 * n_angles]

    pts = np.stack([cx + rs * np.cos(angles), cy + rs * np.sin(angles)], axis=1)
    pts = pts.astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [pts], -1, color=255, thickness=-1)

    area = float((mask > 0).sum())
    if area < 100 or area > 0.98 * H * W:
        return None
    return mask


# ---------------------------------------------------------------------------
# Centroid / farthest edge
# ---------------------------------------------------------------------------

def find_centroid(mask: np.ndarray) -> Optional[tuple[int, int]]:
    """Return (cx, cy) — geometric centroid of the binary mask, or None if empty."""
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    return int(round(M["m10"] / M["m00"])), int(round(M["m01"] / M["m00"]))


def find_intensity_centroid(rgb: np.ndarray, mask: np.ndarray) -> Optional[tuple[int, int]]:
    """Intensity-weighted centroid within ``mask`` (nucleus/primordium estimate).

    The geometric mask centroid (:func:`find_centroid`) can sit away from the true
    nucleus when the otolith grows asymmetrically — the primordium is usually the
    most OPAQUE point, not the middle of the outline. This weights every mask pixel
    by its (polarity-corrected) brightness, pulling the estimate toward that opaque
    core. Polarity-aware (reuses :func:`_detect_polarity`) so both bright-core and
    dark-core preparations weight toward their own opaque region.

    Falls back to :func:`find_centroid` when the weighted sum is degenerate (a
    perfectly uniform mask interior — no intensity signal to weight by).
    """
    m = np.asarray(mask) > 0
    if not m.any():
        return None
    arr = np.asarray(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    gray = gray.astype(np.float64)
    if _detect_polarity(gray.astype(np.uint8)) == "dark":
        gray = 255.0 - gray                          # otolith is the bright region either way
    ys, xs = np.nonzero(m)
    weights = gray[ys, xs]
    total = float(weights.sum())
    if total <= 1e-9:
        return find_centroid((m.astype(np.uint8)) * 255)
    cx = float((xs * weights).sum() / total)
    cy = float((ys * weights).sum() / total)
    return int(round(cx)), int(round(cy))


def resolve_centroid(
    rgb: np.ndarray, mask: np.ndarray, method: str = "geometric",
) -> Optional[tuple[int, int]]:
    """Dispatch to :func:`find_centroid` (default) or :func:`find_intensity_centroid`.

    Single entry point so every caller (fresh segmentation in :func:`detect_axis` and
    the cached-mask path in ``run_pipeline.py``) picks the nucleus the same way.
    """
    if method == "intensity":
        c = find_intensity_centroid(rgb, mask)
        if c is not None:
            return c
    return find_centroid(mask)


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

def detect_axis(
    rgb: np.ndarray, seg_params: Optional[dict] = None, nucleus_method: str = "geometric",
) -> Optional[dict]:
    """Run full pipeline: segment → centroid → farthest edge.

    ``seg_params`` (optional): keyword overrides forwarded to ``segment_otolith``
    (e.g. ``{"method": "radial", "frac": 0.2}``). Typically built from
    ``ProjectConfig.segmentation`` — see ``SegmentationConfig.as_params``.

    ``nucleus_method``: ``"geometric"`` (default, unchanged behaviour) uses the mask's
    geometric centroid; ``"intensity"`` uses :func:`find_intensity_centroid` — see
    ``SegmentationConfig.nucleus_method``.

    Returns:
        dict with keys:
          - ``mask``      : (H, W) uint8 mask
          - ``centroid``  : (cx, cy)
          - ``far_edge``  : (fx, fy)
          - ``contour``   : (N, 1, 2) int32 — for drawing
          - ``length_px`` : Euclidean axis length in pixels
        or ``None`` if any stage fails.
    """
    mask = segment_otolith(rgb, **(seg_params or {}))
    if mask is None:
        return None
    centroid = resolve_centroid(rgb, mask, nucleus_method)
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


def get_or_compute_mask(
    rgb: np.ndarray, cache_path: str | Path, seg_params: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Load ``cache_path`` if present, else segment ``rgb`` and cache the result there.

    Single entry point for "one mask per image, computed once, reused everywhere" —
    used by the input-masking dataset path (``dataset.OtolithDataset``) and can replace
    the equivalent inline cache-or-segment logic in ``scripts/run_pipeline.py``'s card
    generation. Returns ``None`` (without writing anything) when segmentation fails.
    """
    cached = load_mask(cache_path)
    if cached is not None:
        return cached
    mask = segment_otolith(rgb, **(seg_params or {}))
    if mask is not None:
        save_mask(mask, cache_path)
    return mask


# Fill colour for masked-out background (0–255 uint8): the per-channel ImageNet mean.
# After ImageNet normalisation (src/dataset.py, src/utils.py) this becomes ~0 — the
# background contributes essentially no signal, instead of the strong, high-contrast
# edge a flat black fill would create right at the mask boundary (register-token /
# high-norm-patch artifacts already fire on exactly this kind of sharp edge — see
# DINO_proces.md §4.7).
MASK_FILL_RGB = (124, 116, 104)


def apply_background_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of ``rgb`` with pixels outside ``mask`` set to ``MASK_FILL_RGB``."""
    out = rgb.copy()
    out[mask == 0] = MASK_FILL_RGB
    return out
