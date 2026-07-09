"""Candidate increment markers using the biological measurement axis.

Visualisation:
  - Original-resolution image (not the scaled 518×518 model input)
  - Measurement axis: from segmented otolith centroid (≈ nucleus) to the
    farthest contour edge — the post-rostral tip / longest radius
    (see ``src/otolith_axis.py`` for the segmentation pipeline)
  - Filled dots on this axis at importance-profile peaks = candidate annual rings

If segmentation fails for an image, we fall back to a vertical line through
the image centre (the previous behaviour) so the pipeline never breaks.

Caveats:
  - Peaks reflect DINOv2 patch activation, not confirmed annuli
  - For the current CORAL-only architecture, patch importance is the L2 norm of
    patch tokens — a heuristic, not a directly supervised localisation signal
  - The MIL architecture (planned, see plan file) will replace L2 norm with
    directly-trained patch probabilities
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.signal import find_peaks
from torch.utils.data import DataLoader

from src.config import OtolithConfig
from src.interpretation import compute_patch_importance
from src.model import OtolithModel
from src.otolith_axis import (
    detect_axis,
    find_centroid,
    find_farthest_edge,
    load_mask,
    sample_profile_along_axis,
    save_mask,
)
from src.utils import resolve_device, tensor_to_uint8_rgb


# ---------------------------------------------------------------------------
# Fallback profile (when segmentation fails) — vertical centre column, bottom half
# ---------------------------------------------------------------------------

def _fallback_axis(image_h: int, image_w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Vertical line from image centre to bottom edge."""
    cx, cy = image_w // 2, image_h // 2
    return (cx, cy), (cx, image_h - 1)


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def find_candidate_peaks(
    profile: np.ndarray,
    min_distance: int = 1,
    prominence_threshold: float = 0.0,
) -> np.ndarray:
    """Detect local maxima in a 1D importance profile.

    Returns int64 array of peak indices into ``profile``.
    """
    peaks, _ = find_peaks(
        profile,
        distance=max(1, int(min_distance)),
        prominence=float(prominence_threshold),
    )
    return peaks.astype(np.int64)


# ---------------------------------------------------------------------------
# JSON / overlay saving
# ---------------------------------------------------------------------------

def save_candidates_json(
    image_id: str,
    peak_profile_indices: np.ndarray,
    profile: np.ndarray,
    path: str | Path,
    axis_info: Optional[dict] = None,
) -> None:
    """Save candidate increment positions + axis metadata to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "image_id":             image_id,
        "num_candidates":       int(len(peak_profile_indices)),
        "peak_profile_indices": [int(i) for i in peak_profile_indices],
        "radial_profile":       [float(v) for v in profile],
    }
    if axis_info is not None:
        data["axis"] = {
            "centroid":  [int(axis_info["centroid"][0]), int(axis_info["centroid"][1])],
            "far_edge":  [int(axis_info["far_edge"][0]), int(axis_info["far_edge"][1])],
            "length_px": float(axis_info["length_px"]),
            "method":    "centroid_to_farthest",
        }
    else:
        data["axis"] = {"method": "fallback_vertical_centre"}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_candidates_overlay(
    original_image: np.ndarray,
    peak_profile_indices: np.ndarray,
    line_xy: np.ndarray,
    path: str | Path,
    axis_info: Optional[dict] = None,
    contour_color:  tuple = (0,   255, 255),   # cyan
    centroid_color: tuple = (0,   100, 255),   # blue
    axis_color:     tuple = (255, 220, 0),     # yellow
    dot_color:      tuple = (220, 30,  30),    # red
    dot_radius:     Optional[int] = None,
    line_thickness: Optional[int] = None,
) -> None:
    """Draw the otolith contour, measurement axis and increment dots.

    Args:
        original_image       : (H, W, 3) uint8 RGB — full-resolution photo
        peak_profile_indices : indices into the sampled profile (= indices into line_xy)
        line_xy              : (n_samples, 2) int — pixel coords sampled along the axis
        axis_info            : dict from detect_axis(); None ⇒ fallback overlay
        ... colours and sizes auto-scale with image height when not provided
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = original_image.copy()
    H = img.shape[0]
    r  = dot_radius     if dot_radius     is not None else max(8, H // 60)
    lw = line_thickness if line_thickness is not None else max(2, H // 250)

    if axis_info is not None:
        # 1. Otolith contour
        cv2.drawContours(img, [axis_info["contour"]], -1, contour_color, lw)
        cx, cy = axis_info["centroid"]
        fx, fy = axis_info["far_edge"]
        # 2. Centroid cross
        cross = max(8, H // 80)
        cv2.line(img, (cx - cross, cy), (cx + cross, cy), centroid_color, lw)
        cv2.line(img, (cx, cy - cross), (cx, cy + cross), centroid_color, lw)
        # 3. Axis line (centroid → far edge)
        cv2.line(img, (cx, cy), (fx, fy), axis_color, lw)
    else:
        # Fallback overlay — vertical line from centre to bottom edge
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        cv2.line(img, (cx, cy), (cx, img.shape[0] - 1), axis_color, lw)

    # 4. Increment dots at peak positions on the axis
    for idx in peak_profile_indices:
        k = int(idx)
        if 0 <= k < len(line_xy):
            x, y = int(line_xy[k][0]), int(line_xy[k][1])
            cv2.circle(img, (x, y), r, dot_color, -1)

    PILImage.fromarray(img, mode="RGB").save(path)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_candidates(
    cfg: OtolithConfig,
    model: OtolithModel,
    loader: DataLoader,
    output_dir: str | Path,
    image_dir: Optional[Path] = None,
) -> List[Dict]:
    """Detect candidate increments along the biological axis for every sample.

    Outputs per image:
        output_dir/masks/<stem>_mask.png                  — cached binary mask
        output_dir/candidates/<stem>_candidates.json      — peaks + axis metadata
        output_dir/candidates_overlays/<stem>_*.png       — annotated overlay

    Masks are cached on disk; subsequent runs re-use them (segmentation is the
    most expensive step). Delete the masks/ directory to force re-segmentation.
    """
    output_dir  = Path(output_dir)
    mask_dir    = output_dir / "masks"
    json_dir    = output_dir / "candidates"
    overlay_dir = output_dir / "candidates_overlays"
    for d in (mask_dir, json_dir, overlay_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.training.device)
    model.to(device)
    model.eval()

    min_dist   = cfg.candidates.min_peak_distance
    prominence = cfg.candidates.prominence_threshold
    n_samples_axis = 50          # profile samples along the (cx,cy)→(fx,fy) line
    results: List[Dict] = []

    for batch in loader:
        images    = batch["image"].to(device)
        image_ids = batch["image_id"]
        B = images.shape[0]

        for i in range(B):
            image_id = image_ids[i]
            single   = images[i : i + 1]

            importance = compute_patch_importance(model, single)
            imp_np = importance.cpu().numpy() if hasattr(importance, "cpu") else np.asarray(importance)

            # --- Load original ----
            orig_rgb: Optional[np.ndarray] = None
            if image_dir is not None:
                img_path = Path(image_dir) / image_id
                if img_path.exists():
                    try:
                        orig_rgb = np.array(PILImage.open(img_path).convert("RGB"))
                    except Exception:
                        orig_rgb = None
            if orig_rgb is None:
                orig_rgb = tensor_to_uint8_rgb(images[i])
            orig_h, orig_w = orig_rgb.shape[:2]

            stem      = Path(image_id).stem
            mask_path = mask_dir / f"{stem}_mask.png"

            # --- Resolve axis: try cached mask, else segment ---
            axis_info: Optional[dict] = None
            cached = load_mask(mask_path)
            if cached is not None:
                cent = find_centroid(cached)
                far  = find_farthest_edge(cached, cent) if cent else None
                if cent and far:
                    contours, _ = cv2.findContours(cached, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)
                    if contours:
                        contour = max(contours, key=cv2.contourArea)
                        axis_info = {
                            "mask":      cached,
                            "centroid":  cent,
                            "far_edge":  far,
                            "contour":   contour,
                            "length_px": float(np.hypot(far[0] - cent[0], far[1] - cent[1])),
                        }
            if axis_info is None:
                axis_info = detect_axis(orig_rgb)
                if axis_info is not None:
                    save_mask(axis_info["mask"], mask_path)

            # --- Sample profile along axis (or fallback) ---
            if axis_info is not None:
                profile, line_xy = sample_profile_along_axis(
                    imp_np, axis_info["centroid"], axis_info["far_edge"],
                    orig_h, orig_w, n_samples=n_samples_axis,
                )
            else:
                # Fallback — vertical centre→bottom
                cent, far = _fallback_axis(orig_h, orig_w)
                profile, line_xy = sample_profile_along_axis(
                    imp_np, cent, far, orig_h, orig_w, n_samples=n_samples_axis,
                )

            peak_idx = find_candidate_peaks(profile, min_dist, prominence)

            json_path    = json_dir    / f"{stem}_candidates.json"
            overlay_path = overlay_dir / f"{stem}_candidates_overlay.png"

            save_candidates_json(image_id, peak_idx, profile, json_path, axis_info)
            save_candidates_overlay(orig_rgb, peak_idx, line_xy, overlay_path,
                                    axis_info=axis_info)

            # Experimental: rings read straight from the image (light/dark bands),
            # independent of the model. Off unless cfg.candidates.detect_image_rings.
            if getattr(cfg.candidates, "detect_image_rings", False) and axis_info is not None:
                from src.ring_detection import detect_and_draw_rings
                ring_dir = output_dir / "image_rings_overlays"
                ring_dir.mkdir(parents=True, exist_ok=True)
                try:
                    _, ring_overlay = detect_and_draw_rings(orig_rgb, axis_info)
                    PILImage.fromarray(ring_overlay, mode="RGB").save(
                        ring_dir / f"{stem}_image_rings.png")
                except Exception as e:  # experimental — never break the pipeline
                    print(f"    [image-rings] {stem}: {e}")

            results.append({
                "image_id":                image_id,
                "num_candidates":          int(len(peak_idx)),
                "candidate_markers_path":  str(json_path),
                "candidates_overlay_path": str(overlay_path),
                "axis_detected":           bool(axis_info is not None),
            })

    return results
