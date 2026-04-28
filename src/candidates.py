"""Candidate increment markers via 1D radial importance profile and peak detection.

Caveats:
  - Candidate peaks are regions of high patch importance, NOT confirmed annuli.
  - Results are backbone-dependent and not biologically validated.
  - Peak detection parameters (min_distance, prominence) are heuristic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.signal import find_peaks
from torch import Tensor
from torch.utils.data import DataLoader

from src.config import OtolithConfig
from src.interpretation import compute_patch_importance, importance_to_heatmap
from src.model import OtolithModel
from src.utils import resolve_device, tensor_to_uint8_rgb


# ---------------------------------------------------------------------------
# Radial profile
# ---------------------------------------------------------------------------

def extract_radial_profile(
    importance_grid: Union[Tensor, np.ndarray],
) -> np.ndarray:
    """Collapse 2D patch importance to a 1D radial profile.

    Takes the mean across rows (axis=0), yielding shape (W_p,).
    Each element is the average importance at a given horizontal (radial) position.

    Args:
        importance_grid: (H_p, W_p) Tensor or ndarray

    Returns:
        profile: (W_p,) float32 ndarray
    """
    if hasattr(importance_grid, "cpu"):            # Tensor
        importance_grid = importance_grid.cpu().numpy()
    return importance_grid.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def find_candidate_peaks(
    profile: np.ndarray,
    min_distance: int = 1,
    prominence_threshold: float = 0.0,
) -> np.ndarray:
    """Detect local maxima in a 1D importance profile.

    Args:
        profile:              1D float32 array (radial importance profile)
        min_distance:         minimum sample-distance between returned peaks
        prominence_threshold: minimum prominence of a valid peak

    Returns:
        peaks: 1D int64 array of peak indices (may be empty)
    """
    peaks, _ = find_peaks(
        profile,
        distance=max(1, int(min_distance)),
        prominence=float(prominence_threshold),
    )
    return peaks.astype(np.int64)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def peaks_to_pixel_positions(
    peak_indices: np.ndarray,
    image_size: int,
    num_patches: int,
) -> np.ndarray:
    """Convert patch-grid peak indices to pixel x-coordinates.

    Each peak is placed at the centre of its patch.

    Args:
        peak_indices: 1D int array of indices into the radial profile (0..W_p-1)
        image_size:   image width in pixels
        num_patches:  number of patches along the horizontal axis (W_p)

    Returns:
        pixel_positions: 1D int64 array of x-coordinates (0..image_size-1)
    """
    if len(peak_indices) == 0:
        return np.array([], dtype=np.int64)
    patch_width = image_size / num_patches
    return ((peak_indices + 0.5) * patch_width).astype(np.int64)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_candidates_json(
    image_id: str,
    peak_positions: np.ndarray,
    profile: np.ndarray,
    path: str | Path,
) -> None:
    """Save candidate increment markers to JSON.

    Saved fields:
        image_id, num_candidates, peak_pixel_positions, radial_profile

    Note: peak_pixel_positions are candidates only — NOT confirmed annuli.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_id": image_id,
        "num_candidates": int(len(peak_positions)),
        "peak_pixel_positions": [int(x) for x in peak_positions],
        "radial_profile": [float(v) for v in profile],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_candidates_overlay(
    original_image: np.ndarray,
    peak_positions: np.ndarray,
    heatmap: np.ndarray,
    path: str | Path,
    alpha: float = 0.5,
    line_color: tuple = (255, 0, 0),
    line_thickness: int = 2,
) -> None:
    """Blend heatmap over original image and mark candidate peaks with vertical lines.

    Args:
        original_image: (H, W, 3) uint8 RGB
        peak_positions: 1D int array of x pixel positions
        heatmap:        (H, W) float32 in [0, 1]
        path:           output file path
        alpha:          heatmap opacity
        line_color:     RGB colour for peak marker lines (default: red)
        line_thickness: line width in pixels
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Blend JET heatmap over original
    uint8_map = (heatmap * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
    rgb_map = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    blended = (alpha * rgb_map + (1.0 - alpha) * original_image.astype(np.float32))
    blended = blended.clip(0, 255).astype(np.uint8)

    # Draw vertical line at each candidate position
    H = blended.shape[0]
    for x in peak_positions:
        x_int = int(x)
        if 0 <= x_int < blended.shape[1]:
            cv2.line(blended, (x_int, 0), (x_int, H - 1), line_color, line_thickness)

    PILImage.fromarray(blended, mode="RGB").save(path)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_candidates(
    cfg: OtolithConfig,
    model: OtolithModel,
    loader: DataLoader,
    output_dir: str | Path,
) -> List[Dict]:
    """Detect candidate increment markers for all samples in loader.

    For each sample:
      1. Compute patch importance grid
      2. Extract 1D radial profile (mean across rows)
      3. Detect candidate peaks (scipy.signal.find_peaks)
      4. Save markers JSON and annotated overlay PNG

    Returns list of dicts with keys:
        image_id, num_candidates, candidate_markers_path, candidates_overlay_path

    Caveats:
        - Peaks reflect patch activation patterns, not biological annuli.
        - No spatial normalisation — results are backbone- and input-dependent.
        - Prominence/distance thresholds are global; otolith shape is ignored.
    """
    output_dir = Path(output_dir)
    json_dir    = output_dir / "candidates"
    overlay_dir = output_dir / "candidates_overlays"
    json_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.training.device)
    model.to(device)
    model.eval()

    min_dist   = cfg.candidates.min_peak_distance
    prominence = cfg.candidates.prominence_threshold
    results: List[Dict] = []

    for batch in loader:
        images    = batch["image"].to(device)    # (B, 3, H, W)
        image_ids = batch["image_id"]
        B, C, H, W = images.shape

        for i in range(B):
            image_id = image_ids[i]
            single   = images[i : i + 1]         # (1, 3, H, W)

            importance   = compute_patch_importance(model, single)  # (H_p, W_p)
            heatmap      = importance_to_heatmap(importance, H)     # (H, W) [0,1]
            profile      = extract_radial_profile(importance)       # (W_p,)
            peak_idx     = find_candidate_peaks(profile, min_dist, prominence)
            num_patches_w = importance.shape[1]
            peak_px      = peaks_to_pixel_positions(peak_idx, W, num_patches_w)
            orig_rgb     = tensor_to_uint8_rgb(images[i])           # (H, W, 3) uint8

            stem         = Path(image_id).stem
            json_path    = json_dir    / f"{stem}_candidates.json"
            overlay_path = overlay_dir / f"{stem}_candidates_overlay.png"

            save_candidates_json(image_id, peak_px, profile, json_path)
            save_candidates_overlay(
                orig_rgb, peak_px, heatmap, overlay_path,
                alpha=cfg.interpretation.heatmap_alpha,
            )

            results.append({
                "image_id":                image_id,
                "num_candidates":          int(len(peak_px)),
                "candidate_markers_path":  str(json_path),
                "candidates_overlay_path": str(overlay_path),
            })

    return results
