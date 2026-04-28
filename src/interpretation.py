"""Interpretation: patch importance heatmaps and overlays."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torch import Tensor
from torch.utils.data import DataLoader

from src.config import OtolithConfig
from src.model import OtolithModel
from src.utils import resolve_device, tensor_to_uint8_rgb


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_patch_importance(
    model: OtolithModel,
    image_tensor: Tensor,
) -> Tensor:
    """Compute per-patch importance as L2 norm of DINOv2 patch tokens.

    Args:
        model:        OtolithModel (must be in eval mode on the correct device)
        image_tensor: single image — shape (1, 3, H, W) or (3, H, W)

    Returns:
        importance: FloatTensor (H_p, W_p) — higher value = more activated patch
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    _, patches = model.get_cls_and_patches(image_tensor)  # (1, H_p, W_p, D)
    patches = patches.squeeze(0)                           # (H_p, W_p, D)
    importance = patches.norm(dim=-1)                      # (H_p, W_p)
    return importance


def importance_to_heatmap(
    importance_grid: Tensor,
    image_size: int,
) -> np.ndarray:
    """Upsample and normalise patch importance to image resolution.

    Args:
        importance_grid: (H_p, W_p) float tensor
        image_size:      target side length in pixels (square output)

    Returns:
        heatmap: float32 ndarray (image_size, image_size) in [0, 1]
                 uniform input maps to all-zeros (no information to show).
    """
    grid = importance_grid.cpu().float().numpy()   # (H_p, W_p)

    vmin, vmax = float(grid.min()), float(grid.max())
    if vmax > vmin:
        grid = (grid - vmin) / (vmax - vmin)
    else:
        grid = np.zeros_like(grid)

    # cv2.resize expects (W, H) size tuple
    heatmap = cv2.resize(
        grid.astype(np.float32),
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    return heatmap.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_heatmap(heatmap: np.ndarray, path: str | Path) -> None:
    """Save a [0, 1] float heatmap as an 8-bit grayscale PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
    PILImage.fromarray(uint8, mode="L").save(path)


def save_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    path: str | Path,
    alpha: float = 0.5,
) -> None:
    """Blend a JET-colourised heatmap over the original image and save as RGB PNG.

    Args:
        original_image: (H, W, 3) uint8 RGB array
        heatmap:        (H, W)    float32 in [0, 1]
        path:           output file path
        alpha:          heatmap opacity (0 = invisible, 1 = full heatmap)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # cv2 colormap expects uint8; returns BGR → convert to RGB
    uint8_map = (heatmap * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)        # (H, W, 3) BGR
    rgb_map = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    orig_f = original_image.astype(np.float32)
    blended = (alpha * rgb_map + (1.0 - alpha) * orig_f).clip(0, 255).astype(np.uint8)
    PILImage.fromarray(blended, mode="RGB").save(path)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_interpretation(
    cfg: OtolithConfig,
    model: OtolithModel,
    loader: DataLoader,
    output_dir: str | Path,
) -> List[Dict]:
    """Run interpretation for every sample in loader.

    Saves per-sample:
        output_dir/heatmaps/<stem>_heatmap.png
        output_dir/overlays/<stem>_overlay.png

    Returns list of dicts with keys: image_id, heatmap_path, overlay_path.

    Caveats:
        - Heatmap reflects patch token norms, not biological ground truth.
        - Overlay is the reverse-normalised input image — not the original file.
        - Spatially uniform importance maps produce all-black heatmaps.
    """
    output_dir = Path(output_dir)
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.training.device)
    model.to(device)
    model.eval()

    results: List[Dict] = []

    for batch in loader:
        images     = batch["image"].to(device)    # (B, 3, H, W)
        image_ids  = batch["image_id"]            # list[str]
        B, C, H, W = images.shape

        for i in range(B):
            image_id   = image_ids[i]
            single     = images[i : i + 1]        # (1, 3, H, W)

            importance = compute_patch_importance(model, single)   # (H_p, W_p)
            heatmap    = importance_to_heatmap(importance, H)      # (H, H) [0,1]
            orig_rgb   = tensor_to_uint8_rgb(images[i])            # (H, W, 3) uint8

            stem          = Path(image_id).stem
            heatmap_path  = heatmap_dir / f"{stem}_heatmap.png"
            overlay_path  = overlay_dir / f"{stem}_overlay.png"

            save_heatmap(heatmap, heatmap_path)
            save_overlay(orig_rgb, heatmap, overlay_path, alpha=cfg.interpretation.heatmap_alpha)

            results.append(
                {
                    "image_id":     image_id,
                    "heatmap_path": str(heatmap_path),
                    "overlay_path": str(overlay_path),
                }
            )

    return results
