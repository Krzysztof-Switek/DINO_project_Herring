"""Interpretation: patch importance heatmaps and overlays."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

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
    method: str = "auto",
) -> Tensor:
    """Per-patch importance, honouring ``cfg.interpretation.method``.

    Supported methods:
        "auto"                   — MIL patch probabilities if the model has a MIL
                                   head (``patch_head``), else L2 norm of tokens.
        "patch_token_importance" — always the L2 norm of DINOv2 patch tokens
                                   (post-hoc heuristic, no localisation supervision).
        "mil_patch_probs"        — always the trained MIL patch probabilities;
                                   raises if the model has no MIL head.

    Args:
        model:        OtolithModel in eval mode on the right device
        image_tensor: (1, 3, H, W) or (3, H, W)
        method:       one of the strings above

    Returns:
        importance: FloatTensor (H_p, W_p)
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    has_mil = hasattr(model, "patch_head")
    use_mil = method == "mil_patch_probs" or (method == "auto" and has_mil)

    if use_mil:
        if not has_mil:
            raise RuntimeError(
                "method='mil_patch_probs' requires a MIL head "
                "(cfg.model.head_type must be 'mil' or 'both')"
            )
        probs = model.get_patch_probs(image_tensor)        # (1, H_p, W_p)
        return probs.squeeze(0)                             # (H_p, W_p)

    # L2 norm of patch tokens (patch_token_importance, or auto without MIL head)
    _, patches = model.get_cls_and_patches(image_tensor)   # (1, H_p, W_p, D)
    patches = patches.squeeze(0)                            # (H_p, W_p, D)
    return patches.norm(dim=-1)                             # (H_p, W_p)


def compute_coral_gradcam(
    model: OtolithModel,
    image_tensor: Tensor,
) -> Optional[Tensor]:
    """Grad-CAM-style saliency for the CORAL **age verdict** (11.07 Punkt 7).

    Answers „które patche najbardziej wpływają na przewidziany wiek": gradient of
    the CORAL score ``g`` (pre-threshold scalar) w.r.t. the DINOv2 patch tokens,
    ``importance = ReLU(Σ_d grad ⊙ token)``, reshaped to the (H_p, W_p) grid.

    This is the age head's OWN attribution — distinct from the MIL localisation map.
    Returns None if the model has no CORAL head. (With a constant-patch mock backbone
    the gradient may be zero → returns a zero map, which is fine.)
    """
    if not hasattr(model, "head"):
        return None
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    device = next(model.parameters()).device
    image = image_tensor.to(device).clone().requires_grad_(True)
    was_training = model.training
    model.eval()
    cam: Optional[Tensor] = None
    try:
        feats = model.backbone.forward_features(image)
        patches = feats["x_norm_patchtokens"]              # (1, N, D)
        N = patches.shape[1]
        hp = int(round(N ** 0.5))
        if not patches.requires_grad:                      # e.g. constant-patch mock
            cam = torch.zeros(hp, hp)
        else:
            patches.retain_grad()
            g = model.head(feats["x_norm_clstoken"])       # (1, 1) CORAL score
            model.zero_grad(set_to_none=True)
            g.sum().backward()
            grad = patches.grad
            if grad is None:
                cam = torch.zeros(hp, hp)
            else:
                flat = torch.relu((grad * patches).sum(dim=-1)).squeeze(0)  # (N,)
                cam = flat[: hp * hp].reshape(hp, hp).detach().cpu()
    except Exception:
        cam = None
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
    return cam


def compute_cls_attention(
    model: OtolithModel,
    image_tensor: Tensor,
) -> Optional[Tensor]:
    """Mean CLS→patch self-attention from DINOv2's LAST block, as a (H_p, W_p) grid.

    The token CLS (which the CORAL head reads) aggregates the image via attention;
    this shows from which patches it drew. Captured with a forward hook on the last
    attention block's ``attn_drop`` (whose input is the softmax attention matrix).
    Returns None when unavailable — mock backbone, xFormers fused attention, or a
    different ViT layout — so callers can fall back gracefully.
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    backbone = model.backbone
    try:
        attn_drop = backbone.blocks[-1].attn.attn_drop
    except (AttributeError, IndexError, TypeError):
        return None
    # Some DINOv2 builds use fused attention (PyTorch SDPA / xFormers) and keep
    # ``attn_drop`` as a plain float (the dropout probability) with NO attention matrix
    # to hook. Guard so we return None gracefully instead of raising
    # "'float' object has no attribute 'register_forward_hook'" — which otherwise
    # propagates and skips the WHOLE reasoning card (diagnosed on the 14.07 preview).
    if not hasattr(attn_drop, "register_forward_hook"):
        return None

    captured: Dict[str, Tensor] = {}

    def _hook(_m, inp, _out):
        if inp and torch.is_tensor(inp[0]):
            captured["a"] = inp[0].detach()

    try:
        handle = attn_drop.register_forward_hook(_hook)
    except Exception:
        return None
    try:
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            backbone.forward_features(image_tensor.to(device))
    except Exception:
        return None
    finally:
        handle.remove()

    a = captured.get("a")
    if a is None or a.dim() != 4:                          # expect (B, heads, T, T)
        return None
    cls_row = a[0, :, 0, 1:].mean(0)                       # avg heads → (T-1,)
    n_patch = cls_row.shape[0]
    hp = int(round(n_patch ** 0.5))
    if hp < 1:
        return None
    cls_row = cls_row[-(hp * hp):]                         # robust to register tokens
    return cls_row.reshape(hp, hp).float().cpu()


def compute_cls_attention_patched(
    model: OtolithModel,
    image_tensor: Tensor,
) -> Optional[Tensor]:
    """REAL CLS→patch attention via a temporary monkey-patch of the last block (16.07).

    ``compute_cls_attention`` (hook on ``attn_drop``) fails on modern DINOv2: the block
    computes ``scaled_dot_product_attention`` (fused → no softmax matrix) and ``attn_drop``
    is a plain float, so there is nothing to hook. This function instead **temporarily
    replaces** ``backbone.blocks[-1].attn.forward`` with a maths-identical version that
    materialises ``softmax(q·kᵀ · scale)``, captures it, runs ONE forward, and restores the
    original in ``finally``. Returns the mean CLS→patch attention as a ``(H_p, W_p)`` grid,
    or ``None`` (safe fallback to the L2 proxy) when the layout is unexpected — e.g. the mock
    backbone. Weights are untouched; this is post-hoc and read-only.
    """
    import types
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    backbone = model.backbone
    try:
        attn = backbone.blocks[-1].attn
    except (AttributeError, IndexError, TypeError):
        return None
    if not all(hasattr(attn, a) for a in ("qkv", "proj", "num_heads", "scale")):
        return None

    captured: Dict[str, Tensor] = {}

    def _patched_forward(self, x, *args, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]      # (B, heads, N, head_dim)
        a = (q @ k.transpose(-2, -1)) * self.scale            # (B, heads, N, N)
        a = a.softmax(dim=-1)
        captured["a"] = a.detach()
        out = (a @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.proj_drop(self.proj(out))

    orig_forward = attn.forward
    try:
        attn.forward = types.MethodType(_patched_forward, attn)
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            backbone.forward_features(image_tensor.to(device))
    except Exception:
        return None
    finally:
        attn.forward = orig_forward

    a = captured.get("a")
    if a is None or a.dim() != 4:                             # expect (B, heads, T, T)
        return None
    cls_row = a[0, :, 0, 1:].mean(0)                          # avg heads → CLS→(non-CLS)
    n = cls_row.shape[0]
    hp = int(round(n ** 0.5))
    if hp < 1:
        return None
    cls_row = cls_row[-(hp * hp):]                            # drop register tokens → last hp*hp patches
    return cls_row.reshape(hp, hp).float().cpu()


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


def importance_to_heatmap_2d(
    importance_grid: Tensor,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Upsample importance grid to arbitrary (H, W) resolution.

    Returns float32 ndarray (target_h, target_w) in [0, 1].
    """
    grid = importance_grid.cpu().float().numpy() if hasattr(importance_grid, "cpu") else np.asarray(importance_grid, dtype=np.float32)
    vmin, vmax = float(grid.min()), float(grid.max())
    if vmax > vmin:
        grid = (grid - vmin) / (vmax - vmin)
    else:
        grid = np.zeros_like(grid)
    return cv2.resize(grid.astype(np.float32), (target_w, target_h),
                      interpolation=cv2.INTER_LINEAR).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

# INFERNO's low end is near-black — indistinguishable from the dark bands/shadows already
# present in otolith photos, so a low-signal region and "no signal, showing raw dark photo"
# looked identical (21.07 user report). JET's low end is a vivid, saturated blue that never
# occurs naturally in these grayish/tan/dark-blue-grey photos — any colour tint at all means
# real signal, not photo content. (Known JET downsides — perceptual non-uniformity, false
# banding — matter less here: this is a "where does attention/density fall" diagnostic
# overlay, not a precision measurement readout.)
DEFAULT_COLORMAP = cv2.COLORMAP_JET


def apply_colormap_with_mask(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    alpha: float = 0.55,
    colormap: int = DEFAULT_COLORMAP,
) -> np.ndarray:
    """Inferno-colourise ``heatmap`` and blend with ``original_image``.

    When ``mask`` is provided, the blend is applied **only inside the mask**;
    pixels outside the mask retain the raw original image — so the background
    never shows fake activation from edges/rulers/captions.

    Args:
        heatmap:        (H, W) float32 in [0, 1]
        original_image: (H, W, 3) uint8 RGB
        mask:           (H, W) bool or uint8 ({0, 255}); None disables masking
        alpha:          heatmap opacity inside mask (literature default: 0.4–0.55)
        colormap:       cv2 colormap id (default INFERNO — chosen for light otolith
                        photos: dark = low signal, bright yellow = high signal)

    Returns ``(H, W, 3)`` uint8 RGB.
    """
    uint8_map = (heatmap * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(uint8_map, colormap)
    rgb_map = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    orig_f = original_image.astype(np.float32)
    blended = (alpha * rgb_map + (1.0 - alpha) * orig_f).clip(0, 255).astype(np.uint8)

    if mask is None:
        return blended

    out = original_image.copy()
    inside = mask > 0
    out[inside] = blended[inside]
    return out


def save_heatmap(
    heatmap: np.ndarray,
    path: str | Path,
    colormap: int = DEFAULT_COLORMAP,
) -> None:
    """Save a [0, 1] float heatmap as a colourised RGB PNG (inferno by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(uint8, colormap)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    PILImage.fromarray(rgb, mode="RGB").save(path)


def save_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    path: str | Path,
    alpha: float = 0.55,
    mask: Optional[np.ndarray] = None,
    colormap: int = DEFAULT_COLORMAP,
) -> None:
    """Blend a colourised heatmap over the original image and save as RGB PNG.

    Args:
        original_image: (H, W, 3) uint8 RGB array
        heatmap:        (H, W)    float32 in [0, 1]  — must match original_image dims
        path:           output file path
        alpha:          heatmap opacity (0 = invisible, 1 = full heatmap)
        mask:           optional (H, W) binary mask — restrict blending to mask pixels;
                        pixels outside the mask retain the raw original image
        colormap:       cv2 colormap id (default INFERNO)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blended = apply_colormap_with_mask(
        heatmap, original_image, mask=mask, alpha=alpha, colormap=colormap,
    )
    PILImage.fromarray(blended, mode="RGB").save(path)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_interpretation(
    cfg: OtolithConfig,
    model: OtolithModel,
    loader: DataLoader,
    output_dir: str | Path,
    image_dir: Optional[Path] = None,
) -> List[Dict]:
    """Run interpretation for every sample in loader.

    Saves per-sample:
        output_dir/heatmaps/<stem>_heatmap.png   — JET-colourised importance map
        output_dir/overlays/<stem>_overlay.png   — JET heatmap blended over original

    Both files are at the original image resolution when ``image_dir`` is provided
    (falls back to 518×518 model-input resolution if original not found).

    Args:
        cfg        : OtolithConfig
        model      : trained OtolithModel (eval mode)
        loader     : DataLoader (may use scaled 518×518 images for model inference)
        output_dir : directory for heatmap and overlay files
        image_dir  : directory of original-resolution photos; if None or not found,
                     falls back to the de-normalised scaled batch image

    Returns list of dicts with keys: image_id, heatmap_path, overlay_path.
    """
    from src.otolith_axis import detect_axis, load_mask, save_mask

    output_dir = Path(output_dir)
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    mask_dir    = output_dir / "masks"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.training.device)
    model.to(device)
    model.eval()

    results: List[Dict] = []

    for batch in loader:
        images     = batch["image"].to(device)    # (B, 3, H, W) — scaled
        image_ids  = batch["image_id"]
        B, C, H, W = images.shape

        for i in range(B):
            image_id = image_ids[i]
            single   = images[i : i + 1]

            importance = compute_patch_importance(
                model, single, method=cfg.interpretation.method,
            )   # (H_p, W_p)

            # Load original-resolution image if possible
            orig_rgb: Optional[np.ndarray] = None
            if image_dir is not None:
                img_path = Path(image_dir) / image_id
                if img_path.exists():
                    try:
                        orig_rgb = np.array(PILImage.open(img_path).convert("RGB"))
                    except Exception:
                        orig_rgb = None
            if orig_rgb is None:
                orig_rgb = tensor_to_uint8_rgb(images[i])   # 518×518 fallback

            orig_h, orig_w = orig_rgb.shape[:2]
            heatmap = importance_to_heatmap_2d(importance, orig_h, orig_w)

            stem          = Path(image_id).stem
            heatmap_path  = heatmap_dir / f"{stem}_heatmap.png"
            overlay_path  = overlay_dir / f"{stem}_overlay.png"
            mask_path     = mask_dir / f"{stem}_mask.png"

            # Resolve otolith mask: cache first, then segment (saved for reuse by run_candidates)
            mask_arr = load_mask(mask_path)
            if mask_arr is None:
                info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params())
                if info is not None:
                    mask_arr = info["mask"]
                    save_mask(mask_arr, mask_path)

            save_heatmap(heatmap, heatmap_path)
            save_overlay(
                orig_rgb, heatmap, overlay_path,
                alpha=cfg.interpretation.heatmap_alpha,
                mask=mask_arr,
            )

            results.append(
                {
                    "image_id":     image_id,
                    "heatmap_path": str(heatmap_path),
                    "overlay_path": str(overlay_path),
                    "mask_path":    str(mask_path) if mask_arr is not None else None,
                }
            )

    return results