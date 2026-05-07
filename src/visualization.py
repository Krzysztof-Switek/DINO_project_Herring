"""Increment annotation cards for otolith images.

Loads original images from disk, overlays numbered dots at predicted increment positions,
and generates a 3-panel card (original+dots | heatmap+dots | vertical importance profile).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage

_DOT_COLOR = (255, 230, 0)      # yellow fill
_DOT_BORDER = (0, 0, 0)         # black border
_AXIS_COLOR = (220, 30, 30)     # dark red reading axis
_DOT_RADIUS = 8
_BORDER_THICKNESS = 1
_FONT_SCALE = 0.35
_TEXT_COLOR_DARK = (0, 0, 0)
_TEXT_WHITE = (255, 255, 255)
_TEXT_SHADOW = (0, 0, 0)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_original_image(image_id: str, image_dir: Path) -> np.ndarray:
    """Load original image from disk without any model preprocessing.

    Returns RGB uint8 ndarray (H, W, 3).
    Raises FileNotFoundError if the image is not found.
    """
    path = Path(image_dir) / image_id
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = PILImage.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_top_k_samples(
    predictions_csv: Path,
    k_best: int = 10,
    k_worst: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Sort predictions by |predicted_age - age| and return best/worst k samples.

    One entry per image (not per fish — if fish has multiple images, all are included).
    Returns (best_k, worst_k) where each element is a list of row dicts.
    """
    df = pd.read_csv(predictions_csv)
    df["abs_error"] = (df["predicted_age"] - df["age"]).abs()
    df_sorted = df.sort_values("abs_error").reset_index(drop=True)
    best = df_sorted.head(k_best).to_dict(orient="records")
    worst = df_sorted.tail(k_worst).sort_values("abs_error", ascending=False).to_dict(orient="records")
    return best, worst


# ---------------------------------------------------------------------------
# Dot position computation
# ---------------------------------------------------------------------------

def compute_dot_positions(
    importance_grid: np.ndarray,
    predicted_age: int,
    image_height: int,
    profile: np.ndarray | None = None,
) -> list[int]:
    """Detect N peaks in the vertical importance profile and map to y-pixel positions.

    Parameters
    ----------
    importance_grid : (H_p, W_p) float32
    predicted_age   : number of dots to place (N peaks)
    image_height    : original image height in pixels
    profile         : pre-computed vertical profile; computed from grid if None

    Returns
    -------
    Sorted list of y-pixel positions (top=far edge of otolith, bottom=nucleus).
    """
    from scipy.signal import find_peaks

    if profile is None:
        profile = importance_grid.mean(axis=1).astype(np.float32)

    H_p = len(profile)
    n = max(1, int(predicted_age))

    # Find up to n peaks; if fewer peaks than n, pad with evenly spaced positions
    peaks, props = find_peaks(profile, distance=max(1, H_p // (n + 1)))
    if len(peaks) >= n:
        # Keep n highest-prominence peaks
        prom = props.get("prominences", profile[peaks])
        top_n_idx = np.argsort(prom)[-n:]
        selected = np.sort(peaks[top_n_idx])
    else:
        # Fallback: evenly spaced positions
        selected = np.linspace(0, H_p - 1, n).astype(int)

    patch_h = image_height / H_p
    y_pixels = [int((idx + 0.5) * patch_h) for idx in selected]
    return sorted(y_pixels)


# ---------------------------------------------------------------------------
# Card drawing
# ---------------------------------------------------------------------------

def draw_increment_card(
    original_rgb: np.ndarray,
    dot_y_positions: list[int],
    importance_grid: np.ndarray,
    predicted_age: int,
    true_age: int,
    last_sigmoid: float = 0.0,
) -> np.ndarray:
    """Compose a 3-panel increment annotation card.

    Panel A — original image with reading axis and numbered increment dots.
    Panel B — JET heatmap overlay with the same dots.
    Panel C — vertical importance profile with horizontal dashed lines at dot positions.

    Returns
    -------
    np.ndarray (H_card, W_card, 3) uint8 — the composed card.
    """
    H_orig, W_orig = original_rgb.shape[:2]
    H_p, W_p = importance_grid.shape

    # --- Panel A: original + axis + dots ---
    panel_a = original_rgb.copy()
    x_mid = W_orig // 2
    # Reading axis — thin dark-red vertical line
    cv2.line(panel_a, (x_mid, 0), (x_mid, H_orig - 1), _AXIS_COLOR, 1)

    n_dots = len(dot_y_positions)
    for i, y in enumerate(dot_y_positions):
        is_last = (i == n_dots - 1)
        is_partial = is_last and last_sigmoid > 0.3
        number = str(i + 1)

        if is_partial:
            # Hollow circle for incomplete increment
            cv2.circle(panel_a, (x_mid, y), _DOT_RADIUS, _DOT_COLOR, _BORDER_THICKNESS)
            label = f"{number}+"
            cv2.putText(panel_a, label, (x_mid + _DOT_RADIUS + 2, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _TEXT_SHADOW, 2)
            cv2.putText(panel_a, label, (x_mid + _DOT_RADIUS + 2, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _TEXT_WHITE, 1)
        else:
            # Filled circle
            cv2.circle(panel_a, (x_mid, y), _DOT_RADIUS, _DOT_BORDER, _DOT_RADIUS + _BORDER_THICKNESS)
            cv2.circle(panel_a, (x_mid, y), _DOT_RADIUS, _DOT_COLOR, -1)
            # Number inside dot
            (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, 1)
            tx = x_mid - tw // 2
            ty = y + th // 2
            cv2.putText(panel_a, number, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _TEXT_COLOR_DARK, 1)

    # Age annotation in corner
    error = abs(predicted_age - true_age)
    frame_color = (0, 200, 0) if error == 0 else (200, 0, 0)
    label_text = f"Pred:{predicted_age} True:{true_age}"
    cv2.putText(panel_a, label_text, (4, H_orig - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT_SHADOW, 2)
    cv2.putText(panel_a, label_text, (4, H_orig - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT_WHITE, 1)
    cv2.rectangle(panel_a, (0, 0), (W_orig - 1, H_orig - 1), frame_color, 2)

    # --- Panel B: heatmap overlay + dots ---
    norm_grid = importance_grid.copy()
    if norm_grid.max() > norm_grid.min():
        norm_grid = (norm_grid - norm_grid.min()) / (norm_grid.max() - norm_grid.min())
    heatmap_uint8 = (norm_grid * 255).clip(0, 255).astype(np.uint8)
    # Resize grid to original image size
    heatmap_full = cv2.resize(heatmap_uint8, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    bgr = cv2.applyColorMap(heatmap_full, cv2.COLORMAP_JET)
    rgb_map = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    panel_b = (0.5 * rgb_map + 0.5 * original_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    # Same dots on panel B
    for i, y in enumerate(dot_y_positions):
        is_partial = (i == n_dots - 1) and last_sigmoid > 0.3
        if is_partial:
            cv2.circle(panel_b, (x_mid, y), _DOT_RADIUS, _DOT_COLOR, _BORDER_THICKNESS)
        else:
            cv2.circle(panel_b, (x_mid, y), _DOT_RADIUS, _DOT_BORDER, _DOT_RADIUS + _BORDER_THICKNESS)
            cv2.circle(panel_b, (x_mid, y), _DOT_RADIUS, _DOT_COLOR, -1)

    # --- Panel C: vertical importance profile (matplotlib figure → numpy) ---
    profile = importance_grid.mean(axis=1).astype(np.float32)
    fig_h = H_orig / 100
    fig, ax = plt.subplots(figsize=(2.0, fig_h))
    y_positions = np.linspace(0, H_orig, len(profile))
    ax.plot(profile, y_positions, color="steelblue", linewidth=1)
    for y in dot_y_positions:
        ax.axhline(y, color="gold", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.set_ylim(H_orig, 0)   # top of image = y=0 → at top of plot
    ax.set_xlabel("Importance", fontsize=6)
    ax.set_ylabel("y-pixel (top=edge)", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.set_title("Profile", fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    w_fig, h_fig = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    panel_c_raw = buf.reshape(h_fig, w_fig, 4)[:, :, :3]
    plt.close(fig)
    # Resize panel C to match H_orig
    panel_c = cv2.resize(panel_c_raw, (panel_c_raw.shape[1], H_orig))

    # --- Compose: horizontal concatenation A | B | C ---
    card = np.concatenate([panel_a, panel_b, panel_c], axis=1)
    return card


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_increment_cards(
    samples: list[dict],
    image_dir: Path,
    importance_grids: dict,
    last_sigmoids: dict,
    output_dir: Path,
    label: str,
) -> list[Path]:
    """Generate and save increment annotation cards for a list of prediction samples.

    Parameters
    ----------
    samples         : list of row-dicts from predictions CSV (must have image_id, age, predicted_age)
    image_dir       : directory containing original images
    importance_grids: mapping image_id → (H_p, W_p) ndarray
    last_sigmoids   : mapping image_id → float (last ordinal sigmoid value)
    output_dir      : directory to write PNG cards
    label           : 'best' or 'worst' (used in filename)

    Returns
    -------
    List of saved PNG paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for sample in samples:
        image_id = sample["image_id"]
        predicted_age = int(sample["predicted_age"])
        true_age = int(sample["age"])

        try:
            original = load_original_image(image_id, image_dir)
        except FileNotFoundError:
            continue

        grid = importance_grids.get(image_id)
        if grid is None:
            continue

        profile = grid.mean(axis=1).astype(np.float32)
        dot_positions = compute_dot_positions(
            grid, predicted_age, original.shape[0], profile=profile
        )
        last_sig = last_sigmoids.get(image_id, 0.0)

        card = draw_increment_card(
            original, dot_positions, grid, predicted_age, true_age, last_sig
        )

        stem = Path(image_id).stem
        out_path = output_dir / f"{label}_{stem}_card.png"
        PILImage.fromarray(card, mode="RGB").save(out_path)
        saved.append(out_path)

    return saved
