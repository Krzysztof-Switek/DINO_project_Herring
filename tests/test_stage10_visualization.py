"""Tests for src/visualization.py."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synth_image(tmp_path: Path, name: str = "img.png", h: int = 56, w: int = 42) -> Path:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    p = tmp_path / name
    PILImage.fromarray(arr, mode="RGB").save(p)
    return p


def _make_predictions_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "predictions.csv"
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return p


# ---------------------------------------------------------------------------
# test_load_original_image
# ---------------------------------------------------------------------------

def test_load_original_image(tmp_path):
    from src.visualization import load_original_image
    img_path = _make_synth_image(tmp_path, "otolith.png", h=64, w=48)
    result = load_original_image("otolith.png", tmp_path)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape == (64, 48, 3)


# ---------------------------------------------------------------------------
# test_select_top_k
# ---------------------------------------------------------------------------

def test_select_top_k(tmp_path):
    from src.visualization import select_top_k_samples
    rows = [
        {"image_id": f"img_{i}.png", "age": 5, "predicted_age": 5 + i}
        for i in range(10)
    ]
    csv_path = _make_predictions_csv(tmp_path, rows)
    best, worst = select_top_k_samples(csv_path, k_best=3, k_worst=3)
    assert len(best) == 3
    assert len(worst) == 3
    # best: errors 0,1,2; worst: errors 9,8,7
    best_errors = [abs(r["predicted_age"] - r["age"]) for r in best]
    worst_errors = [abs(r["predicted_age"] - r["age"]) for r in worst]
    assert best_errors == sorted(best_errors)
    assert worst_errors == sorted(worst_errors, reverse=True)


# ---------------------------------------------------------------------------
# Reasoning-card pipeline (6 panels)
# ---------------------------------------------------------------------------

def _make_axis_payload(H: int, W: int):
    """Synthetic mask + axis_info + peaks for a horizontal rectangle."""
    import cv2
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask, (W // 5, H // 3), (4 * W // 5, 2 * H // 3), 255, -1)
    centroid = (W // 4, H // 2)
    far_edge = (3 * W // 4, H // 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    axis_info = {
        "mask": mask, "centroid": centroid, "far_edge": far_edge,
        "contour": contour, "length_px": float(W // 2),
    }
    n_samples = 20
    xs = np.linspace(centroid[0], far_edge[0], n_samples).astype(np.int64)
    ys = np.full(n_samples, centroid[1], dtype=np.int64)
    line_xy = np.stack([xs, ys], axis=1)
    profile_1d = np.linspace(0.0, 1.0, n_samples).astype(np.float32)
    peak_indices = np.array([5, 12, 17], dtype=np.int64)
    return mask, axis_info, line_xy, profile_1d, peak_indices


def test_draw_reasoning_card_shape_with_axis():
    from src.visualization import draw_reasoning_card
    H, W = 120, 200
    original = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    mask, axis_info, line_xy, profile_1d, peak_indices = _make_axis_payload(H, W)
    grid = np.random.rand(8, 14).astype(np.float32)
    card = draw_reasoning_card(
        original_rgb=original,
        importance_grid=grid,
        predicted_age=3,
        true_age=3,
        mask=mask,
        axis_info=axis_info,
        peak_indices=peak_indices,
        line_xy=line_xy,
        profile_1d=profile_1d,
    )
    assert card.ndim == 3 and card.shape[2] == 3
    # 3 columns × 2 rows + title bar per panel
    assert card.shape[1] == 3 * W
    assert card.shape[0] > 2 * H   # extra rows for title bars


def test_draw_reasoning_card_fallback_no_axis():
    """When axis_info/mask are None, the function must still produce a card."""
    from src.visualization import draw_reasoning_card
    H, W = 80, 100
    original = np.zeros((H, W, 3), dtype=np.uint8)
    grid = np.random.rand(5, 7).astype(np.float32)
    card = draw_reasoning_card(
        original_rgb=original,
        importance_grid=grid,
        predicted_age=2,
        true_age=4,
        mask=None,
        axis_info=None,
        peak_indices=None,
        line_xy=None,
        profile_1d=None,
    )
    assert card.shape[1] == 3 * W
    assert card.shape[0] > 2 * H


def test_save_reasoning_cards_writes_png(tmp_path):
    from src.visualization import save_reasoning_cards
    H, W = 80, 120
    img_name = "fish99.png"
    (tmp_path / "images").mkdir()
    _make_synth_image(tmp_path / "images", img_name, h=H, w=W)

    mask, axis_info, line_xy, profile_1d, peak_indices = _make_axis_payload(H, W)
    grid = np.random.rand(5, 8).astype(np.float32)

    samples = [{"image_id": img_name, "age": 3, "predicted_age": 3}]
    saved = save_reasoning_cards(
        samples=samples,
        image_dir=tmp_path / "images",
        importance_grids={img_name: grid},
        axis_data={img_name: {
            "mask":         mask,
            "axis_info":    axis_info,
            "peak_indices": peak_indices,
            "line_xy":      line_xy,
            "profile_1d":   profile_1d,
        }},
        output_dir=tmp_path / "cards",
        label="best",
    )
    assert len(saved) == 1
    assert saved[0].exists()
    img = PILImage.open(saved[0])
    assert img.mode == "RGB"
