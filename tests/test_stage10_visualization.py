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
# test_draw_card_shape
# ---------------------------------------------------------------------------

def test_draw_card_shape():
    from src.visualization import draw_increment_card
    H, W = 56, 42
    original = np.zeros((H, W, 3), dtype=np.uint8)
    grid = np.random.rand(4, 3).astype(np.float32)
    dot_positions = [10, 25, 40]
    card = draw_increment_card(original, dot_positions, grid, predicted_age=3, true_age=3)
    assert card.ndim == 3
    assert card.shape[2] == 3
    # 3 panels side by side → width = W*2 + panel_c_width; height = H
    assert card.shape[0] == H
    assert card.shape[1] > W  # at least wider than one panel


# ---------------------------------------------------------------------------
# test_draw_card_last_sigmoid_hollow
# ---------------------------------------------------------------------------

def test_draw_card_last_sigmoid_hollow():
    """When last_sigmoid > 0.3 the last dot should be hollow (not solid yellow fill)."""
    from src.visualization import draw_increment_card
    H, W = 100, 80
    original = np.zeros((H, W, 3), dtype=np.uint8)
    grid = np.ones((7, 6), dtype=np.float32)
    dot_positions = [30, 70]

    card_solid = draw_increment_card(
        original, dot_positions, grid, predicted_age=2, true_age=2, last_sigmoid=0.0
    )
    card_hollow = draw_increment_card(
        original, dot_positions, grid, predicted_age=2, true_age=2, last_sigmoid=0.5
    )
    # The two cards should differ (hollow vs solid last dot)
    assert not np.array_equal(card_solid, card_hollow)


# ---------------------------------------------------------------------------
# test_draw_card_saves_file
# ---------------------------------------------------------------------------

def test_draw_card_saves_file(tmp_path):
    from src.visualization import save_increment_cards, load_original_image
    H, W = 56, 42
    img_name = "fish42.png"
    (tmp_path / "images").mkdir()
    _make_synth_image(tmp_path / "images", img_name, h=H, w=W)

    samples = [{"image_id": img_name, "age": 4, "predicted_age": 4}]
    grid = np.random.rand(4, 3).astype(np.float32)
    importance_grids = {img_name: grid}
    last_sigmoids = {img_name: 0.1}

    saved = save_increment_cards(
        samples=samples,
        image_dir=tmp_path / "images",
        importance_grids=importance_grids,
        last_sigmoids=last_sigmoids,
        output_dir=tmp_path / "cards",
        label="best",
    )
    assert len(saved) == 1
    assert saved[0].exists()
    img = PILImage.open(saved[0])
    assert img.mode == "RGB"
