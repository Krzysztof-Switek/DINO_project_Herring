"""Tests for src/comparison_report.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_predictions(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ages = rng.integers(1, 10, size=n)
    noise = rng.integers(-2, 3, size=n)
    predicted = np.clip(ages + noise, 1, 16).astype(int)
    return pd.DataFrame({
        "image_id": [f"img_{i}.png" for i in range(n)],
        "age": ages,
        "predicted_age": predicted,
    })


# ---------------------------------------------------------------------------
# test_compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics():
    from src.comparison_report import compute_metrics
    y_true = np.array([1, 2, 3, 4, 5], dtype=float)
    y_pred = np.array([1, 3, 3, 5, 5], dtype=float)
    m = compute_metrics(y_true, y_pred)
    assert abs(m["MAE"] - 0.4) < 1e-6   # |0|+|1|+|0|+|1|+|0| / 5 = 0.4
    assert m["RMSE"] >= m["MAE"]
    assert "R2" in m
    assert 0.0 <= m["Acc1yr"] <= 1.0
    assert 0.0 <= m["Acc2yr"] <= 1.0
    assert isinstance(m["Bias"], float)


# ---------------------------------------------------------------------------
# test_cross_comment_good / bad
# ---------------------------------------------------------------------------

def test_cross_comment_good():
    from src.comparison_report import cross_comment
    comment = cross_comment(own_mae=1.0, cross_mae=1.4)
    assert "generalizuje" in comment.lower()


def test_cross_comment_bad():
    from src.comparison_report import cross_comment
    comment = cross_comment(own_mae=1.0, cross_mae=2.0)
    assert "słaba" in comment.lower() or "cross" in comment.lower()


# ---------------------------------------------------------------------------
# test_build_report_creates_file
# ---------------------------------------------------------------------------

def test_build_report_creates_file(tmp_path):
    from src.comparison_report import build_comparison_report
    results = {
        "emb_on_emb":       _make_predictions(seed=0),
        "notemb_on_notemb": _make_predictions(seed=1),
        "emb_on_notemb":    _make_predictions(seed=2),
        "notemb_on_emb":    _make_predictions(seed=3),
    }
    training_logs = {
        "embedded": [
            {"epoch": i, "train_loss": 1.0 / (i + 1), "val_loss": 1.1 / (i + 1),
             "val_mae": 2.0 / (i + 1), "lr": 1e-4}
            for i in range(5)
        ],
        "not_embedded": [
            {"epoch": i, "train_loss": 1.2 / (i + 1), "val_loss": 1.3 / (i + 1),
             "val_mae": 2.2 / (i + 1), "lr": 1e-4}
            for i in range(5)
        ],
    }
    dataset_stats = {
        "counts": {
            "Embedded":    {"train": 100, "val": 20, "test": 20},
            "NotEmbedded": {"train": 90,  "val": 18, "test": 18},
        },
        "orphan_count": 5,
        "age_distributions": {
            "Embedded":    list(range(1, 10)) * 3,
            "NotEmbedded": list(range(1, 10)) * 2,
        },
    }
    out = tmp_path / "report.html"
    build_comparison_report(
        results=results,
        training_logs=training_logs,
        increment_cards={"best": [], "worst": []},
        dataset_stats=dataset_stats,
        output_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


# ---------------------------------------------------------------------------
# test_report_has_all_sections
# ---------------------------------------------------------------------------

def test_report_has_all_sections(tmp_path):
    from src.comparison_report import build_comparison_report
    results = {k: _make_predictions() for k in
               ["emb_on_emb", "notemb_on_notemb", "emb_on_notemb", "notemb_on_emb"]}
    out = tmp_path / "report.html"
    build_comparison_report(
        results=results,
        training_logs={},
        increment_cards={},
        dataset_stats={"counts": {}, "orphan_count": 0, "age_distributions": {}},
        output_path=out,
        model_info={"backbone": "dinov2_vits14"},
    )
    content = out.read_text(encoding="utf-8")
    # All section anchors must be present
    for section_id in ["A", "B", "C", "D", "E", "F"]:
        assert f'id="{section_id}"' in content, f"Section {section_id} missing from HTML"
