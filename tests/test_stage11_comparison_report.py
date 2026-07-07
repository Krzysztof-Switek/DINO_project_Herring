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


# ---------------------------------------------------------------------------
# test_section_e_has_reasoning_card_caption
# ---------------------------------------------------------------------------

def test_section_e_has_reasoning_card_caption(tmp_path):
    """Section E must include the 6-step reasoning caption (ordered list)."""
    from src.comparison_report import build_comparison_report
    out = tmp_path / "report.html"
    build_comparison_report(
        results={k: _make_predictions() for k in
                 ["emb_on_emb", "notemb_on_notemb", "emb_on_notemb", "notemb_on_emb"]},
        training_logs={},
        increment_cards={"best": [], "worst": []},
        dataset_stats={"counts": {}, "orphan_count": 0, "age_distributions": {}},
        output_path=out,
    )
    content = out.read_text(encoding="utf-8")
    assert "Karty rozumowania" in content
    assert "<ol>" in content
    assert "Surowe zdjęcie" in content
    assert "Strefy roczne" in content
    # Section D caption about heatmaps/overlays distinction
    assert "inferno" in content.lower()
    assert "overlays" in content
    # Doprecyzowanie opisu (post-fix sekcji E):
    # — panel 4: jasno wskazana wstawka 1D w rogu zdjęcia
    assert "wstawka" in content.lower()
    # — panele 5/6 i adnotacja: brak peaków w demo to oczekiwane zachowanie
    assert "oczekiwane" in content.lower()
    assert "find_peaks" in content


# ---------------------------------------------------------------------------
# test_cross_prefixed_keys_are_handled  (regresja: run_pipeline daje cross_*)
# ---------------------------------------------------------------------------

def test_normalize_result_keys_maps_cross_prefix():
    from src.comparison_report import normalize_result_keys
    norm = normalize_result_keys({"cross_emb_on_notemb": "X", "emb_on_emb": "Y"})
    assert norm["emb_on_notemb"] == "X"   # prefixed key exposed under canonical name
    assert norm["emb_on_emb"] == "Y"      # already-canonical key untouched


def test_cross_comment_nan_returns_no_data():
    from src.comparison_report import cross_comment
    assert "brak danych" in cross_comment(float("nan"), 1.0).lower()
    assert "brak danych" in cross_comment(1.0, float("nan")).lower()


def test_cross_prefixed_keys_populate_sections(tmp_path):
    """run_pipeline delivers cross_ prefixed keys — sections C/D must resolve them."""
    from src.comparison_report import build_comparison_report
    results = {
        "emb_on_emb":          _make_predictions(seed=0),
        "notemb_on_notemb":    _make_predictions(seed=1),
        "cross_emb_on_notemb": _make_predictions(seed=2),
        "cross_notemb_on_emb": _make_predictions(seed=3),
    }
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
    # Both CROSS labels present in section C metric table
    assert "Emb → NotEmb ★ CROSS" in content
    assert "NotEmb → Emb ★ CROSS" in content
    # Section D cross cells carry numeric MAE, not the "N/A" placeholder
    # (bare "N/A" also occurs inside base64 PNG blobs, so match the cell text).
    assert "← CROSS" in content
    assert "MAE = N/A" not in content


# ---------------------------------------------------------------------------
# Section G — increment-dot gallery
# ---------------------------------------------------------------------------

def _base_kwargs():
    return dict(
        results={k: _make_predictions() for k in
                 ["emb_on_emb", "notemb_on_notemb", "emb_on_notemb", "notemb_on_emb"]},
        training_logs={},
        increment_cards={},
        dataset_stats={"counts": {}, "orphan_count": 0, "age_distributions": {}},
    )


def test_candidate_overlay_gallery_section(tmp_path):
    """Section G embeds the model-drawn dot overlays passed in candidate_overlays."""
    import numpy as np
    from PIL import Image
    from src.comparison_report import build_comparison_report

    ov = tmp_path / "sample_001_candidates_overlay.png"
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(ov)
    out = tmp_path / "report.html"
    build_comparison_report(
        output_path=out,
        candidate_overlays={"Emb → Emb": [ov]},
        **_base_kwargs(),
    )
    content = out.read_text(encoding="utf-8")
    assert 'id="G"' in content
    assert "Galeria" in content
    assert "Emb → Emb — 1 obraz" in content


def test_no_gallery_section_without_overlays(tmp_path):
    """Section G is omitted when no candidate_overlays are supplied."""
    from src.comparison_report import build_comparison_report
    out = tmp_path / "report.html"
    build_comparison_report(output_path=out, **_base_kwargs())
    assert 'id="G"' not in out.read_text(encoding="utf-8")
