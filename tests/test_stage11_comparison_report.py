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
    """Section E must include the 2-head reasoning caption (ordered list)."""
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
    assert "GŁOWICA WIEKU (CORAL)" in content
    assert "GŁOWICA LOKALIZACJI (density)" in content
    # Section D caption about heatmaps/overlays distinction (present in the
    # 4-condition comparison; Section D is dropped only for single-condition).
    assert "inferno" in content.lower()
    # Nowy opis 2-głowicowy: jawne panele density (kandydaci / finalne)
    assert "Kandydaci" in content
    assert "Finalne" in content


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


def test_no_gallery_section_g(tmp_path):
    """Sekcja G (galeria kropek) została usunięta — raport nie może jej zawierać."""
    from src.comparison_report import build_comparison_report
    out = tmp_path / "report.html"
    build_comparison_report(output_path=out, **_base_kwargs())
    assert 'id="G"' not in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Faza C — condition-aware (embedded-only) report
# ---------------------------------------------------------------------------

def test_single_condition_report_drops_cross(tmp_path):
    """One condition ⇒ 'Raport treningu (Embedded)', no Section D, no ★ CROSS."""
    from src.comparison_report import build_comparison_report
    out = tmp_path / "report.html"
    build_comparison_report(
        results={"emb_on_emb": _make_predictions(seed=0)},
        training_logs={},
        increment_cards={},
        dataset_stats={"counts": {"Embedded": {"train": 100, "val": 20, "test": 20}},
                       "orphan_count": 3, "age_distributions": {},
                       "active_ptypes": ["Embedded"]},
        output_path=out,
        model_info={"backbone": "dinov2_vits14"},
    )
    content = out.read_text(encoding="utf-8")
    assert "Raport treningu (Embedded)" in content
    assert 'id="D"' not in content        # cross-eval dropped for a single condition
    assert "★ CROSS" not in content
    for sec in ["A", "B", "C", "E", "F"]:
        assert f'id="{sec}"' in content


def test_section_a_funnel_and_per_split_ages(tmp_path):
    """Section A renders the data funnel, a fish column, and per-split age charts."""
    from src.comparison_report import build_comparison_report
    out = tmp_path / "report.html"
    dataset_stats = {
        "counts": {"Embedded": {"train": 60, "val": 12, "test": 12}},
        "fish_counts": {"Embedded": {"train": 30, "val": 6, "test": 6}},
        "age_by_split": {"Embedded": {
            "train": list(range(0, 16)) * 3,
            "val":   list(range(0, 16)),
            "test":  list(range(0, 16)),
        }},
        "age_distributions": {"Embedded": list(range(0, 16)) * 5},
        "orphan_count": 4,
        "active_ptypes": ["Embedded"],
        "funnel": {"on_disk": 200, "parsed": 180, "labeled": 150, "orphans": 30,
                   "embedded": 100, "notembedded": 80},
    }
    build_comparison_report(
        results={"emb_on_emb": _make_predictions(seed=0)},
        training_logs={}, increment_cards={},
        dataset_stats=dataset_stats, output_path=out,
    )
    content = out.read_text(encoding="utf-8")
    assert "Lejek danych" in content       # funnel present
    assert "200" in content and "Ryby" in content   # disk count + fish column
    # per-split small-multiples caption (chart titles live inside the PNG)
    assert "powinny się pokrywać" in content


def test_section_b_component_charts(tmp_path):
    """Section B renders the CORAL-vs-MIL and #active-vs-age charts when logged."""
    from src.comparison_report import build_comparison_report
    out = tmp_path / "report.html"
    logs = {"embedded": [
        {"epoch": i, "train_loss": 1.0 / (i + 1), "val_loss": 1.1 / (i + 1),
         "val_mae": 2.0 / (i + 1), "lr": 1e-4,
         "coral_loss": 0.6 / (i + 1), "mil_loss": 0.4 / (i + 1),
         "mil_active": float(i), "mean_age": 4.0}
        for i in range(5)
    ]}
    build_comparison_report(
        results={"emb_on_emb": _make_predictions(seed=0)},
        training_logs=logs, increment_cards={},
        dataset_stats={"counts": {}, "orphan_count": 0, "age_distributions": {},
                       "active_ptypes": ["Embedded"]},
        output_path=out,
    )
    content = out.read_text(encoding="utf-8")
    # caption text (chart titles live inside the PNG)
    assert "(CORAL vs MIL)" in content
    assert "lokalizuje" in content.lower()


def test_section_c_confusion_bias_n_per_age(tmp_path):
    """Section C adds confusion matrix + signed bias-per-age + n-per-age (11.07 Punkt 4)."""
    from src.comparison_report import build_comparison_report
    out = tmp_path / "report.html"
    build_comparison_report(
        results={"emb_on_emb": _make_predictions(n=120, seed=1)},
        training_logs={}, increment_cards={},
        dataset_stats={"counts": {}, "orphan_count": 0, "age_distributions": {},
                       "active_ptypes": ["Embedded"]},
        output_path=out,
    )
    content = out.read_text(encoding="utf-8")
    assert "Macierz pomyłek" in content              # confusion matrix caption
    assert "Znakowany bias per wiek" in content       # signed bias-per-age caption
    assert "Ile obrazów testowych" in content         # n-per-age caption


def test_section_opencv_removed(tmp_path):
    """20.07: Section H (interactive OpenCV) is REMOVED from the report even when
    opencv_reference is passed (kept out until cards/decision process are refined)."""
    from src.comparison_report import build_comparison_report
    out = tmp_path / "report.html"
    ref = {"a.jpg": {"img": "data:image/png;base64,iVBORw0KGgo=", "w": 60, "h": 40,
                     "line": [[5, 20], [30, 20], [55, 20]], "profile": [0.1, 0.8, 0.2],
                     "true_age": 4, "pred_age": 4}}
    build_comparison_report(
        results={"emb_on_emb": _make_predictions(n=40, seed=2)},
        training_logs={}, increment_cards={},
        dataset_stats={"counts": {}, "orphan_count": 0, "age_distributions": {},
                       "active_ptypes": ["Embedded"]},
        output_path=out, opencv_reference=ref,
    )
    content = out.read_text(encoding="utf-8")
    assert "H. OpenCV" not in content and "OPENCV_DATA" not in content


def test_section_e_reasoning_cards_embedded(tmp_path):
    """Section E embeds the reasoning-card PNGs passed via increment_cards."""
    import numpy as np
    from PIL import Image
    from src.comparison_report import build_comparison_report

    card = tmp_path / "best_fishA_card.png"
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(card)
    out = tmp_path / "report.html"
    kw = _base_kwargs()
    kw["increment_cards"] = {"best": [card], "worst": []}
    build_comparison_report(output_path=out, **kw)
    content = out.read_text(encoding="utf-8")
    assert 'id="E"' in content
    assert "data:image/png;base64," in content   # karta osadzona


def test_section_localization_methods_removed(tmp_path):
    """20.07: the 20-otolith bake-off (sekcje I/J/K/L) is REMOVED from the report even when
    localization_methods is passed — deferred until the reasoning cards + walkthrough are done."""
    from src.comparison_report import build_comparison_report
    out = tmp_path / "report.html"
    b64 = "data:image/png;base64,AAAA"
    loc = {m: [{"image_id": "fishA.jpg", "true_age": 3, "pred_age": 3, "b64": b64, "n_final": 3}]
           for m in ("density", "classical", "consensus", "dp")}
    build_comparison_report(output_path=out, localization_methods=loc, **_base_kwargs())
    content = out.read_text(encoding="utf-8")
    for sec in ('id="I"', 'id="J"', 'id="K"', 'id="L"'):
        assert sec not in content
    assert "density (model)" not in content
