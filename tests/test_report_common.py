"""Tests for src/report_common.py and the report-generator consolidation."""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest


def test_comparison_reexports_shared_compute_metrics():
    """Both report generators must use the one shared compute_metrics."""
    from src.report_common import compute_metrics as shared
    from src.comparison_report import compute_metrics as reexported
    assert reexported is shared


ORIGINAL_METRIC_KEYS = {"MAE", "RMSE", "R2", "Acc1yr", "Acc2yr", "Bias"}


def test_compute_metrics_values():
    from src.report_common import compute_metrics
    m = compute_metrics(np.array([1, 2, 3, 4, 5]), np.array([1, 3, 3, 5, 5]))
    assert abs(m["MAE"] - 0.4) < 1e-9
    assert m["RMSE"] >= m["MAE"]
    # The original six must never disappear — comparison_report / run_pipeline /
    # entrypoint read them by name. New keys (24.09) are additive, so this is a
    # superset check, not an equality check.
    assert ORIGINAL_METRIC_KEYS <= set(m)


def test_compute_metrics_exact_accuracy():
    """Exact = share of |pred - true| == 0. Hand-counted: 3 of 5 rows match."""
    from src.report_common import compute_metrics
    m = compute_metrics(np.array([1, 2, 3, 4, 5]), np.array([1, 3, 3, 5, 5]))
    assert abs(m["Exact"] - 3 / 5) < 1e-12
    assert abs(m["MedAE"] - 0.0) < 1e-12
    assert m["N"] == 5


def test_compute_metrics_signed_error_decomposition_sums_to_one():
    from src.report_common import compute_metrics
    true = np.array([5, 5, 5, 5, 5, 5, 5])
    pred = np.array([2, 4, 5, 5, 6, 7, 9])       # -3, -1, 0, 0, +1, +2, +4
    m = compute_metrics(true, pred)
    parts = [m["Err_le_m2"], m["Err_m1"], m["Err_0"], m["Err_p1"], m["Err_ge_p2"]]
    assert abs(sum(parts) - 1.0) < 1e-12
    assert m["Err_0"] == m["Exact"]
    assert abs(m["Err_m1"] - 1 / 7) < 1e-12
    assert abs(m["Err_p1"] - 1 / 7) < 1e-12
    assert abs(m["Err_le_m2"] - 1 / 7) < 1e-12   # only the -3
    assert abs(m["Err_ge_p2"] - 2 / 7) < 1e-12   # +2 and +4
    # MAE splits cleanly into the +/-1 band and everything beyond it.
    assert abs(m["MAE_from_pm1"] + m["MAE_from_ge2"] - m["MAE"]) < 1e-12


def test_compute_per_age_metrics_and_macro_exact():
    from src.report_common import compute_per_age_metrics, macro_exact
    true = np.array([1, 1, 1, 1, 9])
    pred = np.array([1, 1, 1, 2, 4])        # age 1: 3/4 exact; age 9: 0/1 exact
    per_age = compute_per_age_metrics(true, pred)
    assert set(per_age) == {1, 9}
    assert per_age[1]["n"] == 4
    assert abs(per_age[1]["Exact"] - 0.75) < 1e-12
    assert abs(per_age[9]["Exact"] - 0.0) < 1e-12
    assert abs(per_age[9]["Bias"] + 5.0) < 1e-12
    # Macro treats the single age-9 fish as heavily as the four age-1 fish, which is
    # exactly why it is reported next to the global number.
    assert abs(macro_exact(per_age) - 0.375) < 1e-12


def test_per_age_metrics_match_comparison_report_definition():
    """The lifted per-age helper must reproduce comparison_report's own formulas."""
    from src.report_common import compute_per_age_metrics
    rng = np.random.default_rng(0)
    true = rng.integers(0, 8, size=200).astype(float)
    pred = np.clip(true + rng.integers(-2, 3, size=200), 0, 16).astype(float)
    per_age = compute_per_age_metrics(true, pred)
    errors = pred - true
    for a in np.unique(true):
        mask = true == a
        assert abs(per_age[int(a)]["MAE"] - float(np.mean(np.abs(errors[mask])))) < 1e-12
        assert abs(per_age[int(a)]["Exact"] - float(np.mean(errors[mask] == 0))) < 1e-12
        assert abs(per_age[int(a)]["Bias"] - float(np.mean(errors[mask]))) < 1e-12


def test_confusion_counts_shape_and_totals():
    from src.report_common import confusion_counts
    true = np.array([0, 1, 1, 3])
    pred = np.array([0, 1, 2, 3])
    cm, ages = confusion_counts(true, pred)
    assert ages == [0, 1, 2, 3]            # union of true and predicted values
    assert cm.shape == (4, 4)
    assert cm.sum() == 4
    assert cm[ages.index(1), ages.index(2)] == 1
    assert int(np.trace(cm)) == 3           # three exact hits


def test_fig_to_b64_is_png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.report_common import fig_to_b64
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    raw = base64.b64decode(fig_to_b64(fig))
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic bytes


def test_img_tag_and_png_to_b64(tmp_path):
    from src.report_common import img_tag, png_to_b64
    tag = img_tag("QQ==", style="width:50%;")
    assert 'src="data:image/png;base64,QQ=="' in tag
    assert "max-width:100%" in tag
    assert png_to_b64(tmp_path / "missing.png") is None


def test_report_build_html_smoke(tmp_path):
    """src/report.py must still assemble an HTML string after the refactor."""
    from src.report import build_html_report
    html = build_html_report(
        labels_csv=None, log_path=None, predictions_csv=None,
        heatmaps_dir=tmp_path, overlays_dir=tmp_path,
        cand_json_dir=tmp_path, cand_overlays_dir=tmp_path,
    )
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# Age target scale (24.09) — shared semantics, used by both CORAL diagnostics
# ---------------------------------------------------------------------------

IDS = [
    "2022_BITS1q_HER_Loc_Embedded_Sharpest_FishIndex1_Single1_Left.jpg",
    "2022_BITS4q_HER_Loc_Embedded_Sharpest_FishIndex2_Single1_Left.jpg",
    "2022_BIAS_HER_Loc_Embedded_Sharpest_FishIndex3_Single1_Left.jpg",
]
RECORDED = np.array([4.0, 5.0, 3.0])


def test_campaign_of_parses_the_filename_convention():
    from src.report_common import campaign_of
    assert list(campaign_of(IDS)) == ["BITS1q", "BITS4q", "BIAS"]
    assert list(campaign_of(["no-underscores.jpg"])) == [""]


def test_target_definition_distinguishes_the_two_scales():
    from src.report_common import target_definition
    assert target_definition(IDS, RECORDED, RECORDED) == "recorded"
    rings = RECORDED - np.array([1.0, 0.0, 0.0])
    assert target_definition(IDS, rings, RECORDED) == "rings(-1 for Q1)"
    # a shift on a non-Q1 row is neither scale and must not be mistaken for one
    odd = RECORDED - np.array([0.0, 1.0, 0.0])
    assert target_definition(IDS, odd, RECORDED) == "unknown-shift"


def test_rebase_adds_the_offset_only_to_q1():
    from src.report_common import rebase_to_recorded
    assert list(rebase_to_recorded(IDS, [3.0, 5.0, 3.0])) == [4.0, 5.0, 3.0]


def test_rebase_leaves_exact_and_mae_unchanged():
    """The invariance the whole cross-scale comparison rests on.

    Adding the same season offset to prediction and target cannot change any error, so a
    quarter-adjusted run's Exact/MAE are the same figures on either scale. What does change
    is forgetting the offset — covered separately.
    """
    from src.report_common import compute_metrics, rebase_to_recorded
    rng = np.random.default_rng(2)
    n = 300
    camp = rng.choice(["BITS1q", "BITS4q", "BIAS"], size=n)
    ids = [f"2022_{c}_HER_Loc_Embedded_Sharpest_FishIndex{i}_Single1_Left.jpg"
           for i, c in enumerate(camp)]
    recorded = rng.integers(1, 12, size=n).astype(float)
    rings_target = recorded - np.isin(camp, ["BITS1q", "BITS2q"]).astype(float)
    rings_pred = np.clip(rings_target + rng.integers(-2, 3, size=n), 0, 16).astype(float)

    on_rings = compute_metrics(rings_target, rings_pred)
    on_recorded = compute_metrics(recorded, rebase_to_recorded(ids, rings_pred))
    for key in ("Exact", "MAE", "Acc1yr", "Bias", "MedAE"):
        assert on_rings[key] == pytest.approx(on_recorded[key])


def test_forgetting_the_offset_costs_accuracy():
    from src.report_common import compute_metrics, rebase_to_recorded
    ids, rings_pred = IDS, np.array([3.0, 5.0, 3.0])       # perfect on the rings scale
    assert compute_metrics(RECORDED, rebase_to_recorded(ids, rings_pred))["Exact"] == 1.0
    assert compute_metrics(RECORDED, rings_pred)["Exact"] < 1.0
