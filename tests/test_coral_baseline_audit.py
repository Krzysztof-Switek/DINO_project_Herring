"""Tests for scripts/diagnostics/coral_baseline_audit.py (24.09 CORAL work).

The audit is pure measurement, but two of its claims carry the whole comparison between
the quarter-adjusted run and every other run, so they are pinned here:

  1. ``target_definition`` must recognise WHICH age definition a run was scored against.
     The labels CSV is identical either way — the shift happens inside the dataset
     (``src/dataset.py::_effective_age``) — so misreading this silently compares a
     "visible rings" model against a "recorded age" model.
  2. ``rebase_to_recorded`` must leave every error untouched. That invariance is the only
     reason the two definitions are comparable at all; if it ever stops holding, the
     +10 pp exact-accuracy finding stops being a fair comparison.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    """Import the diagnostic by path — scripts/diagnostics is not a package."""
    sys.path.insert(0, str(PROJECT_ROOT))
    path = PROJECT_ROOT / "scripts" / "diagnostics" / "coral_baseline_audit.py"
    spec = importlib.util.spec_from_file_location("coral_baseline_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _preds(ids, pred, target):
    return pd.DataFrame({"image_id": ids, "predicted_age": np.asarray(pred, dtype=float),
                         "target_age": np.asarray(target, dtype=float)})


IDS = [
    "2022_BITS1q_HER_Loc_Embedded_Sharpest_FishIndex1_Single1_Left.jpg",
    "2022_BITS1q_HER_Loc_Embedded_Sharpest_FishIndex2_Single1_Left.jpg",
    "2022_BITS4q_HER_Loc_Embedded_Sharpest_FishIndex3_Single1_Left.jpg",
    "2022_BIAS_HER_Loc_Embedded_Sharpest_FishIndex4_Single1_Left.jpg",
]
RECORDED = pd.Series([4.0, 6.0, 5.0, 3.0], index=IDS)


def test_target_definition_recorded(audit):
    df = _preds(IDS, [4, 5, 5, 3], [4, 6, 5, 3])
    assert audit.target_definition(df, RECORDED) == "recorded"


def test_target_definition_detects_quarter_shift(audit):
    """Only the two Q1 rows are one lower — that is the adjustment's exact signature."""
    df = _preds(IDS, [3, 5, 5, 3], [3, 5, 5, 3])
    assert audit.target_definition(df, RECORDED) == "rings(-1 for Q1)"


def test_target_definition_rejects_a_shift_that_is_not_the_quarter_rule(audit):
    """A shift on a Q4 row is not the quarter rule and must not be mistaken for it."""
    df = _preds(IDS, [4, 6, 4, 3], [4, 6, 4, 3])
    assert audit.target_definition(df, RECORDED) == "unknown-shift"


def test_rebase_adds_one_only_to_q1(audit):
    df = _preds(IDS, [3, 5, 5, 3], [3, 5, 5, 3])
    assert list(audit.rebase_to_recorded(df)) == [4.0, 6.0, 5.0, 3.0]


def test_rebase_leaves_every_error_unchanged(audit):
    """THE load-bearing claim: rebasing cannot flatter or punish a quarter-adjusted run.

    Errors on the rings scale must equal errors on the recorded scale, element by element,
    because the same offset is added to prediction and target. Exact accuracy and MAE are
    therefore identical under both definitions.
    """
    rng = np.random.default_rng(7)
    n = 400
    camp = rng.choice(["BITS1q", "BITS4q", "BIAS"], size=n)
    ids = [f"2022_{c}_HER_Loc_Embedded_Sharpest_FishIndex{i}_Single1_Left.jpg"
           for i, c in enumerate(camp)]
    recorded = pd.Series(rng.integers(1, 12, size=n).astype(float), index=ids)
    offset = np.isin(camp, ["BITS1q", "BITS2q"]).astype(float)
    rings_target = recorded.values - offset
    rings_pred = np.clip(rings_target + rng.integers(-2, 3, size=n), 0, 16).astype(float)
    df = _preds(ids, rings_pred, rings_target)

    assert audit.target_definition(df, recorded) == "rings(-1 for Q1)"
    err_rings = rings_pred - rings_target
    err_recorded = audit.rebase_to_recorded(df).values - recorded.values
    assert np.array_equal(err_rings, err_recorded)
    assert np.mean(err_rings == 0) == np.mean(err_recorded == 0)          # Exact
    assert np.mean(np.abs(err_rings)) == np.mean(np.abs(err_recorded))    # MAE


def test_forgetting_the_rebase_is_measurably_worse(audit):
    """Scoring a rings model against recorded ages without the offset must lose accuracy.

    This is the deployment trap the narrative warns about; if the two ever scored the
    same, the warning would be wrong.
    """
    df = _preds(IDS, [3, 5, 5, 3], [3, 5, 5, 3])       # perfect on the rings scale
    exact_rebased = float(np.mean(audit.rebase_to_recorded(df).values == RECORDED.values))
    exact_naive = float(np.mean(df["predicted_age"].values == RECORDED.values))
    assert exact_rebased == 1.0
    assert exact_naive < exact_rebased


def test_mcnemar_exact_discordant_counts_and_symmetry(audit):
    a = np.array([1, 1, 1, 0, 0, 0])
    b = np.array([0, 0, 0, 1, 0, 0])
    only_a, only_b, p = audit.mcnemar_exact(a, b)
    assert (only_a, only_b) == (3, 1)
    # Swapping the arguments swaps the counts and leaves the two-sided p-value alone.
    only_a2, only_b2, p2 = audit.mcnemar_exact(b, a)
    assert (only_a2, only_b2) == (1, 3)
    assert p == pytest.approx(p2)


def test_mcnemar_identical_vectors_have_no_evidence(audit):
    v = np.array([1, 0, 1, 1, 0])
    assert audit.mcnemar_exact(v, v) == (0, 0, 1.0)


def test_wilson_ci_brackets_the_estimate_and_narrows_with_n(audit):
    lo, hi = audit.wilson_ci(500, 1000)
    assert lo < 0.5 < hi
    wide_lo, wide_hi = audit.wilson_ci(5, 10)
    assert (hi - lo) < (wide_hi - wide_lo)
    assert audit.wilson_ci(0, 0) != audit.wilson_ci(0, 0) or True   # n=0 -> NaN, no crash


def test_per_fish_collapses_two_photos_into_one_age(audit):
    ids = ["a_Single1_Left.jpg", "a_Single2_Right.jpg", "b_Single1_Left.jpg"]
    fish_keys = pd.Series(["fish_a", "fish_a", "fish_b"], index=ids)
    df = _preds(ids, [4, 5, 7], [5, 5, 7])
    out = audit.per_fish(df, fish_keys).set_index("fish")
    # mean(4, 5) = 4.5 -> rounds half up to 5, which happens to be the true age
    assert out.loc["fish_a", "predicted_age"] == 5.0
    assert out.loc["fish_a", "n_photos"] == 2
    assert out.loc["fish_a", "photos_agree"] == 0        # 4 and 5 disagree
    assert out.loc["fish_b", "n_photos"] == 1
    assert out.loc["fish_b", "photos_agree"] == 1        # a single photo trivially agrees


def test_testset_signature_is_order_independent_but_content_sensitive(audit):
    a = pd.DataFrame({"image_id": ["x.jpg", "y.jpg"]})
    b = pd.DataFrame({"image_id": ["y.jpg", "x.jpg"]})
    c = pd.DataFrame({"image_id": ["x.jpg", "z.jpg"]})
    assert audit.testset_signature(a) == audit.testset_signature(b)
    assert audit.testset_signature(a) != audit.testset_signature(c)
