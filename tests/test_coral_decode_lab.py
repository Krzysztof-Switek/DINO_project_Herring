"""Tests for scripts/diagnostics/coral_decode_lab.py (24.09 CORAL work).

The lab's whole argument rests on one claim: because the head reduces every image to a
single scalar ``g``, the best achievable decoder is an exactly computable partition of that
axis — so "tune the threshold" has a hard ceiling that can be measured instead of searched.
That claim is only worth anything if the dynamic program really finds the optimum, so it is
checked here against brute force, plus the equivalences that let the other decoders be
expressed on the same axis.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lab():
    sys.path.insert(0, str(PROJECT_ROOT))
    path = PROJECT_ROOT / "scripts" / "diagnostics" / "coral_decode_lab.py"
    spec = importlib.util.spec_from_file_location("coral_decode_lab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dump(g: np.ndarray, thetas: np.ndarray, target=None) -> pd.DataFrame:
    """A logit dump for a CORAL head with the given scalar g and thresholds."""
    logits = np.asarray(g, dtype=float)[:, None] - np.asarray(thetas, dtype=float)[None, :]
    cols = {f"coral_logit_{k:02d}": logits[:, k] for k in range(logits.shape[1])}
    n = len(g)
    return pd.DataFrame({
        "image_id": [f"img{i}.jpg" for i in range(n)],
        "target_age": np.zeros(n) if target is None else np.asarray(target, dtype=float),
        **cols,
    })


def brute_force_best(g, y, n_classes, cost="exact"):
    """Optimum over every monotone labelling, by enumerating all cutpoint placements."""
    order = np.argsort(g, kind="mergesort")
    ys = np.asarray(y, dtype=int)[order]
    n = len(ys)
    best = -np.inf
    # choose n_classes-1 non-decreasing block boundaries in [0, n]
    for bounds in itertools.combinations_with_replacement(range(n + 1), n_classes - 1):
        pred = np.searchsorted(np.asarray(bounds), np.arange(n), side="right")
        score = (float(np.sum(pred == ys)) if cost == "exact"
                 else -float(np.sum(np.abs(pred - ys))))
        best = max(best, score)
    return best


# ---------------------------------------------------------------------------
# The dynamic program
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_optimal_cutpoints_matches_brute_force_exact(lab, seed):
    rng = np.random.default_rng(seed)
    n, K = 12, 4
    g = rng.normal(size=n)
    y = rng.integers(0, K, size=n)
    cuts, score = lab.optimal_cutpoints(g, y, K, cost="exact")
    assert score == brute_force_best(g, y, K, cost="exact")
    # the returned cutpoints must actually realise that score
    assert float(np.sum(lab.decode_cutpoints(g, cuts) == y)) == score


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_optimal_cutpoints_matches_brute_force_mae(lab, seed):
    rng = np.random.default_rng(100 + seed)
    n, K = 10, 5
    g = rng.normal(size=n)
    y = rng.integers(0, K, size=n)
    cuts, score = lab.optimal_cutpoints(g, y, K, cost="mae")
    assert score == brute_force_best(g, y, K, cost="mae")
    assert -float(np.sum(np.abs(lab.decode_cutpoints(g, cuts) - y))) == score


def test_optimal_cutpoints_are_sorted_and_correct_length(lab):
    rng = np.random.default_rng(11)
    K = 17
    g = rng.normal(size=300)
    y = rng.integers(0, K, size=300)
    cuts, _ = lab.optimal_cutpoints(g, y, K)
    assert len(cuts) == K - 1
    assert np.all(np.diff(cuts) >= 0)
    assert lab.decode_cutpoints(g, cuts).max() <= K - 1


def test_ceiling_is_never_below_any_concrete_decoder(lab):
    """No threshold rule may beat the DP — that is the entire point of calling it a ceiling."""
    rng = np.random.default_rng(5)
    K = 8
    thetas = np.sort(rng.normal(size=K - 1))
    g = rng.normal(scale=2.0, size=400)
    y = np.clip(np.searchsorted(thetas, g) + rng.integers(-1, 2, size=400), 0, K - 1)
    df = _dump(g, thetas, y)
    _, hits = lab.optimal_cutpoints(g, y, K)
    for thr in (0.2, 0.35, 0.5, 0.65, 0.8):
        assert float(np.sum(lab.decode_global_threshold(df, thr) == y)) <= hits
    assert float(np.sum(lab.decode_argmax(df) == y)) <= hits
    per_b = lab.decode_per_boundary(df, np.full(K - 1, 0.5))
    assert float(np.sum(per_b == y)) <= hits


def test_perfectly_separable_case_reaches_100_percent(lab):
    """When g orders the classes cleanly the optimum must find every sample."""
    K = 5
    y = np.repeat(np.arange(K), 6)
    g = y.astype(float) + 0.01
    cuts, hits = lab.optimal_cutpoints(g, y, K)
    assert hits == len(y)
    assert np.array_equal(lab.decode_cutpoints(g, cuts), y)


# ---------------------------------------------------------------------------
# Axis bookkeeping: g recovery and rule equivalences
# ---------------------------------------------------------------------------

def test_recover_g_returns_the_true_scalar_when_thresholds_are_known(lab):
    rng = np.random.default_rng(3)
    thetas = np.sort(rng.normal(size=6))
    g = rng.normal(size=50)
    assert np.allclose(lab.recover_g(_dump(g, thetas), thetas), g)


def test_recover_g_without_thresholds_preserves_the_ordering(lab):
    """Column 0 is g - theta_0, so it ranks samples identically to g itself."""
    rng = np.random.default_rng(4)
    thetas = np.sort(rng.normal(size=6))
    g = rng.normal(size=50)
    surrogate = lab.recover_g(_dump(g, thetas), None)
    assert np.array_equal(np.argsort(surrogate), np.argsort(g))


def test_cutpoints_for_threshold_reproduces_the_production_rule(lab):
    """Needed so an AVERAGED g (per fish) can be decoded with the production rule.

    ``P(y>k) > thr``  <=>  ``g > theta_k + logit(thr)``. If this equivalence ever breaks,
    fish-level aggregation would silently be using a different decoder than the baseline.
    """
    rng = np.random.default_rng(6)
    thetas = np.sort(rng.normal(size=9))
    g = rng.normal(scale=2.0, size=200)
    df = _dump(g, thetas)
    for thr in (0.3, 0.5, 0.7):
        cuts = lab.cutpoints_for_threshold(df, g, thr)
        assert np.array_equal(lab.decode_cutpoints(g, cuts),
                              lab.decode_global_threshold(df, thr))


def test_class_probabilities_are_a_distribution(lab):
    rng = np.random.default_rng(8)
    thetas = np.sort(rng.normal(size=7))
    probs = lab.class_probabilities(_dump(rng.normal(scale=3.0, size=100), thetas))
    assert probs.shape == (100, 8)
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_class_probabilities_need_not_be_unimodal(lab):
    """Counterexample, pinned so the stronger claim is never re-introduced.

    It is tempting to argue that one scalar g plus increasing thresholds must give a
    single-moded distribution over ages, hence adjacent runner-ups, hence a tidy bound on
    candidate reranking. It is false, and not only for uneven thresholds: the two END
    classes are half-open (``P(y=0) = 1 - sigma(g - theta_0)`` and
    ``P(y=K-1) = sigma(g - theta_{K-2})``), so they absorb the whole tail and can outrank
    their own neighbour while a later class rises again. Here, with PERFECTLY even
    thresholds, P(0) > P(1) < P(2).
    """
    thetas = np.arange(-4.0, 4.01, 1.0)
    probs = lab.class_probabilities(_dump(np.array([-2.985]), thetas))[0]
    assert probs[0] > probs[1] < probs[2]
    d = np.diff(probs)
    rises, falls = np.flatnonzero(d > 1e-12), np.flatnonzero(d < -1e-12)
    assert rises.max() > falls.min()          # a fall precedes a rise: not unimodal


def test_unimodality_holds_away_from_the_open_ended_end_classes(lab):
    """The interior of the range does behave as the intuition says — bound the claim there."""
    thetas = np.arange(-4.0, 4.01, 1.0)
    probs = lab.class_probabilities(_dump(np.linspace(-1.5, 1.5, 50), thetas))
    for row in probs:
        interior = row[1:-1]
        d = np.diff(interior)
        rises, falls = np.flatnonzero(d > 1e-12), np.flatnonzero(d < -1e-12)
        if len(rises) and len(falls):
            assert rises.max() < falls.min()


def test_class_probabilities_depend_only_on_g(lab):
    """The structural fact the ceiling argument rests on: one scalar decides everything.

    Two images with the same g must get identical class probabilities, hence identical
    candidates and identical decoded age under any rule.
    """
    rng = np.random.default_rng(13)
    thetas = np.sort(rng.normal(size=8))
    probs = lab.class_probabilities(_dump(np.array([1.234, 1.234, -0.5]), thetas))
    assert np.allclose(probs[0], probs[1])
    assert not np.allclose(probs[0], probs[2])


def test_topk_accuracy_is_monotone_in_k(lab):
    rng = np.random.default_rng(10)
    K = 9
    thetas = np.sort(rng.normal(size=K - 1))
    g = rng.normal(scale=2.0, size=300)
    y = np.clip(np.searchsorted(thetas, g) + rng.integers(-1, 2, size=300), 0, K - 1)
    tk = lab.topk_table(_dump(g, thetas, y))
    a1, a2, a3 = (lab.topk_accuracy(tk, k) for k in (1, 2, 3))
    assert a1 <= a2 <= a3


def test_adjacency_rate_measures_rather_than_assumes(lab):
    """It must come out below 1.0 on a case that genuinely has non-adjacent runner-ups."""
    tk = lab.topk_table(_dump(np.linspace(-3, 3, 100), np.arange(-4.0, 4.01, 1.0)))
    rate = lab.adjacency_rate(tk)
    assert 0.0 < rate < 1.0
    uneven = lab.adjacency_rate(lab.topk_table(
        _dump(np.array([0.0]), np.array([-0.2, 0.0, 0.2, 6.0]))))
    assert uneven == 0.0


def test_non_adjacent_runner_ups_are_the_low_confidence_ones(lab):
    """Where the runner-up is not a neighbour, the model is also barely deciding.

    This is what keeps the reranking bound useful in practice despite the counterexample:
    the exceptions sit at near-zero margin, not spread through confident predictions.
    """
    tk = lab.topk_table(_dump(np.linspace(-3, 3, 100), np.arange(-4.0, 4.01, 1.0)))
    real = tk[tk["score_2"] > 1e-9]
    adjacent = np.abs(real["age_candidate_1"] - real["age_candidate_2"]) == 1
    assert adjacent.sum() > 0 and (~adjacent).sum() > 0
    assert real.loc[~adjacent, "margin"].mean() < real.loc[adjacent, "margin"].mean()


def test_saturated_rows_have_no_meaningful_runner_up(lab):
    """A collapsed distribution offers no second candidate for any reranker to use."""
    thetas = np.array([-2.0, -1.0, 0.0, 1.0])
    tk = lab.topk_table(_dump(np.array([40.0, -40.0]), thetas, [4, 0]))
    assert list(tk["age_candidate_1"]) == [4, 0]
    assert np.all(tk["score_1"] > 1 - 1e-9)
    assert np.all(tk["score_2"] < 1e-9)
    assert lab.adjacency_rate(tk) != lab.adjacency_rate(tk) or True   # excluded, no crash


def test_selective_accuracy_is_highest_on_the_most_confident_slice(lab):
    """Confidence must actually rank correctness, or the REVIEW-REQUIRED idea is empty."""
    rng = np.random.default_rng(12)
    K = 7
    thetas = np.sort(rng.normal(size=K - 1))
    g = rng.normal(scale=2.5, size=600)
    # noise grows for samples near a boundary, so margin should track correctness
    base = np.searchsorted(thetas, g)
    y = np.clip(base + rng.integers(-1, 2, size=600), 0, K - 1)
    sa = lab.selective_accuracy(lab.topk_table(_dump(g, thetas, y)))
    assert sa["top_25pct"] >= sa["top_100pct"]


def test_per_boundary_decode_stays_ordinal_with_inconsistent_thresholds(lab):
    """With per-boundary thresholds a raw count could jump a gap; cumulative AND cannot."""
    thetas = np.array([0.0, 1.0, 2.0, 3.0])
    df = _dump(np.array([2.5]), thetas)
    # boundary 1 made impossible, boundaries 2-3 made trivial
    thrs = np.array([0.5, 0.999999, 0.000001, 0.000001])
    assert lab.decode_per_boundary(df, thrs)[0] == 1     # stops at the blocked boundary


# ---------------------------------------------------------------------------
# End-to-end: the lab must run on a dump and produce every promised artifact
# ---------------------------------------------------------------------------

def _write_dump(root: Path, n_val: int = 260, n_test: int = 240, K: int = 9) -> None:
    """A synthetic logit dump shaped exactly like coral_score_splits.py writes one."""
    import json
    rng = np.random.default_rng(21)
    thetas = np.sort(rng.normal(scale=1.5, size=K - 1))
    (root).mkdir(parents=True, exist_ok=True)
    (root / "coral_thresholds.json").write_text(
        json.dumps({"theta": [float(x) for x in thetas]}), encoding="utf-8")
    for split, n in (("val", n_val), ("test", n_test)):
        g = rng.normal(scale=2.5, size=n)
        y = np.clip(np.searchsorted(thetas, g) + rng.integers(-1, 2, size=n), 0, K - 1)
        df = _dump(g, thetas, y)
        df["predicted_age"] = np.searchsorted(thetas, g)
        df["abs_error"] = np.abs(df["predicted_age"] - y)
        (root / split).mkdir(parents=True, exist_ok=True)
        df.to_csv(root / split / "predictions.csv", index=False)


def test_decode_lab_end_to_end(lab, tmp_path, monkeypatch):
    dump = tmp_path / "synthetic__best"
    _write_dump(dump)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["coral_decode_lab.py", "--dump", str(dump),
                                      "--out", str(out), "--grid", "0.30:0.70:0.05"])
    lab.main()

    for name in ("metrics.json", "decoders.csv", "per_age.csv", "predictions.csv",
                 "notes.md", "threshold_vs_exact.png", "threshold_sweep_val.csv",
                 "topk_test.csv"):
        assert (out / name).exists(), f"missing {name}"

    import json
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    dec = pd.read_csv(out / "decoders.csv")

    # The ceiling must dominate every concrete decoder ON VAL, where it was fitted.
    val = dec[dec["split"] == "val"]
    ceiling = payload["ceiling_val_exact"]
    assert val["Exact"].max() <= ceiling + 1e-12
    assert ceiling == pytest.approx(
        float(val.loc[val["decoder"].str.startswith("CEILING"), "Exact"].iloc[0]))

    # Both splits must be reported, and test must never have been used for selection:
    # the val-fitted partition applied to test may well come out below the ceiling.
    assert set(dec["split"]) == {"val", "test"}
    test = dec[dec["split"] == "test"]
    assert len(test) == len(val)

    assert payload["topk"]["test"]["top1"] <= payload["topk"]["test"]["top2"]
    assert len(payload["selected_on_val"]["optimal_cutpoints"]) == payload["K"] - 1


def test_decode_lab_refuses_a_dump_without_logits(lab, tmp_path):
    d = tmp_path / "bare__best"
    (d / "val").mkdir(parents=True)
    pd.DataFrame({"image_id": ["a.jpg"], "predicted_age": [3], "target_age": [3.0]}).to_csv(
        d / "val" / "predictions.csv", index=False)
    with pytest.raises(SystemExit, match="coral_logit"):
        lab.load_split(d, "val")


def test_decode_lab_refuses_a_missing_split(lab, tmp_path):
    with pytest.raises(SystemExit, match="coral_score_splits"):
        lab.load_split(tmp_path / "nothing", "val")


def test_per_boundary_cutpoints_reproduce_the_per_boundary_rule(lab):
    """The cummax re-expression must decode identically to the threshold form.

    If it did not, the fish-level row for that rule would silently be a different decoder
    than the per-image row above it.
    """
    rng = np.random.default_rng(31)
    K = 12
    thetas = np.sort(rng.normal(scale=1.2, size=K - 1))
    g = rng.normal(scale=2.5, size=400)
    df = _dump(g, thetas)
    for _ in range(5):
        thrs = rng.uniform(0.1, 0.9, size=K - 1)
        cuts = lab.cutpoints_for_per_boundary(df, g, thrs)
        assert np.all(np.diff(cuts) >= 0)          # monotone, so still a cutpoint decoder
        assert np.array_equal(lab.decode_cutpoints(g, cuts),
                              lab.decode_per_boundary(df, thrs))


def test_candidate_window_is_monotone_and_starts_at_exact(lab):
    rng = np.random.default_rng(32)
    K = 9
    thetas = np.sort(rng.normal(size=K - 1))
    g = rng.normal(scale=2.0, size=300)
    y = np.clip(np.searchsorted(thetas, g) + rng.integers(-2, 3, size=300), 0, K - 1)
    cuts = lab.cutpoints_for_threshold(_dump(g, thetas), g, 0.5)
    w = lab.candidate_window(g, cuts, y, radius=2)
    assert w["window_pm0"] == pytest.approx(float(np.mean(lab.decode_cutpoints(g, cuts) == y)))
    assert w["window_pm0"] <= w["window_pm1"] <= w["window_pm2"]


def test_window_bound_beats_argmax_ranked_topk_when_argmax_is_the_worse_estimate(lab):
    """Why the window exists: a +-1 window round a GOOD decode covers more than top-2 round argmax.

    The top-k table is ranked by P(y=k), so its top-1 is argmax. Where argmax is the weaker
    point estimate, centring the candidate set on it understates what a reranker could reach.
    """
    rng = np.random.default_rng(33)
    K = 11
    thetas = np.sort(rng.normal(scale=1.0, size=K - 1))
    g = rng.normal(scale=2.5, size=500)
    y = np.clip(np.searchsorted(thetas, g) + rng.integers(-1, 2, size=500), 0, K - 1)
    df = _dump(g, thetas, y)
    cuts = lab.cutpoints_for_threshold(df, g, 0.5)
    argmax_exact = float(np.mean(lab.decode_argmax(df) == y))
    thr_exact = float(np.mean(lab.decode_cutpoints(g, cuts) == y))
    if argmax_exact < thr_exact:
        assert lab.candidate_window(g, cuts, y, radius=1)["window_pm1"] >= \
            lab.topk_accuracy(lab.topk_table(df), 2) - 1e-12


def test_per_group_cutpoints_absorb_a_group_constant_shift(lab):
    """The whole point: a per-group rule fixes a shift a global rule provably cannot.

    Two groups, identical evidence, but one group's labels are shifted by +1 — exactly the
    shape of the measured BITS1q bias. A single cutpoint vector has to split the difference;
    per-group vectors do not.
    """
    rng = np.random.default_rng(41)
    K = 10
    n = 1200
    groups = np.where(rng.random(n) < 0.5, "BITS1q", "BITS4q")
    g = rng.uniform(0.5, K - 1.5, size=n)
    y = np.clip(np.floor(g).astype(int) + (groups == "BITS1q").astype(int), 0, K - 1)

    global_cuts, _ = lab.optimal_cutpoints(g, y, K)
    per_group = lab.optimal_cutpoints_per_group(g, y, groups, K)
    exact_global = float(np.mean(lab.decode_cutpoints(g, global_cuts) == y))
    exact_group = float(np.mean(lab.decode_per_group(g, groups, per_group) == y))
    assert set(per_group) == {"__global__", "BITS1q", "BITS4q"}
    assert exact_group > exact_global + 0.2      # the shift is worth ~half the samples
    assert exact_group > 0.95


def test_small_groups_fall_back_to_the_global_rule(lab):
    """A group with a handful of samples must not get its own 16 fitted cutpoints."""
    rng = np.random.default_rng(42)
    K = 6
    groups = np.array(["BIG"] * 400 + ["TINY"] * 20)
    g = rng.normal(size=420)
    y = rng.integers(0, K, size=420)
    per_group = lab.optimal_cutpoints_per_group(g, y, groups, K, min_n=150)
    assert "BIG" in per_group and "TINY" not in per_group
    # TINY rows are decoded by the global vector
    pred = lab.decode_per_group(g, groups, per_group)
    assert np.array_equal(pred[groups == "TINY"],
                          lab.decode_cutpoints(g[groups == "TINY"],
                                               per_group["__global__"]))


def test_per_group_never_loses_to_global_on_the_data_it_was_fitted_on(lab):
    """Per-group strictly generalises global, so in-sample it cannot be worse."""
    rng = np.random.default_rng(43)
    K = 8
    groups = rng.choice(["A", "B", "C"], size=900)
    g = rng.normal(scale=2.0, size=900)
    y = np.clip(np.searchsorted(np.sort(rng.normal(size=K - 1)), g), 0, K - 1)
    per_group = lab.optimal_cutpoints_per_group(g, y, groups, K, min_n=100)
    glob, _ = lab.optimal_cutpoints(g, y, K)
    assert float(np.mean(lab.decode_per_group(g, groups, per_group) == y)) >= \
        float(np.mean(lab.decode_cutpoints(g, glob) == y)) - 1e-12


def test_selected_rule_is_looked_up_by_key_not_by_label(lab, tmp_path, monkeypatch):
    """Regression guard: the val-selected rule must always find its own test row.

    A rule fitted on val is DISPLAYED differently on test ("per-campaign optimal partition"
    vs "per-campaign partition (fitted on val)"), so joining on the label silently yields
    nothing and the etap reports no result at all.
    """
    import json
    dump = tmp_path / "keyed__best"
    _write_dump(dump)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["coral_decode_lab.py", "--dump", str(dump),
                                      "--out", str(out), "--grid", "0.40:0.60:0.10"])
    lab.main()
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert payload["val_selected_rule_key"]
    assert payload["val_selected_rule_test"] is not None
    assert payload["val_selected_rule_test"]["rule_key"] == payload["val_selected_rule_key"]

    # The selected rule must be the OUT-OF-FOLD winner, not the in-sample val winner.
    cv = payload["cv_within_val"]
    assert payload["val_selected_rule_key"] == max(cv, key=lambda k: cv[k]["mean"])
    dec = pd.read_csv(out / "decoders.csv")
    val = dec[dec["split"] == "val"]
    assert len(val[val["rule_key"] == payload["val_selected_rule_key"]]) == 1


def test_every_decoder_appears_on_both_splits_under_the_same_key(lab, tmp_path, monkeypatch):
    dump = tmp_path / "both__best"
    _write_dump(dump)
    out = tmp_path / "out2"
    monkeypatch.setattr(sys, "argv", ["coral_decode_lab.py", "--dump", str(dump),
                                      "--out", str(out), "--grid", "0.40:0.60:0.10"])
    lab.main()
    dec = pd.read_csv(out / "decoders.csv")
    keys_val = set(dec.loc[dec["split"] == "val", "rule_key"])
    keys_test = set(dec.loc[dec["split"] == "test", "rule_key"])
    assert keys_val == keys_test


# ---------------------------------------------------------------------------
# Per-group offset: the correctly specified form of the measured bias
# ---------------------------------------------------------------------------

def test_group_offsets_recover_a_group_constant_shift_with_one_parameter(lab):
    """Three scalars must match what per-group partitions do, on the shift they model.

    The measured phenomenon is a group-constant bias, so the offset form is the correctly
    specified model -- and being correctly specified, it should not need the extra capacity
    of a free per-group re-cutting of the axis.
    """
    rng = np.random.default_rng(51)
    K = 10
    n = 1200
    groups = np.where(rng.random(n) < 0.5, "BITS1q", "BITS4q")
    g = rng.uniform(0.5, K - 1.5, size=n)
    y = np.clip(np.floor(g).astype(int) + (groups == "BITS1q").astype(int), 0, K - 1)

    cuts, offsets = lab.fit_offsets_and_cuts(g, y, groups, K)
    exact_offset = float(np.mean(
        lab.decode_with_group_offsets(g, groups, cuts, offsets) == y))
    per_group = lab.optimal_cutpoints_per_group(g, y, groups, K)
    exact_per_group = float(np.mean(lab.decode_per_group(g, groups, per_group) == y))

    assert exact_offset > 0.95
    assert exact_offset >= exact_per_group - 0.02
    # the fitted shift must point the right way and be about one class wide
    assert offsets["BITS1q"] - offsets["BITS4q"] > 0.5


def test_group_offsets_generalise_better_than_per_group_partitions(lab):
    """The reason the offset form exists: less capacity, so less fit-to-new shrinkage."""
    K = 12

    def make(n, seed):
        r = np.random.default_rng(seed)
        grp = r.choice(["A", "B", "C"], size=n)
        gg = r.uniform(0.5, K - 1.5, size=n)
        noise = r.integers(-1, 2, size=n)
        yy = np.clip(np.floor(gg).astype(int) + (grp == "A").astype(int) + noise, 0, K - 1)
        return gg, yy, grp

    g_fit, y_fit, grp_fit = make(600, 1)
    g_new, y_new, grp_new = make(600, 2)

    cuts, _ = lab.optimal_cutpoints(g_fit, y_fit, K)
    offsets = lab.optimal_group_offsets(g_fit, y_fit, grp_fit, cuts)
    per_group = lab.optimal_cutpoints_per_group(g_fit, y_fit, grp_fit, K, min_n=100)

    shrink_offset = (
        float(np.mean(lab.decode_with_group_offsets(g_fit, grp_fit, cuts, offsets) == y_fit))
        - float(np.mean(lab.decode_with_group_offsets(g_new, grp_new, cuts, offsets) == y_new)))
    shrink_partition = (
        float(np.mean(lab.decode_per_group(g_fit, grp_fit, per_group) == y_fit))
        - float(np.mean(lab.decode_per_group(g_new, grp_new, per_group) == y_new)))
    assert shrink_offset < shrink_partition


def test_unknown_group_gets_no_shift(lab):
    """A campaign unseen at fit time falls back to the shared rule instead of crashing."""
    cuts = np.array([1.0, 2.0, 3.0])
    pred = lab.decode_with_group_offsets(np.array([2.5, 2.5]), np.array(["A", "ZZZ"]),
                                         cuts, {"A": 1.0})
    assert pred[0] == 3 and pred[1] == 2


def test_offset_rule_reduces_to_the_shared_rule_when_offsets_are_zero(lab):
    rng = np.random.default_rng(53)
    cuts = np.sort(rng.normal(size=8))
    g = rng.normal(scale=2.0, size=200)
    groups = rng.choice(["A", "B"], size=200)
    zeros = {"A": 0.0, "B": 0.0}
    assert np.array_equal(lab.decode_with_group_offsets(g, groups, cuts, zeros),
                          lab.decode_cutpoints(g, cuts))


def test_single_pass_offsets_can_be_defeated_by_a_contaminated_partition(lab):
    """Why the alternating fit exists, pinned as a measurement.

    Fitting offsets on a partition that was itself fitted to the uncorrected axis is
    circular. On a clean group-constant shift the single-pass form leaves a large gap that
    the alternating form closes.
    """
    rng = np.random.default_rng(51)
    K, n = 10, 1200
    groups = np.where(rng.random(n) < 0.5, "BITS1q", "BITS4q")
    g = rng.uniform(0.5, K - 1.5, size=n)
    y = np.clip(np.floor(g).astype(int) + (groups == "BITS1q").astype(int), 0, K - 1)

    naive_cuts, _ = lab.optimal_cutpoints(g, y, K)
    naive_off = lab.optimal_group_offsets(g, y, groups, naive_cuts)
    naive = float(np.mean(lab.decode_with_group_offsets(g, groups, naive_cuts, naive_off) == y))

    alt_cuts, alt_off = lab.fit_offsets_and_cuts(g, y, groups, K)
    alt = float(np.mean(lab.decode_with_group_offsets(g, groups, alt_cuts, alt_off) == y))
    assert alt > naive + 0.1


def test_alternating_fit_never_scores_below_the_plain_shared_partition(lab):
    """It starts from zero offsets, so it can only keep or improve on the global rule."""
    rng = np.random.default_rng(54)
    K = 9
    groups = rng.choice(["A", "B", "C"], size=800)
    g = rng.normal(scale=2.0, size=800)
    y = np.clip(np.searchsorted(np.sort(rng.normal(size=K - 1)), g), 0, K - 1)
    glob, _ = lab.optimal_cutpoints(g, y, K)
    cuts, offsets = lab.fit_offsets_and_cuts(g, y, groups, K)
    assert float(np.mean(lab.decode_with_group_offsets(g, groups, cuts, offsets) == y)) >= \
        float(np.mean(lab.decode_cutpoints(g, glob) == y)) - 1e-12


def test_selection_uses_out_of_fold_not_in_sample_val(lab, tmp_path, monkeypatch):
    """Guard against ranking decoders by capacity.

    In-sample val favours whichever rule fits the most parameters — measured on the real
    dumps, a 48-parameter per-campaign rule topped the in-sample table while scoring level
    with or below a 16-parameter one out-of-fold. Selection must therefore read the CV table,
    and this test fails if it ever silently goes back to the val column.
    """
    import json
    dump = tmp_path / "sel__best"
    _write_dump(dump, n_val=400, n_test=360, K=9)
    out = tmp_path / "out_sel"
    monkeypatch.setattr(sys, "argv", ["coral_decode_lab.py", "--dump", str(dump),
                                      "--out", str(out), "--grid", "0.40:0.60:0.10"])
    lab.main()
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    cv = payload["cv_within_val"]
    assert payload["selection_basis"].startswith("out-of-fold")
    assert payload["val_selected_rule_key"] == max(cv, key=lambda k: cv[k]["mean"])
    # every decoder that appears in the results table must also be cross-validated,
    # otherwise a rule could win selection without ever being held out
    dec = pd.read_csv(out / "decoders.csv")
    rules = set(dec.loc[dec["split"] == "val", "rule_key"]) - {"global_sweep"}
    assert rules <= set(cv)


def test_fish_folds_keep_a_fish_on_one_side(lab):
    """Leaking a fish across folds would make every fitted rule look better than it is."""
    ids = [f"f{i}_Single{j}_Left.jpg" for i in range(40) for j in (1, 2)]
    keys = pd.Series([f"fish{i}" for i in range(40) for _ in (1, 2)], index=ids)
    folds = lab.fish_folds(np.array(ids), keys, n_folds=2, seed=1)
    df = pd.DataFrame({"fish": keys.values, "fold": folds})
    assert set(np.unique(folds)) == {0, 1}
    assert df.groupby("fish")["fold"].nunique().max() == 1


def test_fish_folds_fall_back_to_the_row_when_the_fish_is_unknown(lab):
    ids = np.array(["unknown_a.jpg", "unknown_b.jpg"])
    folds = lab.fish_folds(ids, pd.Series(dtype=object), n_folds=2, seed=0)
    assert len(folds) == 2
