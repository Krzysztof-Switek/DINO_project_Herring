"""Tests for scripts/diagnostics/coral_quarter_ab_report.py.

The report has one job beyond arithmetic: refuse to compare runs that are not what their
directory name claims. A run once used pure defaults because its config never reached the
server (`plans and summaries/22.09_wedge_b_analiza.md`), and a seed-replicate experiment where
one arm silently ran the other arm's target would look like a null result.
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
def rep():
    sys.path.insert(0, str(PROJECT_ROOT))
    path = PROJECT_ROOT / "scripts" / "diagnostics" / "coral_quarter_ab_report.py"
    spec = importlib.util.spec_from_file_location("coral_quarter_ab_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CAMPS = ["BITS1q", "BITS4q", "BIAS"]


def _ids(n):
    return [f"2022_{CAMPS[i % 3]}_HER_Loc_Embedded_Sharpest_FishIndex{i}_Single1_Left.jpg"
            for i in range(n)]


def _make_run(root: Path, arm: str, seed: int, recorded: pd.Series, error: np.ndarray):
    """A fake run directory whose predictions are `recorded + error`, on the arm's own scale."""
    d = root / f"24.09_quarter_ab_{arm}_seed{seed}" / "emb_on_emb"
    d.mkdir(parents=True)
    ids = list(recorded.index)
    is_q1 = np.isin([i.split("_")[1] for i in ids], ("BITS1q", "BITS2q")).astype(int)
    offset = is_q1 if arm == "on" else 0
    target = recorded.to_numpy(dtype=float) - offset
    pd.DataFrame({
        "image_id": ids,
        "predicted_age": (target + error).astype(int),
        "target_age": target,
        "abs_error": np.abs(error).astype(int),
        "metadata_used": False,
    }).to_csv(d / "predictions.csv", index=False)
    return d.parent


@pytest.fixture
def recorded():
    ids = _ids(60)
    return pd.Series(np.arange(len(ids)) % 7 + 2, index=ids, name="age").astype(float)


def test_discovers_both_arms_and_all_seeds(rep, tmp_path, recorded):
    for arm in ("off", "on"):
        for seed in (42, 7):
            _make_run(tmp_path, arm, seed, recorded, np.zeros(len(recorded)))
    runs = rep.discover_runs([tmp_path])
    assert {(r["arm"], r["seed"]) for r in runs} == {("off", 42), ("off", 7),
                                                    ("on", 42), ("on", 7)}


def test_scores_both_arms_on_the_recorded_scale(rep, tmp_path, recorded):
    """A perfect run must score 100 % in BOTH arms — the rebase is what makes that true."""
    for arm in ("off", "on"):
        d = _make_run(tmp_path, arm, 42, recorded, np.zeros(len(recorded)))
        row = rep.score_run({"dir": d, "arm": arm, "seed": 42,
                             "csv": d / "emb_on_emb" / "predictions.csv"}, recorded)
        assert row["flag_as_declared"], row
        assert row["Exact"] == pytest.approx(1.0), arm
        assert row["MAE"] == pytest.approx(0.0), arm


def test_flags_a_run_whose_target_scale_contradicts_its_name(rep, tmp_path, recorded):
    """An 'on' directory that actually trained on recorded age must be caught, not averaged in."""
    d = _make_run(tmp_path, "off", 42, recorded, np.zeros(len(recorded)))
    mis = d.parent / "24.09_quarter_ab_on_seed42"
    d.rename(mis)
    row = rep.score_run({"dir": mis, "arm": "on", "seed": 42,
                         "csv": mis / "emb_on_emb" / "predictions.csv"}, recorded)
    assert row["flag_as_declared"] is False
    assert row["target_scale"] == "recorded"


def test_paired_stats_use_only_seeds_present_in_both_arms(rep):
    rows = pd.DataFrame({
        "arm": ["off", "off", "on", "on"],
        "seed": [42, 7, 42, 99],
        "Exact": [0.40, 0.42, 0.50, 0.60],
    })
    p = rep.paired_stats(rows, "Exact")
    assert p["seeds"] == [42]
    assert p["mean"] == pytest.approx(0.10)
    assert p["n_seeds"] == 1
    assert "p_two_sided" not in p          # one pair carries no test


def test_paired_stats_report_wins_and_a_p_value(rep):
    rows = pd.DataFrame({
        "arm": ["off"] * 3 + ["on"] * 3,
        "seed": [42, 7, 13] * 2,
        "Exact": [0.40, 0.41, 0.39, 0.50, 0.52, 0.48],
    })
    p = rep.paired_stats(rows, "Exact")
    assert p["n_seeds"] == 3
    assert p["wins_for_ON"] == 3
    assert p["mean"] == pytest.approx(0.10, abs=1e-9)
    assert p["p_two_sided"] < 0.05


def test_paired_stats_handle_no_overlapping_seeds(rep):
    rows = pd.DataFrame({"arm": ["off", "on"], "seed": [1, 2], "Exact": [0.4, 0.5]})
    assert rep.paired_stats(rows, "Exact")["n_seeds"] == 0


def test_main_refuses_when_no_runs_exist(rep, tmp_path):
    import sys as _sys
    argv = ["coral_quarter_ab_report.py", "--roots", str(tmp_path), "--out", str(tmp_path / "o")]
    old = _sys.argv
    _sys.argv = argv
    try:
        with pytest.raises(SystemExit, match="main_quarter_ab"):
            rep.main()
    finally:
        _sys.argv = old
