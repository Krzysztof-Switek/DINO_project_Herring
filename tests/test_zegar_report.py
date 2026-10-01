"""zegar_report.py (01.10): controls, bootstrap and the §2.3 verdict do what the plan says."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.zegar_report import (arm_verdicts, boot_ci, hits, population_prior_points,
                                              spearman)


def test_population_prior_is_leave_one_out():
    gt_all = {"a": [0.9, 0.7], "b": [0.9, 0.7], "c": [0.2]}
    pts = population_prior_points(gt_all, "c", 2)          # c's own 0.2 must not count
    assert sorted(round(p, 2) for p in pts) == [0.7, 0.9]
    assert population_prior_points(gt_all, "a", 0) == []


def test_hits_respect_per_ring_tolerance():
    h = hits([0.50, 0.81], [0.5, 0.8, 0.9], [0.01, 0.005, 0.05])
    assert h.tolist() == [True, False]


def test_boot_ci_brackets_the_difference():
    rng = np.random.default_rng(0)
    a, b, n = np.full(30, 3.0), np.full(30, 1.0), np.full(30, 4.0)
    lo, hi = boot_ci(a, b, n, rng)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_spearman_none_for_constant_input():
    assert spearman([1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6]) is None
    assert spearman([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]) == 1.0


def _table(arm, cis, cov=0.3):
    return [{"head": f"{arm}_seed{i}", "arm": arm, "precision": 0.6, "prior_precision": 0.5,
             "rule_precision": 0.4, "oracle_coverage": 0.5, "coverage": cov, "inner_third_share": 0.0,
             "corr_outer": 0.1, "precision_minus_prior_ci95": c} for i, c in enumerate(cis)]


def test_verdict_rules_of_section_2_3():
    pos = [[0.01, 0.2]] * 4 + [[-0.01, 0.1]]
    neg = [[0.01, 0.2]] * 3 + [[-0.1, 0.1]] * 2
    t = pd.DataFrame(_table("A5", [[-0.1, 0.1]] * 5) + _table("B1", pos) + _table("B2", neg)
                     + _table("B3", pos, cov=0.2))
    mat = {h: True for h in t["head"]}
    v = {r["arm"]: r["verdict"] for r in arm_verdicts(t, mat, "A5")}
    assert v["A5"] == "negatywny"
    assert v["B1"] == "H1 potwierdzona"
    assert v["B2"] == "negatywny"
    assert v["B3"] == "niejednoznaczny"       # beats the prior but loses > 5 pt coverage
    mat["B1_seed0"] = False
    assert {r["arm"]: r["verdict"] for r in arm_verdicts(t, mat, "A5")}["B1"] == "niejednoznaczny"
