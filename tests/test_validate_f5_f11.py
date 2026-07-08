"""Tests for scripts/validate_f5_f11.py — F5/F11 post-training validation metrics."""
from __future__ import annotations

import numpy as np


def test_monotonicity_stats_non_increasing_passes():
    from scripts.validate_f5_f11 import monotonicity_stats
    probs = np.array([[0.9, 0.7, 0.5, 0.2], [0.8, 0.6, 0.4, 0.1]])
    s = monotonicity_stats(probs)
    assert s["pairs"] == 6
    assert s["violations"] == 0
    assert s["max_increase"] <= 0.0


def test_monotonicity_stats_detects_rank_violation():
    from scripts.validate_f5_f11 import monotonicity_stats
    probs = np.array([[0.4, 0.6, 0.5]])          # 0.4 → 0.6 is an increase
    s = monotonicity_stats(probs)
    assert s["violations"] >= 1
    assert s["max_increase"] > 0.0


def test_active_patch_stats_tracks_age():
    from scripts.validate_f5_f11 import active_patch_stats
    ages = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    patch = np.zeros((len(ages), 100))
    for i, a in enumerate(ages):
        patch[i, :a] = 0.9                        # exactly `a` active patches
    s = active_patch_stats(patch, ages, threshold=0.5)
    assert s["n_samples"] == 8
    assert s["mae_nactive_vs_age"] < 1e-6         # #active == age exactly
    assert s["corr_nactive_age"] > 0.99
    assert s["mean_nactive_per_age"]["5"] == 5.0


def test_active_patch_stats_uncorrelated_is_low():
    from scripts.validate_f5_f11 import active_patch_stats
    rng = np.random.default_rng(0)
    ages = np.arange(1, 21)
    patch = (rng.random((20, 50)) > 0.5).astype(float)  # random, independent of age
    s = active_patch_stats(patch, ages, threshold=0.5)
    assert abs(s["corr_nactive_age"]) < 0.6       # no real relationship
