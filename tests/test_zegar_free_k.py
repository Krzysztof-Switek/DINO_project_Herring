"""ZEGAR evaluation without a forced number of points (30.09)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnostics.expert_annotation_eval_wedge import (decode_peaks, geometric_t, paired_eval,
                                                              summarize)

DENS = np.array([0.9, 0.8, 0.05, 0.6, 0.02, 0.3, 0.7])
T = np.array([0.10, 0.12, 0.30, 0.50, 0.60, 0.75, 0.90])     # 0.10 and 0.12 are too close
VALID = np.ones_like(DENS)


def test_oracle_takes_the_expert_count():
    assert len(decode_peaks(DENS, T, VALID, "oracle", 3)) == 3


def test_count_uses_the_maps_own_sum():
    idx = decode_peaks(DENS, T, VALID, "count", 99)                  # sum 3.37 -> 3
    assert len(idx) == 3


def test_threshold_keeps_only_active_peaks_with_spacing():
    idx = decode_peaks(DENS, T, VALID, "threshold", 99)
    assert sorted(idx) == [0, 3, 6]                                  # 0.8 at t=0.12 suppressed


def test_count_respects_tissue():
    valid = VALID.copy(); valid[[0, 1]] = 0.0                        # 1.7 of the mass off-tissue
    assert len(decode_peaks(DENS, T, valid, "count", 99)) == 2


def test_geometric_prior_shrinks_outward():
    g = geometric_t(5)
    gaps = np.diff(g)
    assert len(g) == 5 and all(0 < v < 1 for v in g) and all(gaps[i] > gaps[i + 1] for i in range(3))
    assert geometric_t(0) == []


def test_summary_precision_f1_and_count_error():
    rows = [paired_eval([0.2, 0.5, 0.9], [0.2, 0.5], 100.0, 0.05),   # 2 pairs, 1 spurious
            paired_eval([0.3], [0.3, 0.7], 100.0, 0.05)]             # 1 pair, 1 missed
    s = summarize(rows)
    assert s["total_pairs"] == 3 and s["total_model_points"] == 4 and s["total_gt_rings"] == 4
    assert s["precision"] == pytest.approx(0.75) and s["pairing_coverage"] == pytest.approx(0.75)
    assert s["f1"] == pytest.approx(0.75)
    assert s["count_mae"] == pytest.approx(1.0) and s["count_exact"] == 0.0


def test_ring_tolerances_half_the_nearest_gap():
    from scripts.diagnostics.expert_annotation_eval_wedge import ring_tolerances
    tol = ring_tolerances([0.40, 0.60, 0.70, 0.95], 0.10)
    assert tol == pytest.approx([0.10, 0.05, 0.05, 0.10])     # capped at max_gap 0.10
    assert ring_tolerances([0.5], 0.10) == [0.05]


def test_tight_tolerance_rejects_points_far_from_any_ring():
    """A point in a wide gap pairs under the loose 10 %-of-axis rule but not under half spacing;
    under quarter spacing a point must sit in the nearer half of the way to its ring."""
    from scripts.diagnostics.expert_annotation_eval_wedge import ring_tolerances
    gt = [0.40, 0.46]
    loose = paired_eval([0.52], gt, 100.0, 0.10)
    half = paired_eval([0.52], gt, 100.0, 0.10, ring_tolerances(gt, 0.10, 0.5))
    assert loose["n_pairs"] == 1 and half["n_pairs"] == 0
    q = ring_tolerances(gt, 0.10, 0.25)                       # 0.015 each
    assert paired_eval([0.425], gt, 100.0, 0.10, q)["n_pairs"] == 0   # midway: no ring claimed
    assert paired_eval([0.41, 0.455], gt, 100.0, 0.10, q)["n_pairs"] == 2
