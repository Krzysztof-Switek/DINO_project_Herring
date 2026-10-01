"""E4 (01.10): the growth-pattern DP decoder finds the true optimum of its own objective."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from scripts.diagnostics.zegar_growth_decoder import LAMBDA, RATIO, SIGMA, growth_dp


def _objective(p, t, idx, min_gap):
    idx = sorted(idx, key=lambda i: t[i])
    ts = [t[i] for i in idx]
    if any(b - a < min_gap - 1e-12 for a, b in zip(ts, ts[1:])):
        return -np.inf
    s = sum(math.log(max(p[i], 1e-4)) for i in idx)
    g = np.diff(ts)
    for a, b in zip(g, g[1:]):
        s -= LAMBDA * (math.log(b / a) - math.log(RATIO)) ** 2 / (2 * SIGMA ** 2)
    return s


@pytest.mark.parametrize("seed,k", [(0, 3), (1, 4), (2, 5), (3, 2), (4, 1)])
def test_dp_equals_brute_force(seed, k):
    rng = np.random.default_rng(seed)
    n = 11
    t = np.sort(rng.uniform(0, 1, n))
    p = rng.uniform(0.01, 0.99, n)
    gap = 0.03
    got = growth_dp(p, t, k, min_gap=gap)
    best = max(_objective(p, t, c, gap) for c in itertools.combinations(range(n), k))
    assert len(got) == k
    assert _objective(p, t, got, gap) == pytest.approx(best)


def test_flat_map_follows_the_growth_rhythm():
    t = np.linspace(0, 1, 200)
    idx = growth_dp(np.full(200, 0.5), t, 5, min_gap=0.02)
    g = np.diff(t[idx])
    assert np.allclose(g[1:] / g[:-1], RATIO, atol=0.08)


def test_bright_rows_win_over_rhythm():
    t = np.linspace(0, 1, 44)
    p = np.full(44, 0.01)
    on = [10, 25, 33, 38]
    p[on] = 0.95
    assert sorted(growth_dp(p, t, 4)) == on


def test_returns_fewer_when_k_rows_do_not_fit():
    t = np.array([0.0, 0.01, 0.02])
    assert len(growth_dp(np.full(3, 0.5), t, 3, min_gap=0.015)) == 2
