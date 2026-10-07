"""E8 (07.10): the contiguous normalised-association cut finds block boundaries of a row sequence."""
from __future__ import annotations

import numpy as np
import torch

from scripts.diagnostics.zegar_ncut_decoder import affinity, ncut_boundaries, raw_rows


def _blocks(sizes, d=16, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    protos = rng.normal(size=(len(sizes), d))
    return np.concatenate([protos[i] + noise * rng.normal(size=(n, d)) for i, n in enumerate(sizes)])


def test_cut_recovers_block_starts():
    e = _blocks([6, 5, 7, 4])
    t = np.linspace(0, 1, len(e))
    assert ncut_boundaries(affinity(e), t, 3, min_gap=0.0) == [6, 11, 18]


def test_cut_respects_min_gap_and_returns_fewer_when_needed():
    e = _blocks([6, 5, 7, 4])
    t = np.linspace(0, 1, len(e))
    b = ncut_boundaries(affinity(e), t, 3, min_gap=0.3)
    mids = [0.5 * (t[s - 1] + t[s]) for s in b]
    assert all(m2 - m1 >= 0.3 - 1e-9 for m1, m2 in zip(mids, mids[1:]))
    assert len(ncut_boundaries(affinity(e), t, 10, min_gap=0.3)) <= 3


def test_cut_edge_cases():
    e = _blocks([3, 3])
    t = np.linspace(0, 1, 6)
    assert ncut_boundaries(affinity(e), t, 0) == []
    assert ncut_boundaries(affinity(e), t, 1, min_gap=0.0) == [3]


def test_raw_rows_is_tissue_weighted_mean_per_row():
    shapes = [(2, 3), (1, 2)]
    tok = torch.arange(8 * 2, dtype=torch.float32).reshape(1, 8, 2)
    tissue = torch.ones(1, 8)
    tissue[0, 0] = 0.0
    r = raw_rows(tok, tissue, shapes)
    assert r.shape == (3, 2)
    assert np.allclose(r[0], tok[0, 1:3].mean(0).numpy(), atol=1e-4)
    assert np.allclose(r[2], tok[0, 6:8].mean(0).numpy())
