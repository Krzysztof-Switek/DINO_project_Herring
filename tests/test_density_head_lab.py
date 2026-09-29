"""L6 (30.09): density-head lab — the arms test what they claim to test."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.diagnostics.density_head_lab import (
    ARMS, HEAD_KW, CanvasWindowDensityHead, RowDensityHead, TokenCache, arm_loss,
    bce_logit_loss, canvas_dense_mask, canvas_neighbour_counts, local_encoder_layer,
    prior_bias, run_arm, set_prior)
from src.model import RadialAttentionDensityHead, density_count_loss

SHAPES = [(3, 8), (2, 9)]           # small stand-in for the 4 real bands
N_SMALL = sum(r * c for r, c in SHAPES)
REAL_SHAPES = [(11, 97), (12, 129), (9, 145), (12, 161)]


# ---------------------------------------------------------------------------
# A4 — sparse canvas attention is the production layer with a dense mask
# ---------------------------------------------------------------------------

def test_local_layer_equals_production_layer_with_dense_mask():
    torch.manual_seed(0)
    head = RadialAttentionDensityHead(embed_dim=32, **HEAD_KW).eval()
    layer = head.encoder.layers[0]
    x = torch.randn(2, N_SMALL, 32)
    mask = canvas_dense_mask(SHAPES).unsqueeze(0).repeat(2 * HEAD_KW["num_heads"], 1, 1)
    with torch.no_grad():
        ref = head.encoder(x, mask=mask)
        got = local_encoder_layer(layer, x, SHAPES)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_canvas_head_is_production_head_with_canvas_mask():
    """Same weights: A4 head == production head fed the dense canvas mask."""
    torch.manual_seed(1)
    a4 = CanvasWindowDensityHead(32, SHAPES, **HEAD_KW).eval()
    ref = RadialAttentionDensityHead(embed_dim=32, **HEAD_KW).eval()
    ref.load_state_dict(a4.state_dict())
    x, t, th = torch.randn(2, N_SMALL, 32), torch.rand(2, N_SMALL), torch.rand(2, N_SMALL)
    from src.model import polar_fourier_features
    with torch.no_grad():
        h = ref.input_norm(x) + ref.pos_proj(polar_fourier_features(t, th, 4, 4))
        mask = canvas_dense_mask(SHAPES).unsqueeze(0).repeat(8, 1, 1)
        expected = ref.out_head(ref.encoder(h, mask=mask))
        got = a4(x, polar_t=t, polar_theta=th)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_canvas_neighbourhood_is_21_in_the_interior():
    counts = canvas_neighbour_counts(REAL_SHAPES)
    assert counts.shape == (5852,)
    assert counts.max() == 21 and counts.min() == 2 * 4          # corner: 2 rows × 4 cols
    assert np.median(counts) == 21
    dense = (~canvas_dense_mask(SHAPES)).sum(1).numpy()
    np.testing.assert_array_equal(dense, canvas_neighbour_counts(SHAPES))


# ---------------------------------------------------------------------------
# Mechanisms 1 and 2
# ---------------------------------------------------------------------------

def test_prior_bias_value_matches_plan():
    assert prior_bias(3.81, 5852) == pytest.approx(-7.34, abs=0.01)
    assert prior_bias(3.81, 44) == pytest.approx(np.log(3.81 / 44 / (1 - 3.81 / 44)))


def test_prior_init_starts_integral_near_age_not_n_over_2():
    torch.manual_seed(0)
    n, age = 1500, 4.0
    x = torch.randn(4, n, 32)
    default = RadialAttentionDensityHead(embed_dim=32, **HEAD_KW).eval()
    prior = RadialAttentionDensityHead(embed_dim=32, **HEAD_KW).eval()
    set_prior(prior, age, n)
    with torch.no_grad():
        s_def = torch.sigmoid(default(x)).sum(1).mean().item()
        s_pri = torch.sigmoid(prior(x)).sum(1).mean().item()
    assert s_def > n / 4                       # ~N/2 with PyTorch's default init
    assert 0.5 * age < s_pri < 2.0 * age


def test_bce_gradient_does_not_vanish_where_squared_loss_does():
    """An 'on' cell stuck at logit −13: the gradient reaching it (wedge_b's absorbing state)."""
    n = 200
    z = torch.full((1, n), -13.0, requires_grad=True)
    age = torch.tensor([3])
    valid = torch.ones(1, n)
    g = {}
    for name in ("A0", "A3"):
        z.grad = None
        arm_loss(ARMS[name], z, age, valid).backward()
        g[name] = z.grad[0, :3].abs().max().item()        # the top-3 = the "on" cells
    assert g["A0"] < 1e-4
    assert g["A3"] > 0.1


def test_bce_loss_matches_production_structure_on_clean_maps():
    """Perfect map (k cells at +10, rest −10) → both forms ≈ 0; empty map → both clearly > 0."""
    n, k = 100, 4
    z = torch.full((1, n), -10.0); z[0, :k] = 10.0
    age, valid = torch.tensor([k]), torch.ones(1, n)
    assert bce_logit_loss(z, age, valid).item() < 1e-3
    assert density_count_loss(torch.sigmoid(z), age, 1.0, 0.0, valid_mask=valid).item() < 1e-3
    empty = torch.full((1, n), -10.0)
    assert bce_logit_loss(empty, age, valid).item() > 3.0


# ---------------------------------------------------------------------------
# The claim the lab rests on: at N = 5852 the wedge_b recipe does not mature, A3 does
# ---------------------------------------------------------------------------

class _TinyHead(nn.Module):
    """Per-cell MLP with the production out_head layout, so set_prior applies unchanged."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.out_head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.out_head(self.norm(x)).squeeze(-1)


def _synthetic_run(arm_name: str, steps: int = 400, n: int = 5852, d: int = 16, b: int = 8):
    """Synthetic tokens: ⌈age⌉ random cells per image carry a fixed signal direction."""
    torch.manual_seed(0)
    sig = torch.randn(d); sig = sig / sig.norm() * 3
    arm = ARMS[arm_name]
    torch.manual_seed(1)
    head = _TinyHead(d)
    if arm.prior_bias:
        set_prior(head, 4.5, n)
    opt = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(2)

    def batch(g):
        age = torch.randint(3, 7, (b,), generator=g)
        x = torch.randn(b, n, d, generator=g)
        for i in range(b):
            x[i, torch.randperm(n, generator=g)[: int(age[i])]] += sig
        return x, age

    for _ in range(steps):
        x, age = batch(gen)
        loss = arm_loss(arm, head(x), age, torch.ones(b, n))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x, age = batch(torch.Generator().manual_seed(99))
        z = head(x)
    return (torch.sigmoid(z) > 0.5).sum(1).float().mean().item(), z.amax(1).mean().item()


def test_wedge_b_recipe_never_activates_a_cell_but_a3_does():
    active_a0, max_a0 = _synthetic_run("A0")
    active_a3, max_a3 = _synthetic_run("A3")
    assert active_a0 == 0.0 and max_a0 < 0.0          # diffuse map, nothing above 0.5
    assert active_a3 >= 2.0 and max_a3 > 0.0          # mean age 4.5, cells switched on


# ---------------------------------------------------------------------------
# End to end on a tiny fake cache (A4 and A5 — the arms that run locally)
# ---------------------------------------------------------------------------

def _fake_cache(root, n_train=12, n_val=6, dim=32):
    rng = np.random.default_rng(0)
    n = n_train + n_val
    shapes = SHAPES
    P = N_SMALL
    np.save(root / "tokens.npy", rng.standard_normal((n, P, dim)).astype(np.float16))
    t = np.concatenate([np.repeat(np.linspace(0, 1, r), c) for r, c in shapes])
    np.save(root / "polar_t.npy", np.tile(t, (n, 1)).astype(np.float32))
    np.save(root / "polar_theta.npy", rng.uniform(-0.8, 0.8, (n, P)).astype(np.float32))
    np.save(root / "tissue_valid.npy", np.ones((n, P), np.float16))
    np.save(root / "done.npy", np.ones(n, bool))
    import pandas as pd
    ages = rng.integers(1, 6, n)
    pd.DataFrame({"image_id": [f"img{i}" for i in range(n)],
                  "split": ["train"] * n_train + ["val"] * n_val,
                  "age_recorded": ages, "age_quarter": np.maximum(ages - 1, 0)}).to_csv(
        root / "index.csv", index=False)
    (root / "meta.json").write_text(json.dumps({
        "dim": dim, "n_patches": P,
        "data.wedge_band_n_radius_patches": [r for r, _ in shapes],
        "data.wedge_band_n_angle_patches": [c for _, c in shapes]}), encoding="utf-8")


@pytest.mark.parametrize("arm", ["A4", "A5", "A3"])
def test_run_arm_end_to_end(tmp_path, arm):
    (tmp_path / "cache").mkdir()
    _fake_cache(tmp_path / "cache")
    cache = TokenCache(tmp_path / "cache", "quarter")
    s = run_arm(ARMS[arm], cache, seed=0, epochs=2, batch_size=4, device=torch.device("cpu"),
                out_dir=tmp_path / "out", log=lambda m: None)
    assert (tmp_path / "out" / "metrics.csv").exists() and (tmp_path / "out" / "head.pt").exists()
    import pandas as pd
    m = pd.read_csv(tmp_path / "out" / "metrics.csv")
    assert list(m["epoch"]) == [0, 1, 2]
    assert np.isfinite(m["zero_ratio"]).all()
    assert set(s) >= {"matured", "final_zero_ratio"}


def test_row_head_outputs_one_value_per_row():
    torch.manual_seed(0)
    h = RowDensityHead(32, SHAPES).eval()
    x = torch.randn(2, N_SMALL, 32)
    t = torch.cat([torch.linspace(0, 1, r).repeat_interleave(c) for r, c in SHAPES]).expand(2, -1)
    out = h(x, t, torch.ones(2, N_SMALL))
    assert out.shape == (2, sum(r for r, _ in SHAPES), 1)


def test_incomplete_cache_is_refused(tmp_path):
    _fake_cache(tmp_path)
    np.save(tmp_path / "done.npy", np.array([True, False] + [True] * 16))
    with pytest.raises(SystemExit):
        TokenCache(tmp_path, "quarter")
