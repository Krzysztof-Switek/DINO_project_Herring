"""Tests for scripts/diagnostics/coral_head_ab.py — the quarter-adjustment A/B.

The two that matter most are the controls: on synthetic features where the label genuinely
follows "complete rings visible" the ON arm must win, and on features where the label has no
season structure at all it must NOT win. Together they show the design can detect the effect
and does not manufacture one.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ab():
    sys.path.insert(0, str(PROJECT_ROOT))
    path = PROJECT_ROOT / "scripts" / "diagnostics" / "coral_head_ab.py"
    spec = importlib.util.spec_from_file_location("coral_head_ab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _index(campaigns, ages, splits):
    return pd.DataFrame({
        "image_id": [f"2022_{c}_HER_Loc_Embedded_Sharpest_FishIndex{i}_Single1_Left.jpg"
                     for i, c in enumerate(campaigns)],
        "split": list(splits),
        "age": np.asarray(ages, dtype=int),
    })


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def test_targets_for_arm_off_is_the_recorded_age(ab):
    idx = _index(["BITS1q", "BITS4q", "BIAS"], [4, 5, 3], ["train"] * 3)
    assert list(ab.targets_for_arm(idx, quarter=False)) == [4, 5, 3]


def test_targets_for_arm_on_subtracts_one_only_for_q1(ab):
    idx = _index(["BITS1q", "BITS2q", "BITS4q", "BIAS"], [4, 6, 5, 3], ["train"] * 4)
    assert list(ab.targets_for_arm(idx, quarter=True)) == [3, 5, 5, 3]


def test_targets_for_arm_on_clamps_at_zero(ab):
    """Mirrors `max(recorded_age - 1, 0)` in `src/dataset.py::_effective_age`.

    The clamp never fires on the real dataset (no age-0 fish in the Q1 campaigns), but the
    production code has it, so the reproduction must have it too.
    """
    idx = _index(["BITS1q"], [0], ["train"])
    assert list(ab.targets_for_arm(idx, quarter=True)) == [0]


def test_rebase_undoes_the_relabelling_when_the_clamp_does_not_fire(ab):
    from src.report_common import rebase_to_recorded
    idx = _index(["BITS1q", "BITS4q", "BIAS"], [4, 5, 3], ["train"] * 3)
    rings = ab.targets_for_arm(idx, quarter=True)
    assert list(rebase_to_recorded(idx["image_id"].values, rings)) == [4.0, 5.0, 3.0]


def test_to_recorded_is_identity_for_the_off_arm(ab):
    pred = np.array([1, 2, 3])
    assert list(ab._to_recorded(pred, None)) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# The stub backbone must drive the real production head
# ---------------------------------------------------------------------------

def test_cached_backbone_feeds_the_production_coral_head(ab):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.head_type = "coral"
    cfg.model.use_density_head = False
    cfg.model.dropout = 0.0
    model = ab.build_head(cfg, dim=384, device=torch.device("cpu"))
    out = model(torch.randn(5, 384))
    assert "coral_logits" in out
    assert out["coral_logits"].shape == (5, cfg.model.num_age_classes - 1)
    # rank monotonicity comes from the production threshold construction, not from us
    assert torch.all(torch.diff(out["coral_logits"], dim=1) <= 1e-6)


def test_head_uses_the_cls_token_only(ab):
    """Two feature vectors that differ must give different logits; equal ones must not."""
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.head_type = "coral"
    cfg.model.use_density_head = False
    cfg.model.dropout = 0.0
    model = ab.build_head(cfg, dim=384, device=torch.device("cpu")).eval()
    x = torch.randn(1, 384)
    with torch.no_grad():
        a = model(x)["coral_logits"]
        b = model(x.clone())["coral_logits"]
        c = model(torch.randn(1, 384))["coral_logits"]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ---------------------------------------------------------------------------
# The two controls
# ---------------------------------------------------------------------------

def _synthetic(n_per_split, season_effect: bool, seed: int, dim: int = 24, n_ages: int = 8):
    """Features carrying a latent "visible rings" signal, plus a campaign in the filename.

    ``season_effect=True``: a Q1 fish's RECORDED age is one more than what its image shows —
    the hypothesis. ``season_effect=False``: the recorded age is exactly what the image shows
    for every campaign, so subtracting one for Q1 is pure damage.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for split, n in n_per_split.items():
        camp = rng.choice(["BITS1q", "BITS4q", "BIAS"], size=n, p=[0.4, 0.4, 0.2])
        rings = rng.integers(1, n_ages - 1, size=n)
        signal = np.zeros(dim)
        signal[0] = 1.0
        x = rings[:, None] * signal[None, :] * 0.8 + rng.normal(scale=0.25, size=(n, dim))
        recorded = rings + (np.isin(camp, ["BITS1q"]).astype(int) if season_effect else 0)
        parts.append((split, camp, recorded, x.astype(np.float32)))

    campaigns = np.concatenate([p[1] for p in parts])
    recorded = np.concatenate([p[2] for p in parts])
    splits = np.concatenate([[p[0]] * len(p[1]) for p in parts])
    X = np.concatenate([p[3] for p in parts], axis=0)
    index = _index(campaigns, recorded, splits)
    return X, index


def _run_ab(ab, X, index, seeds, epochs=25):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.head_type = "coral"
    cfg.model.use_density_head = False
    cfg.model.num_age_classes = 10
    cfg.training.batch_size = 32
    device = torch.device("cpu")
    splits = index["split"].to_numpy()
    recorded = index["age"].to_numpy(dtype=float)
    ids = index["image_id"].to_numpy()
    out = {}
    for arm, quarter in (("off", False), ("on", True)):
        y = ab.targets_for_arm(index, quarter)
        out[arm] = [ab.train_one(X, y, splits, cfg, s, epochs, device, recorded,
                                 ids if quarter else None) for s in seeds]
    return out


def test_positive_control_on_arm_wins_when_the_season_effect_is_real(ab):
    """If the image really shows age-1 for Q1 fish, the ON arm must come out ahead."""
    X, index = _synthetic({"train": 900, "val": 300, "test": 300},
                          season_effect=True, seed=3)
    res = _run_ab(ab, X, index, seeds=[42, 7, 13])
    d = np.array([on["test"]["Exact"] - off["test"]["Exact"]
                  for on, off in zip(res["on"], res["off"])])
    assert d.mean() > 0.05, f"ON failed to win a real effect: {d}"
    assert (d > 0).sum() >= 2


def test_negative_control_on_arm_does_not_win_without_a_season_effect(ab):
    """With no season structure, relabelling Q1 is pure damage and must not look like a gain.

    This is the guard against the whole design manufacturing an effect: the ON arm trains on a
    target that is wrong for 40 % of the data, then adds the offset back at scoring time. If
    that still came out ahead, the comparison would be measuring something other than the
    hypothesis.
    """
    X, index = _synthetic({"train": 900, "val": 300, "test": 300},
                          season_effect=False, seed=5)
    res = _run_ab(ab, X, index, seeds=[42, 7, 13])
    d = np.array([on["test"]["Exact"] - off["test"]["Exact"]
                  for on, off in zip(res["on"], res["off"])])
    assert d.mean() < 0.02, f"ON won on season-free data: {d}"


def test_both_arms_are_scored_against_the_same_recorded_ages(ab):
    """Per-arm metrics must share one yardstick, or the arms are not comparable at all."""
    X, index = _synthetic({"train": 400, "val": 200, "test": 200},
                          season_effect=True, seed=9)
    res = _run_ab(ab, X, index, seeds=[42], epochs=12)
    assert res["on"][0]["test"]["n"] == res["off"][0]["test"]["n"] == 200


def test_same_seed_gives_the_same_initialisation_in_both_arms(ab):
    """Pairing across arms is only meaningful if the seed fixes the init identically."""
    from src.config import OtolithConfig
    from src.utils import seed_everything
    cfg = OtolithConfig()
    cfg.model.head_type = "coral"
    cfg.model.use_density_head = False
    states = []
    for _ in range(2):
        seed_everything(77)
        m = ab.build_head(cfg, dim=32, device=torch.device("cpu"))
        states.append({k: v.clone() for k, v in m.state_dict().items()})
    for k in states[0]:
        assert torch.equal(states[0][k], states[1][k])


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def test_load_cache_rejects_a_mismatched_index(ab, tmp_path, monkeypatch):
    d = tmp_path / "bad"
    d.mkdir()
    np.save(d / "features.npy", np.zeros((5, 4), dtype=np.float32))
    pd.DataFrame({"image_id": ["a"], "split": ["test"], "age": [1]}).to_csv(
        d / "index.csv", index=False)
    (d / "meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ab, "CACHE_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="inconsistent"):
        ab.load_cache("bad")


def test_load_cache_reports_a_missing_cache(ab, tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "CACHE_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="cache_cls_features"):
        ab.load_cache("nope")


# ---------------------------------------------------------------------------
# Learning-rate pilot and the under-training guard
# ---------------------------------------------------------------------------

def test_pilot_picks_by_the_mean_of_both_arms(ab):
    """The lr must not be chosen on one arm — that could quietly favour it.

    Production's lr is tuned for a head training alongside the backbone; on frozen features
    the head has to travel much further alone, so the rate is a nuisance parameter. Choosing
    it on the mean of both arms keeps it neutral.
    """
    from src.config import OtolithConfig
    import torch as _t
    X, index = _synthetic({"train": 400, "val": 200, "test": 200},
                          season_effect=True, seed=11)
    cfg = OtolithConfig()
    cfg.model.head_type = "coral"
    cfg.model.use_density_head = False
    cfg.model.num_age_classes = 10
    cfg.training.batch_size = 32
    res = ab.pilot_lr(X, index, index["split"].to_numpy(), cfg, _t.device("cpu"),
                      index["age"].to_numpy(float), index["image_id"].to_numpy(),
                      [1e-4, 1e-2], epochs=12)
    assert set(res["scores"]) == {"0.0001", "0.01"}
    for entry in res["scores"].values():
        assert len(entry["per_arm"]) == 2
        assert entry["mean_val_exact"] == pytest.approx(np.mean(entry["per_arm"]))
    best = max(res["scores"], key=lambda k: res["scores"][k]["mean_val_exact"])
    assert res["chosen_lr"] == pytest.approx(float(best))


def test_still_improving_flag_marks_an_undertrained_run(ab):
    """A best epoch at the end of the budget means the budget, not the target, was measured."""
    from src.config import OtolithConfig
    import torch as _t
    X, index = _synthetic({"train": 400, "val": 200, "test": 200},
                          season_effect=True, seed=12)
    cfg = OtolithConfig()
    cfg.model.head_type = "coral"
    cfg.model.use_density_head = False
    cfg.model.num_age_classes = 10
    cfg.training.batch_size = 32
    # one epoch cannot be enough, so the guard must fire
    r = ab.train_one(X, ab.targets_for_arm(index, False), index["split"].to_numpy(), cfg,
                     42, 1, _t.device("cpu"), index["age"].to_numpy(float), None, lr=1e-4)
    assert r["still_improving"] is True
    assert r["epochs"] == 1 and r["lr"] == pytest.approx(1e-4)


def test_lr_is_actually_used(ab):
    """A wildly different lr must change the outcome, or the parameter is not wired through."""
    from src.config import OtolithConfig
    import torch as _t
    X, index = _synthetic({"train": 400, "val": 200, "test": 200},
                          season_effect=True, seed=13)
    cfg = OtolithConfig()
    cfg.model.head_type = "coral"
    cfg.model.use_density_head = False
    cfg.model.num_age_classes = 10
    cfg.training.batch_size = 32
    y = ab.targets_for_arm(index, False)
    args = (X, y, index["split"].to_numpy(), cfg, 42, 10, _t.device("cpu"),
            index["age"].to_numpy(float), None)
    slow = ab.train_one(*args, lr=1e-6)
    fast = ab.train_one(*args, lr=1e-2)
    assert slow["val_exact"] != fast["val_exact"]
