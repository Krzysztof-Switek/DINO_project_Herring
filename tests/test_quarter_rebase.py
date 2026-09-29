"""L2 (30.09): season offset added back at decode time for quarter-adjusted runs.

With ``data.quarter_age_adjustment_enabled`` the model learns "complete rings visible"
(recorded age - 1 for BITS1q/BITS2q), so its raw output is a ring count. Without adding the
offset back, the best age run (06.08_attention_first) scores 40.81 % exact against the
recorded ages instead of 53.85 %. ``inference.rebase_quarter_offset`` fixes the reported scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.dataset import encode_age_ordinal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_Q = PROJECT_ROOT / "outputs" / "06.08_attention_first"

K = 10
IDS = [
    "2022_BITS1q_HER_X_Embedded_Sharpest_FishIndex1_Single1_Left.jpg",
    "2022_BITS1q_HER_X_Embedded_Sharpest_FishIndex1_Single2_Right.jpg",
    "2022_BITS4q_HER_X_Embedded_Sharpest_FishIndex2_Single1_Left.jpg",
    "2022_BITS4q_HER_X_Embedded_Sharpest_FishIndex2_Single2_Right.jpg",
    "2022_BIAS_HER_X_Embedded_Sharpest_FishIndex3_Single1_Left.jpg",
    "2022_BITS2q_HER_X_Embedded_Sharpest_FishIndex4_Single1_Left.jpg",
]
FISH = ["f1", "f1", "f2", "f2", "f3", "f4"]
OFFSET = [1, 1, 0, 0, 0, 1]
RINGS = [3, 3, 5, 5, 2, 4]          # what the dataset yields as "age" with the adjustment on


class _MockDinoBackbone(nn.Module):
    embed_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x.mean(dim=(1, 2, 3)).reshape(x.shape[0], 1))

    def forward_features(self, x: Tensor) -> Dict:
        B, _, H, W = x.shape
        return {"x_norm_clstoken": self.forward(x),
                "x_norm_patchtokens": torch.zeros(B, (H // 14) * (W // 14), self.embed_dim)}


class _QuarterDataset(Dataset):
    """Deterministic images; ``age`` is already on the ring scale, as _effective_age yields."""

    def __len__(self) -> int:
        return len(IDS)

    def __getitem__(self, i: int) -> Dict:
        return {
            "image": torch.full((3, 56, 56), 0.1 * i),
            "age_ordinal": encode_age_ordinal(RINGS[i], K),
            "age": torch.tensor(RINGS[i], dtype=torch.long),
            "image_id": IDS[i],
        }


def _cfg(tmp_path: Path, *, quarter: bool, rebase: bool, per_fish: bool = False):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.num_age_classes = K
    cfg.model.dropout = 0.0
    cfg.training.device = "cpu"
    cfg.data.quarter_age_adjustment_enabled = quarter
    cfg.inference.rebase_quarter_offset = rebase
    cfg.inference.aggregate_per_fish = per_fish
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"image_id": IDS, "neutral_fish_key": FISH,
                  "age": [r + o for r, o in zip(RINGS, OFFSET)]}).to_csv(labels, index=False)
    cfg.data.labels_csv = str(labels)
    return cfg


def _run(tmp_path: Path, name: str, **kw):
    from src.inference import run_inference
    from src.model import OtolithModel
    torch.manual_seed(0)
    cfg = _cfg(tmp_path, **kw)
    model = OtolithModel(cfg, backbone=_MockDinoBackbone())
    out = tmp_path / name
    summary = run_inference(cfg, model, DataLoader(_QuarterDataset(), batch_size=4), out)
    return summary, out


# ---------------------------------------------------------------------------
# quarter_offsets — the inverse of _effective_age
# ---------------------------------------------------------------------------

def test_quarter_offsets_per_campaign(tmp_path):
    from src.inference import quarter_offsets
    cfg = _cfg(tmp_path, quarter=True, rebase=True)
    assert quarter_offsets(cfg, IDS).tolist() == OFFSET


def test_quarter_offsets_follow_configured_campaigns(tmp_path):
    from src.inference import quarter_offsets
    cfg = _cfg(tmp_path, quarter=True, rebase=True)
    cfg.data.quarter_age_adjustment_campaigns = ["BITS1q"]
    assert quarter_offsets(cfg, IDS).tolist() == [1, 1, 0, 0, 0, 0]


def test_offsets_invert_effective_age(tmp_path):
    """rings + offset == recorded for every campaign (no age-0 Q1 fish here)."""
    from src.dataset import OtolithDataset
    from src.inference import quarter_offsets
    cfg = _cfg(tmp_path, quarter=True, rebase=True)
    ds = OtolithDataset.__new__(OtolithDataset)
    ds.cfg, ds._has_campaign_col = cfg, False
    recorded = [r + o for r, o in zip(RINGS, OFFSET)]
    rings = [ds._effective_age(pd.Series({}), iid, a) for iid, a in zip(IDS, recorded)]
    assert rings == RINGS
    assert (np.asarray(rings) + quarter_offsets(cfg, IDS)).tolist() == recorded


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

def test_rebase_moves_predictions_and_targets_to_recorded_scale(tmp_path):
    _, out = _run(tmp_path, "on", quarter=True, rebase=True)
    df = pd.read_csv(out / "predictions.csv")
    off = np.asarray(OFFSET)
    assert (df["predicted_age"].values == df["predicted_rings"].values + off).all()
    assert (df["target_rings"].values == np.asarray(RINGS)).all()
    assert (df["target_age"].values == np.asarray(RINGS) + off).all()


def test_rebase_leaves_errors_unchanged(tmp_path):
    s_on, out_on = _run(tmp_path, "on", quarter=True, rebase=True)
    s_off, out_off = _run(tmp_path, "off", quarter=True, rebase=False)
    on = pd.read_csv(out_on / "predictions.csv")
    off = pd.read_csv(out_off / "predictions.csv")
    assert (on["abs_error"] == off["abs_error"]).all()
    assert (on["predicted_rings"] == off["predicted_age"]).all()
    assert s_on["mean_mae"] == s_off["mean_mae"]


def test_flag_off_keeps_legacy_columns(tmp_path):
    _, out = _run(tmp_path, "off", quarter=True, rebase=False)
    cols = list(pd.read_csv(out / "predictions.csv").columns)
    assert cols == ["image_id", "predicted_age", "target_age", "abs_error", "metadata_used"]


def test_flag_is_noop_without_quarter_adjustment(tmp_path):
    _, out_on = _run(tmp_path, "on", quarter=False, rebase=True)
    _, out_off = _run(tmp_path, "off", quarter=False, rebase=False)
    assert (out_on / "predictions.csv").read_bytes() == (out_off / "predictions.csv").read_bytes()


def test_rebase_per_fish(tmp_path):
    summary, out = _run(tmp_path, "on", quarter=True, rebase=True, per_fish=True)
    fish = pd.read_csv(out / "predictions_per_fish.csv").set_index("fish_id")
    fish_off = {"f1": 1, "f2": 0, "f3": 0, "f4": 1}
    fish_rec = {"f1": 4, "f2": 5, "f3": 2, "f4": 5}
    for f, o in fish_off.items():
        assert fish.loc[f, "predicted_age"] == fish.loc[f, "predicted_rings"] + o
        assert fish.loc[f, "target_age"] == fish_rec[f]
    assert summary["n_fish"] == 4


# ---------------------------------------------------------------------------
# Localization count in cards
# ---------------------------------------------------------------------------

def test_ring_count_prefers_rings():
    from scripts.run_pipeline import _ring_count
    assert _ring_count({"predicted_age": 5, "predicted_rings": 4}) == 4
    assert _ring_count({"predicted_age": 5}) == 5
    assert _ring_count(pd.Series({"predicted_age": 5, "predicted_rings": float("nan")})) == 5


# ---------------------------------------------------------------------------
# Real run: 06.08_attention_first (Run Q, trained with the adjustment ON)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (RUN_Q / "emb_on_emb" / "predictions.csv").exists(),
                    reason="Run Q outputs not present on this machine")
def test_run_q_exact_with_and_without_offset(tmp_path):
    from src.inference import quarter_offsets
    cfg = _cfg(tmp_path, quarter=True, rebase=True)
    p = pd.read_csv(RUN_Q / "emb_on_emb" / "predictions.csv")
    rec = p["image_id"].map(
        pd.read_csv(RUN_Q / "data" / "labels_embedded.csv").set_index("image_id")["age"])
    off = quarter_offsets(cfg, p["image_id"])
    assert len(p) == 1105 and not rec.isna().any()
    assert (p["target_age"].values + off == rec.values).all()
    assert round(100 * float((p["predicted_age"] + off == rec).mean()), 2) == 53.85
    assert round(100 * float((p["predicted_age"] == rec).mean()), 2) == 40.81
