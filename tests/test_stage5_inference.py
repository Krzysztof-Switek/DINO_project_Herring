"""Stage 5 tests: inference — predictions.csv, predictions.json, abs_error, summary."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.dataset import encode_age_ordinal


# ---------------------------------------------------------------------------
# Mock backbone
# ---------------------------------------------------------------------------

class _MockDinoBackbone(nn.Module):
    embed_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        mean_val = x.mean(dim=(1, 2, 3), keepdim=True).reshape(B, 1)
        return self.proj(mean_val)

    def forward_features(self, x: Tensor) -> Dict:
        B, C, H, W = x.shape
        num_patches = (H // 14) * (W // 14)
        cls = self.forward(x)
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": torch.zeros(B, num_patches, self.embed_dim, device=x.device),
        }


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

class _SyntheticDataset(Dataset):
    """Dataset that returns images + labels (age present)."""
    def __init__(self, n: int = 8, num_age_classes: int = 10):
        self.n = n
        self.num_age_classes = num_age_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict:
        age = (idx % (self.num_age_classes - 1)) + 1
        return {
            "image": torch.randn(3, 56, 56),
            "age_ordinal": encode_age_ordinal(age, self.num_age_classes),
            "age": torch.tensor(age, dtype=torch.long),
            "image_id": f"img_{idx:03d}.png",
        }


class _SyntheticDatasetNoAge(Dataset):
    """Dataset without labels — simulates unlabeled inference."""
    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict:
        return {
            "image": torch.randn(3, 56, 56),
            "image_id": f"unlabeled_{idx:03d}.png",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.num_age_classes = 10
    cfg.model.dropout = 0.0
    cfg.training.device = "cpu"
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.training.log_dir = str(tmp_path / "logs")
    return cfg


def _make_model(cfg):
    from src.model import OtolithModel
    return OtolithModel(cfg, backbone=_MockDinoBackbone())


def _make_loader(n: int = 8, with_labels: bool = True) -> DataLoader:
    ds = _SyntheticDataset(n=n) if with_labels else _SyntheticDatasetNoAge(n=n)
    return DataLoader(ds, batch_size=4, shuffle=False)


def _save_checkpoint(model, tmp_path: Path) -> Path:
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "val_loss": 0.5,
            "cfg": {},
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# run_inference — output files
# ---------------------------------------------------------------------------

def test_inference_creates_csv(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    run_inference(cfg, model, _make_loader(), tmp_path / "out")
    assert (tmp_path / "out" / "predictions.csv").exists()


def test_inference_creates_json(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    run_inference(cfg, model, _make_loader(), tmp_path / "out")
    assert (tmp_path / "out" / "predictions.json").exists()


def test_predictions_csv_required_columns(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    run_inference(cfg, model, _make_loader(), tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    for col in ("image_id", "predicted_age", "target_age", "abs_error", "metadata_used"):
        assert col in df.columns, f"Missing column: {col}"


def test_predictions_csv_row_count(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    N = 6
    run_inference(cfg, model, _make_loader(n=N), tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert len(df) == N


def test_predictions_json_parseable(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    run_inference(cfg, model, _make_loader(n=4), tmp_path / "out")
    records = json.loads((tmp_path / "out" / "predictions.json").read_text())
    assert isinstance(records, list)
    assert len(records) == 4
    assert "image_id" in records[0]


def test_predictions_json_has_all_fields(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    run_inference(cfg, model, _make_loader(n=4), tmp_path / "out")
    records = json.loads((tmp_path / "out" / "predictions.json").read_text())
    required = {"image_id", "predicted_age", "target_age", "abs_error", "metadata_used"}
    for rec in records:
        assert required.issubset(rec.keys())


# ---------------------------------------------------------------------------
# abs_error correctness
# ---------------------------------------------------------------------------

def test_abs_error_equals_abs_diff(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    run_inference(cfg, model, _make_loader(n=8), tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    for _, row in df.iterrows():
        expected = abs(int(row["predicted_age"]) - int(row["target_age"]))
        assert int(row["abs_error"]) == expected


def test_predicted_age_is_integer(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    run_inference(cfg, model, _make_loader(n=4), tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    for val in df["predicted_age"]:
        assert float(val) == int(val)


def test_predicted_age_in_valid_range(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    run_inference(cfg, model, _make_loader(n=8), tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert (df["predicted_age"] >= 0).all()
    assert (df["predicted_age"] < cfg.model.num_age_classes).all()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_summary_has_required_keys(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    summary = run_inference(cfg, model, _make_loader(n=8), tmp_path / "out")
    assert "n_samples" in summary
    assert "mean_mae" in summary
    assert "median_mae" in summary


def test_summary_n_samples_correct(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    N = 6
    summary = run_inference(cfg, model, _make_loader(n=N), tmp_path / "out")
    assert summary["n_samples"] == N


def test_summary_mae_finite_and_nonneg(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    summary = run_inference(cfg, model, _make_loader(n=8), tmp_path / "out")
    assert math.isfinite(summary["mean_mae"])
    assert math.isfinite(summary["median_mae"])
    assert summary["mean_mae"] >= 0
    assert summary["median_mae"] >= 0


# ---------------------------------------------------------------------------
# No-label inference (unlabeled data)
# ---------------------------------------------------------------------------

def test_inference_no_labels_target_age_is_null(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    run_inference(cfg, model, _make_loader(n=4, with_labels=False), tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert df["target_age"].isna().all()
    assert df["abs_error"].isna().all()


def test_inference_no_labels_summary_mae_is_none(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    summary = run_inference(cfg, model, _make_loader(n=4, with_labels=False), tmp_path / "out")
    assert summary["mean_mae"] is None
    assert summary["median_mae"] is None


# ---------------------------------------------------------------------------
# load_model_from_checkpoint
# ---------------------------------------------------------------------------

def test_load_model_from_checkpoint_returns_model(tmp_path):
    from src.inference import load_model_from_checkpoint
    cfg = _make_cfg(tmp_path)
    original = _make_model(cfg)
    ckpt_path = _save_checkpoint(original, tmp_path)
    loaded = load_model_from_checkpoint(cfg, ckpt_path, backbone=_MockDinoBackbone())
    assert isinstance(loaded, type(original))


def test_load_model_from_checkpoint_restores_weights(tmp_path):
    from src.inference import load_model_from_checkpoint
    cfg = _make_cfg(tmp_path)
    original = _make_model(cfg)
    ckpt_path = _save_checkpoint(original, tmp_path)

    images = torch.randn(2, 3, 56, 56)
    original.eval()
    with torch.no_grad():
        out_original = original(images)["coral_logits"].clone()

    loaded = load_model_from_checkpoint(cfg, ckpt_path, backbone=_MockDinoBackbone())
    with torch.no_grad():
        out_loaded = loaded(images)["coral_logits"]

    assert torch.allclose(out_original, out_loaded, atol=1e-6)


def test_load_model_is_in_eval_mode(tmp_path):
    from src.inference import load_model_from_checkpoint
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    ckpt_path = _save_checkpoint(model, tmp_path)
    loaded = load_model_from_checkpoint(cfg, ckpt_path, backbone=_MockDinoBackbone())
    assert not loaded.training


def test_load_checkpoint_warns_on_missing_keys(tmp_path):
    """Partial (non-strict) load must emit a RuntimeWarning, not silently succeed."""
    from src.inference import load_model_from_checkpoint
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    state = model.state_dict()
    # Drop a head parameter → forces a missing key on load
    head_key = next(k for k in state if k.startswith("head."))
    del state[head_key]
    ckpt_path = tmp_path / "partial.pt"
    torch.save({"model_state_dict": state, "epoch": 1}, ckpt_path)

    with pytest.warns(RuntimeWarning):
        load_model_from_checkpoint(cfg, ckpt_path, backbone=_MockDinoBackbone())
