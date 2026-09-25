"""Stage 5 tests: inference — predictions.csv, predictions.json, abs_error, summary."""
from __future__ import annotations

import json
import math
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


def test_load_checkpoint_drops_shape_mismatched_keys_instead_of_crashing(tmp_path):
    """22.07: a sub-module architecture change (e.g. a layer inserted inside `head`,
    shifting every subsequent key's shape) must NOT hard-crash the load — the mismatched
    keys are dropped (that sub-module randomly initialised, with a warning), exactly like
    the existing missing-key case, not a RuntimeError from torch's strict loader."""
    from src.inference import load_model_from_checkpoint
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    state = model.state_dict()
    # Corrupt a head weight's shape in place (simulates an architecture change) —
    # plain strict=False would still raise on this (shape mismatch, not a missing key).
    head_key = next(k for k in state if k.startswith("head.") and state[k].dim() >= 1)
    state[head_key] = state[head_key].unsqueeze(0)   # shape now wrong on every dim
    ckpt_path = tmp_path / "shape_mismatch.pt"
    torch.save({"model_state_dict": state, "epoch": 1}, ckpt_path)

    with pytest.warns(RuntimeWarning, match="shape-mismatched"):
        loaded = load_model_from_checkpoint(cfg, ckpt_path, backbone=_MockDinoBackbone())
    assert loaded is not None   # did not raise


# ---------------------------------------------------------------------------
# CORAL logit dump (24.09) — gated, additive, must not change the default output
# ---------------------------------------------------------------------------

def test_logit_dump_is_off_by_default(tmp_path):
    """Default output must stay exactly what every recorded run already produced."""
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    run_inference(cfg, _make_model(cfg), _make_loader(), tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert list(df.columns) == [
        "image_id", "predicted_age", "target_age", "abs_error", "metadata_used"]


def test_logit_dump_adds_one_column_per_boundary(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    cfg.inference.dump_coral_logits = True
    run_inference(cfg, _make_model(cfg), _make_loader(), tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    expected = [f"coral_logit_{k:02d}" for k in range(cfg.model.num_age_classes - 1)]
    assert expected == [c for c in df.columns if c.startswith("coral_logit_")]
    assert df[expected].notna().all().all()


def test_dumped_logits_reproduce_the_predicted_age_at_threshold_0_5(tmp_path):
    """The dump must be the decoder's own input, not a re-derived approximation.

    This is the guard for Etap 1: if counting sigmoid(logit) > 0.5 over the dumped
    columns ever disagrees with predicted_age, then any threshold search run on the
    dumped file is measuring a different model than the pipeline reports.
    """
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    cfg.inference.dump_coral_logits = True
    run_inference(cfg, _make_model(cfg), _make_loader(), tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    cols = [c for c in df.columns if c.startswith("coral_logit_")]
    probs = torch.sigmoid(torch.from_numpy(df[cols].to_numpy())).numpy()
    recomputed = (probs > 0.5).sum(axis=1)
    assert list(recomputed) == list(df["predicted_age"])


def test_dumped_logits_are_monotone_decreasing(tmp_path):
    """Rank consistency is structural (single g minus increasing thetas) — pin it.

    It is the reason `(p > 0.5).sum()` is a valid decoder at all, and the reason every
    candidate age ranking has adjacent runners-up.
    """
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    cfg.inference.dump_coral_logits = True
    run_inference(cfg, _make_model(cfg), _make_loader(), tmp_path / "out")
    df = pd.read_csv(tmp_path / "out" / "predictions.csv")
    vals = df[[c for c in df.columns if c.startswith("coral_logit_")]].to_numpy()
    assert np.all(np.diff(vals, axis=1) <= 1e-9)


def test_logit_dump_leaves_json_and_summary_consistent(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    cfg.inference.dump_coral_logits = True
    summary = run_inference(cfg, _make_model(cfg), _make_loader(), tmp_path / "out")
    records = json.loads((tmp_path / "out" / "predictions.json").read_text(encoding="utf-8"))
    assert summary["n_samples"] == len(records)
    assert "coral_logit_00" in records[0]


# ---------------------------------------------------------------------------
# Per-fish aggregation (24.09) — gated, separate file, predictions.csv untouched
# ---------------------------------------------------------------------------

class _TwoPhotoDataset(Dataset):
    """Two photos per fish, both carrying that fish's single true age."""

    def __init__(self, n_fish: int = 6, num_age_classes: int = 10):
        self.n_fish = n_fish
        self.num_age_classes = num_age_classes

    def __len__(self) -> int:
        return self.n_fish * 2

    def __getitem__(self, idx: int) -> Dict:
        fish = idx // 2
        age = (fish % (self.num_age_classes - 1)) + 1
        return {
            "image": torch.randn(3, 56, 56),
            "age_ordinal": encode_age_ordinal(age, self.num_age_classes),
            "age": torch.tensor(age, dtype=torch.long),
            "image_id": f"2022_BITS4q_HER_Loc_Embedded_Sharpest_FishIndex{fish}"
                        f"_Single{idx % 2 + 1}_Left.jpg",
        }


def _write_two_photo_labels(tmp_path: Path, n_fish: int = 6) -> Path:
    rows = []
    for fish in range(n_fish):
        for s in (1, 2):
            rows.append({
                "image_id": f"2022_BITS4q_HER_Loc_Embedded_Sharpest_FishIndex{fish}"
                            f"_Single{s}_Left.jpg",
                "neutral_fish_key": f"2022_BITS4q_HER_Loc_FishIndex{fish}",
            })
    path = tmp_path / "labels_two_photo.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _cfg_with_fish(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    cfg.inference.aggregate_per_fish = True
    cfg.data.labels_csv = str(_write_two_photo_labels(tmp_path))
    return cfg


def test_per_fish_is_off_by_default(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    summary = run_inference(cfg, _make_model(cfg), _make_loader(), tmp_path / "out")
    assert not (tmp_path / "out" / "predictions_per_fish.csv").exists()
    assert "n_fish" not in summary


def test_per_fish_does_not_touch_predictions_csv(tmp_path):
    """The per-image file must stay exactly what every downstream consumer expects."""
    from src.inference import run_inference
    loader = DataLoader(_TwoPhotoDataset(), batch_size=4, shuffle=False)

    plain = _make_cfg(tmp_path)
    plain.data.labels_csv = str(_write_two_photo_labels(tmp_path))
    torch.manual_seed(0)
    model = _make_model(plain)
    run_inference(plain, model, loader, tmp_path / "a")

    withfish = _cfg_with_fish(tmp_path)
    run_inference(withfish, model, loader, tmp_path / "b")

    assert ((tmp_path / "a" / "predictions.csv").read_bytes()
            == (tmp_path / "b" / "predictions.csv").read_bytes())
    assert (tmp_path / "b" / "predictions_per_fish.csv").exists()


def test_per_fish_collapses_two_photos_into_one_row(tmp_path):
    from src.inference import run_inference
    cfg = _cfg_with_fish(tmp_path)
    loader = DataLoader(_TwoPhotoDataset(n_fish=6), batch_size=4, shuffle=False)
    summary = run_inference(cfg, _make_model(cfg), loader, tmp_path / "out")
    fish = pd.read_csv(tmp_path / "out" / "predictions_per_fish.csv")

    assert len(fish) == 6
    assert summary["n_fish"] == 6
    assert set(fish["n_images"]) == {2}
    assert list(fish.columns) == ["fish_id", "predicted_age", "target_age",
                                 "n_images", "abs_error"]
    # one age per fish, carried through the grouping
    assert fish["target_age"].notna().all()
    assert summary["exact_fish"] is not None


def test_per_fish_decodes_the_mean_of_the_logits(tmp_path):
    """Averaging logits == averaging the scalar g, which is the whole point.

    Every logit is ``g - theta_k`` with constant thetas, so the mean over a fish's photos is
    ``mean(g) - theta_k``. Decoding that is decoding the averaged decision variable — which
    halves its variance. Averaging two already-rounded integers cannot, and a 0.5 mean is not
    even well defined. This reproduces the aggregation by hand and demands a match.
    """
    from src.inference import run_inference
    cfg = _cfg_with_fish(tmp_path)
    cfg.inference.dump_coral_logits = True
    loader = DataLoader(_TwoPhotoDataset(n_fish=5), batch_size=4, shuffle=False)
    run_inference(cfg, _make_model(cfg), loader, tmp_path / "out")

    per_img = pd.read_csv(tmp_path / "out" / "predictions.csv")
    fish = pd.read_csv(tmp_path / "out" / "predictions_per_fish.csv").set_index("fish_id")
    cols = [c for c in per_img.columns if c.startswith("coral_logit_")]
    per_img["fish"] = per_img["image_id"].str.replace(
        r"_Embedded_Sharpest_(FishIndex\d+)_Single\d+_Left\.jpg", r"_\1", regex=True)
    for fish_id, grp in per_img.groupby("fish"):
        mean_logits = grp[cols].to_numpy().mean(axis=0)
        expected = int((mean_logits > 0).sum())        # sigmoid(x) > 0.5  <=>  x > 0
        key = [k for k in fish.index if k.endswith(fish_id.split("FishIndex")[1])]
        assert len(key) == 1
        assert int(fish.loc[key[0], "predicted_age"]) == expected


def test_per_fish_error_never_exceeds_the_worse_of_the_two_photos(tmp_path):
    """Both photos share one true age, so the averaged decode sits between the two errors."""
    from src.inference import run_inference
    n = 8
    cfg = _make_cfg(tmp_path)
    cfg.inference.aggregate_per_fish = True
    cfg.data.labels_csv = str(_write_two_photo_labels(tmp_path, n_fish=n))
    loader = DataLoader(_TwoPhotoDataset(n_fish=n), batch_size=4, shuffle=False)
    run_inference(cfg, _make_model(cfg), loader, tmp_path / "out")
    per_img = pd.read_csv(tmp_path / "out" / "predictions.csv")
    fish = pd.read_csv(tmp_path / "out" / "predictions_per_fish.csv")
    assert len(fish) == n
    worst = per_img.groupby(per_img.index // 2)["abs_error"].max().values
    assert (fish["abs_error"].values <= worst).all()


def test_per_fish_reports_images_it_could_not_map(tmp_path):
    """Images with no fish key are excluded — that must be counted, not silent."""
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    cfg.inference.aggregate_per_fish = True
    # labels cover only 6 of the 8 fish in the loader
    cfg.data.labels_csv = str(_write_two_photo_labels(tmp_path, n_fish=6))
    loader = DataLoader(_TwoPhotoDataset(n_fish=8), batch_size=4, shuffle=False)
    summary = run_inference(cfg, _make_model(cfg), loader, tmp_path / "out")
    assert summary["n_fish"] == 6
    assert summary["n_images_without_fish_key"] == 4


def test_per_fish_skips_gracefully_without_fish_keys(tmp_path):
    """A labels file with no neutral_fish_key must not crash the run."""
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    cfg.inference.aggregate_per_fish = True
    bad = tmp_path / "no_key.csv"
    pd.DataFrame({"image_id": ["x.jpg"], "age": [3]}).to_csv(bad, index=False)
    cfg.data.labels_csv = str(bad)
    summary = run_inference(cfg, _make_model(cfg), _make_loader(), tmp_path / "out")
    assert summary["n_fish"] == 0
    assert summary["exact_fish"] is None
    assert not (tmp_path / "out" / "predictions_per_fish.csv").exists()


def test_per_fish_skips_gracefully_when_labels_csv_is_missing(tmp_path):
    from src.inference import run_inference
    cfg = _make_cfg(tmp_path)
    cfg.inference.aggregate_per_fish = True
    cfg.data.labels_csv = str(tmp_path / "does_not_exist.csv")
    summary = run_inference(cfg, _make_model(cfg), _make_loader(), tmp_path / "out")
    assert summary["n_fish"] == 0


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "experiments" / "logits" /
         "06.08_attention_first__best" / "test" / "predictions.csv").exists(),
    reason="needs the logit dump from scripts/diagnostics/coral_score_splits.py")
def test_production_per_fish_matches_the_diagnostic_on_real_data(tmp_path):
    """The production path and the diagnostic must agree on the real dump, exactly.

    Two independent implementations of the same aggregation (src/inference.py::_write_per_fish
    and coral_decode_lab.py::fish_level_rule) would otherwise be free to drift, and the
    +5.8 pp reported for per-fish aggregation would stop being the number production emits.
    """
    import json
    import numpy as np
    from src.config import OtolithConfig
    from src.inference import _write_per_fish

    root = Path(__file__).resolve().parents[1]
    dump = root / "experiments" / "logits" / "06.08_attention_first__best" / "test" / "predictions.csv"
    lab_json = root / "experiments" / "threshold" / "06.08_attention_first__best" / "metrics.json"
    if not lab_json.exists():
        pytest.skip("needs coral_decode_lab.py output")

    df = pd.read_csv(dump)
    cols = sorted(c for c in df.columns if c.startswith("coral_logit_"))
    cfg = OtolithConfig()
    cfg.data.labels_csv = "data/labels_embedded.csv"
    got = _write_per_fish(cfg, df, df[cols].to_numpy(dtype=np.float32), tmp_path)

    want = json.loads(lab_json.read_text(encoding="utf-8"))["fish_level"]["test_production"]
    assert got["n_fish"] == want["n_fish"]
    assert got["exact_fish"] == pytest.approx(want["Exact"])
    assert got["mean_mae_fish"] == pytest.approx(want["MAE"])
