"""Stage 8: end-to-end mini test — full pipeline with synthetic data and mock backbone."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torch import Tensor
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Mock backbone (spatially varying patch tokens, no DINOv2 download)
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
        H_p = H // 14
        W_p = W // 14
        num_patches = H_p * W_p
        cls = self.forward(x)
        idx = torch.arange(num_patches, dtype=torch.float32, device=x.device)
        scale = (idx + 1.0).reshape(1, num_patches, 1)
        patches = scale.expand(B, num_patches, self.embed_dim).contiguous()
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patches,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 5   # ages 0-4; ordinal head length = 4


def _make_synthetic_data(root: Path) -> tuple[Path, Path]:
    """Create 56x56 RGB PNG images + labels.csv  (8 train / 4 val / 4 test)."""
    img_dir = root / "images"
    img_dir.mkdir()

    rows = []
    idx = 0
    rng = np.random.default_rng(0)
    for split, n in [("train", 8), ("val", 4), ("test", 4)]:
        for i in range(n):
            fname = f"img_{idx:03d}.png"
            age = (i % (NUM_CLASSES - 1)) + 1
            arr = rng.integers(0, 255, (56, 56, 3), dtype=np.uint8)
            PILImage.fromarray(arr, "RGB").save(img_dir / fname)
            rows.append({"image_id": fname, "age": age, "split": split})
            idx += 1

    csv_path = root / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return img_dir, csv_path


def _make_cfg(tmp_path: Path):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.num_age_classes = NUM_CLASSES
    cfg.model.dropout = 0.0
    cfg.data.image_size = 56        # 56 = 4 * 14
    cfg.data.patch_size = 14
    cfg.data.num_workers = 0
    cfg.data.metadata_cols = []
    cfg.training.epochs = 2
    cfg.training.freeze_backbone_epochs = 1
    cfg.training.batch_size = 4
    cfg.training.device = "cpu"
    cfg.training.scheduler = "none"
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.candidates.min_peak_distance = 1
    cfg.candidates.prominence_threshold = 0.0
    return cfg


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

def test_full_pipeline(tmp_path):
    from src.config import OtolithConfig
    from src.dataset import OtolithDataset
    from src.model import OtolithModel
    from src.trainer import Trainer
    from src.inference import run_inference, load_model_from_checkpoint
    from src.interpretation import run_interpretation
    from src.candidates import run_candidates

    img_dir, csv_path = _make_synthetic_data(tmp_path)
    cfg = _make_cfg(tmp_path)

    # ------------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------------
    common = dict(labels_csv=str(csv_path), image_dir=str(img_dir))
    ds_train = OtolithDataset(cfg, split="train", **common)
    ds_val   = OtolithDataset(cfg, split="val",   **common)
    ds_test  = OtolithDataset(cfg, split="test",  **common)

    assert len(ds_train) == 8
    assert len(ds_val)   == 4
    assert len(ds_test)  == 4

    train_loader = DataLoader(ds_train, batch_size=4, shuffle=False)
    val_loader   = DataLoader(ds_val,   batch_size=4, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=4, shuffle=False)

    # ------------------------------------------------------------------
    # Train for 2 epochs (backbone frozen on epoch 1, unfrozen on epoch 2)
    # ------------------------------------------------------------------
    model = OtolithModel(cfg, backbone=_MockDinoBackbone())
    trainer = Trainer(cfg, model, train_loader, val_loader)
    trainer.fit()

    # Backbone should be unfrozen after training (freeze_backbone_epochs=1 < epochs=2)
    assert not model.backbone_is_frozen()

    # Two per-epoch checkpoint files must exist (plus best.pt from early stopping logic)
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_files = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
    assert len(ckpt_files) == 2, f"Expected 2 checkpoints, got {len(ckpt_files)}"

    # ------------------------------------------------------------------
    # Load model from last checkpoint
    # ------------------------------------------------------------------
    loaded_model = load_model_from_checkpoint(
        cfg, ckpt_files[-1], backbone=_MockDinoBackbone()
    )
    assert not loaded_model.training, "Model must be in eval mode after loading"

    # ------------------------------------------------------------------
    # Inference on test split
    # ------------------------------------------------------------------
    out_dir = tmp_path / "outputs"
    summary = run_inference(cfg, loaded_model, test_loader, out_dir)

    assert summary["n_samples"] == 4
    assert summary["mean_mae"] is not None

    csv_out  = out_dir / "predictions.csv"
    json_out = out_dir / "predictions.json"
    assert csv_out.exists(),  "predictions.csv not found"
    assert json_out.exists(), "predictions.json not found"

    pred_df = pd.read_csv(csv_out)
    required_cols = {"image_id", "predicted_age", "target_age", "abs_error", "metadata_used"}
    missing = required_cols - set(pred_df.columns)
    assert not missing, f"predictions.csv missing columns: {missing}"
    assert len(pred_df) == 4

    pred_records = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(pred_records) == 4
    for rec in pred_records:
        for key in ("image_id", "predicted_age"):
            assert key in rec, f"predictions.json record missing key '{key}'"

    # ------------------------------------------------------------------
    # Interpretation: heatmap + overlay PNGs
    # ------------------------------------------------------------------
    interp_dir = tmp_path / "interpretation"
    interp_results = run_interpretation(cfg, loaded_model, test_loader, interp_dir)

    assert len(interp_results) == 4
    for r in interp_results:
        assert Path(r["heatmap_path"]).exists(), f"Heatmap missing: {r['heatmap_path']}"
        assert Path(r["overlay_path"]).exists(), f"Overlay missing: {r['overlay_path']}"

    # ------------------------------------------------------------------
    # Candidates: JSON + annotated overlay PNGs
    # ------------------------------------------------------------------
    cand_dir = tmp_path / "candidates_out"
    cand_results = run_candidates(cfg, loaded_model, test_loader, cand_dir)

    assert len(cand_results) == 4
    for r in cand_results:
        assert "image_id"       in r
        assert "num_candidates" in r

        json_path = Path(r["candidate_markers_path"])
        ov_path   = Path(r["candidates_overlay_path"])
        assert json_path.exists(), f"Candidates JSON missing: {json_path}"
        assert ov_path.exists(),   f"Candidates overlay missing: {ov_path}"

        data = json.loads(json_path.read_text(encoding="utf-8"))
        for key in ("image_id", "num_candidates", "peak_profile_indices", "radial_profile"):
            assert key in data, f"Candidates JSON missing key '{key}'"
        assert isinstance(data["peak_profile_indices"], list)
        assert isinstance(data["radial_profile"],       list)
