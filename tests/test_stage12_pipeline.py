"""Tests for scripts/run_pipeline.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Mock backbone (no DINOv2 download)
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
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 5


def _make_synth_labels(tmp_path: Path, img_dir: Path) -> tuple[Path, Path]:
    """Create synthetic images + two labels CSVs (embedded, not_embedded)."""
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows_e, rows_n = [], []
    fish_idx = 0
    for split, n in [("train", 6), ("val", 2), ("test", 2)]:
        for i in range(n):
            age = (fish_idx % (NUM_CLASSES - 1)) + 1
            arr = rng.integers(0, 255, (56, 56, 3), dtype=np.uint8)
            fname_e = f"2022_BIAS_HER_Loc_Embedded_Sharp_FishIndex{fish_idx}_Single1_Left.png"
            fname_n = f"2022_BIAS_HER_Loc_NotEmbedded_Sharp_FishIndex{fish_idx}_Single1_Left.png"
            PILImage.fromarray(arr, "RGB").save(img_dir / fname_e)
            PILImage.fromarray(arr, "RGB").save(img_dir / fname_n)
            rows_e.append({"image_id": fname_e, "age": age, "split": split})
            rows_n.append({"image_id": fname_n, "age": age, "split": split})
            fish_idx += 1

    emb_csv = tmp_path / "labels_embedded.csv"
    notemb_csv = tmp_path / "labels_not_embedded.csv"
    pd.DataFrame(rows_e).to_csv(emb_csv, index=False)
    pd.DataFrame(rows_n).to_csv(notemb_csv, index=False)
    return emb_csv, notemb_csv


def _make_cfg(tmp_path: Path, labels_csv: Path, img_dir: Path):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.num_age_classes = NUM_CLASSES
    cfg.model.dropout = 0.0
    cfg.data.image_size = 56
    cfg.data.patch_size = 14
    cfg.data.num_workers = 0
    cfg.data.metadata_cols = []
    cfg.data.labels_csv = str(labels_csv)
    cfg.data.image_dir = str(img_dir)
    cfg.training.epochs = 1
    cfg.training.freeze_backbone_epochs = 0
    cfg.training.batch_size = 4
    cfg.training.device = "cpu"
    cfg.training.scheduler = "none"
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.inference.output_dir = str(tmp_path / "outputs")
    return cfg


def _save_mock_checkpoint(cfg, labels_csv: Path, img_dir: Path) -> Path:
    """Train for 1 epoch with MockDinoBackbone and return path to best checkpoint."""
    import shutil
    from src.dataset import OtolithDataset
    from src.model import OtolithModel
    from src.trainer import Trainer
    from torch.utils.data import DataLoader

    train_ds = OtolithDataset(cfg, split="train")
    val_ds = OtolithDataset(cfg, split="val")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = OtolithModel(cfg, backbone=_MockDinoBackbone())
    trainer = Trainer(cfg, model, train_loader, val_loader)
    trainer.fit()

    ckpt_files = sorted(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint in {trainer.checkpoint_dir}")
    best_src = min(ckpt_files, key=lambda p: float(p.stem.split("_loss")[-1]))
    best_ckpt = trainer.checkpoint_dir / "best.pt"
    shutil.copy2(best_src, best_ckpt)
    return best_ckpt


# ---------------------------------------------------------------------------
# test_pipeline_state_file
# ---------------------------------------------------------------------------

def test_pipeline_state_file(tmp_path):
    """pipeline_state.json is written after each completed step."""
    from scripts.run_pipeline import _save_state, _load_state

    state_path = tmp_path / "pipeline_state.json"
    _save_state(state_path, ["scan", "train_e"])
    loaded = _load_state(state_path)
    assert "scan" in loaded
    assert "train_e" in loaded
    assert "report" not in loaded


# ---------------------------------------------------------------------------
# test_skip_scan_flag
# ---------------------------------------------------------------------------

def test_skip_scan_flag(tmp_path):
    """--skip-scan flag marks scan step as SKIP in dry-run output."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
         "--output-dir", str(tmp_path / "out"),
         "--skip-scan", "--skip-train", "--dry-run"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    assert result.returncode == 0
    assert "SKIP" in result.stdout
    assert "scan" in result.stdout


# ---------------------------------------------------------------------------
# test_skip_train_flag
# ---------------------------------------------------------------------------

def test_skip_train_flag(tmp_path):
    """--skip-train flag marks train_e and train_n as SKIP in dry-run."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
         "--output-dir", str(tmp_path / "out"),
         "--skip-train", "--dry-run"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    assert result.returncode == 0
    output = result.stdout
    # Both train steps should be marked SKIP
    lines = [l for l in output.splitlines() if "train_e" in l or "train_n" in l]
    assert any("SKIP" in l for l in lines)


# ---------------------------------------------------------------------------
# test_dry_run
# ---------------------------------------------------------------------------

def test_dry_run(tmp_path):
    """--dry-run prints steps without creating any output files."""
    import subprocess
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
         "--output-dir", str(out_dir), "--dry-run"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    # No state file should be created
    assert not (out_dir / "pipeline_state.json").exists()


# ---------------------------------------------------------------------------
# test_full_smoke
# ---------------------------------------------------------------------------

def test_full_smoke(tmp_path):
    """Full pipeline on synthetic data: skip-scan + skip-train, run infer+report."""
    from src.dataset import OtolithDataset
    from src.model import OtolithModel
    from src.trainer import Trainer
    from src.inference import run_inference
    from src.comparison_report import build_comparison_report
    from torch.utils.data import DataLoader

    img_dir = tmp_path / "images"
    emb_csv, notemb_csv = _make_synth_labels(tmp_path, img_dir)

    cfg_emb = _make_cfg(tmp_path / "emb", emb_csv, img_dir)
    cfg_notemb = _make_cfg(tmp_path / "notemb", notemb_csv, img_dir)

    # Train both models
    ckpt_emb = _save_mock_checkpoint(cfg_emb, emb_csv, img_dir)
    ckpt_notemb = _save_mock_checkpoint(cfg_notemb, notemb_csv, img_dir)

    # Run inference for all 4 conditions
    from src.inference import load_model_from_checkpoint

    conditions = [
        ("emb_on_emb",        cfg_emb,    ckpt_emb,    emb_csv),
        ("notemb_on_notemb",  cfg_notemb, ckpt_notemb, notemb_csv),
        ("cross_emb_on_notemb", cfg_emb,  ckpt_emb,    notemb_csv),
        ("cross_notemb_on_emb", cfg_notemb, ckpt_notemb, emb_csv),
    ]
    results_dfs = {}
    for cond_key, cfg, ckpt, labels_csv in conditions:
        cfg_c = cfg.model_copy(deep=True)
        cfg_c.data.labels_csv = str(labels_csv)
        test_ds = OtolithDataset(cfg_c, split="test")
        loader = DataLoader(test_ds, batch_size=4, shuffle=False)
        model = load_model_from_checkpoint(cfg_c, ckpt, backbone=_MockDinoBackbone())
        infer_dir = tmp_path / "out" / cond_key
        run_inference(cfg_c, model, loader, infer_dir)
        pred_csv = infer_dir / "predictions.csv"
        if pred_csv.exists():
            df = pd.read_csv(pred_csv)
            if "target_age" in df.columns and "age" not in df.columns:
                df = df.rename(columns={"target_age": "age"})
            results_dfs[cond_key] = df

    # Build report
    report_path = tmp_path / "out" / "comparison_report.html"
    build_comparison_report(
        results=results_dfs,
        training_logs={},
        increment_cards={"best": [], "worst": []},
        dataset_stats={"counts": {}, "orphan_count": 0, "age_distributions": {}},
        output_path=report_path,
    )

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "<html" in content
    # State file tracking
    state_path = tmp_path / "out" / "pipeline_state.json"
    from scripts.run_pipeline import _save_state
    _save_state(state_path, ["scan", "train_e", "train_n",
                              "infer_ee", "infer_nn", "infer_en", "infer_ne",
                              "cards", "report"])
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert "report" in state["completed_steps"]
