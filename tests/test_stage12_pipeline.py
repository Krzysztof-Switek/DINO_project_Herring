"""Tests for scripts/run_pipeline.py."""
from __future__ import annotations

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

    # fit() now always writes best.pt (and prunes per-epoch checkpoints by default).
    best_ckpt = trainer.checkpoint_dir / "best.pt"
    if not best_ckpt.exists():
        ckpt_files = sorted(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint in {trainer.checkpoint_dir}")
        best_src = min(ckpt_files, key=lambda p: float(p.stem.split("_loss")[-1]))
        shutil.copy2(best_src, best_ckpt)
    return best_ckpt


# ---------------------------------------------------------------------------
# test_parse_train_log
# ---------------------------------------------------------------------------

def test_parse_train_log(tmp_path):
    """_parse_train_log parses trainer text log into list of epoch dicts."""
    from scripts.run_pipeline import _parse_train_log

    log = tmp_path / "train.log"
    log.write_text(
        "[2026-05-07 10:00:00] epoch=  1  train_loss=0.8234  val_loss=0.7456  val_mae=2.345\n"
        "[2026-05-07 10:01:00] epoch=  2  train_loss=0.6100  val_loss=0.6800  val_mae=1.900\n",
        encoding="utf-8",
    )
    rows = _parse_train_log(log)
    assert len(rows) == 2
    assert rows[0] == {"epoch": 1, "train_loss": 0.8234, "val_loss": 0.7456, "val_mae": 2.345}
    assert rows[1]["epoch"] == 2
    assert rows[1]["val_mae"] == 1.9
    # Missing file returns empty list
    assert _parse_train_log(tmp_path / "nonexistent.log") == []


def test_parse_train_log_with_lr(tmp_path):
    """Newer logs include lr=…; parser must expose it (older logs stay unchanged)."""
    from scripts.run_pipeline import _parse_train_log

    log = tmp_path / "train.log"
    log.write_text(
        "[2026-07-07 10:00:00] epoch=  1  train_loss=0.80  val_loss=0.70  val_mae=2.30  lr=1.00e-04\n",
        encoding="utf-8",
    )
    rows = _parse_train_log(log)
    assert len(rows) == 1
    assert rows[0]["epoch"] == 1
    assert rows[0]["lr"] == 1e-4


def test_select_topk_image_ids_uses_target_age(tmp_path):
    """Top-k selection must fire on the 'target_age' column written by run_inference."""
    import pandas as pd
    from scripts.run_pipeline import _select_topk_image_ids

    df = pd.DataFrame({
        "image_id":      [f"img_{i}.png" for i in range(6)],
        "predicted_age": [3, 3, 3, 3, 3, 3],
        "target_age":    [3, 3, 3, 4, 5, 9],   # abs errors: 0,0,0,1,2,6
    })
    csv = tmp_path / "predictions.csv"
    df.to_csv(csv, index=False)

    ids = _select_topk_image_ids(csv, k_best=1, k_worst=1)
    assert len(ids) == 2
    assert "img_5.png" in ids          # worst (error 6) always selected
    # Missing file / bad columns → empty set (interpretation falls back to all)
    assert _select_topk_image_ids(tmp_path / "nope.csv", 1, 1) == set()


def test_embedded_only_dry_run(tmp_path, capsys):
    """--embedded-only runs only Embedded steps; NotEmbedded + cross are SKIP."""
    from scripts.run_pipeline import main as rp_main

    rp_main([
        "--output-dir", str(tmp_path),
        "--base-config", str(PROJECT_ROOT / "configs" / "config_demo.yaml"),
        "--embedded-only", "--dry-run",
    ])
    out = capsys.readouterr().out
    assert "[RUN ] train_e" in out
    assert "[RUN ] infer_ee" in out
    for skipped in ("train_n", "infer_nn", "infer_en", "infer_ne"):
        assert f"[SKIP] {skipped}" in out


# ---------------------------------------------------------------------------
# test_reload_cards_from_disk
# ---------------------------------------------------------------------------

def test_reload_cards_from_disk(tmp_path):
    """_reload_cards_from_disk should reconstruct {best, worst} from PNGs on disk
       so that idempotent re-runs of the pipeline don't drop section E content."""
    from scripts.run_pipeline import _reload_cards_from_disk

    # Empty output dir → empty dict (graceful)
    out = tmp_path / "out"
    out.mkdir()
    cards = _reload_cards_from_disk(out)
    assert cards == {"best": [], "worst": []}

    # Layout: out/cards/<cond>/{best,worst}/*.png
    for cond in ("emb_on_emb", "notemb_on_notemb"):
        for label, n in [("best", 2), ("worst", 1)]:
            d = out / "cards" / cond / label
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{label}_img{i}_card.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    cards = _reload_cards_from_disk(out)
    # 2 conds × 2 best = 4; 2 conds × 1 worst = 2
    assert len(cards["best"]) == 4
    assert len(cards["worst"]) == 2
    assert all(p.suffix == ".png" for p in cards["best"])


# ---------------------------------------------------------------------------
# test_embedded_only_skips_notembedded_in_dry_run
# ---------------------------------------------------------------------------

def test_embedded_only_flag_marks_skips(tmp_path):
    """--embedded-only marks NotEmbedded + cross steps as SKIP in dry-run."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
         "--output-dir", str(tmp_path / "out"),
         "--embedded-only", "--dry-run"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    assert result.returncode == 0
    lines = [l for l in result.stdout.splitlines() if "train_n" in l]
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


# ---------------------------------------------------------------------------
# test_walkthrough_panel_b64_is_raw (regression, 20.07)
# ---------------------------------------------------------------------------

def test_walkthrough_panel_b64_is_raw(tmp_path, monkeypatch):
    """panel_*_b64 in the walkthrough payload must be RAW base64 (no "data:image/..."
    prefix) — src/comparison_report.py renders them via _img_tag(), which adds the
    prefix itself. A pre-prefixed string here silently double-prefixes the <img src>,
    producing an invalid data URI (broken image, no exception — bug found 20.07 via
    screenshots showing Krok 0/1/3b/5 panels missing while the matplotlib panels,
    which already used raw base64, rendered fine)."""
    import base64
    import cv2
    from src.inference import load_model_from_checkpoint as _real_load
    from scripts.run_pipeline import _compute_axis_data_for_samples

    # _compute_axis_data_for_samples always calls load_model_from_checkpoint(cfg, ckpt)
    # with the REAL DINOv2 backbone; inject the mock backbone (matches the checkpoint
    # trained below) so this test stays offline/fast like the rest of the suite.
    monkeypatch.setattr(
        "src.inference.load_model_from_checkpoint",
        lambda cfg, ckpt_path, backbone=None: _real_load(cfg, ckpt_path, backbone=_MockDinoBackbone()),
    )

    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    # A real segmentable otolith-like shape (dark ellipse on light background) so
    # detect_axis succeeds and the walkthrough branch actually runs.
    img = np.full((300, 220, 3), 255, dtype=np.uint8)
    cv2.ellipse(img, (110, 150), (60, 100), angle=0, startAngle=0, endAngle=360,
                color=(40, 40, 40), thickness=-1)

    rows = []
    for i, split in enumerate(["train", "val", "test"]):
        fname = f"2022_BIAS_HER_Loc_Embedded_Sharp_FishIndex{i}_Single1_Left.png"
        PILImage.fromarray(img, "RGB").save(img_dir / fname)
        rows.append({"image_id": fname, "age": 4, "split": split})
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)
    test_fname = rows[-1]["image_id"]

    cfg = _make_cfg(tmp_path, labels_csv, img_dir)
    ckpt = _save_mock_checkpoint(cfg, labels_csv, img_dir)

    samples = [{"image_id": test_fname, "age": 4, "predicted_age": 4}]
    _grids, _axis_data, wt = _compute_axis_data_for_samples(
        samples, img_dir, cfg, ckpt, tmp_path / "cond",
    )

    assert wt is not None, "walkthrough payload was not built — otolith failed to segment"
    for key in ("panel_patchgrid_b64", "panel_rays_b64", "panel_rings_b64", "panel_final_b64"):
        b64 = wt[key]
        assert b64, f"{key} is empty"
        assert not b64.startswith("data:image"), (
            f"{key} is pre-prefixed with a data URI — will be double-prefixed by _img_tag()")
        assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n", f"{key} is not a valid PNG"

    # Krok 2 companion images (one per sample_profiles entry) — same raw-base64 contract.
    ray_imgs = wt["panel_ray_examples_b64"]
    assert len(ray_imgs) == len(wt["data"]["sample_profiles"]) >= 1
    for b64 in ray_imgs:
        assert b64 and not b64.startswith("data:image")
        assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"
