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
# test_pick_walkthrough_iid (ring-visibility-aware example selection, 20.07)
# ---------------------------------------------------------------------------

def test_pick_walkthrough_iid_without_image_dir_falls_back_to_age():
    from scripts.run_pipeline import _pick_walkthrough_iid
    samples = [
        {"image_id": "a.png", "predicted_age": 1},
        {"image_id": "b.png", "predicted_age": 4},
        {"image_id": "c.png", "predicted_age": 9},
    ]
    assert _pick_walkthrough_iid(samples) == "b.png"
    assert _pick_walkthrough_iid(samples, image_dir=None) == "b.png"


def test_pick_walkthrough_iid_rejects_badly_wrong_prediction():
    """A sample closer to age 4 but where the model is WAY off (predicted=2, true=7)
    must lose to a worse-age-distance sample the model actually got right — otherwise
    the walkthrough shows the model confidently placing 2 rings on a stated-true-age-7
    otolith, which is self-contradictory (20.07 user report)."""
    from scripts.run_pipeline import _pick_walkthrough_iid
    samples = [
        {"image_id": "wrong_but_near4.png", "predicted_age": 2, "age": 7},
        {"image_id": "right_but_far.png", "predicted_age": 6, "age": 6},
    ]
    assert _pick_walkthrough_iid(samples) == "right_but_far.png"


def test_pick_walkthrough_iid_accuracy_tolerance_widens_gracefully():
    """No exact (err=0) prediction exists → widen to err<=1, not straight to 'anything'."""
    from scripts.run_pipeline import _pick_walkthrough_iid
    samples = [
        {"image_id": "off_by_one.png", "predicted_age": 4, "age": 5},
        {"image_id": "off_by_five.png", "predicted_age": 4, "age": 9},
    ]
    assert _pick_walkthrough_iid(samples) == "off_by_one.png"


def test_pick_walkthrough_iid_prefers_sharper_image(tmp_path):
    """Given two images at the same (age-4) distance, the sharper one (higher
    Laplacian-variance — a real texture, not a flat/blurry patch) must win."""
    import cv2
    from scripts.run_pipeline import _pick_walkthrough_iid

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    blurry = np.full((80, 80, 3), 180, dtype=np.uint8)   # flat — near-zero Laplacian variance
    PILImage.fromarray(blurry).save(img_dir / "blurry.png")

    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)   # high-frequency noise — sharp
    PILImage.fromarray(sharp, "RGB").save(img_dir / "sharp.png")

    samples = [
        {"image_id": "blurry.png", "predicted_age": 4},
        {"image_id": "sharp.png", "predicted_age": 4},
    ]
    assert _pick_walkthrough_iid(samples, img_dir) == "sharp.png"


def _make_faded_disk(H=400, W=400, center=(200, 200), r_core=80, r_outer=140,
                     bg=10, fg=220) -> np.ndarray:
    """Bright disk fading to a dark background — segmentable by the radial method
    (mirrors tests/test_otolith_axis.py's fixture of the same name)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - center[0], yy - center[1])
    inten = np.full((H, W), float(bg), dtype=np.float32)
    inten[d <= r_core] = fg
    ramp = (d > r_core) & (d <= r_outer)
    inten[ramp] = fg - (fg - bg) * (d[ramp] - r_core) / (r_outer - r_core)
    img = np.clip(inten, 0, 255).astype(np.uint8)
    return np.stack([img] * 3, axis=2)


def _add_dark_rings(img: np.ndarray, center=(200, 200), radii=(30, 50, 70),
                    width=4, dark=60) -> np.ndarray:
    import cv2
    out = img.copy()
    for r in radii:
        cv2.circle(out, center, r, (dark, dark, dark), width)
    return out


def test_pick_walkthrough_iid_prefers_visible_rings_over_flat_image(tmp_path):
    """Given two segmentable otoliths at the same age distance and similar sharpness,
    the one with clearly visible concentric bands (classical_increments finds strong,
    well-supported clusters) must win over a flat one with no ring signal at all —
    this is the actual pipeline detector, not a proxy like raw image sharpness."""
    import cv2
    from scripts.run_pipeline import _pick_walkthrough_iid

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    flat = _make_faded_disk()
    PILImage.fromarray(flat).save(img_dir / "flat.png")

    banded = _add_dark_rings(_make_faded_disk())
    PILImage.fromarray(banded).save(img_dir / "banded.png")

    samples = [
        {"image_id": "flat.png", "predicted_age": 4},
        {"image_id": "banded.png", "predicted_age": 4},
    ]
    assert _pick_walkthrough_iid(samples, img_dir) == "banded.png"


def test_pick_walkthrough_iid_widens_window_only_as_needed(tmp_path):
    """No sample at age exactly 4 → widen just far enough (age 5, window 1), not all the
    way to a sharper but far-off age (age 8, window 4) — few-ring young/old fish are a
    worse teaching example than a slightly-off-4 one (user feedback, 20.07)."""
    from scripts.run_pipeline import _pick_walkthrough_iid

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(2)
    blurry_age5 = np.full((80, 80, 3), 180, dtype=np.uint8)
    PILImage.fromarray(blurry_age5).save(img_dir / "age5.png")
    sharp_age8 = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    PILImage.fromarray(sharp_age8, "RGB").save(img_dir / "age8.png")

    samples = [
        {"image_id": "age5.png", "predicted_age": 5},
        {"image_id": "age8.png", "predicted_age": 8},
    ]
    assert _pick_walkthrough_iid(samples, img_dir) == "age5.png"


def test_pick_walkthrough_iid_restricts_to_age_window_first(tmp_path):
    """A SHARPER image at a wildly different age must NOT be picked over a blurrier one
    near age 4 — the age window is a pre-filter, sharpness only ranks within it."""
    from scripts.run_pipeline import _pick_walkthrough_iid

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    blurry_near_age = np.full((80, 80, 3), 180, dtype=np.uint8)
    PILImage.fromarray(blurry_near_age).save(img_dir / "near_age.png")
    rng = np.random.default_rng(1)
    sharp_far_age = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    PILImage.fromarray(sharp_far_age, "RGB").save(img_dir / "far_age.png")

    samples = [
        {"image_id": "near_age.png", "predicted_age": 4},
        {"image_id": "far_age.png", "predicted_age": 15},   # far outside the |age-4|<=3 window
    ]
    assert _pick_walkthrough_iid(samples, img_dir) == "near_age.png"


def test_pick_walkthrough_iid_missing_files_falls_back_gracefully(tmp_path):
    """No image on disk for anyone → sharpness is -inf for all → falls back to age
    distance / image_id tie-break without crashing."""
    from scripts.run_pipeline import _pick_walkthrough_iid
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    samples = [
        {"image_id": "missing_a.png", "predicted_age": 1},
        {"image_id": "missing_b.png", "predicted_age": 4},
    ]
    assert _pick_walkthrough_iid(samples, img_dir) == "missing_b.png"


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
    for key in ("panel_patchgrid_b64", "panel_rays_b64", "panel_rings_b64"):
        b64 = wt[key]
        assert b64, f"{key} is empty"
        assert not b64.startswith("data:image"), (
            f"{key} is pre-prefixed with a data URI — will be double-prefixed by _img_tag()")
        assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n", f"{key} is not a valid PNG"
    # Krok 5 was merged into the Krok 4 interactive widget (20.07) — no standalone panel.
    assert "panel_final_b64" not in wt

    # Krok 2 companion images (one per sample_profiles entry) — same raw-base64 contract.
    ray_imgs = wt["panel_ray_examples_b64"]
    assert len(ray_imgs) == len(wt["data"]["sample_profiles"]) >= 1
    for b64 in ray_imgs:
        assert b64 and not b64.startswith("data:image")
        assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"

    # Krok 4 interactive: far_edge needed so JS can project chosen t onto the SINGLE
    # measurement axis (merged Krok 4/5, 20.07) — not just the 48-direction ring.
    interactive = wt["krok4_interactive"]
    assert "far_edge" in interactive and len(interactive["far_edge"]) == 2


# ---------------------------------------------------------------------------
# Cards feed the model a MASKED input when cfg.data.mask_background (regression, 21.07)
# ---------------------------------------------------------------------------

def _bg_pixel_seen_by_model(tmp_path, monkeypatch, mask_background: bool):
    """Build a segmentable otolith on a distinctive background, run
    _compute_axis_data_for_samples, and return the background-region pixel (0,0) of the
    image actually handed to build_transforms()'s transform — i.e. what the model sees."""
    import cv2
    from src.dataset import build_transforms
    from src.inference import load_model_from_checkpoint as _real_load
    from scripts.run_pipeline import _compute_axis_data_for_samples

    monkeypatch.setattr(
        "src.inference.load_model_from_checkpoint",
        lambda cfg, ckpt_path, backbone=None: _real_load(cfg, ckpt_path, backbone=_MockDinoBackbone()),
    )

    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    bg_color = (200, 200, 200)
    img = np.full((300, 220, 3), bg_color, dtype=np.uint8)
    cv2.ellipse(img, (110, 150), (60, 100), angle=0, startAngle=0, endAngle=360,
                color=(40, 40, 40), thickness=-1)

    fname = "2022_BIAS_HER_Loc_Embedded_Sharp_FishIndex0_Single1_Left.png"
    PILImage.fromarray(img, "RGB").save(img_dir / fname)
    rows = [{"image_id": fname, "age": 4, "split": s} for s in ("train", "val", "test")]
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)

    cfg = _make_cfg(tmp_path, labels_csv, img_dir)
    cfg.data.mask_background = mask_background
    ckpt = _save_mock_checkpoint(cfg, labels_csv, img_dir)

    captured = {}
    real_transform = build_transforms(cfg.data.image_size, "test")

    def _spy_transform(image):
        captured["bg_pixel"] = tuple(int(v) for v in np.array(image)[0, 0])
        return real_transform(image)

    monkeypatch.setattr("src.dataset.build_transforms", lambda *a, **kw: _spy_transform)

    samples = [{"image_id": fname, "age": 4, "predicted_age": 4}]
    _compute_axis_data_for_samples(samples, img_dir, cfg, ckpt, tmp_path / "cond")
    assert "bg_pixel" in captured, "transform() was never called — model forward pass skipped"
    return captured["bg_pixel"], bg_color


def test_cards_mask_model_input_when_mask_background_enabled(tmp_path, monkeypatch):
    """With mask_background=True, the image fed to the model's forward pass must have its
    background replaced with MASK_FILL_RGB — matching what OtolithDataset feeds at training
    time (src/dataset.py). Before this fix, run_pipeline.py always ran the model on the RAW
    unmasked photo, so density/attention/attn_bg_frac on cards silently measured a different
    input distribution than the model actually trained/infers on (20.07 gap, fixed 21.07)."""
    from src.otolith_axis import MASK_FILL_RGB
    bg_pixel, _orig_bg = _bg_pixel_seen_by_model(tmp_path, monkeypatch, mask_background=True)
    assert bg_pixel == tuple(MASK_FILL_RGB)


def test_cards_leave_model_input_unmasked_when_mask_background_disabled(tmp_path, monkeypatch):
    """With mask_background=False (default), cards must keep feeding the model the RAW
    photo — no behaviour change for runs/configs that never opted into masking."""
    bg_pixel, orig_bg = _bg_pixel_seen_by_model(tmp_path, monkeypatch, mask_background=False)
    assert bg_pixel == orig_bg


# ---------------------------------------------------------------------------
# Higher-resolution / cropped density for candidate detection (22.07)
# ---------------------------------------------------------------------------

def test_hires_cropped_density_uses_shifted_axis_info_and_bigger_grid(tmp_path, monkeypatch):
    """With candidates.density_image_size set + density_crop_to_otolith=True, the density
    forward pass driving select_increments() must: (1) use a grid shaped for
    density_image_size (not data.image_size), (2) be called with axis_info shifted INTO
    the crop's coordinate frame (centroid == full centroid - crop offset), (3) with
    image_h/image_w equal to the CROP's dimensions, not the full photo's. This is the
    exact mechanism a coordinate-frame bug would silently break — asserted directly via
    a spy on select_increments, not inferred from a symptom."""
    import cv2
    from src.inference import load_model_from_checkpoint as _real_load
    from src.otolith_axis import detect_axis, mask_bbox
    import src.ring_extraction as _re_module
    from scripts.run_pipeline import _compute_axis_data_for_samples

    monkeypatch.setattr(
        "src.inference.load_model_from_checkpoint",
        lambda cfg, ckpt_path, backbone=None: _real_load(cfg, ckpt_path, backbone=_MockDinoBackbone()),
    )

    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    img = np.full((300, 220, 3), 255, dtype=np.uint8)
    cv2.ellipse(img, (110, 150), (60, 100), angle=0, startAngle=0, endAngle=360,
                color=(40, 40, 40), thickness=-1)
    fname = "2022_BIAS_HER_Loc_Embedded_Sharp_FishIndex0_Single1_Left.png"
    PILImage.fromarray(img, "RGB").save(img_dir / fname)
    rows = [{"image_id": fname, "age": 4, "split": s} for s in ("train", "val", "test")]
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)

    cfg = _make_cfg(tmp_path, labels_csv, img_dir)
    cfg.model.use_density_head = True
    cfg.candidates.density_image_size = 112          # != data.image_size (56) -> triggers hi-res path
    cfg.candidates.density_crop_to_otolith = True
    cfg.candidates.density_crop_pad_frac = 0.05
    ckpt = _save_mock_checkpoint(cfg, labels_csv, img_dir)

    # Independently reproduce the expected crop (same image, same deterministic segmentation).
    mask_arr = detect_axis(img, seg_params=cfg.segmentation.as_params())["mask"]
    exp_x0, exp_y0, exp_cw, exp_ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)

    real_select_increments = _re_module.select_increments
    captured: dict = {}

    def _spy(grid, axis_info, age, image_h, image_w, **kwargs):
        captured["grid_shape"] = grid.shape
        captured["centroid"] = axis_info["centroid"]
        captured["dims"] = (image_h, image_w)
        return real_select_increments(grid, axis_info, age, image_h, image_w, **kwargs)

    monkeypatch.setattr("src.ring_extraction.select_increments", _spy)

    samples = [{"image_id": fname, "age": 4, "predicted_age": 4}]
    grids, axis_data, _wt = _compute_axis_data_for_samples(
        samples, img_dir, cfg, ckpt, tmp_path / "cond",
    )

    assert "grid_shape" in captured, "select_increments (spied) was never called"
    assert captured["grid_shape"] == (112 // 14, 112 // 14)          # density_image_size grid, not 56/14
    assert captured["dims"] == (exp_ch, exp_cw)                      # crop dims, not full (300, 220)
    full_centroid = detect_axis(img, seg_params=cfg.segmentation.as_params())["centroid"]
    assert captured["centroid"] == (full_centroid[0] - exp_x0, full_centroid[1] - exp_y0)

    # Output points must be shifted BACK to full-image coordinates for drawing.
    iid = fname
    for (x, y) in axis_data[iid]["candidate_pts"] + axis_data[iid]["final_axis_pts"]:
        assert 0 <= x < 220 and 0 <= y < 300

    # Panel-4 heatmap / grids[iid] stays at the ORIGINAL resolution (data.image_size),
    # unaffected by the density-only hi-res path (56/14=4).
    assert grids[iid].shape == (4, 4)


def test_hires_density_off_by_default_matches_today(tmp_path, monkeypatch):
    """density_image_size=None, density_crop_to_otolith=False (defaults) -> select_increments
    is called with the ORIGINAL grid/axis_info/dims, zero behaviour change."""
    import cv2
    from src.inference import load_model_from_checkpoint as _real_load
    import src.ring_extraction as _re_module
    from scripts.run_pipeline import _compute_axis_data_for_samples

    monkeypatch.setattr(
        "src.inference.load_model_from_checkpoint",
        lambda cfg, ckpt_path, backbone=None: _real_load(cfg, ckpt_path, backbone=_MockDinoBackbone()),
    )

    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    img = np.full((300, 220, 3), 255, dtype=np.uint8)
    cv2.ellipse(img, (110, 150), (60, 100), angle=0, startAngle=0, endAngle=360,
                color=(40, 40, 40), thickness=-1)
    fname = "2022_BIAS_HER_Loc_Embedded_Sharp_FishIndex0_Single1_Left.png"
    PILImage.fromarray(img, "RGB").save(img_dir / fname)
    rows = [{"image_id": fname, "age": 4, "split": s} for s in ("train", "val", "test")]
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)

    cfg = _make_cfg(tmp_path, labels_csv, img_dir)
    cfg.model.use_density_head = True
    ckpt = _save_mock_checkpoint(cfg, labels_csv, img_dir)

    real_select_increments = _re_module.select_increments
    captured: dict = {}

    def _spy(grid, axis_info, age, image_h, image_w, **kwargs):
        captured["grid_shape"] = grid.shape
        captured["dims"] = (image_h, image_w)
        return real_select_increments(grid, axis_info, age, image_h, image_w, **kwargs)

    monkeypatch.setattr("src.ring_extraction.select_increments", _spy)

    samples = [{"image_id": fname, "age": 4, "predicted_age": 4}]
    _compute_axis_data_for_samples(samples, img_dir, cfg, ckpt, tmp_path / "cond")

    assert captured["grid_shape"] == (56 // 14, 56 // 14)   # data.image_size grid, unchanged
    assert captured["dims"] == (300, 220)                    # full image dims, unchanged
