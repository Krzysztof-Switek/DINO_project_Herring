"""Standalone smoke test for the 05.08 "weak-supervision, attention-first" changes:
Change A (attention density head), Change B (angle-windowed E9), Change C
(quarter/"-1" age adjustment) — run TOGETHER, exactly as intended for the next real
training run. Unit tests (tests/test_stage3_model.py, test_stage2_dataset.py,
test_stage9_scan_labels.py) already verify each change in isolation; this script's
only job is to catch a bug in the INTERACTION between all three before queuing an
expensive multi-hour server run — see plans and summaries/5.08_plan_TO_DO.md /
the session plan's "Sequencing & validation" section.

Usage:
    python scripts/smoke_test_attention_first.py

No pytest, no network, no GPU required (mock backbone). Exits 0 on success, 1 on
any failure.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from PIL import Image as PILImage
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Mock backbone — same shape contract as src/model.py expects, but with
# CONTENT-dependent patch tokens (unlike scripts/smoke_test.py's purely
# positional mock) so the attention head has non-degenerate input to mix.
# ---------------------------------------------------------------------------

class _ContentMockBackbone(nn.Module):
    embed_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, self.embed_dim)

    def forward_features(self, x: Tensor) -> Dict:
        B, C, H, W = x.shape
        H_p, W_p = H // 14, W // 14
        num_patches = H_p * W_p
        # Per-patch mean colour -> embed_dim, so different image content
        # (the drawn ellipse vs background) produces different patch tokens.
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (H_p, W_p))   # (B,3,H_p,W_p)
        flat = pooled.permute(0, 2, 3, 1).reshape(B, num_patches, C)
        patches = self.proj(flat)
        cls = patches.mean(dim=1)
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}


# ---------------------------------------------------------------------------
# Synthetic data — a real segmentable blob (not random noise), so
# mask_background / compute_polar_grid produce actual geometry rather than
# hitting the "segmentation failed" all-zero fallback.
# ---------------------------------------------------------------------------

def _make_data(root: Path):
    img_dir = root / "images"
    img_dir.mkdir()
    rows = []
    # Alternate BITS1q (adjustable) / BITS4q (not adjustable) so Change C's
    # effect is actually exercised both ways, ages spanning 1-4 so the
    # floor-at-zero path isn't hit by accident.
    campaigns = ["BITS1q", "BITS4q"]
    for split, n in [("train", 8), ("val", 4), ("test", 4)]:
        for i in range(n):
            campaign = campaigns[i % 2]
            age = (i % 4) + 1
            name = f"2022_{campaign}_HER_Loc_Embedded_Sharpest_FishIndex{i}_Single1_Left.png"
            arr = np.full((140, 140, 3), 235, dtype=np.uint8)
            cv2.ellipse(arr, (70, 70), (45, 60), 0, 0, 360, (40, 40, 40), -1)
            PILImage.fromarray(arr, "RGB").save(img_dir / name)
            rows.append({"image_id": name, "age": age, "split": split})
    csv_path = root / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return img_dir, csv_path


CONFIG_PATH = PROJECT_ROOT / "configs" / "config_attention_first.yaml"


def _make_cfg(tmp: Path):
    """Load the REAL configs/config_attention_first.yaml (not a hand-built
    equivalent) and override ONLY what's required to run a tiny, network-free,
    GPU-free local smoke pass: data paths (Z: drive isn't mounted here), device,
    epoch count, and log/checkpoint dirs. Every field that defines the three new
    mechanisms (density_head_type, density_attn_num_*, density_concentricity_*,
    quarter_age_adjustment_*) is left EXACTLY as written in the file — this is
    what makes it a test of the actual artifact, not a parallel approximation of
    it. cfg.model.backbone is also left untouched (dinov2_vits14_reg); the real
    backbone is simply never instantiated because OtolithModel(cfg, backbone=...)
    is called below with an explicit mock override.
    """
    from src.config import load_config
    cfg = load_config(CONFIG_PATH)

    cfg.data.image_dir = str(tmp / "images")
    cfg.data.labels_csv = str(tmp / "labels.csv")
    cfg.data.mask_cache_dir = str(tmp / "masks_cache")
    cfg.data.num_workers = 0

    cfg.training.epochs = 2                 # file says 50 (real run) — smoke test needs seconds, not hours
    cfg.training.freeze_backbone_epochs = 1
    cfg.training.min_epochs = 0             # file says 12 — would never fail, but keep the loop itself short
    cfg.training.device = "cpu"             # file says auto — force cpu, no GPU assumed in this smoke run
    cfg.training.checkpoint_dir = str(tmp / "checkpoints")
    cfg.training.log_dir = str(tmp / "logs")
    cfg.training.keep_only_best = False     # so step "checkpoint round-trip" can find a per-epoch file too
    return cfg


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def step(name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                _results.append((name, True, ""))
                return result
            except Exception:
                _results.append((name, False, traceback.format_exc()))
                return None
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_smoke_test():
    from src.dataset import OtolithDataset
    from src.model import OtolithModel, AttentionDensityHead
    from src.trainer import Trainer
    from src.inference import load_model_from_checkpoint

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        img_dir, csv_path = _make_data(tmp)
        cfg = _make_cfg(tmp)
        common = dict(labels_csv=str(csv_path), image_dir=str(img_dir))

        ds_train = ds_val = None

        @step("1. Build datasets with all 3 flags on")
        def build_datasets():
            nonlocal ds_train, ds_val
            ds_train = OtolithDataset(cfg, split="train", **common)
            ds_val = OtolithDataset(cfg, split="val", **common)
            assert len(ds_train) == 8 and len(ds_val) == 4

        build_datasets()

        @step("2. Change C: age actually shifts for BITS1q, not for BITS4q")
        def check_quarter_adjustment():
            df = pd.read_csv(csv_path)
            df_train = df[df["split"] == "train"].reset_index(drop=True)
            saw_adjusted = saw_unadjusted = False
            for i in range(len(ds_train)):
                item = ds_train[i]
                recorded = int(item["age_original"])
                effective = int(item["age"])
                is_q1 = "BITS1q" in df_train.loc[i, "image_id"]
                if is_q1:
                    assert effective == max(recorded - 1, 0), "BITS1q row must be adjusted"
                    saw_adjusted = True
                else:
                    assert effective == recorded, "BITS4q row must NOT be adjusted"
                    saw_unadjusted = True
            assert saw_adjusted and saw_unadjusted, "fixture should exercise both branches"

        check_quarter_adjustment()

        @step("3. Change B: real polar geometry computed (not segmentation-failure fallback)")
        def check_polar_geometry():
            item = ds_train[0]
            assert "polar_grid" in item and "polar_theta" in item
            assert bool(item["polar_valid"].any()), \
                "segmentable ellipse should mark some patches valid — if this fails, " \
                "geometry silently fell back to all-zero and Change B's window " \
                "logic was never actually exercised below"

        check_polar_geometry()

        train_loader = DataLoader(ds_train, batch_size=4, shuffle=False)
        val_loader = DataLoader(ds_val, batch_size=4, shuffle=False)

        model = trainer = None

        @step("4. Change A: density_head is AttentionDensityHead")
        def build_model():
            nonlocal model, trainer
            model = OtolithModel(cfg, backbone=_ContentMockBackbone())
            assert isinstance(model.density_head, AttentionDensityHead)
            trainer = Trainer(cfg, model, train_loader, val_loader)

        build_model()

        @step("5. Train 2 epochs with A+B+C all enabled together")
        def train():
            trainer.fit()
            assert "density_conc_loss" in trainer.last_val_metrics
            val = trainer.last_val_metrics["density_conc_loss"]
            assert np.isfinite(val), f"density_conc_loss is not finite: {val}"
            ckpt = Path(cfg.training.checkpoint_dir) / "best.pt"
            assert ckpt.exists(), "best.pt was not written"

        train()

        @step("6. Checkpoint round-trip: attention-trained best.pt loads back cleanly")
        def reload_attention():
            reloaded = load_model_from_checkpoint(cfg, Path(cfg.training.checkpoint_dir) / "best.pt",
                                                   backbone=_ContentMockBackbone())
            assert isinstance(reloaded.density_head, AttentionDensityHead)

        reload_attention()

        @step("7. Checkpoint round-trip: attention checkpoint into an 'mlp' model warns, doesn't crash")
        def reload_into_mlp():
            mlp_cfg = _make_cfg(tmp)
            mlp_cfg.model.density_head_type = "mlp"   # old default architecture
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reloaded = load_model_from_checkpoint(
                    mlp_cfg, Path(cfg.training.checkpoint_dir) / "best.pt",
                    backbone=_ContentMockBackbone(),
                )
            assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
                "expected a non-strict-load warning when architectures mismatch"
            assert isinstance(reloaded.density_head, nn.Sequential)

        reload_into_mlp()


def main() -> int:
    print("=" * 70)
    print("OtolithDino -- attention-first smoke test (Change A + B + C together)")
    print("=" * 70)
    run_smoke_test()

    ok = True
    for name, passed, tb in _results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            ok = False
            print("         " + tb.replace("\n", "\n         "))
    print("=" * 70)
    print("RESULT:", "ALL STEPS PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
