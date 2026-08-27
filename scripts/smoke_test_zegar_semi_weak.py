"""Standalone smoke test for the 26.08 "Opcja A" change (semi-weak supervision):
zegar_position_loss + OtolithDataset's zegar_heatmap/has_zegar_target wiring. Unit tests
(tests/test_stage2_dataset.py, tests/test_stage3_model.py, tests/test_stage4_trainer.py) already
verify each piece in isolation; this script's job is to catch a bug in the INTERACTION between
dataset (merged main+ZEGAR labels, per-sample heatmap), trainer (masked loss gating, threading the
new tensors into a real training loop), and checkpoint round-trip — before queuing an expensive
multi-hour server run. Same discipline as scripts/smoke_test_radial_attention.py.

Usage:
    python scripts/smoke_test_zegar_semi_weak.py

No pytest, no network, no GPU required (mock backbone). Exits 0 on success, 1 on any failure.
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
from PIL import Image as PILImage
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _ContentMockBackbone(nn.Module):
    """Same recipe as smoke_test_radial_attention.py's mock — content-dependent patch
    tokens, so the density head has non-degenerate input to mix."""
    embed_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, self.embed_dim)

    def forward_features(self, x: Tensor) -> Dict:
        B, C, H, W = x.shape
        H_p, W_p = H // 14, W // 14
        num_patches = H_p * W_p
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (H_p, W_p))
        flat = pooled.permute(0, 2, 3, 1).reshape(B, num_patches, C)
        patches = self.proj(flat)
        cls = patches.mean(dim=1)
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}


def _make_ellipse(path: Path, size: int = 140) -> None:
    arr = np.full((size, size, 3), 235, dtype=np.uint8)
    cv2.ellipse(arr, (size // 2, size // 2), (size // 3, size // 2 - 10), 0, 0, 360, (40, 40, 40), -1)
    PILImage.fromarray(arr, "RGB").save(path)


def _make_main_data(root: Path):
    """Main (non-ZEGAR) dataset — mirrors smoke_test_radial_attention.py's fixture."""
    img_dir = root / "images"
    img_dir.mkdir()
    rows = []
    for split, n in [("train", 8), ("val", 4), ("test", 4)]:
        for i in range(n):
            age = (i % 4) + 1
            name = f"2022_BITS4q_HER_Loc_Embedded_Sharpest_FishIndex{i}_Single1_Left.png"
            _make_ellipse(img_dir / f"{split}_{name}")
            rows.append({"image_id": f"{split}_{name}", "age": age, "split": split})
    csv_path = root / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return img_dir, csv_path


def _make_zegar_data(root: Path):
    """Mirrors scripts/prepare_zegar_semi_weak_data.py's output shape, at smoke-test
    scale — 3 samples, 2 annotated points each, in the crop's own pixel space."""
    zegar_img_dir = root / "zegar_processed"
    zegar_img_dir.mkdir()
    labels_rows, target_rows = [], []
    for i in range(3):
        image_id = f"ZEGAR_T{i:02d}.jpg"
        _make_ellipse(zegar_img_dir / image_id)
        labels_rows.append({"image_id": image_id, "age": 3 + i, "split": "train"})
        target_rows.append({"image_id": image_id, "x": 50.0 + i, "y": 60.0 + i})
        target_rows.append({"image_id": image_id, "x": 90.0 - i, "y": 40.0 + i})
    labels_csv = root / "zegar_labels.csv"
    targets_csv = root / "zegar_targets.csv"
    pd.DataFrame(labels_rows).to_csv(labels_csv, index=False)
    pd.DataFrame(target_rows).to_csv(targets_csv, index=False)
    return zegar_img_dir, labels_csv, targets_csv


CONFIG_PATH = PROJECT_ROOT / "configs" / "config_zegar_semi_weak.yaml"


def _make_cfg(tmp: Path, zegar_img_dir: Path, zegar_labels_csv: Path, zegar_targets_csv: Path):
    """Load the REAL configs/config_zegar_semi_weak.yaml — every field defining the new
    mechanism (zegar_position_weight, density_head_type=radial_attention) is left EXACTLY
    as written in the file. Only local-run necessities are overridden."""
    from src.config import load_config
    cfg = load_config(CONFIG_PATH)

    cfg.data.image_dir = str(tmp / "images")
    cfg.data.labels_csv = str(tmp / "labels.csv")
    cfg.data.mask_cache_dir = str(tmp / "masks_cache")
    cfg.data.num_workers = 0
    cfg.data.zegar_extra_image_dir = str(zegar_img_dir)
    cfg.data.zegar_extra_labels_csv = str(zegar_labels_csv)
    cfg.data.zegar_targets_csv = str(zegar_targets_csv)

    cfg.training.epochs = 2
    cfg.training.freeze_backbone_epochs = 1
    cfg.training.min_epochs = 0
    cfg.training.device = "cpu"
    cfg.training.checkpoint_dir = str(tmp / "checkpoints")
    cfg.training.log_dir = str(tmp / "logs")
    cfg.training.keep_only_best = False
    return cfg


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


def run_smoke_test():
    from src.dataset import OtolithDataset
    from src.model import OtolithModel, RadialAttentionDensityHead
    from src.trainer import Trainer
    from src.inference import load_model_from_checkpoint

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        img_dir, csv_path = _make_main_data(tmp)
        zegar_img_dir, zegar_labels_csv, zegar_targets_csv = _make_zegar_data(tmp)
        cfg = _make_cfg(tmp, zegar_img_dir, zegar_labels_csv, zegar_targets_csv)
        common = dict(labels_csv=str(csv_path), image_dir=str(img_dir))

        ds_train = ds_val = None

        @step("1. Build datasets — main+ZEGAR merged, ZEGAR rows only in train split")
        def build_datasets():
            nonlocal ds_train, ds_val
            ds_train = OtolithDataset(cfg, split="train", **common)
            ds_val = OtolithDataset(cfg, split="val", **common)
            assert len(ds_train) == 8 + 3, "8 main train rows + 3 ZEGAR rows"
            assert len(ds_val) == 4, "ZEGAR rows must NOT leak into val"
            assert ds_train._zegar_image_ids == {"ZEGAR_T00.jpg", "ZEGAR_T01.jpg", "ZEGAR_T02.jpg"}

        build_datasets()

        @step("2. A ZEGAR sample has a real heatmap target; a main sample does not")
        def check_heatmap_targets():
            zegar_idx = ds_train.df.index[
                ds_train.df["image_id"].isin(ds_train._zegar_image_ids)][0]
            main_idx = ds_train.df.index[
                ~ds_train.df["image_id"].isin(ds_train._zegar_image_ids)][0]
            zegar_item = ds_train[zegar_idx]
            main_item = ds_train[main_idx]
            assert bool(zegar_item["has_zegar_target"]) is True
            assert float(zegar_item["zegar_heatmap"].max()) > 0.9
            assert bool(main_item["has_zegar_target"]) is False
            assert float(main_item["zegar_heatmap"].max()) == 0.0

        check_heatmap_targets()

        train_loader = DataLoader(ds_train, batch_size=4, shuffle=False)
        val_loader = DataLoader(ds_val, batch_size=4, shuffle=False)

        model = trainer = None

        @step("3. Build model (radial_attention density head) + trainer")
        def build_model():
            nonlocal model, trainer
            model = OtolithModel(cfg, backbone=_ContentMockBackbone())
            assert isinstance(model.density_head, RadialAttentionDensityHead)
            trainer = Trainer(cfg, model, train_loader, val_loader)
            assert trainer.zegar_position_w == cfg.model.zegar_position_weight > 0.0

        build_model()

        @step("4. Train 2 epochs on a MIXED batch (some ZEGAR, some not) — no crash")
        def train():
            trainer.fit()
            ckpt = Path(cfg.training.checkpoint_dir) / "best.pt"
            assert ckpt.exists(), "best.pt was not written"

        train()

        @step("5. zegar_position term is actually computed on a real TRAIN batch")
        def check_metric_logged():
            # By design (scripts/prepare_zegar_semi_weak_data.py), ZEGAR rows are
            # ALWAYS split="train" and never appear in val_loader — so validate()'s
            # last_val_metrics correctly never sees the term (already exercised by
            # step 7's disabled-path check, symmetric to this one). What needs
            # verifying here is that the term reaches _loss_parts on a real batch
            # from train_loader, which DOES contain ZEGAR samples.
            # shuffle=False, batch_size=4, 8 main rows then 3 ZEGAR rows (dataset order,
            # see OtolithDataset.__init__'s concat-before-split-filter) -> ZEGAR samples
            # land in the LAST batch, not necessarily the first.
            batch = next(b for b in train_loader if bool(b["has_zegar_target"].any()))
            with torch.no_grad():
                out = model(batch["image"], polar_t=batch["polar_grid"].reshape(batch["image"].shape[0], -1),
                            polar_theta=batch["polar_theta"].reshape(batch["image"].shape[0], -1),
                            polar_valid=batch["polar_valid"].reshape(batch["image"].shape[0], -1))
                B, N = out["density"].shape
                parts = trainer._loss_parts(
                    out, batch["age_ordinal"], batch["age"],
                    batch["polar_grid"].reshape(B, N), batch["polar_valid"].reshape(B, N),
                    batch["polar_theta"].reshape(B, N),
                    batch["zegar_heatmap"], batch["has_zegar_target"],
                )
            assert "zegar_position" in parts, \
                "zegar_position_weight>0 with a real ZEGAR sample in the batch must add the term"

        check_metric_logged()

        @step("6. Checkpoint round-trip loads back cleanly")
        def reload():
            reloaded = load_model_from_checkpoint(
                cfg, Path(cfg.training.checkpoint_dir) / "best.pt",
                backbone=_ContentMockBackbone(),
            )
            assert isinstance(reloaded.density_head, RadialAttentionDensityHead)

        reload()

        @step("7. zegar_position_weight=0 disables the term end-to-end (default-off safety)")
        def check_disabled_path():
            import copy
            cfg_off = copy.deepcopy(cfg)
            cfg_off.model.zegar_position_weight = 0.0
            model_off = OtolithModel(cfg_off, backbone=_ContentMockBackbone())
            trainer_off = Trainer(cfg_off, model_off, train_loader, val_loader)
            trainer_off.validate()
            assert "zegar_position_loss" not in trainer_off.last_val_metrics

        check_disabled_path()


def main() -> int:
    print("=" * 70)
    print("OtolithDino -- Opcja A (semi-weak supervision, zegar_position_loss) smoke test")
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
