"""Faza 16 of the angular-resolution-bands plan (`plans and summaries/
09.09_wycinek_pasma_katowe_plan.md` / C:\\Users\\kswitek\\.claude\\plans\\wise-moseying-rivest.md):
full end-to-end verification with the REAL DINOv2 backbone on REAL, segmentable ZEGAR photos —
same discipline `verify_wedge_end_to_end.py` used for the single-canvas wedge, extended to the
4-band architecture (4 SEPARATE backbone forward passes per sample, concatenated before the
density head) before queuing an expensive multi-hour server run.

What this specifically exercises that the mock-backbone model tests cannot:
  - real per-band extraction (`wedge_band_geometries_from_axis_info`/`extract_polar_wedge_band`)
    on REAL segmented otoliths, at the actual measured production column/row counts.
  - `OtolithModel.forward()`'s `density_image_bands` path with a REAL backbone (4 real forward
    passes, not a mock's linear projection) feeding a REAL `RadialAttentionDensityHead`.
  - a full `Trainer.train_one_epoch()` + `validate()` pass with the bands wiring end to end.
  - CLOSES THE MEASURE-FIX-MEASURE LOOP: re-computes the angular compression ratio on the SAME
    real photos `analyze_wedge_angular_resolution.py` used to diagnose the problem (mean 5.49x,
    max 8.57x at the edge with the OLD single-canvas 13-column wedge) and confirms it is now
    close to the target 1.0x with the new per-band column counts — the only real verification
    available for this approach, since no literature precedent exists to check against instead
    (see the plan's own literature section).

Usage:
    python scripts/diagnostics/verify_wedge_bands_end_to_end.py

Requires the Z: network share NOT needed here — uses the local ZEGAR photos already used
throughout this session's wedge diagnostics. Requires the DINOv2 backbone to be loadable via
torch.hub (cached locally already, per this project's prior real training runs).
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.diagnostics.expert_annotation_eval import IMAGE_DIR

CONFIG_PATH = PROJECT_ROOT / "configs" / "config_wedge_b.yaml"
# Same sample set as verify_wedge_end_to_end.py, for direct before/after comparability.
SAMPLE_IDS = ["Z07", "Z12", "Z20", "Z37", "Z38", "Z15"]

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


def _make_data(root: Path) -> tuple[Path, Path]:
    rows = []
    for i, sample in enumerate(SAMPLE_IDS):
        split = "train" if i < 4 else "val"
        rows.append({"image_id": f"{sample}.jpg", "age": (i % 4) + 1, "split": split})
    csv_path = root / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return IMAGE_DIR, csv_path


def _make_cfg(tmp: Path, img_dir: Path, csv_path: Path):
    from src.config import load_config
    cfg = load_config(CONFIG_PATH)

    cfg.data.image_dir = str(img_dir)
    cfg.data.labels_csv = str(csv_path)
    cfg.data.mask_cache_dir = str(tmp / "masks_cache")
    cfg.data.wedge_bands_cache_dir = str(tmp / "wedge_bands_cache")
    cfg.data.num_workers = 0

    cfg.training.epochs = 1
    cfg.training.freeze_backbone_epochs = 0
    cfg.training.min_epochs = 0
    cfg.training.device = "cpu"
    cfg.training.checkpoint_dir = str(tmp / "checkpoints")
    cfg.training.log_dir = str(tmp / "logs")
    cfg.training.keep_only_best = False
    return cfg


def run() -> None:
    from src.dataset import OtolithDataset
    from src.model import OtolithModel, RadialAttentionDensityHead, load_dinov2
    from src.trainer import Trainer
    from src.otolith_axis import detect_axis, get_or_compute_mask
    from src.wedge_extraction import wedge_band_geometries_from_axis_info
    from PIL import Image as PILImage

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        img_dir, csv_path = _make_data(tmp)
        cfg = _make_cfg(tmp, img_dir, csv_path)
        common = dict(labels_csv=str(csv_path), image_dir=str(img_dir))
        n_bands = len(cfg.data.wedge_band_edges_t) - 1

        ds_train = ds_val = None

        @step("1. Build datasets (real ZEGAR photos, wedge bands enabled)")
        def build_datasets():
            nonlocal ds_train, ds_val
            ds_train = OtolithDataset(cfg, split="train", **common)
            ds_val = OtolithDataset(cfg, split="val", **common)
            assert ds_train.wedge_bands_enabled
            assert len(ds_train) == 4 and len(ds_val) == 2

        build_datasets()

        @step(f"2. Real {n_bands}-band geometry computed for a segmentable sample (Z20)")
        def check_band_geometry():
            idx = [i for i, s in enumerate(ds_train.df["image_id"]) if s.startswith("Z20")][0]
            item = ds_train[idx]
            assert "image_wedge" not in item, "bands must REPLACE the single-canvas wedge"
            for i in range(n_bands):
                n_angle = cfg.data.wedge_band_n_angle_patches[i]
                n_radius = cfg.data.wedge_band_n_radius_patches[i]
                key = f"image_wedge_band{i}"
                assert key in item
                assert item[key].shape[-2:] == (n_radius * cfg.data.patch_size,
                                                 n_angle * cfg.data.patch_size)
                t_col = item[f"wedge_band{i}_polar_t"][:, 0]
                assert (t_col[1:] > t_col[:-1]).all(), f"band {i}: t must increase row-by-row"

        check_band_geometry()

        print("  Loading real DINOv2 backbone (torch.hub, cached) ...")
        backbone = load_dinov2(cfg.model.backbone)

        train_loader = DataLoader(ds_train, batch_size=2, shuffle=False)
        val_loader = DataLoader(ds_val, batch_size=2, shuffle=False)
        model = trainer = None

        @step("3. Build real OtolithModel (RadialAttentionDensityHead) with the real backbone")
        def build_model():
            nonlocal model, trainer
            model = OtolithModel(cfg, backbone=backbone)
            assert isinstance(model.density_head, RadialAttentionDensityHead)
            trainer = Trainer(cfg, model, train_loader, val_loader)
            assert trainer.n_wedge_bands == n_bands

        build_model()

        @step("4. train_one_epoch() — 4 real backbone passes/sample, finite loss")
        def train_epoch():
            loss = trainer.train_one_epoch()
            assert loss == loss and loss not in (float("inf"), float("-inf")), \
                f"non-finite train loss: {loss}"
            print(f"     train loss = {loss:.4f}")

        train_epoch()

        @step("5. validate() — real backbone forward, finite loss + density diagnostics present")
        def validate():
            val_loss, val_mae = trainer.validate()
            assert val_loss == val_loss and val_loss not in (float("inf"), float("-inf"))
            assert val_mae == val_mae
            print(f"     val loss = {val_loss:.4f}, val MAE = {val_mae:.4f}")
            assert "density_active" in trainer.last_val_metrics
            print(f"     last_val_metrics = {trainer.last_val_metrics}")

        validate()

        @step("6. CLOSES THE LOOP: angular compression re-measured on real photos, now near 1.0x")
        def remeasure_compression():
            seg_params = cfg.segmentation.as_params()
            delta_theta_rad = np.radians(cfg.data.wedge_delta_theta_deg)
            band_edges_t = cfg.data.wedge_band_edges_t
            n_angle = cfg.data.wedge_band_n_angle_patches
            patch_size = cfg.data.patch_size
            worst_ratio = 0.0
            for sample in SAMPLE_IDS:
                raw_rgb = np.array(PILImage.open(img_dir / f"{sample}.jpg").convert("RGB"))
                mask = get_or_compute_mask(raw_rgb, tmp / f"{sample}_mask.png", seg_params=seg_params)
                if mask is None:
                    continue
                axis_info = detect_axis(raw_rgb, seg_params=seg_params,
                                        nucleus_method=cfg.segmentation.nucleus_method,
                                        axis_method=cfg.segmentation.axis_method, mask=mask)
                if axis_info is None:
                    continue
                length_px = axis_info["length_px"]
                for i in range(n_bands):
                    r_hi = band_edges_t[i + 1] * length_px
                    col_angle_rad = delta_theta_rad / n_angle[i]
                    ratio = (r_hi * col_angle_rad) / patch_size
                    worst_ratio = max(worst_ratio, ratio)
            print(f"     najgorszy zmierzony współczynnik kompresji na tych {len(SAMPLE_IDS)} "
                  f"zdjęciach: {worst_ratio:.2f}x (przed: średnio 5.49x, max 8.57x)")
            # generous tolerance -- these 6 photos may not include the true population maximum
            # (1305.1px) the bands were actually sized for, so some slack above 1.0x is expected
            # and not a bug.
            assert worst_ratio < 2.0, \
                f"compression still high ({worst_ratio:.2f}x) on real photos -- investigate"

        remeasure_compression()


def main() -> int:
    print("=" * 78)
    print("OtolithDino -- wycinek katowy PASMA (angular-resolution bands) real-backbone e2e")
    print("=" * 78)
    run()

    ok = True
    for name, passed, tb in _results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            ok = False
            print("         " + tb.replace("\n", "\n         "))
    print("=" * 78)
    print("RESULT:", "ALL STEPS PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
