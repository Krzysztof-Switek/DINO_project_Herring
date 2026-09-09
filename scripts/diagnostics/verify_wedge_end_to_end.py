"""Faza 6 of the wycinek-katowy plan (`plans and summaries/09.09_wycinek_katowy_plan.md` /
C:\\Users\\kswitek\\.claude\\plans\\wise-moseying-rivest.md): full end-to-end verification with the
REAL DINOv2 backbone (not the mock backbone unit tests and scripts/smoke_test_radial_attention.py
use) on REAL, segmentable photos (ZEGAR) — the same discipline the strip experiment (02.09)
required before its first server run: prove OtolithDataset -> DataLoader -> Trainer.
train_one_epoch()/validate() produces the right tensor shapes and a finite, sane loss BEFORE
queuing an expensive multi-hour server run, not after.

What this specifically exercises that the mock-backbone smoke test cannot:
  - a REAL RadialAttentionDensityHead forward pass on the wedge's own (density_image,
    density_polar_t, density_polar_theta, density_polar_valid) — real backbone features, not a
    content-mock's linear projection.
  - the wedge extraction/cache pipeline (get_or_compute_wedge, extract_polar_wedge_validity) on
    REAL segmented otoliths, including the non-uniform radial warp (09.09 follow-up).
  - a full Trainer.train_one_epoch() + validate() pass with dual_branch_wedge=True end to end.

Usage:
    python scripts/diagnostics/verify_wedge_end_to_end.py

Requires the Z: network share NOT needed here — uses the local ZEGAR photos already used
throughout this session's wedge-geometry diagnostics (scripts/diagnostics/
analyze_zegar_wedge_geometry.py, prototype_polar_wedge.py). Requires the DINOv2 backbone to be
loadable via torch.hub (cached locally already, per this project's prior real training runs).
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.diagnostics.expert_annotation_eval import IMAGE_DIR

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
# A handful of real, previously-verified-segmentable ZEGAR photos (Z20/Z38/Z37 succeeded in the
# 09.09 wedge-geometry report; Z15 is the documented Δθ-outside-coverage case — included on
# purpose so this run also exercises the "reading axis exists but some/all real increments fall
# outside the wedge" path, not just the easy cases).
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
    cfg.data.num_workers = 0

    # Faza 5's own target configuration: wedge branch on, radial_attention density head
    # (the "solution docelowe" the plan calls for — no intermediate mlp step for the wedge).
    cfg.data.dual_branch_wedge = True
    cfg.data.mask_background = True
    cfg.data.wedge_cache_dir = str(tmp / "wedges_cache")
    cfg.model.use_density_head = True
    cfg.model.density_head_type = "radial_attention"
    # E9 concentricity loss deliberately OFF, same accepted scope boundary as the strip
    # (config_strip_a.yaml) — square-image polar_grid's patch count does not match the
    # wedge's own, see Faza 5's plan note.
    cfg.model.density_concentricity_weight = 0.0

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

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        img_dir, csv_path = _make_data(tmp)
        cfg = _make_cfg(tmp, img_dir, csv_path)
        common = dict(labels_csv=str(csv_path), image_dir=str(img_dir))

        ds_train = ds_val = None

        @step("1. Build datasets (real ZEGAR photos, dual_branch_wedge=True)")
        def build_datasets():
            nonlocal ds_train, ds_val
            ds_train = OtolithDataset(cfg, split="train", **common)
            ds_val = OtolithDataset(cfg, split="val", **common)
            assert len(ds_train) == 4 and len(ds_val) == 2

        build_datasets()

        @step("2. Real wedge geometry computed for a segmentable sample (Z20)")
        def check_wedge_geometry():
            idx = [i for i, s in enumerate(ds_train.df["image_id"]) if s.startswith("Z20")][0]
            item = ds_train[idx]
            assert "image_wedge" in item and "wedge_polar_t" in item
            h_p, w_p = cfg.data.wedge_n_radius_patches, cfg.data.wedge_n_angle_patches
            assert item["image_wedge"].shape[-2:] == (h_p * cfg.data.patch_size,
                                                        w_p * cfg.data.patch_size)
            assert item["wedge_polar_t"].shape == (h_p, w_p)
            assert bool((item["wedge_polar_valid"] > 0).any()), \
                "a real segmented otolith must have SOME real-tissue wedge patches"
            # non-uniform radial warp: t must increase FASTER in the last few rows than
            # the first few (09.09 follow-up) — same property test_wedge_extraction.py pins.
            t_col = item["wedge_polar_t"][:, 0]
            assert (t_col[4] - t_col[0]) > 3 * (t_col[-1] - t_col[-5])

        check_wedge_geometry()

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

        build_model()

        @step("4. train_one_epoch() — real backbone forward, finite loss")
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


def main() -> int:
    print("=" * 78)
    print("OtolithDino -- wycinek katowy (polar wedge) real-backbone end-to-end verification")
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
