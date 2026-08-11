"""Standalone smoke test for the 10.08 "Zmiana A v2" change: RadialAttentionDensityHead
(density_head_type="radial_attention") — polar positional encoding + locally-windowed
attention. Unit tests (tests/test_stage3_model.py) already verify the head's internals in
isolation; this script's job is to catch a bug in the INTERACTION between dataset (polar
grid), trainer (threading polar tensors into the model call), and inference
(get_density_probs with polar args) before queuing an expensive multi-hour server run —
same discipline as scripts/smoke_test_attention_first.py.

NOT covered here (accepted, documented gap — see plans and summaries/9.08_uwaga_plan_TO.DO.md):
scripts/run_pipeline.py's own polar-tensor wiring (`_polar_tensors_for`, the two
get_density_probs call sites in `_compute_axis_data_for_samples`) is NOT exercised by this
script, because that function loads the REAL DINOv2 backbone internally (no mock-backbone
injection point) and this smoke test deliberately has no network/GPU dependency. That
function's own surrounding logic is covered by tests/test_stage12_pipeline.py (unaffected/
passing); the get_density_probs call itself is proven correct by step 6 below. The first
real server run is therefore also the first true end-to-end exercise of run_pipeline.py's
specific wiring — flagged explicitly, not silently assumed covered.

Usage:
    python scripts/smoke_test_radial_attention.py

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
    """Same recipe as smoke_test_attention_first.py's mock — content-dependent patch
    tokens (not purely positional), so the head has non-degenerate input to mix."""
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


def _make_data(root: Path):
    """Real segmentable ellipse (not random noise) — mask_background/compute_polar_grid
    must produce actual geometry, not hit the all-zero segmentation-failure fallback."""
    img_dir = root / "images"
    img_dir.mkdir()
    rows = []
    for split, n in [("train", 8), ("val", 4), ("test", 4)]:
        for i in range(n):
            age = (i % 4) + 1
            name = f"2022_BITS4q_HER_Loc_Embedded_Sharpest_FishIndex{i}_Single1_Left.png"
            arr = np.full((140, 140, 3), 235, dtype=np.uint8)
            cv2.ellipse(arr, (70, 70), (45, 60), 0, 0, 360, (40, 40, 40), -1)
            PILImage.fromarray(arr, "RGB").save(img_dir / f"{split}_{name}")
            rows.append({"image_id": f"{split}_{name}", "age": age, "split": split})
    csv_path = root / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return img_dir, csv_path


CONFIG_PATH = PROJECT_ROOT / "configs" / "config_radial_attention.yaml"


def _make_cfg(tmp: Path):
    """Load the REAL configs/config_radial_attention.yaml — same discipline as
    smoke_test_attention_first.py: every field defining the new mechanism
    (density_head_type, density_attn_num_*, density_attn_radial_bins/window_deg) is left
    EXACTLY as written in the file. Only local-run necessities are overridden."""
    from src.config import load_config
    cfg = load_config(CONFIG_PATH)

    cfg.data.image_dir = str(tmp / "images")
    cfg.data.labels_csv = str(tmp / "labels.csv")
    cfg.data.mask_cache_dir = str(tmp / "masks_cache")
    cfg.data.num_workers = 0

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
    from src.otolith_axis import compute_polar_grid, get_or_compute_mask, resolve_centroid

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        img_dir, csv_path = _make_data(tmp)
        cfg = _make_cfg(tmp)
        common = dict(labels_csv=str(csv_path), image_dir=str(img_dir))

        ds_train = ds_val = None

        @step("1. Build datasets with radial_attention (_need_polar must trigger from head_type alone)")
        def build_datasets():
            nonlocal ds_train, ds_val
            ds_train = OtolithDataset(cfg, split="train", **common)
            ds_val = OtolithDataset(cfg, split="val", **common)
            assert ds_train._need_polar, \
                "_need_polar must be True for radial_attention even with density_concentricity_weight=0"
            assert len(ds_train) == 8 and len(ds_val) == 4

        build_datasets()

        @step("2. Real polar geometry computed (not segmentation-failure fallback)")
        def check_polar_geometry():
            item = ds_train[0]
            assert "polar_grid" in item and "polar_theta" in item
            assert bool(item["polar_valid"].any()), \
                "segmentable ellipse should mark some patches valid"

        check_polar_geometry()

        train_loader = DataLoader(ds_train, batch_size=4, shuffle=False)
        val_loader = DataLoader(ds_val, batch_size=4, shuffle=False)

        model = trainer = None

        @step("3. density_head is RadialAttentionDensityHead")
        def build_model():
            nonlocal model, trainer
            model = OtolithModel(cfg, backbone=_ContentMockBackbone())
            assert isinstance(model.density_head, RadialAttentionDensityHead)
            trainer = Trainer(cfg, model, train_loader, val_loader)

        build_model()

        @step("4. Train 2 epochs — polar tensors reach the model (not just the loss)")
        def train():
            trainer.fit()
            ckpt = Path(cfg.training.checkpoint_dir) / "best.pt"
            assert ckpt.exists(), "best.pt was not written"
            # density_conc_loss is absent by design (density_concentricity_weight=0 in
            # this config, Zmiana B deliberately off) — assert its ABSENCE, catching an
            # accidental config drift instead of silently passing either way.
            assert "density_conc_loss" not in trainer.last_val_metrics, \
                "density_concentricity_weight=0 should mean no E9 loss component logged"
            assert "density_active" in trainer.last_val_metrics

        train()

        @step("5. Checkpoint round-trip: radial_attention best.pt loads back cleanly")
        def reload():
            reloaded = load_model_from_checkpoint(
                cfg, Path(cfg.training.checkpoint_dir) / "best.pt",
                backbone=_ContentMockBackbone(),
            )
            assert isinstance(reloaded.density_head, RadialAttentionDensityHead)

        reload()

        @step("6. get_density_probs WITH real polar args (the run_pipeline.py call shape) works")
        def check_get_density_probs_with_polar():
            # Mirrors exactly what scripts/run_pipeline.py's _polar_tensors_for + the two
            # get_density_probs call sites do: resolve mask/centroid, compute_polar_grid,
            # flatten to (1, N), pass through. Real compute_polar_grid on the real
            # synthetic ellipse (not synthetic tensors like the unit tests) — the closest
            # this script gets to the run_pipeline.py integration point without a real
            # backbone (see module docstring for why that's not covered here).
            raw = np.array(PILImage.open(img_dir / "train_2022_BITS4q_HER_Loc_Embedded_Sharpest_FishIndex0_Single1_Left.png").convert("RGB"))
            mask = get_or_compute_mask(raw, tmp / "smoke_mask.png", seg_params=cfg.segmentation.as_params())
            assert mask is not None, "synthetic ellipse must be segmentable"
            centroid = resolve_centroid(raw, mask, cfg.segmentation.nucleus_method)
            h_p = w_p = cfg.data.image_size // cfg.data.patch_size
            t_grid, valid_grid, theta_grid = compute_polar_grid(mask, centroid, h_p, w_p)
            to_flat = lambda a, dt: torch.from_numpy(a.reshape(1, -1).copy()).to(dt)
            p_t, p_th, p_v = (to_flat(t_grid, torch.float32), to_flat(theta_grid, torch.float32),
                              to_flat(valid_grid, torch.bool))

            tensor = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)
            with_polar = model.get_density_probs(tensor, polar_t=p_t, polar_theta=p_th, polar_valid=p_v)
            without_polar = model.get_density_probs(tensor)
            assert with_polar.shape == without_polar.shape
            assert not torch.allclose(with_polar, without_polar), \
                "passing real polar args must change get_density_probs' output " \
                "(otherwise the wiring is silently not reaching the head)"

        check_get_density_probs_with_polar()


def main() -> int:
    print("=" * 70)
    print("OtolithDino -- radial_attention (Change A v2) smoke test")
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
