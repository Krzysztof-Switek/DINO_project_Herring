"""Standalone smoke test: runs the full mini pipeline and prints PASS / FAIL.

Usage:
    python scripts/smoke_test.py

No pytest required. Exits with code 0 on success, 1 on any failure.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torch import Tensor
from torch.utils.data import DataLoader

# Make sure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
        return self.proj(x.mean(dim=(1, 2, 3), keepdim=True).reshape(B, 1))

    def forward_features(self, x: Tensor) -> Dict:
        B, C, H, W = x.shape
        H_p, W_p = H // 14, W // 14
        num_patches = H_p * W_p
        cls = self.forward(x)
        idx = torch.arange(num_patches, dtype=torch.float32, device=x.device)
        scale = (idx + 1.0).reshape(1, num_patches, 1)
        patches = scale.expand(B, num_patches, self.embed_dim).contiguous()
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 5


def _make_data(root: Path):
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


def _make_cfg(tmp: Path):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.num_age_classes = NUM_CLASSES
    cfg.model.dropout = 0.0
    cfg.data.image_size = 56
    cfg.data.patch_size = 14
    cfg.data.num_workers = 0
    cfg.data.metadata_cols = []
    cfg.training.epochs = 2
    cfg.training.freeze_backbone_epochs = 1
    cfg.training.batch_size = 4
    cfg.training.device = "cpu"
    cfg.training.scheduler = "none"
    cfg.training.checkpoint_dir = str(tmp / "checkpoints")
    cfg.training.log_dir = str(tmp / "logs")
    cfg.candidates.min_peak_distance = 1
    cfg.candidates.prominence_threshold = 0.0
    return cfg


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def step(name: str):
    """Decorator that catches exceptions and logs PASS / FAIL."""
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
# Pipeline steps
# ---------------------------------------------------------------------------

def run_smoke_test():
    from src.dataset import OtolithDataset
    from src.model import OtolithModel
    from src.trainer import Trainer
    from src.inference import run_inference, load_model_from_checkpoint
    from src.interpretation import run_interpretation
    from src.candidates import run_candidates

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        img_dir, csv_path = _make_data(tmp)
        cfg = _make_cfg(tmp)
        common = dict(labels_csv=str(csv_path), image_dir=str(img_dir))

        # Step 1: build datasets
        ds_train = ds_val = ds_test = None

        @step("1. Build datasets")
        def build_datasets():
            nonlocal ds_train, ds_val, ds_test
            ds_train = OtolithDataset(cfg, split="train", **common)
            ds_val   = OtolithDataset(cfg, split="val",   **common)
            ds_test  = OtolithDataset(cfg, split="test",  **common)
            assert len(ds_train) == 8 and len(ds_val) == 4 and len(ds_test) == 4

        build_datasets()

        train_loader = DataLoader(ds_train or [], batch_size=4, shuffle=False)
        val_loader   = DataLoader(ds_val   or [], batch_size=4, shuffle=False)
        test_loader  = DataLoader(ds_test  or [], batch_size=4, shuffle=False)

        # Step 2: train
        model = trainer = None

        @step("2. Train 2 epochs (mock backbone)")
        def train():
            nonlocal model, trainer
            model = OtolithModel(cfg, backbone=_MockDinoBackbone())
            trainer = Trainer(cfg, model, train_loader, val_loader)
            trainer.fit()
            ckpts = sorted(Path(cfg.training.checkpoint_dir).glob("*.pt"))
            assert len(ckpts) == 2

        train()

        ckpt_files = sorted(Path(cfg.training.checkpoint_dir).glob("*.pt")) if Path(cfg.training.checkpoint_dir).exists() else []

        # Step 3: load checkpoint
        loaded_model = None

        @step("3. Load model from checkpoint")
        def load_ckpt():
            nonlocal loaded_model
            assert ckpt_files, "No checkpoint files found"
            loaded_model = load_model_from_checkpoint(
                cfg, ckpt_files[-1], backbone=_MockDinoBackbone()
            )
            assert not loaded_model.training

        load_ckpt()

        out_dir = tmp / "outputs"

        # Step 4: inference
        @step("4. run_inference -> predictions.csv + predictions.json")
        def inference():
            summary = run_inference(cfg, loaded_model, test_loader, out_dir)
            assert summary["n_samples"] == 4
            assert (out_dir / "predictions.csv").exists()
            assert (out_dir / "predictions.json").exists()
            df = pd.read_csv(out_dir / "predictions.csv")
            missing = {"image_id", "predicted_age", "target_age", "abs_error", "metadata_used"} - set(df.columns)
            assert not missing, f"Missing columns: {missing}"

        inference()

        # Step 5: interpretation
        interp_dir = tmp / "interpretation"

        @step("5. run_interpretation -> heatmap + overlay PNGs")
        def interpretation():
            results = run_interpretation(cfg, loaded_model, test_loader, interp_dir)
            assert len(results) == 4
            for r in results:
                assert Path(r["heatmap_path"]).exists()
                assert Path(r["overlay_path"]).exists()

        interpretation()

        # Step 6: candidates
        cand_dir = tmp / "candidates_out"

        @step("6. run_candidates -> JSON + overlay PNGs")
        def candidates():
            results = run_candidates(cfg, loaded_model, test_loader, cand_dir)
            assert len(results) == 4
            for r in results:
                jp = Path(r["candidate_markers_path"])
                op = Path(r["candidates_overlay_path"])
                assert jp.exists() and op.exists()
                data = json.loads(jp.read_text(encoding="utf-8"))
                for k in ("image_id", "num_candidates", "peak_pixel_positions", "radial_profile"):
                    assert k in data

        candidates()

        # Step 7: output schema validation
        @step("7. Validate output schema")
        def validate_schema():
            records = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
            assert len(records) == 4
            json_files = list((cand_dir / "candidates").glob("*.json"))
            assert len(json_files) == 4
            for jf in json_files:
                d = json.loads(jf.read_text(encoding="utf-8"))
                assert isinstance(d["peak_pixel_positions"], list)
                assert isinstance(d["radial_profile"], list)
                assert d["num_candidates"] == len(d["peak_pixel_positions"])

        validate_schema()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("OtolithDino -- smoke test")
    print("=" * 55)

    run_smoke_test()

    any_fail = False
    for name, passed, err in _results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            any_fail = True
            for line in err.strip().splitlines():
                print(f"         {line}")

    print("=" * 55)
    if any_fail:
        print("RESULT: FAILED")
        sys.exit(1)
    else:
        print("RESULT: ALL STEPS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
