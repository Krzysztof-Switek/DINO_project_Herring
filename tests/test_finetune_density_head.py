"""Tests for scripts/finetune_density_head.py — freeze correctness is the critical
guarantee here (density_head is the ONLY thing allowed to change)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image as PILImage
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class _MockDinoBackbone(nn.Module):
    embed_dim = 32

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        mean_val = x.mean(dim=(1, 2, 3), keepdim=True).reshape(B, 1)
        return self.proj(mean_val)

    def forward_features(self, x: Tensor) -> Dict:
        B, C, H, W = x.shape
        H_p, W_p = H // 14, W // 14
        num_patches = H_p * W_p
        cls = self.forward(x)
        idx = torch.arange(num_patches, dtype=torch.float32, device=x.device)
        scale = (idx + 1.0).reshape(1, num_patches, 1)
        patches = scale.expand(B, num_patches, self.embed_dim).contiguous()
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}


def _make_segmentable_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """4 train + 2 val synthetic segmentable (ellipse) otolith images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    splits = ["train", "train", "train", "train", "val", "val"]
    for i, split in enumerate(splits):
        arr = np.full((200, 160, 3), 255, dtype=np.uint8)
        cv2.ellipse(arr, (80, 100), (50, 80), 0, 0, 360, (40, 40, 40), -1)
        name = f"fish_{i}.png"
        PILImage.fromarray(arr).save(img_dir / name)
        rows.append({"image_id": name, "age": (i % 4) + 1, "split": split})
    labels_csv = tmp_path / "labels_embedded.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)
    return labels_csv, img_dir


def _make_config_yaml(tmp_path: Path) -> Path:
    cfg_dict = {
        "project": {"seed": 42},
        "model": {
            "backbone": "dinov2_vits14",
            "num_age_classes": 5,
            "dropout": 0.0,
            "head_type": "both",
            "mil_hidden_dim": 16,
            "use_density_head": True,
            "density_conc_weight": 1.0,
            "density_tv_weight": 0.0,
        },
        "data": {
            "image_size": 56,
            "patch_size": 14,
            "mask_background": True,
            "num_workers": 0,
        },
        "training": {"device": "cpu"},
        "candidates": {
            "density_image_size": 112,
            "density_crop_to_otolith": True,
            "density_crop_pad_frac": 0.05,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    return path


def _make_initial_checkpoint(tmp_path: Path, cfg) -> Path:
    from src.model import OtolithModel
    model = OtolithModel(cfg, backbone=_MockDinoBackbone())
    ckpt_path = tmp_path / "initial.pt"
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_loss": 1.0,
        "cfg": {},
    }, ckpt_path)
    return ckpt_path


def test_finetune_only_moves_density_head_weights(tmp_path, monkeypatch):
    from src.config import load_config
    from src.inference import load_model_from_checkpoint as _real_load
    import scripts.finetune_density_head as ft

    monkeypatch.setattr(
        "src.inference.load_model_from_checkpoint",
        lambda cfg, ckpt_path, backbone=None: _real_load(cfg, ckpt_path, backbone=_MockDinoBackbone()),
    )

    labels_csv, img_dir = _make_segmentable_dataset(tmp_path)
    cfg_path = _make_config_yaml(tmp_path)
    cfg = load_config(cfg_path)
    initial_ckpt = _make_initial_checkpoint(tmp_path, cfg)
    out_ckpt = tmp_path / "finetuned.pt"

    ft.main([
        "--config", str(cfg_path),
        "--config-embedded", str(tmp_path / "does_not_exist.yaml"),
        "--checkpoint", str(initial_ckpt),
        "--output", str(out_ckpt),
        "--labels", str(labels_csv),
        "--image-dir", str(img_dir),
        "--epochs", "1",
        "--batch-size", "2",
        "--patience", "0",
    ])

    assert out_ckpt.exists()
    before = torch.load(initial_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    after = torch.load(out_ckpt, map_location="cpu", weights_only=False)["model_state_dict"]

    changed, unchanged = [], []
    for key in before:
        if torch.equal(before[key], after[key]):
            unchanged.append(key)
        else:
            changed.append(key)

    assert changed, "density_head should have changed after fine-tuning"
    assert all(k.startswith("density_head.") for k in changed), (
        f"non-density_head weights changed: {[k for k in changed if not k.startswith('density_head.')]}")
    # backbone / CORAL (head) / MIL (patch_head) must be byte-for-byte identical
    assert any(k.startswith("backbone.") for k in unchanged)
    assert any(k.startswith("head.") for k in unchanged)
    assert any(k.startswith("patch_head.") for k in unchanged)


def test_finetune_output_checkpoint_has_metadata(tmp_path, monkeypatch):
    from src.config import load_config
    from src.inference import load_model_from_checkpoint as _real_load
    import scripts.finetune_density_head as ft

    monkeypatch.setattr(
        "src.inference.load_model_from_checkpoint",
        lambda cfg, ckpt_path, backbone=None: _real_load(cfg, ckpt_path, backbone=_MockDinoBackbone()),
    )

    labels_csv, img_dir = _make_segmentable_dataset(tmp_path)
    cfg_path = _make_config_yaml(tmp_path)
    cfg = load_config(cfg_path)
    initial_ckpt = _make_initial_checkpoint(tmp_path, cfg)
    out_ckpt = tmp_path / "finetuned.pt"

    ft.main([
        "--config", str(cfg_path),
        "--config-embedded", str(tmp_path / "does_not_exist.yaml"),
        "--checkpoint", str(initial_ckpt),
        "--output", str(out_ckpt),
        "--labels", str(labels_csv),
        "--image-dir", str(img_dir),
        "--epochs", "1",
        "--batch-size", "2",
        "--patience", "0",
    ])

    ckpt = torch.load(out_ckpt, map_location="cpu", weights_only=False)
    meta = ckpt["density_finetune"]
    assert meta["density_image_size"] == 112
    assert meta["density_crop_to_otolith"] is True
    assert meta["base_checkpoint"] == str(initial_ckpt)


def test_finetune_requires_density_head(tmp_path, monkeypatch):
    from src.config import load_config
    from src.inference import load_model_from_checkpoint as _real_load
    import scripts.finetune_density_head as ft

    monkeypatch.setattr(
        "src.inference.load_model_from_checkpoint",
        lambda cfg, ckpt_path, backbone=None: _real_load(cfg, ckpt_path, backbone=_MockDinoBackbone()),
    )

    labels_csv, img_dir = _make_segmentable_dataset(tmp_path)
    cfg_dict_path = _make_config_yaml(tmp_path)
    raw = yaml.safe_load(cfg_dict_path.read_text(encoding="utf-8"))
    raw["model"]["use_density_head"] = False
    cfg_dict_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    from src.config import load_config
    cfg = load_config(cfg_dict_path)
    initial_ckpt = _make_initial_checkpoint(tmp_path, cfg)

    with pytest.raises(SystemExit, match="use_density_head"):
        ft.main([
            "--config", str(cfg_dict_path),
            "--config-embedded", str(tmp_path / "does_not_exist.yaml"),
            "--checkpoint", str(initial_ckpt),
            "--output", str(tmp_path / "out.pt"),
            "--labels", str(labels_csv),
            "--image-dir", str(img_dir),
        ])
