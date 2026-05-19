"""Stage 13: tryb demo — limit datasetu i ścieżki cards.

Demo mode ma tylko jeden cel: szybko przejść CAŁY pipeline na małej próbce,
żeby zweryfikować poprawność techniczną przed kosztownym pełnym treningiem.
Limity działają w OtolithDataset, więc train/val/test/inferencja/cards/raport
od razu pracują na ograniczonych danych.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synth_labels(tmp_path: Path, n_per_split: dict[str, int]) -> tuple[Path, Path]:
    """Create N synthetic PNGs and matching labels.csv with given split counts."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for split, n in n_per_split.items():
        for _ in range(n):
            name = f"img_{idx:04d}.png"
            Image.new("RGB", (56, 56), color=(idx % 250, 100, 50)).save(img_dir / name)
            rows.append({"image_id": name, "age": idx % 4, "split": split})
            idx += 1
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, img_dir


def _make_cfg(csv_path: Path, img_dir: Path):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.num_age_classes = 5
    cfg.data.image_size = 56
    cfg.data.patch_size = 14
    cfg.data.labels_csv = str(csv_path)
    cfg.data.image_dir = str(img_dir)
    cfg.data.metadata_cols = []
    return cfg


# ---------------------------------------------------------------------------
# DemoConfig
# ---------------------------------------------------------------------------

def test_demo_config_defaults():
    """OtolithConfig has a demo section, disabled by default, no limits."""
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    assert cfg.demo.enabled is False
    assert cfg.demo.max_train_samples is None
    assert cfg.demo.max_val_samples is None
    assert cfg.demo.max_test_samples is None


def test_demo_config_yaml_load():
    """configs/config_demo.yaml deserialises with demo.enabled=True and the limits."""
    from src.config import load_config
    cfg = load_config(PROJECT_ROOT / "configs" / "config_demo.yaml")
    assert cfg.demo.enabled is True
    assert cfg.demo.max_train_samples is not None and cfg.demo.max_train_samples > 0
    assert cfg.demo.max_val_samples  is not None and cfg.demo.max_val_samples  > 0
    assert cfg.demo.max_test_samples is not None and cfg.demo.max_test_samples > 0


# ---------------------------------------------------------------------------
# Dataset subsampling
# ---------------------------------------------------------------------------

def test_demo_disabled_returns_full_split(tmp_path):
    """With demo.enabled=False the dataset is NOT limited."""
    from src.dataset import OtolithDataset
    csv, img_dir = _make_synth_labels(tmp_path, {"train": 20, "val": 5, "test": 5})
    cfg = _make_cfg(csv, img_dir)
    cfg.demo.enabled = False
    cfg.demo.max_train_samples = 3
    cfg.demo.max_val_samples = 2
    cfg.demo.max_test_samples = 2

    ds_train = OtolithDataset(cfg, split="train")
    ds_val   = OtolithDataset(cfg, split="val")
    ds_test  = OtolithDataset(cfg, split="test")
    assert len(ds_train) == 20
    assert len(ds_val)   == 5
    assert len(ds_test)  == 5


def test_demo_enabled_limits_each_split(tmp_path):
    """With demo.enabled=True dataset is cut to the configured limit per split."""
    from src.dataset import OtolithDataset
    csv, img_dir = _make_synth_labels(tmp_path, {"train": 20, "val": 10, "test": 10})
    cfg = _make_cfg(csv, img_dir)
    cfg.demo.enabled = True
    cfg.demo.max_train_samples = 4
    cfg.demo.max_val_samples = 2
    cfg.demo.max_test_samples = 3

    assert len(OtolithDataset(cfg, split="train")) == 4
    assert len(OtolithDataset(cfg, split="val"))   == 2
    assert len(OtolithDataset(cfg, split="test"))  == 3


def test_demo_limit_above_split_size_is_noop(tmp_path):
    """Limit ≥ split size returns the full split (no over-subsampling)."""
    from src.dataset import OtolithDataset
    csv, img_dir = _make_synth_labels(tmp_path, {"train": 3, "val": 2, "test": 2})
    cfg = _make_cfg(csv, img_dir)
    cfg.demo.enabled = True
    cfg.demo.max_train_samples = 100  # > 3
    cfg.demo.max_val_samples = None    # unbounded
    cfg.demo.max_test_samples = 100

    assert len(OtolithDataset(cfg, split="train")) == 3
    assert len(OtolithDataset(cfg, split="val"))   == 2
    assert len(OtolithDataset(cfg, split="test"))  == 2


def test_demo_subsample_is_deterministic(tmp_path):
    """Same seed → same subset across re-instantiations (cross-condition consistency)."""
    from src.dataset import OtolithDataset
    csv, img_dir = _make_synth_labels(tmp_path, {"train": 30, "val": 4, "test": 4})
    cfg = _make_cfg(csv, img_dir)
    cfg.demo.enabled = True
    cfg.demo.max_train_samples = 5

    ids_a = sorted(OtolithDataset(cfg, split="train").df["image_id"].tolist())
    ids_b = sorted(OtolithDataset(cfg, split="train").df["image_id"].tolist())
    assert ids_a == ids_b


# ---------------------------------------------------------------------------
# select_top_k_samples — bug fix (predictions.csv carries target_age, not age)
# ---------------------------------------------------------------------------

def test_select_top_k_samples_accepts_target_age(tmp_path):
    """select_top_k_samples must work on the actual CSV format written by run_inference,
    which stores the true label as ``target_age`` (not ``age``)."""
    from src.visualization import select_top_k_samples
    rows = [
        {"image_id": f"img_{i}.png", "predicted_age": p, "target_age": t, "abs_error": abs(p - t)}
        for i, (p, t) in enumerate([(2, 2), (3, 5), (4, 4), (0, 6), (1, 1)])
    ]
    csv = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    best, worst = select_top_k_samples(csv, k_best=2, k_worst=2)
    assert len(best)  == 2
    assert len(worst) == 2
    # Best = smallest |pred - true| (= 0): img_0, img_2, img_4
    best_ids = {r["image_id"] for r in best}
    assert best_ids.issubset({"img_0.png", "img_2.png", "img_4.png"})
    # Worst = largest |pred - true| (img_3: |0-6|=6, img_1: |3-5|=2)
    assert any(r["image_id"] == "img_3.png" for r in worst)
