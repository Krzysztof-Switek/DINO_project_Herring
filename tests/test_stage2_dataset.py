"""Stage 2 tests: data format, OtolithDataset, ordinal encoding, metadata."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_data(tmp_path):
    """6 synthetic PNG images + matching labels.csv."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    rows = []
    splits = ["train", "train", "train", "train", "val", "test"]
    for i in range(6):
        name = f"img_{i:03d}.png"
        Image.new("RGB", (64, 64), color=(i * 40, 100, 200)).save(img_dir / name)
        rows.append({
            "image_id": name,
            "age": i + 1,
            "length_cm": 20.0 + i,
            "weight_g": 100.0 + i * 10,
            "sex": "F" if i % 2 == 0 else "M",
            "population": ["North", "South", "East"][i % 3],
            "split": splits[i],
        })

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, img_dir


def _make_cfg(use_metadata: bool = False):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.use_metadata = use_metadata
    cfg.model.num_age_classes = 15
    cfg.data.image_size = 56   # 56 = 14 * 4, divisible by patch_size
    cfg.data.metadata_cols = ["length_cm", "weight_g", "sex", "population"]
    return cfg


# ---------------------------------------------------------------------------
# Ordinal encoding
# ---------------------------------------------------------------------------

def test_encode_age_zero():
    from src.dataset import encode_age_ordinal
    vec = encode_age_ordinal(0, num_classes=10)
    assert vec.shape == (9,)
    assert vec.sum().item() == 0.0


def test_encode_age_middle():
    from src.dataset import encode_age_ordinal
    vec = encode_age_ordinal(3, num_classes=10)
    assert vec.shape == (9,)
    assert vec[:3].sum().item() == 3.0
    assert vec[3:].sum().item() == 0.0


def test_encode_age_max():
    from src.dataset import encode_age_ordinal
    vec = encode_age_ordinal(9, num_classes=10)
    assert vec.sum().item() == 9.0


def test_encode_age_overflow_clamped():
    from src.dataset import encode_age_ordinal
    # age > num_classes-1 should not raise, just fill all ones
    vec = encode_age_ordinal(100, num_classes=10)
    assert vec.sum().item() == 9.0


def test_decode_age_ordinal():
    from src.dataset import decode_age_ordinal
    logits = torch.tensor([5.0, 5.0, 5.0, -5.0, -5.0])
    assert decode_age_ordinal(logits).item() == 3


def test_decode_age_all_positive():
    from src.dataset import decode_age_ordinal
    logits = torch.tensor([5.0, 5.0, 5.0])
    assert decode_age_ordinal(logits).item() == 3


def test_encode_decode_roundtrip():
    from src.dataset import encode_age_ordinal, decode_age_ordinal
    for age in range(10):
        vec = encode_age_ordinal(age, num_classes=12)
        # convert to logits: 1 → +10, 0 → -10
        logits = vec * 20.0 - 10.0
        recovered = decode_age_ordinal(logits).item()
        assert recovered == age, f"roundtrip failed for age={age}: got {recovered}"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def test_dataset_split_sizes(dummy_data):
    csv_path, img_dir = dummy_data
    from src.dataset import OtolithDataset
    cfg = _make_cfg()
    train_ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    val_ds   = OtolithDataset(cfg, "val",   labels_csv=str(csv_path), image_dir=str(img_dir))
    test_ds  = OtolithDataset(cfg, "test",  labels_csv=str(csv_path), image_dir=str(img_dir))
    assert len(train_ds) == 4
    assert len(val_ds)   == 1
    assert len(test_ds)  == 1


def test_dataset_image_shape(dummy_data):
    csv_path, img_dir = dummy_data
    from src.dataset import OtolithDataset
    cfg = _make_cfg()
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image"].shape == (3, 56, 56)
    assert item["image"].dtype == torch.float32


def test_dataset_has_nucleus(dummy_data):
    """Each sample carries a normalized (x, y) nucleus in [0,1] for the radial loss."""
    csv_path, img_dir = dummy_data
    from src.dataset import OtolithDataset
    cfg = _make_cfg()
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["nucleus"].shape == (2,)
    assert item["nucleus"].dtype == torch.float32
    assert bool((item["nucleus"] >= 0).all()) and bool((item["nucleus"] <= 1).all())


def test_dataset_age_ordinal_shape(dummy_data):
    csv_path, img_dir = dummy_data
    from src.dataset import OtolithDataset
    cfg = _make_cfg()
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["age_ordinal"].shape == (14,)  # num_age_classes - 1 = 14


def test_dataset_age_dtype(dummy_data):
    csv_path, img_dir = dummy_data
    from src.dataset import OtolithDataset
    ds = OtolithDataset(_make_cfg(), "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["age"].dtype == torch.long


def test_dataset_image_id_is_string(dummy_data):
    csv_path, img_dir = dummy_data
    from src.dataset import OtolithDataset
    ds = OtolithDataset(_make_cfg(), "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    assert isinstance(ds[0]["image_id"], str)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_not_present_when_disabled(dummy_data):
    csv_path, img_dir = dummy_data
    from src.dataset import OtolithDataset
    ds = OtolithDataset(_make_cfg(use_metadata=False), "train",
                        labels_csv=str(csv_path), image_dir=str(img_dir))
    assert "metadata" not in ds[0]


def test_metadata_tensor_when_enabled(dummy_data):
    csv_path, img_dir = dummy_data
    from src.dataset import OtolithDataset
    ds = OtolithDataset(_make_cfg(use_metadata=True), "train",
                        labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert "metadata" in item
    assert item["metadata"].shape == (4,)
    assert item["metadata"].dtype == torch.float32


def test_metadata_dim_property(dummy_data):
    csv_path, img_dir = dummy_data
    from src.dataset import OtolithDataset
    ds_no  = OtolithDataset(_make_cfg(use_metadata=False), "train",
                            labels_csv=str(csv_path), image_dir=str(img_dir))
    ds_yes = OtolithDataset(_make_cfg(use_metadata=True), "train",
                            labels_csv=str(csv_path), image_dir=str(img_dir))
    assert ds_no.metadata_dim == 0
    assert ds_yes.metadata_dim == 4


def test_sex_encoding_values(dummy_data):
    csv_path, img_dir = dummy_data
    from src.dataset import OtolithDataset
    ds = OtolithDataset(_make_cfg(use_metadata=True), "train",
                        labels_csv=str(csv_path), image_dir=str(img_dir))
    # Collect sex encodings from all train rows
    sex_vals = set()
    for i in range(len(ds)):
        meta = ds[i]["metadata"]
        sex_vals.add(meta[2].item())  # sex is 3rd column
    assert sex_vals == {0.0, 1.0}   # both F and M present in train split


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_missing_required_column_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"image_id": ["x.png"], "split": ["train"]}).to_csv(csv, index=False)
    from src.dataset import OtolithDataset
    with pytest.raises(ValueError, match="age"):
        OtolithDataset(_make_cfg(), "train", labels_csv=str(csv), image_dir=str(tmp_path))


def test_negative_age_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"image_id": ["x.png"], "age": [-1], "split": ["train"]}).to_csv(csv, index=False)
    from src.dataset import OtolithDataset
    with pytest.raises(ValueError, match="negative"):
        OtolithDataset(_make_cfg(), "train", labels_csv=str(csv), image_dir=str(tmp_path))


def test_missing_csv_raises():
    from src.dataset import OtolithDataset
    with pytest.raises(FileNotFoundError):
        OtolithDataset(_make_cfg(), "train", labels_csv="/no/such/file.csv")


# ---------------------------------------------------------------------------
# Sample CSV
# ---------------------------------------------------------------------------

def test_labels_sample_csv_exists():
    assert (PROJECT_ROOT / "data" / "labels_sample.csv").exists()


def test_labels_sample_csv_schema():
    from src.dataset import REQUIRED_COLUMNS
    df = pd.read_csv(PROJECT_ROOT / "data" / "labels_sample.csv")
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert len(df) >= 5
    assert df["age"].ge(0).all()
