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
# Input masking (20.07 pre-training item)
# ---------------------------------------------------------------------------

@pytest.fixture
def ellipse_data(tmp_path):
    """One real segmentable otolith-like image (dark ellipse on light background)."""
    import cv2
    import numpy as np

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    arr = np.full((200, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(arr, (80, 100), (50, 80), 0, 0, 360, (40, 40, 40), -1)
    name = "fish_ellipse.png"
    Image.fromarray(arr).save(img_dir / name)

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([{"image_id": name, "age": 4, "split": "train"}]).to_csv(csv_path, index=False)
    return csv_path, img_dir


def test_mask_background_disabled_by_default(dummy_data):
    from src.config import OtolithConfig
    assert OtolithConfig().data.mask_background is False


def test_mask_background_gracefully_skips_unsegmentable_image(dummy_data, tmp_path):
    """dummy_data's images are flat solid colour — no foreground to segment. Masking
    must fall back to the unmasked image, never crash the dataset."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = dummy_data
    cfg = _make_cfg()
    cfg.data.mask_background = True
    cfg.data.mask_cache_dir = str(tmp_path / "masks_cache")
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image"].shape == (3, 56, 56)   # produced normally, no crash


def test_mask_background_changes_pixels_for_segmentable_image(ellipse_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data

    cfg_plain = _make_cfg()
    cfg_plain.data.image_size = 200
    ds_plain = OtolithDataset(cfg_plain, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    cfg_masked = _make_cfg()
    cfg_masked.data.image_size = 200
    cfg_masked.data.mask_background = True
    cfg_masked.data.mask_cache_dir = str(tmp_path / "masks_cache")
    ds_masked = OtolithDataset(cfg_masked, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    plain_img = ds_plain[0]["image"]
    masked_img = ds_masked[0]["image"]
    assert not torch.allclose(plain_img, masked_img)
    assert any((tmp_path / "masks_cache").glob("*_mask.png"))


def test_mask_background_cache_reused_on_second_access(tmp_path, monkeypatch):
    """Second access to the same image must hit the on-disk cache, not re-segment.

    Uses split="test" deliberately — the "train" transform pipeline applies random
    flips/jitter per call, which would make two accesses differ regardless of caching.
    """
    import cv2
    import numpy as np
    from src.dataset import OtolithDataset

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    arr = np.full((200, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(arr, (80, 100), (50, 80), 0, 0, 360, (40, 40, 40), -1)
    name = "fish_ellipse.png"
    Image.fromarray(arr).save(img_dir / name)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([{"image_id": name, "age": 4, "split": "test"}]).to_csv(csv_path, index=False)

    cfg = _make_cfg()
    cfg.data.image_size = 200
    cfg.data.mask_background = True
    cfg.data.mask_cache_dir = str(tmp_path / "masks_cache")
    ds = OtolithDataset(cfg, "test", labels_csv=str(csv_path), image_dir=str(img_dir))
    first = ds[0]["image"]                          # populates the cache file

    def _boom(*a, **kw):
        raise AssertionError("segment_otolith should NOT run again on a cache hit")
    monkeypatch.setattr("src.otolith_axis.segment_otolith", _boom)

    second = ds[0]["image"]
    assert torch.allclose(first, second)


# ---------------------------------------------------------------------------
# E9: polar_grid (concentricity-prior geometry)
# ---------------------------------------------------------------------------

def _e9_cfg(tmp_path):
    cfg = _make_cfg()
    cfg.data.image_size = 196          # 14*14, divisible by patch_size
    cfg.data.mask_background = True
    cfg.data.mask_cache_dir = str(tmp_path / "masks_cache")
    cfg.model.use_density_head = True
    cfg.model.density_concentricity_weight = 0.5
    return cfg


def test_polar_grid_absent_by_default(ellipse_data, tmp_path):
    """density_concentricity_weight=0.0 (default) → zero behaviour change: no
    polar_grid/polar_valid key at all, no extra segmentation-derived geometry."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _make_cfg()
    cfg.data.image_size = 196
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert "polar_grid" not in item
    assert "polar_valid" not in item


def test_polar_grid_absent_without_density_head(ellipse_data, tmp_path):
    """density_concentricity_weight>0 alone is not enough — use_density_head must
    also be on (the concentricity term modifies the density head's own loss)."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _e9_cfg(tmp_path)
    cfg.model.use_density_head = False
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    assert "polar_grid" not in ds[0]


def test_polar_grid_present_and_shaped_when_enabled(ellipse_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _e9_cfg(tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    h_p = w_p = cfg.data.image_size // cfg.data.patch_size
    assert item["polar_grid"].shape == (h_p, w_p)
    assert item["polar_valid"].shape == (h_p, w_p)
    assert item["polar_valid"].dtype == torch.bool
    assert item["polar_valid"].any(), "segmentable ellipse should mark some patches valid"


def test_polar_grid_flip_synced_with_image(ellipse_data, tmp_path, monkeypatch):
    """The random flip applied to the image on the train split must be applied
    IDENTICALLY to the polar grid — otherwise the E9 loss would see density
    predictions and geometry that disagree about where the otolith rotated to."""
    import numpy as np
    import torchvision.transforms as T
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data   # split="train" in this fixture's csv

    # ColorJitter samples its own random factors on every __call__, independent of
    # the flip decision under test — neutralise it so the only difference between
    # the two ds[0] calls below is the flip, not also unrelated jitter noise.
    monkeypatch.setattr(T.ColorJitter, "forward", lambda self, img: img)

    ds = OtolithDataset(_e9_cfg(tmp_path), "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    monkeypatch.setattr(ds, "_decide_flip", lambda: (False, False))
    baseline = ds[0]

    monkeypatch.setattr(ds, "_decide_flip", lambda: (True, False))
    hflipped = ds[0]

    assert torch.allclose(hflipped["image"], torch.flip(baseline["image"], dims=[2]), atol=1e-5)
    expected_polar = np.fliplr(baseline["polar_grid"].numpy())
    assert np.allclose(hflipped["polar_grid"].numpy(), expected_polar, atol=1e-4)
    expected_valid = np.fliplr(baseline["polar_valid"].numpy())
    assert np.array_equal(hflipped["polar_valid"].numpy(), expected_valid)


# ---------------------------------------------------------------------------
# Change B (05.08): polar_theta (angle-windowed/"local" E9 geometry)
# ---------------------------------------------------------------------------

def test_polar_theta_absent_by_default(ellipse_data, tmp_path):
    """Same gate as polar_grid/polar_valid — density_concentricity_weight=0.0
    (default) means no polar geometry of any kind is computed."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _make_cfg()
    cfg.data.image_size = 196
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    assert "polar_theta" not in ds[0]


def test_polar_theta_absent_without_density_head(ellipse_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _e9_cfg(tmp_path)
    cfg.model.use_density_head = False
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    assert "polar_theta" not in ds[0]


def test_polar_theta_present_and_shaped_when_enabled(ellipse_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _e9_cfg(tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    h_p = w_p = cfg.data.image_size // cfg.data.patch_size
    assert item["polar_theta"].shape == (h_p, w_p)
    assert item["polar_theta"].dtype == torch.float32
    assert item["polar_theta"].min() >= -3.1416 and item["polar_theta"].max() <= 3.1416


def test_polar_theta_flip_synced_with_image(ellipse_data, tmp_path, monkeypatch):
    """Unlike polar_grid/polar_valid (plain scalar fields, correctly synced by a
    simple np.fliplr), polar_theta is a signed DIRECTION — a horizontal flip must
    transform its VALUE (wrap(pi - theta)), not just reverse the array. A naive
    fliplr (as used for polar_grid above) would attach each patch a pre-flip
    direction that no longer matches its post-flip geometry; this test would catch
    exactly that class of bug."""
    import numpy as np
    import torchvision.transforms as T
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data

    monkeypatch.setattr(T.ColorJitter, "forward", lambda self, img: img)
    ds = OtolithDataset(_e9_cfg(tmp_path), "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    monkeypatch.setattr(ds, "_decide_flip", lambda: (False, False))
    baseline = ds[0]
    monkeypatch.setattr(ds, "_decide_flip", lambda: (True, False))
    hflipped = ds[0]

    raw_reversed = np.fliplr(baseline["polar_theta"].numpy())
    expected_theta = ((np.pi - raw_reversed + np.pi) % (2.0 * np.pi)) - np.pi
    actual = hflipped["polar_theta"].numpy()
    # Compare on the unit circle (cos/sin) rather than raw radians so the +-pi
    # branch cut doesn't produce a spurious ~2*pi mismatch for angles near it.
    assert np.allclose(np.cos(actual), np.cos(expected_theta), atol=1e-3)
    assert np.allclose(np.sin(actual), np.sin(expected_theta), atol=1e-3)
    # And explicitly NOT the naive (un-transformed) reversal — pins down that the
    # value transform is actually happening, not just the array reversal.
    assert not np.allclose(actual, raw_reversed, atol=1e-2)


# ---------------------------------------------------------------------------
# Change C (05.08): quarter/campaign "-1" age adjustment
# ---------------------------------------------------------------------------

def _quarter_data(tmp_path, with_campaign_col: bool = True):
    """One BITS1q-campaign image (age=4) + one BITS4q-campaign image (age=5).
    Filenames carry the campaign token at the real production position (2nd
    underscore-separated token) so extract_campaign_token's re-parsing fallback
    works even when ``with_campaign_col=False`` (no materialized column)."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    name_q1 = "2022_BITS1q_HER_Loc_Embedded_Sharpest_FishIndex1_Single1_Left.png"
    name_q4 = "2022_BITS4q_HER_Loc_Embedded_Sharpest_FishIndex2_Single1_Left.png"
    Image.new("RGB", (56, 56), color=(10, 100, 200)).save(img_dir / name_q1)
    Image.new("RGB", (56, 56), color=(20, 100, 200)).save(img_dir / name_q4)
    rows = [
        {"image_id": name_q1, "age": 4, "split": "train"},
        {"image_id": name_q4, "age": 5, "split": "train"},
    ]
    if with_campaign_col:
        rows[0]["campaign"] = "BITS1q"
        rows[1]["campaign"] = "BITS4q"
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, img_dir


def test_quarter_age_adjustment_off_by_default_unchanged(tmp_path):
    """quarter_age_adjustment_enabled=False (default) => byte-identical to today —
    regression test for every existing config that never sets this flag."""
    from src.dataset import OtolithDataset, encode_age_ordinal
    csv_path, img_dir = _quarter_data(tmp_path)
    ds = OtolithDataset(_make_cfg(), "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert int(item["age"]) == 4
    assert int(item["age_original"]) == 4
    assert torch.equal(item["age_ordinal"], encode_age_ordinal(4, ds.num_age_classes))


def test_quarter_age_adjustment_applies_for_adjustable_campaign(tmp_path):
    from src.dataset import OtolithDataset, encode_age_ordinal
    csv_path, img_dir = _quarter_data(tmp_path)
    cfg = _make_cfg()
    cfg.data.quarter_age_adjustment_enabled = True
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]   # BITS1q row, recorded age=4
    assert int(item["age"]) == 3
    assert int(item["age_original"]) == 4
    assert torch.equal(item["age_ordinal"], encode_age_ordinal(3, ds.num_age_classes))


def test_quarter_age_adjustment_skips_non_adjustable_campaign(tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = _quarter_data(tmp_path)
    cfg = _make_cfg()
    cfg.data.quarter_age_adjustment_enabled = True
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[1]   # BITS4q row, recorded age=5 — not in the default adjustable set
    assert int(item["age"]) == 5
    assert int(item["age_original"]) == 5


def test_quarter_age_adjustment_floors_at_zero(tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = _quarter_data(tmp_path)
    df = pd.read_csv(csv_path)
    df.loc[0, "age"] = 0
    df.to_csv(csv_path, index=False)
    cfg = _make_cfg()
    cfg.data.quarter_age_adjustment_enabled = True
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert int(item["age"]) == 0, "age must floor at 0, never go negative"
    assert int(item["age_original"]) == 0


def test_age_original_always_equals_recorded(tmp_path):
    """age_original must equal the raw recorded value regardless of the flag —
    it's a diagnostic field, never itself adjusted."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = _quarter_data(tmp_path)
    for enabled in (False, True):
        cfg = _make_cfg()
        cfg.data.quarter_age_adjustment_enabled = enabled
        ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
        assert int(ds[0]["age_original"]) == 4
        assert int(ds[1]["age_original"]) == 5


def test_quarter_age_adjustment_prefers_campaign_column_over_reparsing(tmp_path):
    """A materialized 'campaign' column that CONTRADICTS what image_id parsing
    would give must win — validates the precedence design in _effective_age."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = _quarter_data(tmp_path, with_campaign_col=True)
    df = pd.read_csv(csv_path)
    df.loc[0, "campaign"] = "BITS4q"   # image_id says BITS1q, column says BITS4q
    df.to_csv(csv_path, index=False)
    cfg = _make_cfg()
    cfg.data.quarter_age_adjustment_enabled = True
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert int(item["age"]) == 4, "materialized column (BITS4q, not adjustable) must win over image_id (BITS1q)"


def test_quarter_age_adjustment_falls_back_when_campaign_column_absent(tmp_path):
    """No 'campaign' column at all (e.g. a CSV from before the label pipeline was
    updated) — must still adjust correctly via the image_id re-parsing fallback."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = _quarter_data(tmp_path, with_campaign_col=False)
    cfg = _make_cfg()
    cfg.data.quarter_age_adjustment_enabled = True
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    assert int(ds[0]["age"]) == 3   # BITS1q, adjusted via fallback
    assert int(ds[1]["age"]) == 5   # BITS4q, unchanged


# ---------------------------------------------------------------------------
# DensityFineTuneDataset (22.07 — fine-tune density_head at density_image_size/crop)
# ---------------------------------------------------------------------------

def test_density_finetune_dataset_requires_mask_background(ellipse_data, tmp_path):
    from src.dataset import DensityFineTuneDataset
    csv_path, img_dir = ellipse_data
    cfg = _make_cfg()
    cfg.data.mask_background = False
    with pytest.raises(ValueError, match="mask_background"):
        DensityFineTuneDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))


def test_density_finetune_dataset_uses_density_image_size(ellipse_data, tmp_path):
    """Output image shape follows density_image_size, not cfg.data.image_size."""
    from src.dataset import DensityFineTuneDataset
    csv_path, img_dir = ellipse_data
    cfg = _make_cfg()
    cfg.data.image_size = 56
    cfg.data.mask_background = True
    cfg.data.mask_cache_dir = str(tmp_path / "masks_cache")
    ds = DensityFineTuneDataset(cfg, "train", density_image_size=112, crop_to_otolith=False,
                                labels_csv=str(csv_path), image_dir=str(img_dir))
    assert ds[0]["image"].shape == (3, 112, 112)


def test_density_finetune_dataset_none_size_falls_back_to_data_image_size(ellipse_data, tmp_path):
    from src.dataset import DensityFineTuneDataset
    csv_path, img_dir = ellipse_data
    cfg = _make_cfg()
    cfg.data.image_size = 56
    cfg.data.mask_background = True
    cfg.data.mask_cache_dir = str(tmp_path / "masks_cache")
    ds = DensityFineTuneDataset(cfg, "train", density_image_size=None,
                                labels_csv=str(csv_path), image_dir=str(img_dir))
    assert ds[0]["image"].shape == (3, 56, 56)


def test_density_finetune_dataset_crop_changes_output_vs_uncropped(ellipse_data, tmp_path):
    """crop_to_otolith=True must actually change what's fed to the model (the crop
    removes background the uncropped path would still include) — split='test' to
    avoid train-time random flip/jitter making the two runs differ for unrelated reasons."""
    import pandas as pd
    csv_path, img_dir = ellipse_data
    df = pd.read_csv(csv_path)
    df["split"] = "test"
    df.to_csv(csv_path, index=False)

    from src.dataset import DensityFineTuneDataset
    cfg = _make_cfg()
    cfg.data.mask_background = True
    cfg.data.mask_cache_dir = str(tmp_path / "masks_cache")

    ds_uncropped = DensityFineTuneDataset(cfg, "test", density_image_size=112,
                                          crop_to_otolith=False,
                                          labels_csv=str(csv_path), image_dir=str(img_dir))
    ds_cropped = DensityFineTuneDataset(cfg, "test", density_image_size=112,
                                        crop_to_otolith=True,
                                        labels_csv=str(csv_path), image_dir=str(img_dir))
    assert not torch.allclose(ds_uncropped[0]["image"], ds_cropped[0]["image"])
    assert ds_cropped[0]["image"].shape == (3, 112, 112)


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


# ---------------------------------------------------------------------------
# Semi-weak supervision (26.08, "Opcja A"): zegar_heatmap / has_zegar_target
# ---------------------------------------------------------------------------

@pytest.fixture
def zegar_semi_weak_data(tmp_path):
    """A main (non-ZEGAR) dataset of 2 dummy images, plus a SEPARATE ZEGAR extra
    labels/targets CSV and image dir with 1 annotated sample — mirrors exactly what
    scripts/prepare_zegar_semi_weak_data.py produces, at unit-test scale. The ZEGAR
    image doesn't need to be segmentable (unlike ellipse_data above) since
    _build_zegar_heatmap never depends on segmentation succeeding."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64), color=(i * 40, 100, 200)).save(img_dir / f"img_{i}.png")
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([
        {"image_id": "img_0.png", "age": 5, "split": "train"},
        {"image_id": "img_1.png", "age": 6, "split": "train"},
    ]).to_csv(csv_path, index=False)

    zegar_img_dir = tmp_path / "zegar_processed"
    zegar_img_dir.mkdir()
    # Crop is 140x196 (H x W) so a non-square crop exercises the row/col scale
    # factors independently — a bug that swapped them would still pass on a square
    # crop by coincidence.
    Image.new("RGB", (196, 140), color=(10, 10, 10)).save(zegar_img_dir / "ZEGAR_T01.jpg")
    zegar_labels_csv = tmp_path / "zegar_labels.csv"
    pd.DataFrame([{"image_id": "ZEGAR_T01.jpg", "age": 4, "split": "train"}]).to_csv(
        zegar_labels_csv, index=False)
    zegar_targets_csv = tmp_path / "zegar_targets.csv"
    # Two annotated points near the crop's bottom-right corner (x close to W=196,
    # y close to H=140) — deliberately off-center so a flip visibly relocates them.
    pd.DataFrame([
        {"image_id": "ZEGAR_T01.jpg", "x": 176.0, "y": 126.0},
        {"image_id": "ZEGAR_T01.jpg", "x": 168.0, "y": 112.0},
    ]).to_csv(zegar_targets_csv, index=False)

    return {
        "csv_path": csv_path, "img_dir": img_dir,
        "zegar_labels_csv": zegar_labels_csv, "zegar_img_dir": zegar_img_dir,
        "zegar_targets_csv": zegar_targets_csv,
    }


def _zegar_cfg(data, tmp_path, image_size: int = 196):
    cfg = _make_cfg()
    cfg.data.image_size = image_size   # 196 = 14*14, divisible by patch_size
    cfg.data.mask_background = True
    cfg.data.mask_cache_dir = str(tmp_path / "masks_cache")
    cfg.model.use_density_head = True
    cfg.data.zegar_extra_labels_csv = str(data["zegar_labels_csv"])
    cfg.data.zegar_extra_image_dir = str(data["zegar_img_dir"])
    cfg.data.zegar_targets_csv = str(data["zegar_targets_csv"])
    return cfg


def test_zegar_heatmap_absent_by_default(zegar_semi_weak_data):
    """No zegar_extra_* configured (default None) → zero behaviour change: every
    sample still gets zegar_heatmap/has_zegar_target, but always zero/False."""
    from src.dataset import OtolithDataset
    data = zegar_semi_weak_data
    cfg = _make_cfg()
    ds = OtolithDataset(cfg, "train", labels_csv=str(data["csv_path"]), image_dir=str(data["img_dir"]))
    item = ds[0]
    assert item["has_zegar_target"].item() is False
    assert item["zegar_heatmap"].max().item() == 0.0


def test_zegar_extra_rows_merged_into_train_split(zegar_semi_weak_data, tmp_path):
    """The ZEGAR-extra CSV's rows must appear in the dataset alongside the main
    CSV's rows (concatenated before the split filter, src/dataset.py __init__)."""
    from src.dataset import OtolithDataset
    data = zegar_semi_weak_data
    cfg = _zegar_cfg(data, tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(data["csv_path"]), image_dir=str(data["img_dir"]))
    assert len(ds) == 3   # 2 main + 1 zegar
    assert "ZEGAR_T01.jpg" in set(ds.df["image_id"])
    assert "ZEGAR_T01.jpg" in ds._zegar_image_ids


def test_zegar_heatmap_present_for_annotated_sample(zegar_semi_weak_data, tmp_path):
    from src.dataset import OtolithDataset
    data = zegar_semi_weak_data
    cfg = _zegar_cfg(data, tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(data["csv_path"]), image_dir=str(data["img_dir"]))
    idx = ds.df.index[ds.df["image_id"] == "ZEGAR_T01.jpg"][0]
    item = ds[idx]
    h_p = w_p = cfg.data.image_size // cfg.data.patch_size
    assert item["has_zegar_target"].item() is True
    assert item["zegar_heatmap"].shape == (h_p, w_p)
    assert item["zegar_heatmap"].max().item() > 0.9   # a real peak, not a flat/near-zero map


def test_zegar_heatmap_zero_for_non_zegar_sample_in_mixed_dataset(zegar_semi_weak_data, tmp_path):
    """A normal, non-ZEGAR sample sitting in the SAME dataset/batch as a ZEGAR one
    must still get an all-zero heatmap and has_zegar_target=False — this is exactly
    the per-sample masking the trainer's loss gating relies on."""
    from src.dataset import OtolithDataset
    data = zegar_semi_weak_data
    cfg = _zegar_cfg(data, tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(data["csv_path"]), image_dir=str(data["img_dir"]))
    idx = ds.df.index[ds.df["image_id"] == "img_0.png"][0]
    item = ds[idx]
    assert item["has_zegar_target"].item() is False
    assert item["zegar_heatmap"].max().item() == 0.0


def test_zegar_image_dir_dispatch(zegar_semi_weak_data, tmp_path):
    """ZEGAR-derived rows must load from zegar_extra_image_dir, NOT the main
    image_dir (which doesn't even contain a file with that name)."""
    from src.dataset import OtolithDataset
    data = zegar_semi_weak_data
    cfg = _zegar_cfg(data, tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(data["csv_path"]), image_dir=str(data["img_dir"]))
    assert not (data["img_dir"] / "ZEGAR_T01.jpg").exists()
    idx = ds.df.index[ds.df["image_id"] == "ZEGAR_T01.jpg"][0]
    item = ds[idx]   # must not raise (would FileNotFoundError if dispatch were wrong)
    assert item["image"].shape == (3, cfg.data.image_size, cfg.data.image_size)


def test_zegar_heatmap_flip_synced_with_image(zegar_semi_weak_data, tmp_path, monkeypatch):
    """Same requirement as polar_grid (test_polar_grid_flip_synced_with_image above):
    the shared explicit flip decision must relocate the heatmap's peak in lockstep
    with the image, not leave it pointing at the pre-flip position."""
    import numpy as np
    import torchvision.transforms as T
    from src.dataset import OtolithDataset
    data = zegar_semi_weak_data

    monkeypatch.setattr(T.ColorJitter, "forward", lambda self, img: img)
    cfg = _zegar_cfg(data, tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(data["csv_path"]), image_dir=str(data["img_dir"]))
    idx = ds.df.index[ds.df["image_id"] == "ZEGAR_T01.jpg"][0]

    monkeypatch.setattr(ds, "_decide_flip", lambda: (False, False))
    baseline = ds[idx]
    monkeypatch.setattr(ds, "_decide_flip", lambda: (True, False))
    hflipped = ds[idx]
    monkeypatch.setattr(ds, "_decide_flip", lambda: (False, True))
    vflipped = ds[idx]

    assert torch.allclose(hflipped["image"], torch.flip(baseline["image"], dims=[2]), atol=1e-5)
    expected_h = np.fliplr(baseline["zegar_heatmap"].numpy())
    assert np.allclose(hflipped["zegar_heatmap"].numpy(), expected_h, atol=1e-4)
    expected_v = np.flipud(baseline["zegar_heatmap"].numpy())
    assert np.allclose(vflipped["zegar_heatmap"].numpy(), expected_v, atol=1e-4)
    # And explicitly NOT unchanged — pins down that some real transform happened,
    # not a no-op that would trivially "match" a wrong expectation.
    assert not np.allclose(hflipped["zegar_heatmap"].numpy(), baseline["zegar_heatmap"].numpy())


# ---------------------------------------------------------------------------
# Dendrochronology-strip experiment (02.09) — dual_branch_density
# ---------------------------------------------------------------------------

def _strip_cfg(tmp_path, strip_length_px: int = 140, strip_width_px: int = 42):
    cfg = _make_cfg()
    cfg.data.image_size = 56
    cfg.data.mask_background = True
    cfg.data.mask_cache_dir = str(tmp_path / "masks_cache")
    cfg.data.dual_branch_density = True
    cfg.data.strip_length_px = strip_length_px
    cfg.data.strip_width_px = strip_width_px
    cfg.data.strips_cache_dir = str(tmp_path / "strips_cache")
    cfg.model.use_density_head = True
    return cfg


def test_dual_branch_density_off_leaves_image_untouched(ellipse_data, tmp_path):
    """dual_branch_density=False (default) must be a complete no-op: no image_strip key,
    "image" itself unaffected — pins the plan's "one isolated variable" guarantee."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _make_cfg()
    cfg.data.image_size = 56
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert "image_strip" not in item
    assert item["image"].shape == (3, 56, 56)


def test_dual_branch_density_adds_correctly_shaped_strip(ellipse_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _strip_cfg(tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image"].shape == (3, 56, 56)                 # square (age) branch untouched
    assert item["image_strip"].shape == (3, 42, 140)           # (3, strip_width_px, strip_length_px)


def test_dual_branch_density_gracefully_falls_back_for_unsegmentable_image(dummy_data, tmp_path):
    """dummy_data's images are flat solid colour — no foreground to segment. The strip
    branch must fall back to a plain resize, never crash the dataset (same philosophy
    as test_mask_background_gracefully_skips_unsegmentable_image)."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = dummy_data
    cfg = _strip_cfg(tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image_strip"].shape == (3, 42, 140)


def test_dual_branch_density_strip_cache_reused_on_second_access(tmp_path, monkeypatch):
    """Second access to the same image must hit the on-disk strip cache, not recompute
    the axis/warp (mirrors test_mask_background_cache_reused_on_second_access).

    Builds its own split="test" image (rather than the ellipse_data fixture, which is
    split="train") deliberately — the "train" transform pipeline applies a random
    vertical flip per call, which would make two accesses differ regardless of caching.
    """
    import cv2
    import numpy as np
    from src.dataset import OtolithDataset

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    arr = np.full((200, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(arr, (80, 100), (50, 80), 0, 0, 360, (40, 40, 40), -1)
    name = "fish_ellipse.png"
    Image.fromarray(arr).save(img_dir / name)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([{"image_id": name, "age": 4, "split": "test"}]).to_csv(csv_path, index=False)

    cfg = _strip_cfg(tmp_path)
    ds = OtolithDataset(cfg, "test", labels_csv=str(csv_path), image_dir=str(img_dir))
    first = ds[0]["image_strip"]

    def _boom(*a, **kw):
        raise AssertionError("detect_axis should NOT be called on a strip-cache hit")
    monkeypatch.setattr("src.dataset.detect_axis", _boom)

    second = ds[0]["image_strip"]
    assert torch.allclose(first, second)


def test_build_transforms_strip_drops_horizontal_flip_only():
    from src.dataset import build_transforms
    tf = build_transforms(140, "train", include_flips=True, strip=True)
    op_types = [type(op).__name__ for op in tf.transforms]
    assert "RandomHorizontalFlip" not in op_types
    assert "RandomVerticalFlip" in op_types
    assert "Resize" not in op_types


def test_dual_branch_density_age_ordinal_unaffected(ellipse_data, tmp_path):
    """The strip branch must never touch age encoding — same age, same ordinal vector,
    with or without dual_branch_density."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data

    cfg_plain = _make_cfg()
    cfg_plain.data.image_size = 56
    ds_plain = OtolithDataset(cfg_plain, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    cfg_strip = _strip_cfg(tmp_path)
    ds_strip = OtolithDataset(cfg_strip, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    assert torch.equal(ds_plain[0]["age_ordinal"], ds_strip[0]["age_ordinal"])
    assert ds_plain[0]["age"].item() == ds_strip[0]["age"].item()


# ---------------------------------------------------------------------------
# Background-activation penalty (08.09) — strip_mask_background_loss
# ---------------------------------------------------------------------------

def _masked_strip_cfg(tmp_path, strip_length_px: int = 140, strip_width_px: int = 42):
    cfg = _strip_cfg(tmp_path, strip_length_px, strip_width_px)
    cfg.data.strip_mask_background_loss = True
    return cfg


def test_strip_mask_background_loss_off_leaves_no_valid_mask_key(ellipse_data, tmp_path):
    """False (default) must be a complete no-op: no strip_valid_mask key at all."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _strip_cfg(tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert "strip_valid_mask" not in item


def test_strip_mask_background_loss_adds_correctly_shaped_valid_mask(ellipse_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _masked_strip_cfg(tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image_strip"].shape == (3, 42, 140)
    assert item["strip_valid_mask"].shape == (3, 10)   # (Wp/patch_size, Lp/patch_size)
    assert item["strip_valid_mask"].min() >= 0.0
    assert item["strip_valid_mask"].max() <= 1.0


def test_strip_mask_background_loss_gracefully_falls_back_for_unsegmentable_image(dummy_data, tmp_path):
    """dummy_data's images are flat solid colour — no foreground to segment. The
    valid mask must fall back to all-ones (unknown geometry -> don't penalise
    anything), never crash the dataset."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = dummy_data
    cfg = _masked_strip_cfg(tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["strip_valid_mask"].shape == (3, 10)
    assert torch.allclose(item["strip_valid_mask"], torch.ones(3, 10))


def test_strip_mask_background_loss_vflip_synced_with_image(ellipse_data, tmp_path, monkeypatch):
    """The same explicit vertical-flip decision must relocate the valid mask in
    lockstep with the strip image — same requirement already enforced for the E9
    polar grid (test_polar_grid_flip_synced_with_image) and the zegar heatmap."""
    import numpy as np
    import torchvision.transforms as T
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data

    monkeypatch.setattr(T.ColorJitter, "forward", lambda self, img: img)
    cfg = _masked_strip_cfg(tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    monkeypatch.setattr(ds, "_decide_strip_vflip", lambda: False)
    baseline = ds[0]
    monkeypatch.setattr(ds, "_decide_strip_vflip", lambda: True)
    flipped = ds[0]

    assert torch.allclose(flipped["image_strip"], torch.flip(baseline["image_strip"], dims=[1]), atol=1e-5)
    expected = np.flipud(baseline["strip_valid_mask"].numpy())
    assert np.allclose(flipped["strip_valid_mask"].numpy(), expected, atol=1e-6)
    # Not a no-op — the ellipse fixture's mask isn't symmetric top/bottom by
    # construction coincidence check would be flaky, so just assert the flip ran.
    assert not torch.allclose(flipped["image_strip"], baseline["image_strip"])


def test_strip_mask_background_loss_cache_reused_on_second_access(tmp_path, monkeypatch):
    """Second access must hit the on-disk valid-mask cache, not recompute the
    axis/warp (mirrors test_dual_branch_density_strip_cache_reused_on_second_access)."""
    import cv2
    import numpy as np
    from src.dataset import OtolithDataset

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    arr = np.full((200, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(arr, (80, 100), (50, 80), 0, 0, 360, (40, 40, 40), -1)
    name = "fish_ellipse.png"
    Image.fromarray(arr).save(img_dir / name)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([{"image_id": name, "age": 4, "split": "test"}]).to_csv(csv_path, index=False)

    cfg = _masked_strip_cfg(tmp_path)
    ds = OtolithDataset(cfg, "test", labels_csv=str(csv_path), image_dir=str(img_dir))
    first_img = ds[0]["image_strip"]
    first_valid = ds[0]["strip_valid_mask"]

    def _boom(*a, **kw):
        raise AssertionError("detect_axis should NOT be called on a cache hit")
    monkeypatch.setattr("src.dataset.detect_axis", _boom)

    second_img = ds[0]["image_strip"]
    second_valid = ds[0]["strip_valid_mask"]
    assert torch.allclose(first_img, second_img)
    assert torch.allclose(first_valid, second_valid)


# ---------------------------------------------------------------------------
# Multi-wycinek experiment (03.09) — multi_wycinek_k > 1
# ---------------------------------------------------------------------------

def _multi_strip_cfg(tmp_path, k: int, strip_length_px: int = 140, strip_width_px: int = 42):
    cfg = _strip_cfg(tmp_path, strip_length_px=strip_length_px, strip_width_px=strip_width_px)
    cfg.data.multi_wycinek_k = k
    cfg.data.multi_wycinek_min_angle_sep_deg = 8.0
    return cfg


def test_multi_wycinek_k1_matches_single_strip_shape(ellipse_data, tmp_path):
    """multi_wycinek_k=1 (default) must keep the single-wycinek (3, Wp, Lp) shape —
    zero behaviour change relative to Track A."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _multi_strip_cfg(tmp_path, k=1)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image_strip"].shape == (3, 42, 140)


def test_multi_wycinek_k3_stacks_k_candidates(ellipse_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _multi_strip_cfg(tmp_path, k=3)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image_strip"].shape == (3, 3, 42, 140)   # (K, 3, Wp, Lp)


def test_multi_wycinek_gracefully_falls_back_for_unsegmentable_image(dummy_data, tmp_path):
    """dummy_data's images are flat solid colour — no candidates found. Must fall back
    to K copies of the plain-resize fallback, never crash."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = dummy_data
    cfg = _multi_strip_cfg(tmp_path, k=3)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image_strip"].shape == (3, 3, 42, 140)


def test_multi_wycinek_cache_reused_on_second_access(tmp_path, monkeypatch):
    """Second access must hit the per-candidate strip cache, not recompute
    detect_axis_candidates (mirrors the single-wycinek cache-reuse test)."""
    import cv2
    import numpy as np
    from src.dataset import OtolithDataset

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    arr = np.full((200, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(arr, (80, 100), (50, 80), 0, 0, 360, (40, 40, 40), -1)
    name = "fish_ellipse.png"
    Image.fromarray(arr).save(img_dir / name)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([{"image_id": name, "age": 4, "split": "test"}]).to_csv(csv_path, index=False)

    cfg = _multi_strip_cfg(tmp_path, k=3)
    ds = OtolithDataset(cfg, "test", labels_csv=str(csv_path), image_dir=str(img_dir))
    first = ds[0]["image_strip"]

    def _boom(*a, **kw):
        raise AssertionError("detect_axis_candidates should NOT be called on a cache hit")
    monkeypatch.setattr("src.dataset.detect_axis_candidates", _boom)

    second = ds[0]["image_strip"]
    assert torch.allclose(first, second)


def test_multi_wycinek_pads_by_repeating_best_when_fewer_candidates_found(ellipse_data, tmp_path, monkeypatch):
    """When detect_axis_candidates returns fewer than K candidates, the stack must
    still have shape (K, 3, Wp, Lp), padded by repeating the BEST (index 0) candidate.

    ellipse_data's row is split="train", whose transform applies an independent
    random vertical flip per candidate tensor — that would make two copies of the
    same underlying strip compare unequal regardless of whether padding itself is
    correct, so the dataset's strip_transform is swapped for a deterministic
    (split="test") one after construction, purely for this equality check.
    """
    from src.dataset import OtolithDataset, build_transforms
    import src.dataset as dataset_module

    csv_path, img_dir = ellipse_data
    cfg = _multi_strip_cfg(tmp_path, k=4)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    ds.strip_transform = build_transforms(cfg.data.strip_length_px, split="test", strip=True)

    real = dataset_module.detect_axis_candidates
    def _only_two(*a, **kw):
        kw["k"] = 2
        return real(*a, **kw)
    monkeypatch.setattr(dataset_module, "detect_axis_candidates", _only_two)

    item = ds[0]
    assert item["image_strip"].shape == (4, 3, 42, 140)
    # candidates 2 and 3 (0-indexed) should be padding copies of candidate 0
    assert torch.equal(item["image_strip"][2], item["image_strip"][0])


# ---------------------------------------------------------------------------
# Polar-wedge experiment (09.09) — dual_branch_wedge
# ---------------------------------------------------------------------------

def _wedge_cfg(tmp_path, n_angle_patches: int = 3, n_radius_patches: int = 10):
    cfg = _make_cfg()
    cfg.data.image_size = 56
    cfg.data.mask_background = True
    cfg.data.mask_cache_dir = str(tmp_path / "masks_cache")
    cfg.data.dual_branch_wedge = True
    cfg.data.wedge_n_angle_patches = n_angle_patches
    cfg.data.wedge_n_radius_patches = n_radius_patches
    cfg.data.wedge_delta_theta_deg = 90.0
    cfg.data.wedge_cache_dir = str(tmp_path / "wedges_cache")
    cfg.model.use_density_head = True
    return cfg


def test_dual_branch_wedge_off_leaves_image_untouched(ellipse_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _make_cfg()
    cfg.data.image_size = 56
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert "image_wedge" not in item
    assert item["image"].shape == (3, 56, 56)


def test_dual_branch_wedge_adds_correctly_shaped_tensors(ellipse_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _wedge_cfg(tmp_path, n_angle_patches=3, n_radius_patches=10)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image"].shape == (3, 56, 56)                 # square (age) branch untouched
    assert item["image_wedge"].shape == (3, 10 * 14, 3 * 14)  # (3, canvas_h, canvas_w)
    assert item["wedge_polar_t"].shape == (10, 3)
    assert item["wedge_polar_theta"].shape == (10, 3)
    assert item["wedge_polar_valid"].shape == (10, 3)
    # t increases monotonically with radius (row), same invariant as the standalone
    # wedge_extraction unit tests.
    t = item["wedge_polar_t"].numpy()
    assert (t[1:, 0] > t[:-1, 0]).all()


def test_dual_branch_wedge_gracefully_falls_back_for_unsegmentable_image(dummy_data, tmp_path):
    """dummy_data's images are flat solid colour — no foreground to segment. The wedge
    branch must fall back to a plain resize + all-valid mask, never crash."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = dummy_data
    cfg = _wedge_cfg(tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image_wedge"].shape == (3, 10 * 14, 3 * 14)
    assert torch.all(item["wedge_polar_valid"] == 1.0)


def test_dual_branch_wedge_cache_reused_on_second_access(tmp_path, monkeypatch):
    """Second access must hit the on-disk wedge cache, not recompute the axis/warp
    (mirrors test_dual_branch_density_strip_cache_reused_on_second_access). Uses
    split="test" (deterministic transform, no random flip) for the same reason that
    test does."""
    import cv2
    import numpy as np
    from src.dataset import OtolithDataset

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    arr = np.full((200, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(arr, (80, 100), (50, 80), 0, 0, 360, (40, 40, 40), -1)
    name = "fish_ellipse.png"
    Image.fromarray(arr).save(img_dir / name)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([{"image_id": name, "age": 4, "split": "test"}]).to_csv(csv_path, index=False)

    cfg = _wedge_cfg(tmp_path)
    ds = OtolithDataset(cfg, "test", labels_csv=str(csv_path), image_dir=str(img_dir))
    first = ds[0]["image_wedge"]

    def _boom(*a, **kw):
        raise AssertionError("detect_axis should NOT be called on a wedge-cache hit")
    monkeypatch.setattr("src.dataset.detect_axis", _boom)

    second = ds[0]["image_wedge"]
    assert torch.allclose(first, second)


def test_build_transforms_wedge_drops_vertical_flip_only():
    from src.dataset import build_transforms
    tf = build_transforms(42, "train", include_flips=True, wedge=True)
    op_types = [type(op).__name__ for op in tf.transforms]
    assert "RandomVerticalFlip" not in op_types
    assert "RandomHorizontalFlip" in op_types
    assert "Resize" not in op_types


def test_dual_branch_wedge_hflip_synced_with_image_and_polar(ellipse_data, tmp_path, monkeypatch):
    """The shared explicit flip decision must mirror the wedge image AND relocate its
    polar_theta/valid columns in lockstep — not leave them describing the pre-flip
    layout (same requirement as the strip's vflip-sync test, mirrored to columns
    instead of rows since the wedge's flip axis is angle, not radius)."""
    import torchvision.transforms as T
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    monkeypatch.setattr(T.ColorJitter, "forward", lambda self, img: img)
    cfg = _wedge_cfg(tmp_path, n_angle_patches=3, n_radius_patches=10)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    monkeypatch.setattr(ds, "_decide_wedge_hflip", lambda: False)
    baseline = ds[0]
    monkeypatch.setattr(ds, "_decide_wedge_hflip", lambda: True)
    flipped = ds[0]

    assert torch.allclose(flipped["image_wedge"], torch.flip(baseline["image_wedge"], dims=[2]), atol=1e-5)
    assert torch.allclose(flipped["wedge_polar_theta"], torch.flip(baseline["wedge_polar_theta"], dims=[1]))
    assert torch.allclose(flipped["wedge_polar_valid"], torch.flip(baseline["wedge_polar_valid"], dims=[1]))
    # explicitly NOT unchanged — pins that a real transform happened
    assert not torch.allclose(flipped["image_wedge"], baseline["image_wedge"])


def test_dual_branch_wedge_age_ordinal_unaffected(ellipse_data, tmp_path):
    """The wedge branch must never touch age encoding — same age, same ordinal
    vector, with or without dual_branch_wedge."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data

    cfg_plain = _make_cfg()
    cfg_plain.data.image_size = 56
    ds_plain = OtolithDataset(cfg_plain, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    cfg_wedge = _wedge_cfg(tmp_path)
    ds_wedge = OtolithDataset(cfg_wedge, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    assert torch.equal(ds_plain[0]["age_ordinal"], ds_wedge[0]["age_ordinal"])
    assert ds_plain[0]["age"].item() == ds_wedge[0]["age"].item()


# ---------------------------------------------------------------------------
# Angular-resolution bands (09.09 follow-up) — wedge_band_edges_t etc.
# ---------------------------------------------------------------------------

def _wedge_bands_cfg(tmp_path, band_edges_t=None, n_angle=None, n_radius=None):
    cfg = _wedge_cfg(tmp_path)
    cfg.data.wedge_band_edges_t = band_edges_t or [0.0, 0.5, 1.0]
    cfg.data.wedge_band_n_angle_patches = n_angle or [3, 5]
    cfg.data.wedge_band_n_radius_patches = n_radius or [4, 6]
    cfg.data.wedge_bands_cache_dir = str(tmp_path / "wedge_bands_cache")
    return cfg


def test_wedge_bands_off_single_wedge_untouched(ellipse_data, tmp_path):
    """dual_branch_wedge=True alone (no band fields) must keep behaving exactly like before —
    single image_wedge key, no band keys."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _wedge_cfg(tmp_path, n_angle_patches=3, n_radius_patches=10)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert "image_wedge" in item
    assert "image_wedge_band0" not in item


def test_wedge_bands_replace_single_wedge_when_enabled(ellipse_data, tmp_path):
    """Bands REPLACE the single-canvas wedge, not add to it — no image_wedge key when band
    fields are set, only the per-band keys, at each band's own correct shape."""
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    cfg = _wedge_bands_cfg(tmp_path, band_edges_t=[0.0, 0.5, 1.0], n_angle=[3, 5], n_radius=[4, 6])
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert "image_wedge" not in item
    assert item["image"].shape == (3, 56, 56)  # square (age) branch untouched

    assert item["image_wedge_band0"].shape == (3, 4 * 14, 3 * 14)
    assert item["wedge_band0_polar_t"].shape == (4, 3)
    assert item["wedge_band0_polar_theta"].shape == (4, 3)
    assert item["wedge_band0_polar_valid"].shape == (4, 3)

    assert item["image_wedge_band1"].shape == (3, 6 * 14, 5 * 14)
    assert item["wedge_band1_polar_t"].shape == (6, 5)

    # radius increases row-by-row within each band, and band 0 stays entirely below band 1's t
    t0 = item["wedge_band0_polar_t"].numpy()
    t1 = item["wedge_band1_polar_t"].numpy()
    assert (t0[1:, 0] > t0[:-1, 0]).all()
    assert (t1[1:, 0] > t1[:-1, 0]).all()
    assert t0.max() < t1.min()


def test_wedge_bands_gracefully_falls_back_for_unsegmentable_image(dummy_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = dummy_data
    cfg = _wedge_bands_cfg(tmp_path)
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))
    item = ds[0]
    assert item["image_wedge_band0"].shape == (3, 4 * 14, 3 * 14)
    assert item["image_wedge_band1"].shape == (3, 6 * 14, 5 * 14)
    assert torch.all(item["wedge_band0_polar_valid"] == 1.0)
    assert torch.all(item["wedge_band1_polar_valid"] == 1.0)


def test_wedge_bands_cache_reused_on_second_access(tmp_path, monkeypatch):
    """Mirrors test_dual_branch_wedge_cache_reused_on_second_access — a second access must hit
    every band's on-disk cache, never re-run detect_axis."""
    import cv2
    import numpy as np
    from src.dataset import OtolithDataset

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    arr = np.full((200, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(arr, (80, 100), (50, 80), 0, 0, 360, (40, 40, 40), -1)
    name = "fish_ellipse.png"
    Image.fromarray(arr).save(img_dir / name)
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame([{"image_id": name, "age": 4, "split": "test"}]).to_csv(csv_path, index=False)

    cfg = _wedge_bands_cfg(tmp_path)
    ds = OtolithDataset(cfg, "test", labels_csv=str(csv_path), image_dir=str(img_dir))
    first0 = ds[0]["image_wedge_band0"]
    first1 = ds[0]["image_wedge_band1"]

    def _boom(*a, **kw):
        raise AssertionError("detect_axis should NOT be called on a wedge-bands cache hit")
    monkeypatch.setattr("src.dataset.detect_axis", _boom)

    second = ds[0]
    assert torch.allclose(first0, second["image_wedge_band0"])
    assert torch.allclose(first1, second["image_wedge_band1"])


def test_wedge_bands_hflip_synced_across_all_bands(ellipse_data, tmp_path, monkeypatch):
    """The ONE shared flip decision must mirror EVERY band's image and relocate its polar_theta/
    valid columns in lockstep — not just one band, and not independently per band."""
    import torchvision.transforms as T
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data
    monkeypatch.setattr(T.ColorJitter, "forward", lambda self, img: img)
    cfg = _wedge_bands_cfg(tmp_path, band_edges_t=[0.0, 0.5, 1.0], n_angle=[3, 5], n_radius=[4, 6])
    ds = OtolithDataset(cfg, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    monkeypatch.setattr(ds, "_decide_wedge_hflip", lambda: False)
    baseline = ds[0]
    monkeypatch.setattr(ds, "_decide_wedge_hflip", lambda: True)
    flipped = ds[0]

    any_image_changed = False
    for i in (0, 1):
        img_key, t_key, theta_key, valid_key = (
            f"image_wedge_band{i}", f"wedge_band{i}_polar_t",
            f"wedge_band{i}_polar_theta", f"wedge_band{i}_polar_valid")
        assert torch.allclose(flipped[img_key], torch.flip(baseline[img_key], dims=[2]), atol=1e-5)
        assert torch.allclose(flipped[theta_key], torch.flip(baseline[theta_key], dims=[1]))
        assert torch.allclose(flipped[valid_key], torch.flip(baseline[valid_key], dims=[1]))
        if not torch.allclose(flipped[img_key], baseline[img_key]):
            any_image_changed = True
    # explicitly NOT unchanged overall — pins that a real transform happened (band 0's tiny 3-col
    # canvas can coincidentally look flip-invariant on this synthetic symmetric ellipse; theta/
    # valid column-reversal above is the real per-band correctness check regardless)
    assert any_image_changed


def test_wedge_bands_age_ordinal_unaffected(ellipse_data, tmp_path):
    from src.dataset import OtolithDataset
    csv_path, img_dir = ellipse_data

    cfg_plain = _make_cfg()
    cfg_plain.data.image_size = 56
    ds_plain = OtolithDataset(cfg_plain, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    cfg_bands = _wedge_bands_cfg(tmp_path)
    ds_bands = OtolithDataset(cfg_bands, "train", labels_csv=str(csv_path), image_dir=str(img_dir))

    assert torch.equal(ds_plain[0]["age_ordinal"], ds_bands[0]["age_ordinal"])
    assert ds_plain[0]["age"].item() == ds_bands[0]["age"].item()
