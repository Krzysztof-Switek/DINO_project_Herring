"""Tests for src/scan_labels.py — parse_filename and build_combined_labels."""
import pandas as pd
import pytest

from src.scan_labels import parse_filename, build_combined_labels

# ---------------------------------------------------------------------------
# Helpers — synthetic filenames
# ---------------------------------------------------------------------------

EMB_FISH1 = "2022_BIAS_HER_ZatokaGdanska_Embedded_Sharpest_FishIndex1_Single1_Left.jpg"
EMB_FISH2 = "2022_BIAS_HER_ZatokaGdanska_Embedded_Sharpest_FishIndex2_Single1_Right.jpg"
EMB_FISH3 = "2023_BITS1q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex3_Single1_Left.jpg"

NOTEMB_FISH1 = "2022_BIAS_HER_ZatokaGdanska_NotEmbedded_withoutPostproc_FishIndex1_Single1_Left.jpg"
NOTEMB_FISH2 = "2022_BIAS_HER_ZatokaGdanska_NotEmbedded_withoutPostproc_FishIndex2_Single1_Right.jpg"
# orphan — no matching Excel entry
NOTEMB_ORPHAN = "2024_NEW_HER_NewLocation_NotEmbedded_withoutPostproc_FishIndex99_Single1_Left.jpg"


def _make_meta_lookup(keys_ages: dict[str, int]) -> pd.DataFrame:
    """Build minimal metadata DataFrame indexed by neutral_fish_key."""
    return pd.DataFrame(
        {
            "age":          list(keys_ages.values()),
            "length_mm":    [15.0] * len(keys_ages),
            "weight_g":     [10.0] * len(keys_ages),
            "sex":          ["M"] * len(keys_ages),
            "population":   ["P1"] * len(keys_ages),
            "subdivision":  ["S1"] * len(keys_ages),
            "otolith_type": ["Left"] * len(keys_ages),
            "year":         [2022] * len(keys_ages),
        },
        index=pd.Index(list(keys_ages.keys()), name="neutral_fish_key"),
    )


# ---------------------------------------------------------------------------
# Tests: parse_filename
# ---------------------------------------------------------------------------

def test_parse_embedded():
    result = parse_filename(EMB_FISH1)
    assert result is not None
    assert result["image_id"] == EMB_FISH1
    assert result["preprocessing_type"] == "Embedded"
    assert result["neutral_fish_key"] == "2022_BIAS_HER_ZatokaGdanska_FishIndex1"
    assert result["side"] == "Left"


def test_parse_not_embedded():
    result = parse_filename(NOTEMB_FISH1)
    assert result is not None
    assert result["preprocessing_type"] == "NotEmbedded"
    assert result["neutral_fish_key"] == "2022_BIAS_HER_ZatokaGdanska_FishIndex1"
    assert result["side"] == "Left"


def test_parse_short_filename():
    assert parse_filename("2022_BIAS_HER_ZatokaGdanska_Embedded.jpg") is None


def test_parse_unknown_type():
    unknown = "2022_BIAS_HER_ZatokaGdanska_RawPair_Sharpest_FishIndex1_Single1_Left.jpg"
    assert parse_filename(unknown) is None


def test_neutral_key_same_fish():
    emb_key = parse_filename(EMB_FISH1)["neutral_fish_key"]
    notemb_key = parse_filename(NOTEMB_FISH1)["neutral_fish_key"]
    assert emb_key == notemb_key


# ---------------------------------------------------------------------------
# Tests: build_combined_labels
# ---------------------------------------------------------------------------

def _neutral_key(fname):
    return parse_filename(fname)["neutral_fish_key"]


def test_build_combined_no_split_leak():
    filenames = [EMB_FISH1, EMB_FISH2, EMB_FISH3, NOTEMB_FISH1, NOTEMB_FISH2]
    meta = _make_meta_lookup({
        _neutral_key(EMB_FISH1): 3,
        _neutral_key(EMB_FISH2): 5,
        _neutral_key(EMB_FISH3): 7,
    })
    df = build_combined_labels(
        image_dir=None, excel_path=None,
        _image_filenames=filenames, _excel_df=meta,
    )
    with_split = df.dropna(subset=["split"])
    leak = with_split.groupby("neutral_fish_key")["split"].nunique()
    assert (leak > 1).sum() == 0, "Niektóre ryby mają >1 split (przeciek)!"


def test_build_combined_shared_split():
    filenames = [EMB_FISH1, EMB_FISH2, EMB_FISH3, NOTEMB_FISH1, NOTEMB_FISH2]
    meta = _make_meta_lookup({
        _neutral_key(EMB_FISH1): 3,
        _neutral_key(EMB_FISH2): 5,
        _neutral_key(EMB_FISH3): 7,
    })
    df = build_combined_labels(
        image_dir=None, excel_path=None,
        _image_filenames=filenames, _excel_df=meta,
    )
    # For each neutral_fish_key that has both Embedded and NotEmbedded rows,
    # both must have the same split.
    with_split = df.dropna(subset=["split"])
    pivot = (
        with_split.groupby(["neutral_fish_key", "preprocessing_type"])["split"]
        .first()
        .unstack()
    )
    both = pivot.dropna()  # rows that have both Embedded and NotEmbedded
    assert len(both) > 0, "Brak ryb z obu typami — test nieprawidłowy"
    mismatches = (both["Embedded"] != both["NotEmbedded"]).sum()
    assert mismatches == 0, f"{mismatches} ryb ma różne splity dla Embedded vs NotEmbedded!"


def test_build_combined_orphan_flag():
    filenames = [EMB_FISH1, NOTEMB_ORPHAN]
    meta = _make_meta_lookup({_neutral_key(EMB_FISH1): 3})
    df = build_combined_labels(
        image_dir=None, excel_path=None,
        _image_filenames=filenames, _excel_df=meta,
    )
    orphan_rows = df[df["image_id"] == NOTEMB_ORPHAN]
    assert len(orphan_rows) == 1
    assert orphan_rows.iloc[0]["orphan"] is True or orphan_rows.iloc[0]["orphan"] == True


def test_assign_split_is_age_balanced():
    """train/val/test must have similar age distributions and each span the full
    age range — regression for the age-sorted-split bug (test = oldest fish only)."""
    import numpy as np
    import pandas as pd
    from scripts.prepare_labels import assign_split_by_fish

    rng = np.random.default_rng(0)
    n_fish = 600
    ages = rng.integers(0, 17, n_fish)          # ages 0..16
    rows = []
    for i, a in enumerate(ages):
        rows += [{"fish": f"fish_{i}", "age": int(a)}] * 2   # 2 images/fish
    df = pd.DataFrame(rows)

    df["split"] = assign_split_by_fish(
        df, fish_col="fish", age_col="age", train=0.7, val=0.15, seed=42).values

    means = df.groupby("split")["age"].mean()
    assert means.max() - means.min() < 1.0                  # balanced, no age skew

    test_ages = df[df["split"] == "test"]["age"]
    assert test_ages.min() <= 2 and test_ages.max() >= 14   # test spans full range

    fish_frac = df.groupby("fish")["split"].first().value_counts(normalize=True)
    assert 0.6 < fish_frac["train"] < 0.8
    assert 0.08 < fish_frac["val"] < 0.22
    assert 0.08 < fish_frac["test"] < 0.22
