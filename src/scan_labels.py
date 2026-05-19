"""Scan image directory and build combined labels for Embedded and NotEmbedded images.

Extends prepare_labels.py to handle both Embedded and NotEmbedded otolith preparations.
Assigns splits at the fish (neutral_fish_key) level so both variants of the same fish
land in the same split — enabling fair cross-evaluation.

Usage:
    python src/scan_labels.py \\
        --image-dir "Z:/Photo/Otolithes/HER/Processed" \\
        --excel data/analysisWithOtolithPhoto.xlsx \\
        --output data/labels_combined.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_labels import scan_image_dir, assign_split_by_fish, GOOD_TYPES

UNKNOWN_AGE = -9

_META_COLS = [
    "age", "length_mm", "weight_g", "sex",
    "population", "subdivision", "otolith_type", "year",
]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def parse_filename(name: str) -> dict | None:
    """Parse otolith image filename into structured components.

    Expected format (9+ underscore-separated tokens):
        {YEAR}_{CAMPAIGN}_{SPECIES}_{LOCATION}_{TYPE}_{POSTPROC}_FishIndex{N}_Single{N}_{SIDE}.ext

    Returns dict with keys: image_id, preprocessing_type, neutral_fish_key, side.
    Returns None if name does not match (too short or unknown type token).
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 9:
        return None

    type_token = parts[4].lower()
    if type_token == "embedded":
        preprocessing_type = "Embedded"
    elif type_token == "notembedded":
        preprocessing_type = "NotEmbedded"
    else:
        return None

    neutral_fish_key = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[6]}"
    side_token = parts[8].lower()
    if side_token == "wrong":
        return None                     # explicitly rejected image — skip
    side = parts[8] if side_token in {"left", "right"} else None

    return {
        "image_id": name,
        "preprocessing_type": preprocessing_type,
        "neutral_fish_key": neutral_fish_key,
        "side": side,
    }


def load_excel_metadata(excel_path: Path) -> pd.DataFrame:
    """Load Excel, filter GOOD_TYPES, return DataFrame indexed by neutral_fish_key.

    Columns: age, length_mm, weight_g, sex, population, subdivision, otolith_type, year.
    Deduplicated — one row per unique neutral_fish_key (keeps first occurrence).
    """
    df = pd.read_excel(excel_path)
    df_filtered = df[df["Typ otolitu"].isin(GOOD_TYPES)].copy()

    records = pd.DataFrame({
        "image_id":     df_filtered["FilePath"].str.strip(),
        "age":          df_filtered["Wiek"],
        "length_mm":    df_filtered["Klasa_dlugosci_mm"],
        "weight_g":     df_filtered["Masa_g"],
        "sex":          df_filtered["Plec"],
        "population":   df_filtered["Populacja"],
        "subdivision":  df_filtered["Subdivision"],
        "otolith_type": df_filtered["Typ otolitu"],
        "year":         df_filtered["Rok"],
    })

    parsed_keys = records["image_id"].apply(
        lambda n: parse_filename(n)["neutral_fish_key"] if parse_filename(n) else None
    )
    records["neutral_fish_key"] = parsed_keys
    records = records.dropna(subset=["neutral_fish_key"])
    return (
        records.drop_duplicates(subset=["neutral_fish_key"])
        .set_index("neutral_fish_key")[_META_COLS]
    )


def build_combined_labels(
    image_dir: Path,
    excel_path: Path,
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 42,
    _image_filenames: list[str] | None = None,
    _excel_df: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Scan image directory, match to Excel metadata, assign splits.

    Parameters
    ----------
    image_dir, excel_path : used in production mode
    _image_filenames : inject list of filenames directly (bypasses dir scan; for tests)
    _excel_df : inject pre-built lookup DataFrame indexed by neutral_fish_key (for tests)

    Returns
    -------
    DataFrame with columns:
        image_id, neutral_fish_key, preprocessing_type,
        age, length_mm, weight_g, sex, population, subdivision, otolith_type, year,
        split, orphan
    """
    # --- Scan / inject images ---
    if _image_filenames is not None:
        filenames = list(_image_filenames)
    else:
        filenames = sorted(scan_image_dir(image_dir))

    # --- Parse filenames ---
    rows: list[dict] = []
    type4_tokens: set[str] = set()
    type5_tokens: set[str] = set()

    for name in filenames:
        parsed = parse_filename(name)
        if parsed is None:
            continue
        rows.append(parsed)
        parts = Path(name).stem.split("_")
        type4_tokens.add(parts[4])
        if len(parts) > 5:
            type5_tokens.add(parts[5])

    print(f"  Unikalne tokeny [4] (typ preparacji): {sorted(type4_tokens)}")
    print(f"  Unikalne tokeny [5] (postproc):       {sorted(type5_tokens)}")

    if not rows:
        raise ValueError("Brak plików pasujących do wzorca nazwy.")

    img_df = pd.DataFrame(rows)
    print(f"  Sparsowane pliki: {len(img_df)}")
    print(f"  Embedded:    {(img_df.preprocessing_type == 'Embedded').sum()}")
    print(f"  NotEmbedded: {(img_df.preprocessing_type == 'NotEmbedded').sum()}")

    # --- Load / inject Excel metadata ---
    if _excel_df is not None:
        meta_lookup = _excel_df
    else:
        print("\n  Wczytywanie metadanych z Excel ...")
        meta_lookup = load_excel_metadata(excel_path)

    # --- Match and merge ---
    img_df = img_df.join(meta_lookup, on="neutral_fish_key", how="left")
    img_df["orphan"] = img_df["age"].isna()

    n_orphan = int(img_df["orphan"].sum())
    print(f"  Sieroty (brak metadanych w Excel): {n_orphan}")

    # --- Assign split on labeled, non-unknown-age rows ---
    labeled = img_df[~img_df["orphan"]].copy()
    labeled["age"] = labeled["age"].astype(int)
    labeled = labeled[labeled["age"] != UNKNOWN_AGE].copy()

    img_df["split"] = None
    if not labeled.empty:
        split_series = assign_split_by_fish(
            labeled, "neutral_fish_key", "age", train, val, seed
        )
        img_df.loc[labeled.index, "split"] = split_series.values

    # --- Age distribution report ---
    if not labeled.empty:
        print(f"\n  Rozkład wiekowy (labeled, n={len(labeled)}):")
        for ptype in ["Embedded", "NotEmbedded"]:
            subset = labeled[labeled.preprocessing_type == ptype]
            if not subset.empty:
                print(
                    f"    {ptype}: wiek {int(subset.age.min())}–{int(subset.age.max())}, "
                    f"mediana={subset.age.median():.1f}, n={len(subset)}"
                )

    # --- Leak check ---
    with_split = img_df.dropna(subset=["split"])
    if not with_split.empty:
        leak = with_split.groupby("neutral_fish_key")["split"].nunique()
        n_leaking = int((leak > 1).sum())
        status = f"UWAGA: {n_leaking} ryb ma >1 split!" if n_leaking > 0 else "0 (brak wycieku)"
        print(f"  Ryby w >1 zbiorze: {status}")

    # --- Ensure all metadata columns exist ---
    for col in _META_COLS:
        if col not in img_df.columns:
            img_df[col] = None

    return img_df[[
        "image_id", "neutral_fish_key", "preprocessing_type",
        "age", "length_mm", "weight_g", "sex", "population",
        "subdivision", "otolith_type", "year", "split", "orphan",
    ]].copy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Skan katalogu + budowanie labels_combined.csv (Embedded + NotEmbedded)"
    )
    p.add_argument(
        "--image-dir", required=True,
        help="Katalog ze zdjęciami (np. Z:/Photo/Otolithes/HER/Processed)"
    )
    p.add_argument(
        "--excel", required=True,
        help="Ścieżka do pliku Excel (analysisWithOtolithPhoto.xlsx)"
    )
    p.add_argument(
        "--output", default="data/labels_combined.csv",
        help="Ścieżka wyjściowa dla labels_combined.csv"
    )
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val",   type=float, default=0.15)
    p.add_argument("--seed",  type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    image_dir = Path(args.image_dir)
    excel_path = Path(args.excel)
    output_path = Path(args.output)

    print("=" * 60)
    print("scan_labels.py — Embedded + NotEmbedded")
    print("=" * 60)
    print(f"Image dir: {image_dir}")
    print(f"Excel:     {excel_path}")
    print(f"Output:    {output_path}")

    combined = build_combined_labels(
        image_dir=image_dir,
        excel_path=excel_path,
        train=args.train,
        val=args.val,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8")

    emb_path = output_path.parent / "labels_embedded.csv"
    notemb_path = output_path.parent / "labels_not_embedded.csv"
    combined[combined.preprocessing_type == "Embedded"].to_csv(emb_path, index=False)
    combined[combined.preprocessing_type == "NotEmbedded"].to_csv(notemb_path, index=False)

    print(f"\n  Zapisano {len(combined)} wierszy → {output_path}")
    print(f"  Embedded    → {emb_path}")
    print(f"  NotEmbedded → {notemb_path}")
    print("\nOK")
