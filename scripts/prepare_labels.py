"""Prepare data/labels.csv from the biological Excel file + image directory scan.

Usage:
    python scripts/prepare_labels.py ^
        --excel data/analysisWithOtolithPhoto.xlsx ^
        --image_dir Z:/Photo/Otolithes/HER/Processed ^
        --out data/labels.csv

Optional flags (see --help for all):
    --include_raw_pair          also include Raw Pair images (default: excluded)
    --unknown_age exclude       what to do with Wiek=-9:
                                  exclude   = drop completely (default)
                                  unlabeled = keep with split=unlabeled
                                  test      = put in test split (predict only)
    --train 0.7 --val 0.15 --test 0.15   split ratios (labeled data only)
    --seed 42

Split is assigned PER FISH (not per image) to prevent data leakage between
Left/Right otolith pairs from the same individual.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Otolith types considered usable for training
GOOD_TYPES   = {"Left", "Right"}
PAIR_TYPE    = "Raw Pair"
BAD_TYPES    = {"LowQuality", "Wrong", "Broken"}
UNKNOWN_AGE  = -9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scan_image_dir(image_dir: Path) -> set[str]:
    """Return set of filenames (stem+ext) present in image_dir (non-recursive)."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    found = {p.name for p in image_dir.iterdir() if p.suffix.lower() in exts}
    print(f"  Zdjecia na dysku: {len(found)}")
    return found


def extract_fish_id(image_id: str) -> str:
    """Derive a fish-level identifier from the image filename.

    Strips the _Single{N}_{Left|Right} or _{Left|Right} suffix so that all
    otolith images from the same individual share the same fish_id.

    Example:
        "2022_BIAS_HER_ZatokaGdanska_Embedded_Sharpest_FishIndex2_Single1_Left.jpg"
        -> "2022_BIAS_HER_ZatokaGdanska_Embedded_Sharpest_FishIndex2"
    """
    stem = Path(image_id).stem
    # Primary: remove _Single{N}_{Left|Right}
    clean = re.sub(r"_Single\d+_(Left|Right)$", "", stem, flags=re.IGNORECASE)
    if clean == stem:
        # Fallback: remove just _{Left|Right}
        clean = re.sub(r"_(Left|Right)$", "", stem, flags=re.IGNORECASE)
    return clean


def assign_split_by_fish(
    df: pd.DataFrame,
    fish_col: str,
    age_col: str,
    train: float,
    val: float,
    seed: int,
) -> pd.Series:
    """Assign train/val/test splits at the fish level (not image level).

    All images belonging to the same fish always land in the same split.

    **Age-stratified**: fish are sorted by median age, then within each block of
    similar-age fish a jittered within-block quantile decides the split. Every age
    band is therefore represented in all three splits. (The previous version sliced
    the sorted list globally, which put the OLDEST fish entirely in test — so the
    model trained on young fish only and failed catastrophically on the older test
    set. See plans and summaries/09.07_after_training_TO.DO.md.)

    Prints a per-split age summary and warns loudly if the splits are still
    age-imbalanced. Returns a Series indexed like df with split labels.
    """
    rng = np.random.default_rng(seed)

    groups = (
        df.groupby(fish_col)[age_col]
        .median()
        .reset_index()
        .rename(columns={age_col: "med_age"})
        .sort_values("med_age")
        .reset_index(drop=True)
    )
    n = len(groups)

    # Within each age block, assign a jittered within-block quantile u∈[0,1),
    # then split by u. u is ~uniform per block and independent of age → the split
    # is age-stratified AND converges to the train/val/test ratios over the blocks.
    block = 10
    u = np.empty(n, dtype=float)
    for start in range(0, n, block):
        idx = np.arange(start, min(start + block, n))
        b = len(idx)
        u[idx] = (rng.permutation(b) + rng.random(b)) / b
    groups["split"] = np.where(u < train, "train",
                               np.where(u < train + val, "val", "test"))

    # Defensive sanity check: the split must be age-balanced. Warn if not.
    stats = groups.groupby("split")["med_age"].agg(["mean", "median", "min", "max", "count"])
    print("  Rozkład wieku per split (mediana wieku ryby):")
    for sp in ("train", "val", "test"):
        if sp in stats.index:
            r = stats.loc[sp]
            print(f"    {sp:5s}: n_ryb={int(r['count']):5d}  mean={r['mean']:.2f}  "
                  f"median={r['median']:.1f}  min={r['min']:.0f}  max={r['max']:.0f}")
    if {"train", "test"}.issubset(stats.index):
        gap = abs(float(stats.loc["train", "mean"]) - float(stats.loc["test", "mean"]))
        if gap > 0.5:
            print(f"  UWAGA: sredni wiek train vs test rozni sie o {gap:.2f} roku "
                  f"- podzial moze byc NIEZBALANSOWANY wiekowo!")

    split_map = groups.set_index(fish_col)["split"]
    return df[fish_col].map(split_map)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_labels(
    excel_path: Path,
    image_dir: Path,
    out_path: Path,
    include_raw_pair: bool = False,
    unknown_age: str = "exclude",   # exclude | unlabeled | test
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    seed: int = 42,
) -> pd.DataFrame:

    print("\n[1] Wczytywanie Excel ...")
    df = pd.read_excel(excel_path)
    print(f"  Zaladowano {len(df)} wierszy, {df.shape[1]} kolumn")

    # ------------------------------------------------------------------ #
    # [2] Filtrowanie typów otolitu
    # ------------------------------------------------------------------ #
    print("\n[2] Filtrowanie typow otolitu ...")
    allowed = set(GOOD_TYPES)
    if include_raw_pair:
        allowed.add(PAIR_TYPE)
    df_filtered = df[df["Typ otolitu"].isin(allowed)].copy()
    removed = len(df) - len(df_filtered)
    print(f"  Zachowano typy: {sorted(allowed)}")
    print(f"  Usunieto {removed} wierszy ({BAD_TYPES | ({PAIR_TYPE} if not include_raw_pair else set())})")
    print(f"  Pozostalo: {len(df_filtered)} wierszy")

    # ------------------------------------------------------------------ #
    # [3] Mapowanie kolumn + wyciagnięcie fish_id
    # ------------------------------------------------------------------ #
    print("\n[3] Mapowanie kolumn ...")
    records = pd.DataFrame({
        "image_id":    df_filtered["FilePath"].str.strip(),
        "age":         df_filtered["Wiek"],
        "length_mm":   df_filtered["Klasa_dlugosci_mm"],
        "weight_g":    df_filtered["Masa_g"],
        "sex":         df_filtered["Plec"],
        "population":  df_filtered["Populacja"],
        "subdivision": df_filtered["Subdivision"],
        "otolith_type":df_filtered["Typ otolitu"],
        "year":        df_filtered["Rok"],
    })

    # Fish-level identifier: all images of the same fish share this value
    records.insert(1, "fish_id", records["image_id"].apply(extract_fish_id))
    n_fish = records["fish_id"].nunique()
    print(f"  Unikalnych ryb (fish_id): {n_fish}")
    print(f"  Srednio zdjec/ryba: {len(records) / n_fish:.1f}")

    # ------------------------------------------------------------------ #
    # [4] Obsługa Wiek = -9
    # ------------------------------------------------------------------ #
    print(f"\n[4] Wiek=-9 -> strategia: '{unknown_age}' ...")
    mask_unknown = records["age"] == UNKNOWN_AGE
    n_unknown = mask_unknown.sum()
    print(f"  Rekordow z Wiek=-9: {n_unknown}")

    if unknown_age == "exclude":
        records = records[~mask_unknown].copy()
        print(f"  Usunieto. Pozostalo: {len(records)}")
    elif unknown_age in ("unlabeled", "test"):
        pass
    else:
        print(f"  UWAGA: nieznana opcja '{unknown_age}', uzywam 'exclude'")
        records = records[~mask_unknown].copy()

    # ------------------------------------------------------------------ #
    # [5] Skanowanie dysku i JOIN
    # ------------------------------------------------------------------ #
    if image_dir.exists():
        print(f"\n[5] Skanowanie {image_dir} ...")
        available = scan_image_dir(image_dir)
        before = len(records)
        records = records[records["image_id"].isin(available)].copy()
        print(f"  Dopasowano {len(records)} / {before} rekordow (brak na dysku: {before - len(records)})")
    else:
        print(f"\n[5] UWAGA: katalog {image_dir} niedostepny -- pomijam sprawdzanie plikow.")
        print("  Wszystkie rekordy z Excel zostana zachowane.")

    if len(records) == 0:
        print("\nBLAD: po filtrowaniu nie pozostaly zadne rekordy!")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # [6] Przypisanie splitu PER RYBA
    # ------------------------------------------------------------------ #
    print("\n[6] Przypisywanie splitu (per ryba, stratified by age) ...")

    records["split"] = ""

    if unknown_age == "unlabeled":
        mask_u = records["age"] == UNKNOWN_AGE
        records.loc[mask_u, "split"] = "unlabeled"
        labeled = records[records["split"] == ""].copy()
    elif unknown_age == "test":
        mask_u = records["age"] == UNKNOWN_AGE
        records.loc[mask_u, "split"] = "test"
        labeled = records[records["split"] == ""].copy()
    else:
        labeled = records.copy()

    if len(labeled) > 0:
        split_series = assign_split_by_fish(
            labeled, "fish_id", "age", train_frac, val_frac, seed
        )
        records.loc[labeled.index, "split"] = split_series.values

    # Print split counts
    split_counts = records["split"].value_counts().to_dict()
    for k in sorted(split_counts):
        n_img  = split_counts[k]
        n_f    = records[records["split"] == k]["fish_id"].nunique()
        print(f"  {k:12s}: {n_img:5d} zdjec  {n_f:5d} ryb")

    # Data-leakage check
    fish_splits = records.groupby("fish_id")["split"].nunique()
    leaking = int((fish_splits > 1).sum())
    if leaking == 0:
        print(f"  Ryby w >1 zbiorze: 0  (BRAK WYCIEKU)")
    else:
        print(f"  UWAGA: {leaking} ryb ma zdjecia w >1 zbiorze!")

    # ------------------------------------------------------------------ #
    # [7] Statystyki wiekowe
    # ------------------------------------------------------------------ #
    labeled_only = records[records["age"] >= 0]
    print(f"\n[7] Statystyki wiekowe (labeled, n={len(labeled_only)}) ...")
    print(f"  Min wiek: {labeled_only['age'].min()}")
    print(f"  Max wiek: {labeled_only['age'].max()}")
    print(f"  Sredni:   {labeled_only['age'].mean():.2f}")
    print(f"  Mediana:  {labeled_only['age'].median():.1f}")
    print("  Rozklad:")
    vc = labeled_only["age"].value_counts().sort_index()
    for age, cnt in vc.items():
        bar = "#" * min(cnt // 30, 40)
        print(f"    wiek {age:2d}: {cnt:4d}  {bar}")

    # ------------------------------------------------------------------ #
    # [8] Zapis
    # ------------------------------------------------------------------ #
    print(f"\n[8] Zapis -> {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Gotowe. Wierszy: {len(records)}  Kolumn: {len(records.columns)}")
    print(f"  Kolumny: {records.columns.tolist()}")

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Przygotuj data/labels.csv")
    p.add_argument("--excel",    default="data/analysisWithOtolithPhoto.xlsx",
                   help="Sciezka do pliku Excel")
    p.add_argument("--image_dir", default="Z:/Photo/Otolithes/HER/Processed",
                   help="Katalog ze zdjeciami (lokalny lub sieciowy)")
    p.add_argument("--out",      default="data/labels.csv",
                   help="Sciezka wyjsciowa dla labels.csv")
    p.add_argument("--include_raw_pair", action="store_true",
                   help="Uwzgledniej zdjecia 'Raw Pair' (domyslnie: nie)")
    p.add_argument("--unknown_age", default="exclude",
                   choices=["exclude", "unlabeled", "test"],
                   help="Co zrobic z Wiek=-9 (domyslnie: exclude)")
    p.add_argument("--train", type=float, default=0.70, dest="train_frac")
    p.add_argument("--val",   type=float, default=0.15, dest="val_frac")
    p.add_argument("--seed",  type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    excel_path = PROJECT_ROOT / args.excel if not Path(args.excel).is_absolute() \
                 else Path(args.excel)
    image_dir  = Path(args.image_dir)
    out_path   = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() \
                 else Path(args.out)

    print("=" * 60)
    print("OtolithDino -- przygotowanie labels.csv")
    print("=" * 60)
    print(f"Excel:      {excel_path}")
    print(f"Zdjecia:    {image_dir}")
    print(f"Wyjscie:    {out_path}")
    print(f"Raw Pair:   {'tak' if args.include_raw_pair else 'nie'}")
    print(f"Wiek=-9:    {args.unknown_age}")
    print(f"Podzialy:   train={args.train_frac} val={args.val_frac} "
          f"test={round(1-args.train_frac-args.val_frac, 2)}")
    print(f"Split:      per ryba (fish-level, stratified by age)")

    build_labels(
        excel_path       = excel_path,
        image_dir        = image_dir,
        out_path         = out_path,
        include_raw_pair = args.include_raw_pair,
        unknown_age      = args.unknown_age,
        train_frac       = args.train_frac,
        val_frac         = args.val_frac,
        seed             = args.seed,
    )

    print("\nOK labels.csv gotowy.")
    print(f"\nNastepny krok:")
    print(f"  python scripts/generate_report.py --labels {out_path} --out outputs/report_data.html")


if __name__ == "__main__":
    main()
