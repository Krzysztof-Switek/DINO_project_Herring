"""Faza 8, krok 1 planu pasm kątowych (`plans and summaries/09.09_wycinek_pasma_katowe_plan.md`):
measures the REAL distribution of otolith size (`length_px`, nucleus-to-edge axis length) across a
broad random sample of the actual training population — not just the 42 ZEGAR-annotated images.

Why this needs a broader sample than ZEGAR: `length_px` is a purely geometric quantity from
segmentation (`detect_axis`) — it does NOT require expert-marked ring positions, unlike the
ring-spacing and angular-deviation measurements that genuinely needed ZEGAR's annotations. The
angular-band design target is "zero loss even for the single largest real otolith in the dataset"
(user's explicit choice, 09.09) — 42 images is too small a sample to safely bound a population
maximum; the real training population (`Z:/Photo/Otolithes/HER/Processed`, Embedded only per this
project's standing scope) has ~10,900 candidate images to sample from instead.

Unlike the ZEGAR pipeline, images here need no `resolve_and_crop_target_otolith` step — real
Processed/ images are already single-otolith crops (confirmed: `OtolithDataset._load_raw_rgb` loads
files directly with no multi-otolith disambiguation) — so this script runs `get_or_compute_mask` +
`detect_axis` directly on each sampled file, same segmentation params as everywhere else.

Usage:
    python scripts/diagnostics/analyze_real_otolith_size_distribution.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image as PILImage

from scripts.diagnostics.expert_annotation_eval import REFERENCE_CONFIG
from scripts.run_pipeline import load_merged_config
from src.otolith_axis import detect_axis, get_or_compute_mask

IMAGE_DIR = Path("Z:/Photo/Otolithes/HER/Processed")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "09.09_masked_wizualizacje" / "wedge_angular_resolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_CACHE_DIR = OUTPUT_DIR / "broad_sample_masks_cache"

N_SAMPLE = 400
SEED = 42


def main() -> None:
    print("=" * 78)
    print("Rozkład realnego rozmiaru otolitu (length_px) — szeroka próba, nie tylko ZEGAR")
    print("=" * 78)

    cfg = load_merged_config(REFERENCE_CONFIG, None)
    seg_params = cfg.segmentation.as_params()

    all_files = [f for f in IMAGE_DIR.iterdir()
                 if "_Embedded_" in f.name and f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"\n[1/3] {len(all_files)} realnych zdjęć Embedded w {IMAGE_DIR}")

    rng = np.random.default_rng(SEED)
    sample_files = list(rng.choice(all_files, size=min(N_SAMPLE, len(all_files)), replace=False))
    print(f"  Losowa próba: {len(sample_files)} zdjęć (seed={SEED})")

    print(f"\n[2/3] Segmentacja + wyznaczanie osi (ta sama metodologia co wszędzie w projekcie)...")
    rows = []
    n_failed = 0
    for i, path in enumerate(sample_files, 1):
        raw_rgb = np.array(PILImage.open(path).convert("RGB"), dtype=np.uint8)
        mask_cache_path = MASK_CACHE_DIR / f"{path.stem}_mask.png"
        mask = get_or_compute_mask(raw_rgb, mask_cache_path, seg_params=seg_params)
        if mask is None:
            n_failed += 1
            continue
        axis_info = detect_axis(raw_rgb, seg_params=seg_params,
                                 nucleus_method=cfg.segmentation.nucleus_method,
                                 axis_method=cfg.segmentation.axis_method, mask=mask)
        if axis_info is None:
            n_failed += 1
            continue
        rows.append({"file": path.name, "length_px": axis_info["length_px"]})
        if i % 50 == 0:
            print(f"    {i}/{len(sample_files)}...")

    print(f"\n  {len(rows)}/{len(sample_files)} zdjęć z policzoną osią ({n_failed} pominiętych)")
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "real_otolith_size_sample.csv", index=False)

    print("\n[3/3] Statystyki length_px (px):")
    desc = df["length_px"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    print(desc.to_string())
    print(f"\n  Prawdziwe MAKSIMUM w tej próbie: {df['length_px'].max():.1f}px "
          f"({df.loc[df['length_px'].idxmax(), 'file']})")
    print(f"  Dla porównania, maksimum z 42 ZEGAR (Z37): 905.3px")

    summary = {
        "n_sampled": len(sample_files), "n_scored": len(rows), "n_failed": n_failed,
        "seed": SEED,
        "length_px_stats": {k: float(v) for k, v in desc.items()},
        "length_px_max": float(df["length_px"].max()),
        "length_px_max_file": str(df.loc[df["length_px"].idxmax(), "file"]),
    }
    (OUTPUT_DIR / "real_otolith_size_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nZapisano: {OUTPUT_DIR / 'real_otolith_size_sample.csv'}, "
          f"{OUTPUT_DIR / 'real_otolith_size_summary.json'}")


if __name__ == "__main__":
    main()