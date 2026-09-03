"""Krok 0 planu paska dendrologicznego (plans and summaries/02.09_wycinki_plan.md) — mierzy
REALNY odstęp między sąsiednimi przyrostami w danych ZEGAR i wyznacza z niego minimalne
``strip_length_px``, zamiast zgadywać tę liczbę przez analogię do pełnego kwadratu (518px).

Dlaczego to w ogóle liczymy: `patch_size` DINOv2 jest ustalone (14px, wagi pretrenowane), więc
jedynym sposobem na to, żeby pojedynczy patch nie obejmował dwóch sąsiednich przyrostów naraz,
jest dobranie takiej długości paska, przy której nawet najciaśniejsza para pierścieni w
rzeczywistych danych ląduje na osobnych patchach — wymuszając, że lokalizację robi INTERAKCJA
między sąsiednimi patchami (samo-uwaga backbone'u), a nie tekstura jednego patcha z osobna.

Metoda: `t` (pozycja pierścienia znormalizowana do [0,1] wzdłuż osi jądro->brzeg,
`otolith_axis.point_to_axis_t` via `expert_annotation_eval.point_to_axis_t`) jest SKALO-
NIEZMIENNA — dokładnie ta sama wielkość, którą zachowuje transformacja paska (`strip_transform_
matrix` liniowo mapuje [0, length_px] na [0, strip_length_px]). Więc minimalne Δt (różnica t
między sąsiednimi przyrostami tego samego czytelnika) po wszystkich 42 obrazach i obu
czytelnikach (KK, SS) daje wprost: strip_length_px >= margines * patch_size / min(Δt).

Najgorszy przypadek to zwykle starsze ryby z wieloma gęsto upakowanymi zewnętrznymi przyrostami —
a to akurat ryby z NAJDŁUŻSZYM length_px, czyli tam gdzie skalowanie do stałego strip_length_px
ściska najmocniej, dokładnie tam gdzie odstępy i tak są już najciaśniejsze. Ta analiza na
realnych danych łapie ten przypadek wprost, zamiast go przeoczyć przy zgadywaniu.

Nie dotyka treningu ani `src/` — czysto diagnostyczny, tylko do odczytu (poza zapisem wyników do
`outputs/`). Reużywa bez zmian `load_expert_annotations`, `point_to_axis_t`,
`resolve_and_crop_target_otolith` z `expert_annotation_eval.py` — te same 42 obrazy, ta sama
metodologia GT (ostatni przyrost per próbka/czytelnik już wykluczony), więc wynik jest od razu
spójny z każdą inną liczbą ZEGAR w projekcie.

Usage:
    python scripts/diagnostics/analyze_zegar_ring_spacing.py
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

from scripts.diagnostics.expert_annotation_eval import (
    IMAGE_DIR,
    REFERENCE_CONFIG,
    load_expert_annotations,
    point_to_axis_t,
    resolve_and_crop_target_otolith,
)
from scripts.run_pipeline import load_merged_config
from src.otolith_axis import detect_axis

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "02.09_zegar_ring_spacing"

# Ile patchy odstępu wymagamy nawet dla najciaśniejszej realnej pary (nie 1 — na styk to wciąż
# ryzyko, że oba przyrosty trafiają do sąsiadujących patchy bez wyraźnej przerwy między nimi).
MARGIN_PATCHES = 2


def _round_up_to_patch(value: float, patch_size: int) -> int:
    return int(np.ceil(value / patch_size) * patch_size)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Analiza odstępów między przyrostami — ZEGAR (Krok 0, pasek dendrologiczny)")
    print("=" * 70)

    print("\n[1/4] Wczytywanie adnotacji...")
    ann = load_expert_annotations()
    samples = sorted(ann["Sample"].unique())
    print(f"  {len(samples)} próbek, {len(ann)} zaadnotowanych przyrostów (KK+SS łącznie, "
          f"core/edge/ostatni-przyrost już wykluczone)")

    cfg = load_merged_config(REFERENCE_CONFIG, None)
    seg_params = cfg.segmentation.as_params()
    patch_size = cfg.data.patch_size

    print("\n[2/4] Przycinanie do pojedynczego otolitu, wyznaczanie osi, rzutowanie punktów...")
    rows: list[dict] = []
    n_seg_failed = 0
    for sample in samples:
        image_id = f"{sample}.jpg"
        sub = ann[ann.Sample == sample]
        mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
        raw_rgb = np.array(PILImage.open(IMAGE_DIR / image_id).convert("RGB"), dtype=np.uint8)
        cropped, x0, y0, _used_second = resolve_and_crop_target_otolith(raw_rgb, mean_xy, seg_params)
        if cropped is None:
            print(f"  [pomiń] {image_id}: segmentacja nieudana")
            n_seg_failed += 1
            continue

        axis_info = detect_axis(
            cropped, seg_params=seg_params,
            nucleus_method=cfg.segmentation.nucleus_method,
            axis_method=cfg.segmentation.axis_method,
        )
        if axis_info is None:
            print(f"  [pomiń] {image_id}: oś nie policzona")
            n_seg_failed += 1
            continue

        for reader in ("KK", "SS"):
            pts = sub[sub.annotator == reader]
            if len(pts) < 2:
                continue
            xs = (pts.x - x0).to_numpy()
            ys = (pts.y - y0).to_numpy()
            t_vals = sorted(point_to_axis_t(x, y, axis_info) for x, y in zip(xs, ys))
            for i in range(len(t_vals) - 1):
                dt = t_vals[i + 1] - t_vals[i]
                rows.append({
                    "Sample": sample,
                    "annotator": reader,
                    "t_lo": t_vals[i],
                    "t_hi": t_vals[i + 1],
                    "delta_t": dt,
                    "length_px": axis_info["length_px"],
                    "delta_px": dt * axis_info["length_px"],
                })

    n_ok = len(samples) - n_seg_failed
    print(f"  {n_ok}/{len(samples)} obrazów z policzoną osią ({n_seg_failed} pominiętych)")

    if not rows:
        sys.exit("Brak pomierzonych par przyrostów — sprawdź dane wejściowe.")

    gaps = pd.DataFrame(rows)
    gaps.to_csv(OUTPUT_DIR / "ring_gaps.csv", index=False)

    print(f"\n[3/4] Rozklad dt (odstep miedzy sasiednimi przyrostami, ulamek dlugosci osi), "
          f"n={len(gaps)} par:")
    dt = gaps["delta_t"]
    stats = {
        "n_pairs": int(len(dt)),
        "min": float(dt.min()),
        "p1": float(dt.quantile(0.01)),
        "p5": float(dt.quantile(0.05)),
        "median": float(dt.median()),
        "mean": float(dt.mean()),
    }
    for k, v in stats.items():
        print(f"   {k}: {v}" if k == "n_pairs" else f"   {k}: {v:.5f}")

    worst = gaps.nsmallest(10, "delta_t")[
        ["Sample", "annotator", "delta_t", "delta_px", "length_px"]
    ]
    print("\n  10 najciaśniejszych par (najgorszy przypadek — do wizualnej weryfikacji):")
    print(worst.to_string(index=False))

    print(f"\n[4/4] Wyznaczanie strip_length_px (margines={MARGIN_PATCHES} patchy, "
          f"patch_size={patch_size}px):")
    min_dt, p1_dt = stats["min"], stats["p1"]
    recommended_min = MARGIN_PATCHES * patch_size / min_dt
    recommended_p1 = MARGIN_PATCHES * patch_size / p1_dt
    rec_min_rounded = _round_up_to_patch(recommended_min, patch_size)
    rec_p1_rounded = _round_up_to_patch(recommended_p1, patch_size)
    print(f"   Na podstawie MINIMUM (najciaśniejsza para w całym zbiorze): "
          f"strip_length_px >= {recommended_min:.0f}px -> zaokrąglone {rec_min_rounded}px "
          f"({rec_min_rounded // patch_size} patchy)")
    print(f"   Na podstawie p1 (odporniejsze na jeden odstający punkt, REKOMENDOWANE): "
          f"strip_length_px >= {recommended_p1:.0f}px -> zaokrąglone {rec_p1_rounded}px "
          f"({rec_p1_rounded // patch_size} patchy)")
    if rec_p1_rounded <= 518:
        print(f"   -> mieści się w domyślnych 518px (37 patchy) z planu — 518px wystarcza.")
    else:
        print(f"   -> WIĘKSZE niż domyślne 518px z planu — użyć {rec_p1_rounded}px jako "
              f"strip_length_px w configs/config_strip_a.yaml.")

    result = {
        "stats": stats,
        "patch_size": patch_size,
        "margin_patches": MARGIN_PATCHES,
        "recommended_strip_length_px_min": rec_min_rounded,
        "recommended_strip_length_px_p1": rec_p1_rounded,
        "n_images_ok": n_ok,
        "n_images_total": len(samples),
        "n_seg_failed": n_seg_failed,
    }
    (OUTPUT_DIR / "ring_spacing_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    print(f"\nZapisano: {OUTPUT_DIR / 'ring_gaps.csv'}, {OUTPUT_DIR / 'ring_spacing_summary.json'}")


if __name__ == "__main__":
    main()
