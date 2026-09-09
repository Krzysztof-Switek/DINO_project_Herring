"""09.09 follow-up to the wycinek-katowy plan (plans and summaries/09.09_wycinek_katowy_plan.md,
Faza 0 addendum) — checks a specific claim the user raised mid-implementation: ring spacing is
NOT uniform along the radius, so a LINEAR row->t mapping in ``wedge_extraction.py`` allocates the
same resolution near the nucleus (where consecutive increments barely exist) as near the edge
(where they pack together tightly). That claim is checked here against the SAME real measurement
already produced for the strip's own resolution, `outputs/02.09_zegar_ring_spacing/ring_gaps.csv`
(335 real consecutive-increment gaps, 42 ZEGAR images x 2 readers, KK/SS) — not new data, since
`t` (axis-relative radius) is the same scale-invariant quantity the wedge and the strip both use.

Method: this is `analyze_zegar_ring_spacing.py`'s own MARGIN_PATCHES-based rule (require the
tightest real gap to still span >=2 patches, so localisation is forced to come from INTER-patch
attention, not one patch's own texture), made LOCAL instead of global. The old strip/wedge
resolution (95 radius patches) used ONE global percentile (p1 across all 335 gaps) applied
uniformly to the whole [0, 1] radius range. That protects the single tightest gap anywhere in the
dataset, but forces the SAME density everywhere else too -- including near the nucleus, where
zero of the 335 measured gaps even fall (see the printed "t_mid < 0.5" check below).

This script instead bins gaps by their radial position (t_mid), takes the same robust p1 THRESHOLD
per bin, enforces the (already-expected, Pearson r=-0.53) monotonic non-increasing trend via a
running cumulative-min left-to-right (radius increasing => required gap tolerance only tightens),
and integrates the resulting local row-density into a monotonic warp: t = f(row / (canvas_h - 1)).
The warp is emitted as (t, row_frac) control points meant to be pasted into
``src/wedge_extraction.py``'s ``_RADIAL_WARP_T`` / ``_RADIAL_WARP_ROW_FRAC`` constants -- this
script does not touch src/, it only re-derives and prints/saves the numbers that module hardcodes,
so the derivation stays reproducible and auditable independently of the production code.

Usage:
    python scripts/diagnostics/analyze_wedge_radial_resolution_profile.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RING_GAPS_CSV = PROJECT_ROOT / "outputs" / "02.09_zegar_ring_spacing" / "ring_gaps.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "09.09_masked_wizualizacje"

PATCH_SIZE = 14          # src.config default data.patch_size — DINOv2 patch, fixed
MARGIN_PATCHES = 2       # same choice as analyze_zegar_ring_spacing.py, same justification
# Zone edges chosen so every zone except the first (which the data shows is genuinely empty)
# has a two-digit sample count -- see the printed per-zone counts.
ZONE_EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Profil promieniowej rozdzielczości wycinka — realne odstępy przyrostów ZEGAR")
    print("=" * 78)

    if not RING_GAPS_CSV.exists():
        sys.exit(f"Brak {RING_GAPS_CSV} — uruchom najpierw analyze_zegar_ring_spacing.py")
    df = pd.read_csv(RING_GAPS_CSV)
    df["t_mid"] = (df["t_lo"] + df["t_hi"]) / 2
    print(f"\n[1/4] Wczytano {len(df)} par przyrostów (ta sama pula co strip, "
          f"outputs/02.09_zegar_ring_spacing/ring_gaps.csv)")

    below_half = df[df.t_mid < 0.5]
    print(f"\n[2/4] Par z t_mid < 0.5 (bliżej jądra niż połowa promienia): {len(below_half)}/{len(df)}")
    if len(below_half):
        print(below_half[["Sample", "annotator", "t_mid", "delta_t"]].to_string(index=False))
    print("  -> potwierdza hipotezę: w pierwszej połowie promienia realnie prawie nie ma par "
          "przyrostów do rozróżnienia.")

    labels = list(range(len(ZONE_EDGES) - 1))
    df["zone"] = pd.cut(df["t_mid"], bins=ZONE_EDGES, labels=labels, include_lowest=True)
    counts = df.groupby("zone", observed=True)["delta_t"].count().reindex(labels, fill_value=0)
    p1_raw = df.groupby("zone", observed=True)["delta_t"].quantile(0.01).reindex(labels)

    print(f"\n[3/4] p1(delta_t) na strefę (próg odporny na pojedynczy odstający punkt, "
          f"ta sama metoda co strip):")
    for i in labels:
        lo, hi = ZONE_EDGES[i], ZONE_EDGES[i + 1]
        n = int(counts[i])
        val = p1_raw[i]
        print(f"   t in [{lo:.2f}, {hi:.2f}): n={n:3d}  p1={val if pd.notna(val) else float('nan'):.5f}"
              if pd.notna(val) else f"   t in [{lo:.2f}, {hi:.2f}): n={n:3d}  p1=brak danych")

    # Strefa 0 (t<0.5) ma za mało par (n=3, brak danych w praktyce) -> nie zgadujemy liczby,
    # przenosimy próg z najbliższej strefy z realnymi danymi (bfill), potem wymuszamy
    # monotoniczność (rosnący promień -> tolerancja tylko się zawęża, zgodnie z r=-0.53).
    g = p1_raw.bfill().cummin()
    print("\n   Po uzupełnieniu pustej strefy najbliższą realną wartością i wymuszeniu "
          "monotoniczności (cummin):")
    print("   g(t) =", np.round(g.values, 5).tolist())

    rho = MARGIN_PATCHES * PATCH_SIZE / g.values           # required local rows per unit t
    widths = np.diff(ZONE_EDGES)
    row_span = rho * widths
    total_px_unrounded = float(row_span.sum())
    canvas_h = int(np.ceil(total_px_unrounded / PATCH_SIZE) * PATCH_SIZE)
    n_radius_patches = canvas_h // PATCH_SIZE

    cum_row = np.concatenate([[0.0], np.cumsum(row_span)])
    row_frac_bp = (cum_row / cum_row[-1]).tolist()
    t_bp = list(ZONE_EDGES)

    print(f"\n[4/4] Całkowita wysokość kanwy: {total_px_unrounded:.1f}px -> zaokrąglone "
          f"{canvas_h}px = {n_radius_patches} patchy")
    print(f"   (dla porównania: stary JEDNOLITY schemat, oparty o GLOBALNY p1 z pełnego zbioru, "
          f"dawał 95 patchy — jednolicie wysoką gęstość WSZĘDZIE, żeby ochronić jedną "
          f"najciaśniejszą parę w całym zbiorze)")
    print(f"\n   t breakpoints:        {[round(v, 4) for v in t_bp]}")
    print(f"   row_frac breakpoints: {[round(v, 6) for v in row_frac_bp]}")
    print("\n   -> wklej powyższe dwie listy jako _RADIAL_WARP_T / _RADIAL_WARP_ROW_FRAC "
          "w src/wedge_extraction.py")

    # ---- wizualizacja: stary (liniowy) vs nowy (nierównomierny) warp, plus rozkład realnych luk ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    row_frac_fine = np.linspace(0, 1, 400)
    t_new = np.interp(row_frac_fine, row_frac_bp, t_bp)
    ax.plot(row_frac_fine, row_frac_fine, "--", color="#999999", label="stary schemat (liniowy)")
    ax.plot(row_frac_fine, t_new, color="#1b6f6f", lw=2.2, label="nowy schemat (nierównomierny)")
    ax.scatter(row_frac_bp, t_bp, color="#1b6f6f", zorder=5, s=22)
    ax.set_xlabel("pozycja wiersza kanwy (znormalizowana)")
    ax.set_ylabel("promień t (jądro=0, brzeg=1)")
    ax.set_title("Odwzorowanie wiersz -> promień")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    ax.scatter(df["t_mid"], df["delta_t"], s=10, alpha=0.35, color="#555555", label="realne pary przyrostów")
    zone_mid = [(ZONE_EDGES[i] + ZONE_EDGES[i + 1]) / 2 for i in labels]
    ax.step(list(ZONE_EDGES), list(g.values) + [g.values[-1]], where="post",
             color="#c0472c", lw=2, label="g(t) — próg użyty do warpu (p1/strefę, monotoniczny)")
    ax.set_xlabel("promień t (jądro=0, brzeg=1)")
    ax.set_ylabel("delta_t (odstęp do następnego przyrostu)")
    ax.set_title("Realne odstępy przyrostów vs zastosowany próg")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    fig_path = OUTPUT_DIR / "radial_resolution_profile.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    result = {
        "n_pairs": int(len(df)),
        "n_pairs_t_mid_below_0_5": int(len(below_half)),
        "zone_edges": ZONE_EDGES,
        "zone_counts": [int(v) for v in counts.values],
        "zone_p1_raw": [None if pd.isna(v) else float(v) for v in p1_raw.values],
        "zone_g_final": [float(v) for v in g.values],
        "margin_patches": MARGIN_PATCHES,
        "patch_size": PATCH_SIZE,
        "canvas_h_px": canvas_h,
        "n_radius_patches": int(n_radius_patches),
        "old_uniform_n_radius_patches": 95,
        "t_breakpoints": t_bp,
        "row_frac_breakpoints": row_frac_bp,
    }
    (OUTPUT_DIR / "radial_resolution_profile.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nZapisano: {OUTPUT_DIR / 'radial_resolution_profile.json'}, "
          f"{fig_path}")


if __name__ == "__main__":
    main()
