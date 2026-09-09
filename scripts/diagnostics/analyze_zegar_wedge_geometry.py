"""Faza 0 planu wycinka katowego (polar wedge) -- mierzy REALNE katowe odchylenie prawdziwych
przyrostow (KK/SS) od kierunku odczytu (`find_reading_edge`), zeby wyznaczyc szerokosc wycinka
(Delta-theta) z pomiaru, nie ze strzalu "45 stopni".

Rozdzielczosc PROMIENIOWA jest SKALO-NIEZMIENNA wzgledem tego, czy tniemy pasek czy wycinek
katowy -- to ta sama wielkosc `t` (znormalizowany promien), juz zmierzona w
`02.09_zegar_ring_spacing/ring_spacing_summary.json` (p1 -> 1330px / 95 patchy). Ponownie
uzywana tutaj wprost, NIE przeliczana od nowa.

Rozdzielczosc KATOWA to inny rodzaj pomiaru: nie chodzi o unikniecie zderzenia dwoch przyrostow w
jednym patchu (przyrosty roznia sie promieniem/wierszem, nie kolumna), tylko o dopasowanie
gestosci katowej do gestosci, jaka Run N sam demonstruje jako dzialajaca (48 promieni / 360 st.
= ok. 7.5 st./promien) -- zwezone do Delta-theta.

Metoda pomiaru Delta-theta: dla kazdego zaadnotowanego przyrostu (KK i SS, 42 obrazy) liczymy KAT
wektora (centroid -> punkt) wzgledem KATA osi odczytu (centroid -> far_edge), zawinietym do
[-180, 180]. Rozklad |odchylenia| po wszystkich punktach daje percentyl pokrywajacy 90-95%
prawdziwych przyrostow.

Nie dotyka treningu ani `src/` -- czysto diagnostyczny.

Usage:
    python scripts/diagnostics/analyze_zegar_wedge_geometry.py
"""
from __future__ import annotations

import json
import math
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
    resolve_and_crop_target_otolith,
)
from scripts.run_pipeline import load_merged_config
from src.otolith_axis import detect_axis

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "09.09_masked_wizualizacje" / "wedge_geometry"
RADIUS_SPACING_RESULT = PROJECT_ROOT / "outputs" / "02.09_zegar_ring_spacing" / "ring_spacing_summary.json"

RUN_N_RAY_SPACING_DEG = 360.0 / 48.0  # Run N's own demonstrated angular density


def angle_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def wrap_deg(a: float) -> float:
    """Wrap to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Pomiar geometrii wycinka katowego -- ZEGAR (Faza 0, wycinek katowy)")
    print("=" * 70)

    print("\n[1/4] Wczytywanie adnotacji...")
    ann = load_expert_annotations()
    samples = sorted(ann["Sample"].unique())
    print(f"  {len(samples)} probek")

    cfg = load_merged_config(REFERENCE_CONFIG, None)
    seg_params = cfg.segmentation.as_params()

    print("\n[2/4] Przycinanie, wyznaczanie osi, pomiar odchylenia katowego kazdego przyrostu...")
    rows: list[dict] = []
    n_seg_failed = 0
    for sample in samples:
        image_id = f"{sample}.jpg"
        sub = ann[ann.Sample == sample]
        mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
        raw_rgb = np.array(PILImage.open(IMAGE_DIR / image_id).convert("RGB"), dtype=np.uint8)
        cropped, x0, y0, _ = resolve_and_crop_target_otolith(raw_rgb, mean_xy, seg_params)
        if cropped is None:
            n_seg_failed += 1
            continue
        axis_info = detect_axis(cropped, seg_params=seg_params,
                                 nucleus_method=cfg.segmentation.nucleus_method,
                                 axis_method=cfg.segmentation.axis_method)
        if axis_info is None:
            n_seg_failed += 1
            continue
        cx, cy = axis_info["centroid"]
        fx, fy = axis_info["far_edge"]
        axis_angle = angle_deg(fx - cx, fy - cy)

        for reader in ("KK", "SS"):
            pts = sub[sub.annotator == reader]
            for _, r in pts.iterrows():
                x, y = r.x - x0, r.y - y0
                t = float(np.hypot(x - cx, y - cy) / max(axis_info["length_px"], 1.0))
                pt_angle = angle_deg(x - cx, y - cy)
                dev = abs(wrap_deg(pt_angle - axis_angle))
                rows.append({"Sample": sample, "annotator": reader,
                             "angle_dev_deg": dev, "t_radius": t})

    n_ok = len(samples) - n_seg_failed
    print(f"  {n_ok}/{len(samples)} obrazow z policzona osia ({n_seg_failed} pominietych)")
    if not rows:
        sys.exit("Brak pomierzonych punktow.")

    devs = pd.DataFrame(rows)
    devs.to_csv(OUTPUT_DIR / "angle_deviations.csv", index=False)

    d = devs["angle_dev_deg"]
    stats = {
        "n_points": int(len(d)), "min": float(d.min()), "median": float(d.median()),
        "mean": float(d.mean()), "p90": float(d.quantile(0.90)),
        "p95": float(d.quantile(0.95)), "max": float(d.max()),
    }
    print("\n[3/4] Rozklad odchylenia katowego przyrostow od osi odczytu (stopnie):")
    for k, v in stats.items():
        print(f"   {k}: {v:.3f}" if k != "n_points" else f"   {k}: {v}")

    worst = devs.nlargest(10, "angle_dev_deg")[["Sample", "annotator", "angle_dev_deg", "t_radius"]]
    print("\n  10 najbardziej odchylonych przyrostow (najgorszy przypadek):")
    print(worst.to_string(index=False))

    # Delta-theta: PELNA szerokosc wycinka wokol osi odczytu musi pokryc odchylenie w OBIE strony
    # (przyrost moze odchylac sie w lewo LUB w prawo od osi) -> pelne Delta-theta = 2 * percentyl.
    delta_theta_p90 = 2.0 * stats["p90"]
    delta_theta_p95 = 2.0 * stats["p95"]
    print(f"\n[4/4] Wyznaczanie Delta-theta (pelna szerokosc wycinka, pokrywajaca odchylenie w obie strony):")
    print(f"   p90 -> Delta-theta = {delta_theta_p90:.1f} deg (pokrywa 90% prawdziwych przyrostow)")
    print(f"   p95 -> Delta-theta = {delta_theta_p95:.1f} deg (pokrywa 95% prawdziwych przyrostow, REKOMENDOWANE)")

    n_cols_p90 = max(1, round(delta_theta_p90 / RUN_N_RAY_SPACING_DEG))
    n_cols_p95 = max(1, round(delta_theta_p95 / RUN_N_RAY_SPACING_DEG))
    print(f"\n   Rozdzielczosc katowa (dopasowana do gestosci promieni Run N, {RUN_N_RAY_SPACING_DEG:.2f} st./kierunek):")
    print(f"   p90 Delta-theta -> {n_cols_p90} kolumn katowych")
    print(f"   p95 Delta-theta -> {n_cols_p95} kolumn katowych (REKOMENDOWANE)")

    radius_result = json.loads(RADIUS_SPACING_RESULT.read_text(encoding="utf-8"))
    print(f"\n   Rozdzielczosc promieniowa (PONOWNIE UZYTA z {RADIUS_SPACING_RESULT.name}, "
          f"NIE przeliczana od nowa -- ta sama wielkosc t): "
          f"{radius_result['recommended_strip_length_px_p1']}px "
          f"({radius_result['recommended_strip_length_px_p1'] // radius_result['patch_size']} patchy)")

    result = {
        "angle_stats_deg": stats,
        "delta_theta_p90_deg": delta_theta_p90,
        "delta_theta_p95_deg": delta_theta_p95,
        "run_n_ray_spacing_deg": RUN_N_RAY_SPACING_DEG,
        "n_angle_columns_p90": n_cols_p90,
        "n_angle_columns_p95_recommended": n_cols_p95,
        "radius_resolution_reused_from": str(RADIUS_SPACING_RESULT),
        "radius_px_recommended": radius_result["recommended_strip_length_px_p1"],
        "radius_patches_recommended": radius_result["recommended_strip_length_px_p1"] // radius_result["patch_size"],
        "n_images_ok": n_ok, "n_images_total": len(samples),
    }
    (OUTPUT_DIR / "wedge_geometry_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nZapisano: {OUTPUT_DIR / 'angle_deviations.csv'}, {OUTPUT_DIR / 'wedge_geometry_summary.json'}")


if __name__ == "__main__":
    main()
