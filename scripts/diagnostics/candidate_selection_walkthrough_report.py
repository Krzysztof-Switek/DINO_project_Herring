"""29.07 — standalone, self-contained explainer: from the model's raw density output
to the final ring-increment candidates, on ONE otolith from the most recent
lower-resolution (518px) run (``outputs/28.07_e9_w0.1``), several perspectives, in
one page (per user request — a dedicated document, not a section buried inside the
31MB training comparison_report.html).

Reuses the REAL production walkthrough machinery so what this shows == what the
pipeline actually does, nothing reimplemented/approximated:
  - scripts.run_pipeline._pick_walkthrough_iid       (deterministic example selection)
  - scripts.run_pipeline._compute_axis_data_for_samples (grid/axis/DP walkthrough data)
  - src.comparison_report._section_localization_walkthrough (Krok 1-4 + interactive
    widget — same HTML already shipped in every training report's section G)

Adds two NEW panels the production walkthrough does not have, to close the gap the
user asked about ("od modelu przechodzimy do kandydatów"):
  - Krok 0: the model's raw density map itself (JET heatmap overlay) BEFORE any
    ring-detection logic runs — i.e. exactly what src/interpretation.py would save
    as outputs/heatmaps + outputs/overlays for this image.
  - Krok 5: the final chosen increments overlaid on the photo, closing the loop.

Does NOT retrain or modify any src/ code, and does NOT touch outputs/28.07_e9_w0.1/
(read-only: checkpoint + predictions.csv only; its own mask cache lives under this
script's own output dir).

Usage: python scripts/diagnostics/candidate_selection_walkthrough_report.py
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image as PILImage

from src.comparison_report import _section_localization_walkthrough
from src.interpretation import importance_to_heatmap_2d, apply_colormap_with_mask
from src.otolith_axis import apply_background_mask
from src.report_common import img_tag
from src.visualization import select_top_k_samples, render_localization_overlay
from scripts.run_pipeline import (load_merged_config, _compute_axis_data_for_samples,
                                  _pick_walkthrough_iid)

RUN_DIR = PROJECT_ROOT / "outputs" / "28.07_e9_w0.1"     # ostatni bieg, 518px (Faza 1 E9, w=0.1)
CFG_PATH = PROJECT_ROOT / "configs" / "config_e9_w0.1.yaml"
CKPT = RUN_DIR / "checkpoints" / "embedded" / "best.pt"
PRED_CSV = RUN_DIR / "emb_on_emb" / "predictions.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "29.07_candidate_selection_walkthrough"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "report.html"
COND_DIR = OUT_DIR / "emb_on_emb"                         # własny cache masek — nie dotyka RUN_DIR


def _b64_from_rgb(arr: np.ndarray, target_w: int = 480) -> str:
    img = PILImage.fromarray(np.ascontiguousarray(arr[..., :3]).astype(np.uint8))
    if img.width > target_w:
        scale = target_w / img.width
        img = img.resize((target_w, max(1, int(round(img.height * scale)))), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    print("Wczytywanie configu (config_e9_w0.1.yaml, samodzielny — bez merge)...", flush=True)
    cfg = load_merged_config(CFG_PATH, None)
    image_dir = Path(cfg.data.image_dir)

    best, worst = select_top_k_samples(
        PRED_CSV, cfg.inference.increment_samples.top_k_best,
        cfg.inference.increment_samples.top_k_worst)
    all_samples = list(best) + list(worst)
    print(f"  {len(all_samples)} kart (best+worst) do wyboru przykładu.", flush=True)

    print("Wybór przykładowego otolitu (ta sama logika co w raporcie treningowym, "
          "deterministyczna)...", flush=True)
    walkthrough_iid = _pick_walkthrough_iid(
        all_samples, image_dir, seg_params=cfg.segmentation.as_params())
    chosen = [s for s in all_samples if str(s["image_id"]) == walkthrough_iid]
    if not chosen:
        raise RuntimeError(f"Nie znaleziono próbki dla {walkthrough_iid}")
    print(f"  Wybrany otolit: {walkthrough_iid}", flush=True)

    print("Liczenie gridu, osi, promieni i danych DP (jeden przebieg modelu)...", flush=True)
    grids, axis_data, walkthrough_payload = _compute_axis_data_for_samples(
        chosen, image_dir, cfg, CKPT, COND_DIR)
    if walkthrough_payload is None:
        raise RuntimeError("Segmentacja/walkthrough nie powiodły się dla wybranego otolitu")

    card = axis_data[walkthrough_iid]
    grid = grids[walkthrough_iid]
    mask_arr = card["mask"]
    axis_info = card["axis_info"]

    # --- Krok 0 (NOWY): surowy wynik modelu, ZANIM zadziała jakakolwiek logika
    # wykrywania pierścieni — dokładnie to, co run_interpretation zapisałoby jako
    # outputs/heatmaps + outputs/overlays dla tego zdjęcia. ---
    print("Renderowanie Kroku 0 (surowa mapa density)...", flush=True)
    orig_rgb = np.array(PILImage.open(image_dir / walkthrough_iid).convert("RGB"), dtype=np.uint8)
    H, W = orig_rgb.shape[:2]
    model_input_rgb = (apply_background_mask(orig_rgb, mask_arr)
                       if (cfg.data.mask_background and mask_arr is not None) else orig_rgb)
    heatmap = importance_to_heatmap_2d(grid, H, W)
    overlay_rgb = apply_colormap_with_mask(heatmap, model_input_rgb, mask=mask_arr, alpha=0.55)

    mask_note = (' Tło jest wygaszone — model dostaje dokładnie to wejście podczas treningu '
                'i inferencji (<code>mask_background=true</code>).'
                if cfg.data.mask_background else '')
    krok0_html = (
        '<section id="krok0"><h2>Krok 0 — co dokładnie „widzi" i zwraca model</h2>'
        f'<p>Otolit <code>{walkthrough_iid}</code> — wiek modelu (CORAL) '
        f'<b>{walkthrough_payload["pred_age"]}</b>, prawdziwy wiek '
        f'<b>{walkthrough_payload["true_age"]}</b>. Cała reszta tej strony pokazuje krok po kroku, '
        'jak z tej jednej mapy liczb (plus klasyczny sygnał jasności obrazu) powstają finalne '
        f'przyrosty.{mask_note}</p>'
        '<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;">'
        f'<div style="max-width:480px;"><b>Zdjęcie wejściowe</b><br>'
        f'{img_tag(_b64_from_rgb(model_input_rgb), style="width:480px;")}</div>'
        f'<div style="max-width:480px;"><b>Mapa density modelu (nałożona, JET)</b><br>'
        f'{img_tag(_b64_from_rgb(overlay_rgb), style="width:480px;")}<br>'
        '<span style="font-size:88%;color:#555;">Niebieski = niska wartość, czerwony = wysoka '
        '(znormalizowane do zakresu TEGO JEDNEGO zdjęcia, wyłącznie dla czytelności — nie porównuj '
        'kolorów między różnymi otolitami). To jedna liczba na kwadracik patcha 14×14&nbsp;px '
        '(siatka 37×37 — patrz Krok 1). Model NIE zwraca gotowych pierścieni, tylko tę '
        'z grubsza-przestrzenną mapę „gdzie prawdopodobnie jest przyrost" — cała reszta tej '
        'strony to logika POST-HOC (bez treningu), która z tej mapy wydobywa konkretne punkty.'
        '</span></div>'
        '</div></section>'
    )

    print("Składanie Kroków 1-4 (produkcyjna sekcja walkthrough, bez zmian)...", flush=True)
    core_html = _section_localization_walkthrough(walkthrough_payload)

    # --- Krok 5 (NOWY): wynik końcowy tej samej ścieżki (Krok 0→4). ---
    print("Renderowanie Kroku 5 (wynik końcowy)...", flush=True)
    wd = walkthrough_payload["data"]
    final_overlay = render_localization_overlay(
        model_input_rgb, axis_info, wd.get("final_axis_pts"), None,
        inner_margin=walkthrough_payload.get("inner_margin", 0.05))
    n_final = len(wd.get("chosen_t") or [])
    age = walkthrough_payload["pred_age"]
    shortfall_note = (
        "" if n_final >= age else
        f' <b>Uwaga:</b> wybrano tylko {n_final} odrębnych pierścieni (mniej niż wiek {age}) — '
        'nie znaleziono wystarczająco wielu wystarczająco odrębnych kandydatów.')
    krok5_html = (
        '<section id="krok5"><h2>Krok 5 — wynik końcowy</h2>'
        f'<p>Z mapy w Kroku 0 i sygnału klasycznego, przez wszystkie kroki 1–4, wybrano '
        f'<b>{n_final}</b> z <b>{age}</b> (przewidywany wiek) odrębnych przyrostów — czerwone '
        f'kropki poniżej.{shortfall_note} To dokładnie ta sama metoda (DP + rozstaw), która '
        'trafia na finalne karty raportu treningowego (sekcja E) — ta strona pokazuje wyłącznie, '
        'SKĄD te punkty się wzięły.</p>'
        f'<div style="text-align:center;">'
        f'{img_tag(_b64_from_rgb(final_overlay, target_w=560), style="width:560px;")}</div>'
        '</section>'
    )

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Jak model wybiera przyrosty — krok po kroku ({walkthrough_iid})</title>
<style>
body {{font-family:sans-serif;max-width:1400px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
h1 {{color:#1a237e;}}
h2 {{color:#1a237e;}}
h3 {{color:#283593;}}
p.cap {{font-size:88%;color:#555;margin:2px 0 12px;}}
</style>
</head>
<body>
<h1>Jak model wybiera przyrosty na otolicie — krok po kroku</h1>
<p>Bieg: <code>{RUN_DIR.name}</code> — ostatni bieg na niższej rozdzielczości (518px; Faza 1 E9,
density_concentricity_weight=0.1). Jeden przykładowy otolit, cały łańcuch od surowego wyjścia
modelu (Krok 0) do finalnie zaznaczonych przyrostów (Krok 5), z kilku perspektyw na każdym etapie
(zdjęcie, wykres promienia, histogram głosowania, siatka pierścieni, widget interaktywny).</p>
{krok0_html}
{core_html}
{krok5_html}
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
