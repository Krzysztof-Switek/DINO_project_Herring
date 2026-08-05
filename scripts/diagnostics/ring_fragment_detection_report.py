"""31.07 (cd. 4) — Wykrywanie PEWNYCH FRAGMENTÓW łuku: eksploracja i strojenie (bez fuzji/RANSAC).

Kontynuacja `ring_shortest_path_report.py` (v1-v5) i `ring_fragment_fusion_report.py` (v6) — TE
ustalenia (density zdegenerowane, prior szerokości, RANSAC-fuzja) NIE są tu powtarzane, patrz
`outputs/31.07_ring_shortest_path/report_v5.html` / `report_v6_fragment_fusion.html` i pamięć
projektu. Użytkownik: scalanie fragmentów w pełny pierścień (RANSAC) na razie ODŁOŻONE — skupiamy
się WYŁĄCZNIE na samym wykrywaniu fragmentów (metody, które już wyglądały obiecująco: gradient
promieniowy i Frangi).

Dwie rzeczy zrobione w tej iteracji:
1. **Naprawiona różnorodność promieniowa w `find_confident_open_arcs`** (nowy param `min_gap_t`
   w `ring_shortest_path_report.py`) — dotąd "top K" bywało zdominowane przez prawie-duplikaty
   TEGO SAMEGO miejsca (przesunięte o kilka stopni kąta), maskując inny, osobny pewny obszar przy
   innym promieniu ("co jest powyżej łuku" — pytanie użytkownika z poprzedniej sesji).
2. **Test mechanizmu "kwartał → K-1"** na poziomie fragmentów (nie tylko zamkniętych pierścieni
   jak w v5): 3 DOPASOWANE PARY ryb (ta sama etykieta wieku w PRAWIE-Q4 i Q1+1 — np. BITS4q
   wiek=1 vs BITS1q wiek=2 — jeśli hipoteza "Q1 = wiek-1 realnych pierścieni" jest słuszna, obie
   ryby w parze powinny wyglądać PODOBNIE pod względem liczby/rozkładu pewnych fragmentów).

Każdy obraz podpisany (na obrazie i w HTML): wiek etykiety, kwartał połowu, i czy to wariant
"-1" (Q1/Q2 → tegoroczny przyrost jeszcze się formuje).

src/ CAŁKOWICIE niedotknięte.

Usage: PYTHONIOENCODING=utf-8 python scripts/diagnostics/ring_fragment_detection_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import cv2
import torch
from PIL import Image as PILImage

from src.report_common import img_tag
from src.otolith_axis import detect_axis, apply_background_mask, mask_bbox, shift_axis_info
from src.dataset import build_transforms, decode_age_ordinal
from src.inference import load_model_from_checkpoint
from scripts.run_pipeline import load_merged_config

from scripts.diagnostics.ring_shortest_path_report import (
    ray_cast_R_theta, polar_unwrap, apply_margins, frangi_cost_field,
    find_confident_open_arcs, render_open_arcs_unwrap, render_open_arcs_on_photo,
    _b64_from_rgb, MODEL_VARIANTS,
    N_ANGLE, N_RADIUS, WINDOW, JUMP_PENALTY, INNER_MARGIN, EDGE_MARGIN, ARC_STRIDE_DEG,
    _quarter_of, _quarter_adjustment,
)

ARC_LEN_DEG = 60
ARC_TOP_K = 8
ARC_MIN_GAP_DEG = 15
ARC_MIN_GAP_T = 0.06   # NOWE — różnorodność też po promieniu, patrz docstring modułu

# 3 DOPASOWANE PARY: (BITS4q, wiek=N) vs (BITS1q, wiek=N+1) — testują "czy Q1 wiek=N+1 wygląda
# jak Q4 wiek=N" (hipoteza -1 roku). Wybrane z data/labels_combined.csv (_Embedded_,
# Single2_Right, wiek 0-6 — zakres, o który prosił użytkownik), zweryfikowane że pliki
# fizycznie istnieją na Z:.
PAIRS = [
    {"q4": "2022_BITS4q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex12_Single2_Right.jpg",
     "q4_age": 1,
     "q1": "2023_BITS1q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex18_Single2_Right.jpg",
     "q1_age": 2},
    {"q4": "2022_BITS4q_HER_GotlandzkieS_Embedded_Sharpest_FishIndex55_Single2_Right.jpg",
     "q4_age": 2,
     "q1": "2023_BITS1q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex45_Single2_Right.jpg",
     "q1_age": 3},
    {"q4": "2022_BITS4q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex19_Single2_Right.jpg",
     "q4_age": 4,
     "q1": "2023_BITS1q_HER_GlebiaGdanskaE_Embedded_Sharpest_FishIndex3_Single2_Right.jpg",
     "q1_age": 5},
]

OUT_DIR = PROJECT_ROOT / "outputs" / "31.07_ring_shortest_path"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION = "v7"
OUT_PATH = OUT_DIR / f"report_{VERSION}_fragment_detection.html"


def _caption_photo(photo: np.ndarray, label_age: int, quarter: str, is_minus1: bool) -> np.ndarray:
    out = photo.copy()
    H, W = out.shape[:2]
    txt1 = f"wiek etykiety: {label_age}  ({quarter})"
    txt2 = ("wariant: -1 (tegoroczny formuje sie)" if is_minus1
           else "wariant: bez korekty (pelny rok)")
    scale = max(0.5, W / 700)
    thick = max(1, W // 350)
    for i, (txt, color) in enumerate([(txt1, (255, 255, 255)), (txt2, (255, 230, 120))]):
        y = 24 + i * int(26 * scale)
        cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return out


def process_one(model, cfg, device, image_id: str, label_age: int) -> tuple[str, dict]:
    quarter = _quarter_of(image_id) or "?"
    is_minus1 = _quarter_adjustment(image_id) > 0
    print(f"\n--- {image_id} (etykieta wieku={label_age}, {quarter}, "
         f"wariant={'−1' if is_minus1 else 'pelny'}) ---", flush=True)

    image_dir = Path(cfg.data.image_dir)
    orig_rgb = np.array(PILImage.open(image_dir / image_id).convert("RGB"), dtype=np.uint8)
    axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                            nucleus_method=cfg.segmentation.nucleus_method)
    if axis_info is None:
        print("  Segmentacja nie powiodła się — pomijam.", flush=True)
        return "", {}
    mask_arr = axis_info["mask"]
    model_input_rgb = (apply_background_mask(orig_rgb, mask_arr)
                       if cfg.data.mask_background else orig_rgb)
    crop_x0, crop_y0, cw, ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)
    crop_rgb = model_input_rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    mask_cropped = mask_arr[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    d_axis_info = shift_axis_info(axis_info, -crop_x0, -crop_y0)
    cx, cy = d_axis_info["centroid"]

    age_transform = build_transforms(cfg.data.image_size, "test")
    age_tensor = age_transform(PILImage.fromarray(model_input_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        coral_logits = model(age_tensor)["coral_logits"]
    predicted_age = int(decode_age_ordinal(coral_logits).item())

    gray_crop = crop_rgb.mean(axis=2).astype(np.float32)
    R_theta, angles = ray_cast_R_theta(mask_cropped, cx, cy, N_ANGLE)

    def _normalize(field: np.ndarray) -> np.ndarray:
        rng = float(field.max() - field.min())
        return (field - field.min()) / rng if rng > 1e-6 else field * 0.0

    polar_classical = _normalize(polar_unwrap(gray_crop, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))
    polar_classical_grad = _normalize(np.abs(np.gradient(polar_classical, axis=0)))
    print("  Liczenie filtru Frangiego...", flush=True)
    frangi_field = frangi_cost_field(gray_crop)
    polar_frangi = _normalize(polar_unwrap(frangi_field, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))

    field_html = []
    stats = {}
    for field_name, polar_norm in [
        ("gradient promieniowy", polar_classical_grad),
        ("Frangi", polar_frangi),
    ]:
        cost = apply_margins(1.0 - polar_norm, N_RADIUS, INNER_MARGIN, EDGE_MARGIN)
        arcs = find_confident_open_arcs(cost, ARC_LEN_DEG, ARC_STRIDE_DEG, WINDOW, JUMP_PENALTY,
                                        ARC_TOP_K, ARC_MIN_GAP_DEG, ARC_MIN_GAP_T)
        ts = sorted(round(a["mean_t"], 3) for a in arcs)
        print(f"  [{field_name}] {len(arcs)} fragmentow (roznorodne po promieniu), t={ts}", flush=True)
        stats[field_name] = ts

        unwrap_b64 = render_open_arcs_unwrap(polar_norm, arcs, N_ANGLE,
                                            f"{field_name} — wiek={label_age} ({quarter})")
        photo = render_open_arcs_on_photo(crop_rgb, d_axis_info, arcs, R_theta, angles, N_RADIUS)
        photo = _caption_photo(photo, label_age, quarter, is_minus1)

        field_html.append(f"""<div style="display:inline-block;vertical-align:top;width:49%;margin:0 0.5% 12px 0;">
<h4 style="margin:4px 0;">{field_name} &mdash; {len(arcs)} fragmentów, t={ts}</h4>
{img_tag(unwrap_b64, style="width:100%;")}
<div style="text-align:center;">{img_tag(_b64_from_rgb(photo, 380), style="width:380px;")}</div>
</div>""")

    html = f"""<div style="border:1px solid #ccc;border-radius:6px;padding:10px;margin-bottom:14px;">
<h3 style="margin:2px 0;">{image_id}</h3>
<p style="font-size:90%;margin:2px 0;">Etykieta wieku: <b>{label_age}</b> &nbsp;|&nbsp;
Kwartał: <b>{quarter}</b> &nbsp;|&nbsp; CORAL przewiduje: <b>{predicted_age}</b> &nbsp;|&nbsp;
Wariant: <b>{"-1 (tegoroczny formuje się)" if is_minus1 else "pełny rok"}</b></p>
{"".join(field_html)}
</div>"""
    return html, stats


def main() -> None:
    base = MODEL_VARIANTS[0]
    cfg = load_merged_config(base["cfg_path"], None)
    device = torch.device(cfg.training.device if cfg.training.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Wczytywanie modelu ({base['name']})...", flush=True)
    model = load_model_from_checkpoint(cfg, base["run_dir"] / "checkpoints" / "embedded" / "best.pt")
    model.eval()
    device = next(model.parameters()).device

    pair_sections = []
    for pi, pair in enumerate(PAIRS):
        print(f"\n=== Para {pi + 1}: BITS4q wiek={pair['q4_age']}  vs  "
             f"BITS1q wiek={pair['q1_age']} (hipoteza: powinny wyglądać podobnie) ===", flush=True)
        html_q4, stats_q4 = process_one(model, cfg, device, pair["q4"], pair["q4_age"])
        html_q1, stats_q1 = process_one(model, cfg, device, pair["q1"], pair["q1_age"])
        pair_sections.append(f"""<section>
<h2>Para {pi + 1}: BITS4q wiek={pair['q4_age']} (pełny rok) &harr; BITS1q wiek={pair['q1_age']}
(wariant &minus;1)</h2>
<p style="font-size:90%;">Jeśli hipoteza kwartału jest słuszna, te dwie ryby powinny wykazywać
PODOBNY wzorzec pewnych fragmentów (liczba, rozkład promieniowy) — mimo różnicy 1 roku w
etykiecie wieku.</p>
{html_q4}
{html_q1}
</section>""")

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Wykrywanie fragmentów łuku — eksploracja ({VERSION})</title>
<style>
body {{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
h1 {{color:#1a237e;}} h2 {{color:#1a237e;}}
code {{background:#f0f0f5;padding:1px 4px;border-radius:3px;}}
</style>
</head>
<body>
<h1>Wykrywanie pewnych fragmentów łuku — eksploracja i test kwartału</h1>
<p>Kontynuacja v1-v6 (patrz <code>outputs/31.07_ring_shortest_path/report_v5.html</code>,
<code>report_v6_fragment_fusion.html</code>, pamięć projektu) — TYLKO nowa treść: (1) naprawiona
różnorodność promieniowa w wykrywaniu fragmentów (<code>min_gap_t={ARC_MIN_GAP_T}</code>), (2) 3
dopasowane pary ryb (BITS4q wiek=N vs BITS1q wiek=N+1) testujące, czy mechanizm "kwartał &rarr;
wiek&minus;1" jest widoczny już na poziomie SAMYCH fragmentów, nie tylko zamkniętych pierścieni.
Scalanie fragmentów w pełny pierścień (RANSAC) świadomie ODŁOŻONE na razie. Nic w <code>src/</code>
nie zostało zmienione.</p>
{"".join(pair_sections)}
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
