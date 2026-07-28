"""22.07: single comparative HTML report for the density-resolution/crop-to-bbox change
(Zmiana A+B) — localization_quality.json before/after (chart + table), a visual side-by-side
of the patch grid on the full image (today) vs the cropped-to-otolith image (new), and a few
sample reasoning cards baseline vs hires+crop.

Reuses outputs/22.07_baseline_local/ and outputs/22.07_hires_local/, produced by the ad hoc
comparison run (see plans and summaries/22.07_TO_DO.MD) — rerun that first if these are stale.
"""
import base64
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from src.comparison_report import _style_ax, _INK, _MUTED, _GRID
from src.config import OtolithConfig
from src.otolith_axis import (apply_background_mask, find_farthest_edge, load_mask,
                              mask_bbox, resolve_centroid, shift_axis_info)
from src.report_common import fig_to_b64, img_tag, png_to_b64
from src.visualization import draw_nucleus_exclusion_zone, render_patch_grid

INNER_MARGIN = OtolithConfig().candidates.inner_margin   # single source of truth (config.py default)

BASELINE_DIR = PROJECT_ROOT / "outputs" / "22.07_baseline_local"
HIRES_DIR = PROJECT_ROOT / "outputs" / "22.07_hires_local"
IMAGE_DIR = Path("Z:/Photo/Otolithes/HER/Processed")
OUT_PATH = PROJECT_ROOT / "outputs" / "22.07_density_resolution_report.html"

BLUE, ORANGE = "#2a78d6", "#eb6834"


def _arr_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --- 1. localization_quality.json comparison ---
baseline_q = json.loads((BASELINE_DIR / "localization_quality.json").read_text(encoding="utf-8"))["emb_on_emb"]
hires_q = json.loads((HIRES_DIR / "localization_quality.json").read_text(encoding="utf-8"))["emb_on_emb"]

methods = ["density", "classical", "consensus", "dp"]
b_vals = [baseline_q["per_method"][m]["mean_dist_final_to_classical_px"] for m in methods]
h_vals = [hires_q["per_method"][m]["mean_dist_final_to_classical_px"] for m in methods]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(methods))
w = 0.35
bars_b = ax.bar(x - w / 2, b_vals, width=w, color=BLUE, label="baseline (dziś)")
bars_h = ax.bar(x + w / 2, h_vals, width=w, color=ORANGE, label="hires+crop (Zmiana A+B)")
for bars in (bars_b, bars_h):
    for rect in bars:
        hgt = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, hgt + 2, f"{hgt:.0f}", ha="center",
                va="bottom", fontsize=8, color=_INK)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("mean_dist_final_to_classical_px (niżej = lepiej)")
ax.set_title("Lokalizacja przyrostów: baseline vs wyższa rozdzielczość density (30 kart)")
ax.legend(fontsize=8)
_style_ax(ax)
fig.tight_layout()
chart_b64 = fig_to_b64(fig)


def _pct(b, h):
    return f"{(h - b) / b * 100:+.1f}%"


rows_html = ""
for m in methods:
    b, h = baseline_q["per_method"][m], hires_q["per_method"][m]
    rows_html += (
        f"<tr><td>{m}</td>"
        f"<td>{b['mean_dist_final_to_classical_px']:.2f}</td>"
        f"<td>{h['mean_dist_final_to_classical_px']:.2f}</td>"
        f"<td>{_pct(b['mean_dist_final_to_classical_px'], h['mean_dist_final_to_classical_px'])}</td>"
        f"<td>{b['mean_abs_n_final_minus_age']:.3f}</td>"
        f"<td>{h['mean_abs_n_final_minus_age']:.3f}</td></tr>"
    )

# --- 2. patch grid: full image (37x37) vs cropped-to-otolith (52x52) ---
sample_iid = "2023_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg"
img = np.array(PILImage.open(IMAGE_DIR / sample_iid).convert("RGB"), dtype=np.uint8)
mask_path = BASELINE_DIR / "emb_on_emb" / "masks" / f"{Path(sample_iid).stem}_mask.png"
mask = load_mask(mask_path)
masked_img = apply_background_mask(img, mask) if mask is not None else img

# Build axis_info from the CACHED mask directly (mirrors run_pipeline.py::
# _compute_axis_data_for_samples's cached-mask path) — do NOT re-segment masked_img:
# detect_axis() on an already-masked image (flat MASK_FILL_RGB background, which can be
# tonally close to the otolith body in places) finds a small, wrong region instead of
# the true outline (bug caught from the first version of this report by the user).
axis_info = None
if mask is not None:
    cent = resolve_centroid(img, mask, "geometric")
    far = find_farthest_edge(mask, cent) if cent else None
    if cent and far:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            axis_info = {"mask": mask, "centroid": cent, "far_edge": far, "contour": contour,
                        "length_px": float(np.hypot(far[0] - cent[0], far[1] - cent[1]))}

full_h, full_w = masked_img.shape[:2]
_full_base = masked_img.copy()
draw_nucleus_exclusion_zone(_full_base, axis_info, inner_margin=INNER_MARGIN)
full_grid_panel = render_patch_grid(_full_base, axis_info, n_patches=37)
full_grid_b64 = _arr_to_b64(full_grid_panel)

x0, y0, cw, ch = mask_bbox(mask, pad_frac=0.02)   # matches configs/config.yaml (22.07 correction)
print(f"Full image: {full_w}x{full_h}. Crop bbox: x0={x0} y0={y0} w={cw} h={ch} "
      f"({cw / full_w:.1%} x {ch / full_h:.1%} of full frame). inner_margin={INNER_MARGIN}", flush=True)
cropped = masked_img[y0:y0 + ch, x0:x0 + cw]
cropped_axis_info = shift_axis_info(axis_info, -x0, -y0) if axis_info is not None else None
_crop_base = cropped.copy()
draw_nucleus_exclusion_zone(_crop_base, cropped_axis_info, inner_margin=INNER_MARGIN)
crop_grid_panel = render_patch_grid(_crop_base, cropped_axis_info, n_patches=52)
crop_grid_b64 = _arr_to_b64(crop_grid_panel)

# Display BOTH panels at the SAME px-per-original-px scale (not independently stretched
# to equal-width flex columns) so the crop's actual size reduction is visible on screen,
# not hidden by CSS scaling both images to the same displayed width regardless of content.
_disp_scale = 500 / full_w
full_disp_w, full_disp_h = round(full_w * _disp_scale), round(full_h * _disp_scale)
crop_disp_w, crop_disp_h = round(cw * _disp_scale), round(ch * _disp_scale)

# --- 3. side-by-side sample cards (best) ---
cards_html = ""
for p in sorted((BASELINE_DIR / "cards" / "emb_on_emb" / "best").glob("*.png"))[:3]:
    hires_p = HIRES_DIR / "cards" / "emb_on_emb" / "best" / p.name
    if not hires_p.exists():
        continue
    cards_html += f"""
    <div style="margin-bottom:24px;">
      <p style="font-size:90%;color:{_MUTED};">{p.name}</p>
      <p style="font-weight:bold;color:{BLUE};margin:4px 0 2px;">baseline</p>
      {img_tag(png_to_b64(p), style="width:900px;")}
      <p style="font-weight:bold;color:{ORANGE};margin:12px 0 2px;">hires+crop</p>
      {img_tag(png_to_b64(hires_p), style="width:900px;")}
    </div>"""

html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>22.07 — density resolution/crop: baseline vs Zmiana A+B</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width:1100px; margin:24px auto; color:{_INK}; }}
h1,h2 {{ color:{_INK}; }}
table {{ border-collapse:collapse; margin:12px 0; }}
td, th {{ border:1px solid {_GRID}; padding:6px 10px; font-size:90%; }}
th {{ background:#f4f3f0; }}
.cap {{ color:{_MUTED}; font-size:85%; }}
</style></head>
<body>
<h1>22.07 — wyższa rozdzielczość density (Zmiana A+B): baseline vs test</h1>
<p class="cap">Checkpoint: <code>outputs/20.07_reg/checkpoints/embedded/best.pt</code>
(density_head przywrócony do architektury SPRZED dzisiejszego LayerNorm — izolacja tylko Zmiany A+B,
patrz plans and summaries/22.07_TO_DO.MD). Próba: 30 kart (15 najlepszych + 15 najgorszych predykcji
wieku). <b>baseline</b> = density_image_size=None, density_crop_to_otolith=False (dzisiejszy kod, bez
A/B). <b>hires+crop</b> = density_image_size=728, density_crop_to_otolith=True.</p>

<h2>1. Jakość lokalizacji (localization_quality.json)</h2>
{img_tag(chart_b64)}
<table>
<tr><th>Metoda</th><th>dist baseline</th><th>dist hires+crop</th><th>zmiana</th>
    <th>|n_final−age| baseline</th><th>|n_final−age| hires+crop</th></tr>
{rows_html}
</table>
<p class="cap">mean_attn_bg_frac: baseline {baseline_q.get('mean_attn_bg_frac')} vs hires+crop
{hires_q.get('mean_attn_bg_frac')} — identyczne, potwierdza że CORAL/uwaga CLS są nietknięte.</p>
<p><b>Wniosek:</b> metoda produkcyjna <code>dp</code> poprawia się ({_pct(b_vals[3], h_vals[3])}), ale
sam surowy sygnał <code>density</code> się pogarsza ({_pct(b_vals[0], h_vals[0])}) — spójne z hipotezą
niedopasowania rozdzielczości trening/inferencja density_head (był trenowany na siatce 37×37 z pełnego
zdjęcia, teraz dostaje 52×52 z przyciętego kadru). Rekomendacja: krótkie doszkolenie density_head na
nowej rozdzielczości.</p>

<h2>2. Nowy podział na patche — pełne zdjęcie (37×37) vs przycięty kadr (52×52)</h2>
<p class="cap">Oba panele w TEJ SAMEJ skali piksele-na-oryginalny-piksel (nie niezależnie rozciągnięte do
tej samej szerokości) — jeśli crop faktycznie coś przycina, będzie fizycznie mniejszy na ekranie.
Crop bbox: {cw}×{ch} px z {full_w}×{full_h} px pełnego zdjęcia ({cw / full_w:.0%}×{ch / full_h:.0%}
oryginalnej ramki).</p>
<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;">
  <div>
    <p style="font-weight:bold;">Dziś: 37×37 na CAŁYM (zmaskowanym) zdjęciu ({full_w}×{full_h}px)</p>
    {img_tag(full_grid_b64, style=f"width:{full_disp_w}px;")}
  </div>
  <div>
    <p style="font-weight:bold;">Zmiana A+B: 52×52 na PRZYCIĘTYM kadrze ({cw}×{ch}px)</p>
    {img_tag(crop_grid_b64, style=f"width:{crop_disp_w}px;")}
  </div>
</div>
<p class="cap">Ten sam otolit ({sample_iid}). Po lewej: siatka patchy dzisiejszego pipeline'u — spora
część kratek pada na (zamaskowane) tło. Po prawej: ta sama fizyczna przestrzeń otolitu pokryta gęstszą
siatką — wejście jest większe (728 zamiast 518) I tło nie zajmuje już miejsca w siatce. <b>Czerwona
strefa</b> wokół jądra na obu panelach = <code>inner_margin={INNER_MARGIN}</code> — ta sama strefa
wykluczenia kandydatów, w tej samej skali co siatka patchy, żeby było widać, ile "miejsca" na siatce
ona zajmuje.</p>

<h2>3. Przykładowe karty — bok o bok</h2>
{cards_html}

</body></html>
"""

OUT_PATH.write_text(html, encoding="utf-8")
print(f"Report: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
