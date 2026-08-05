"""30.07 — Ilustracja: promienie -> patche -> głosowanie, w DWÓCH rozdzielczościach modelu.

Użytkownik poprosił o TRZY obrazy obok siebie, na dokładnie tej samej mapie density:
  1. Mapa density + WSZYSTKICH 48 promieni, w KOLORZE lepiej widocznym niż dotychczasowy
     ciemnoszary ``_RAY_COLOR=(110,110,110)`` ze ``src/visualization.py`` (użytkownik:
     "popraw kolor promieni, nie ich grubość" — na realnym zrzucie ekranu promienie
     prawie znikają na tle żółto-zielonych patchy mapy JET). Tu: biały, 1px, bez zmiany
     grubości — jedyna różnica względem oryginału to kolor.
  2. Ten sam widok, ale WYGASZONE (przyciemnione, odbarwione do szarości) wszystkie
     patche, których ŻADEN z 48 promieni nie dotyka/nie przecina po drodze — zostają
     kolorowe TYLKO te patche, które faktycznie wchodzą do sygnału JAKIEGOKOLWIEK
     promienia (dokładnie ten sam nearest-neighbour sposób próbkowania, którego używa
     ``sample_profile_along_axis`` / Krok 2-3 w ``density_candidate_mechanism_report.py``).
  3. To samo, ale kolorowy zostaje TYLKO JEDEN, pojedynczy przykładowy promień (magenta) —
     wszystkie patche NIE leżące na TYM promieniu są wygaszone, nawet jeśli leżą na innych
     z 48 promieni. Ten przykładowy promień to NIE oś pomiaru (ta ma osobne, już ustalone
     znaczenie w reszcie raportów) — to zwykły, dowolny promień (stały indeks, żeby wynik
     był powtarzalny), żeby ilustracja "jeden promień = jego własny, wąski zestaw patchy"
     nie mieszała się z pojęciem osi pomiaru.

Na wyraźną prośbę użytkownika (30.07): powtórzone w DWÓCH wersjach rozdzielczości modelu —
  A) NISKA rozdzielczość: ``config_e9_w0.1.yaml`` (trening natywnie 518px; density liczone
     post-hoc na 728px / 52x52 patchy) — ten sam checkpoint, którego używa
     ``density_candidate_mechanism_report.py`` (``outputs/28.07_e9_w0.1``).
  B) WYSOKA rozdzielczość: ``config_hires966.yaml`` (trening NATYWNIE na 966px; density
     966px / 69x69 patchy od epoki 1, nie post-hoc) — osobno trenowany checkpoint
     (``outputs/23.07_hires966``).
Segmentacja/kontur/bbox liczone RAZ na oryginalnym zdjęciu (nie zależą od rozdzielczości
modelu) — każda rozdzielczość dostaje własny, osobny forward pass i własną siatkę density
(52x52 vs 69x69 nad TYM SAMYM wyciętym fragmentem zdjęcia).

v2 (30.07, na prośbę użytkownika po obejrzeniu v1): układ 2×2 zamiast 3-w-rzędzie —
DODANY panel (a) "wygładzona" mapa density, DOKŁADNIE tym samym przepisem co produkcyjny
``draw_reasoning_card``'s ``_heat_panel`` (``src/visualization.py:450-472`` — INTER_LINEAR,
percentyl 2-98 liczony nad erodowaną maską, alfa per-piksel z gamma=1.5) — czysty punkt
odniesienia "jak to naprawdę wygląda w raporcie", BEZ siatki patchy/promieni. Wiersz 1 =
(a) wygładzona mapa, (b) dawny Obraz 1 (blokowa siatka + 48 białych promieni). Wiersz 2 =
(c) dawny Obraz 2 (wygaszone bez żadnego promienia), (d) dawny Obraz 3 (zostaje 1 promień).
Dopisany też JAWNY procent wygaszenia pod (c) i (d) (liczby już były liczone w v1, teraz
widoczne wprost przy każdym obrazie, nie tylko w akapicie sekcji).

Nie trenuje ani nie modyfikuje żadnego kodu w src/.

Usage: PYTHONIOENCODING=utf-8 python scripts/diagnostics/ray_patch_voting_report.py
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image as PILImage

from src.report_common import fig_to_b64, img_tag
from src.visualization import (draw_nucleus_exclusion_zone, _CONTOUR_COLOR, _AXIS_COLOR,
                               _robust_grid_range)
from src.otolith_axis import detect_axis, apply_background_mask, mask_bbox, shift_axis_info
from src.dataset import build_transforms
from src.inference import load_model_from_checkpoint
from scripts.run_pipeline import load_merged_config

IMAGE_ID = "2023_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg"

OUT_DIR = PROJECT_ROOT / "outputs" / "29.07_candidate_selection_walkthrough"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION = "v4"
OUT_PATH = OUT_DIR / f"ray_patch_voting_{VERSION}.html"

N_DIRS, N_SAMPLES = 48, 64
EXAMPLE_RAY_IDX = 12   # dowolny, ustalony indeks (1/4 obwodu) — NIE oś pomiaru, celowo osobny promień

RESOLUTIONS = [
    {"label": "Niska rozdzielczość", "slug": "low",
     "cfg_path": PROJECT_ROOT / "configs" / "config_e9_w0.1.yaml",
     "run_dir": PROJECT_ROOT / "outputs" / "28.07_e9_w0.1",
     "note": "trening natywnie 518px; density liczone post-hoc na 728px (crop-to-bbox)"},
    {"label": "Wysoka rozdzielczość", "slug": "high",
     "cfg_path": PROJECT_ROOT / "configs" / "config_hires966.yaml",
     "run_dir": PROJECT_ROOT / "outputs" / "23.07_hires966",
     "note": "trening NATYWNIE na 966px — CORAL/MIL/density widzą tę samą siatkę od epoki 1"},
]

_RAY_COLOR_VISIBLE = (255, 255, 255)     # biały — dużo bardziej widoczny na mapie JET niż
                                          # dotychczasowy ciemnoszary (110,110,110)
_RAY_HIGHLIGHT_COLOR = (255, 0, 255)     # magenta — kolor spoza palety JET, jak reszta raportów


def _b64_from_rgb(arr: np.ndarray, target_w: int = 480) -> str:
    img = PILImage.fromarray(np.ascontiguousarray(arr[..., :3]).astype(np.uint8))
    if img.width > target_w:
        scale = target_w / img.width
        img = img.resize((target_w, max(1, int(round(img.height * scale)))), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ray_cells(grid_shape: tuple[int, int], cx: float, cy: float, fx: float, fy: float,
               cw: int, ch: int, n_samples: int = N_SAMPLES) -> set[tuple[int, int]]:
    """Zbiór (col, row) komórek siatki dotkniętych przez PROMIEŃ centroid->(fx,fy) —
    dokładnie ten sam nearest-neighbour sposób próbkowania (64 punkty), którego używa
    ``sample_profile_along_axis``/Krok 2-3 innego raportu, żeby liczby się zgadzały."""
    H_p, W_p = grid_shape
    xs = np.linspace(cx, fx, n_samples)
    ys = np.linspace(cy, fy, n_samples)
    px = np.clip((xs / max(cw, 1) * W_p).astype(np.int64), 0, W_p - 1)
    py = np.clip((ys / max(ch, 1) * H_p).astype(np.int64), 0, H_p - 1)
    return set(zip(px.tolist(), py.tolist()))


def render_density_heatmap_smooth(grid: np.ndarray, crop_rgb: np.ndarray,
                                  mask_cropped: np.ndarray, axis_info: dict
                                  ) -> tuple[np.ndarray, float, float]:
    """Odtwarza DOKŁADNIE przepis produkcyjnego ``_heat_panel`` (closure wewnątrz
    ``draw_reasoning_card``, ``src/visualization.py:450-472``) jako samodzielną funkcję —
    to jest "wygładzona" mapa density, jaką użytkownik widzi w prawdziwych
    kartach/raportach, w odróżnieniu od celowo "surowej, blokowej" siatki patchy
    (``render_density_heatmap_masked`` niżej). Różnice względem tamtej: ``INTER_LINEAR``
    (nie ``NEAREST``), zakres percentyla 2-98 liczony TYLKO nad wnętrzem maski otolitu po
    erozji (``_robust_grid_range``, nie nad całym wyciętym obrazem), alfa PER-PIKSEL z
    gamma=1.5 (tylko silne piki są prawie nieprzezroczyste), twardo wyzerowana poza maską.
    BEZ siatki patchy, BEZ promieni — to czysty punkt odniesienia "jak to wygląda
    naprawdę w raporcie", nie narzędzie do liczenia pokrycia promieni. ``mask_cropped``
    musi być już przycięta do tej samej ramki co ``crop_rgb`` — ``shift_axis_info`` NIE
    przycina maski (``src/otolith_axis.py:437-454``), więc wołający musi to zrobić sam.
    """
    H, W = crop_rgb.shape[:2]
    arr = np.asarray(grid, dtype=np.float32)
    lo, hi = _robust_grid_range(arr, mask_cropped)
    norm = np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    hm = cv2.resize(norm, (W, H), interpolation=cv2.INTER_LINEAR).clip(0.0, 1.0)
    bgr = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    rgb_map = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = np.power(hm, 1.5) * 0.9
    alpha = alpha * (np.asarray(mask_cropped) > 0)
    a = alpha[..., None]
    panel = (a * rgb_map + (1.0 - a) * crop_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    if axis_info.get("contour") is not None:
        cv2.drawContours(panel, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    return panel, lo, hi


def render_density_heatmap_masked(grid: np.ndarray, crop_rgb: np.ndarray, axis_info: dict,
                                  keep_mask: np.ndarray | None, vmin: float, vmax: float,
                                  dim_gray_factor: float = 0.30) -> np.ndarray:
    """Ta sama kolorystyka (JET, percentyl 1-98) co Krok 1 innego raportu, ale patche
    gdzie ``keep_mask`` (H_p, W_p bool) jest False dostają PRZYCIEMNIONĄ, ODBARWIONĄ
    (szarość zdjęcia × ``dim_gray_factor``) wersję zamiast koloru density — "wygaszone".
    ``keep_mask=None`` = wszystkie patche kolorowe (zwykła, niefiltrowana mapa)."""
    H, W = crop_rgb.shape[:2]
    H_p, W_p = grid.shape
    norm = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0) if vmax > vmin else np.zeros_like(grid)
    up_norm = cv2.resize(norm.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    uint8_map = (up_norm * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
    rgb_map = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    colored = (0.6 * rgb_map + 0.4 * crop_rgb.astype(np.float32))
    if keep_mask is None:
        blended = colored.clip(0, 255).astype(np.uint8)
    else:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dimmed = np.stack([gray, gray, gray], axis=-1) * dim_gray_factor
        up_keep = cv2.resize(keep_mask.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        blended = np.where(up_keep[..., None], colored, dimmed).clip(0, 255).astype(np.uint8)
    for i in range(1, W_p):
        x = int(round(i * W / W_p))
        cv2.line(blended, (x, 0), (x, H), (60, 60, 60), 1, cv2.LINE_AA)
    for i in range(1, H_p):
        y = int(round(i * H / H_p))
        cv2.line(blended, (0, y), (W, y), (60, 60, 60), 1, cv2.LINE_AA)
    if axis_info.get("contour") is not None:
        cv2.drawContours(blended, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    return blended


def draw_rays(img: np.ndarray, axis_info: dict, n_dirs: int = N_DIRS,
             inner_margin: float = 0.05, draw_all: bool = True,
             single_ray_endpoint: tuple[float, float] | None = None) -> np.ndarray:
    """Rysuje promienie na KOPII ``img``. ``draw_all=True`` -> wszystkich ``n_dirs``
    promieni w ``_RAY_COLOR_VISIBLE`` (jedyna zmiana względem
    ``src.visualization.render_rays_and_candidates`` to KOLOR, nie grubość — nadal 1px,
    cv2.LINE_AA). ``single_ray_endpoint`` (jeśli podany) rysuje TYLKO ten jeden promień
    (centroid->punkt) w ``_RAY_HIGHLIGHT_COLOR`` zamiast wszystkich 48 — dla Obrazu 3,
    gdzie chodzi o TEN JEDEN konkretny promień, nie o kontekst pozostałych."""
    out = np.ascontiguousarray(img[..., :3]).copy()
    if axis_info is None:
        return out
    draw_nucleus_exclusion_zone(out, axis_info, inner_margin=inner_margin, n_dirs=n_dirs)
    contour, centroid = axis_info.get("contour"), axis_info.get("centroid")
    H = out.shape[0]
    lt = max(2, H // 300)
    if contour is None or centroid is None:
        return out
    cx, cy = int(centroid[0]), int(centroid[1])
    if draw_all:
        cpts = contour.reshape(-1, 2)
        idx_sel = np.linspace(0, len(cpts) - 1, min(n_dirs, len(cpts)), dtype=int)
        for ci in idx_sel:
            cv2.line(out, (cx, cy), (int(cpts[ci][0]), int(cpts[ci][1])),
                     _RAY_COLOR_VISIBLE, 1, cv2.LINE_AA)
    cv2.drawContours(out, [contour], -1, _CONTOUR_COLOR, lt)
    if single_ray_endpoint is not None:
        fx, fy = single_ray_endpoint
        cv2.line(out, (cx, cy), (int(fx), int(fy)), _RAY_HIGHLIGHT_COLOR, 1, cv2.LINE_AA)
    elif axis_info.get("far_edge"):
        fx, fy = axis_info["far_edge"]
        cv2.line(out, (cx, cy), (int(fx), int(fy)), _AXIS_COLOR, lt, cv2.LINE_AA)
    return out


def process_resolution(res: dict, orig_rgb: np.ndarray, axis_info: dict, mask_arr: np.ndarray,
                       model_input_rgb: np.ndarray) -> str:
    label, cfg_path, run_dir = res["label"], res["cfg_path"], res["run_dir"]
    print(f"[{label}] wczytywanie configu i modelu ({cfg_path.name})...", flush=True)
    cfg = load_merged_config(cfg_path, None)
    device = torch.device(cfg.training.device if cfg.training.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = run_dir / "checkpoints" / "embedded" / "best.pt"
    model = load_model_from_checkpoint(cfg, ckpt)
    model.eval()
    device = next(model.parameters()).device

    crop_x0, crop_y0, cw, ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)
    crop_rgb = model_input_rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    mask_cropped = mask_arr[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    d_axis_info = shift_axis_info(axis_info, -crop_x0, -crop_y0)
    density_transform = build_transforms(cfg.candidates.density_image_size, "test")
    density_tensor = density_transform(PILImage.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        density_grid = model.get_density_probs(density_tensor).squeeze(0).cpu().numpy()
    H_p, W_p = density_grid.shape
    print(f"  [{label}] siatka: {H_p}x{W_p} = {H_p*W_p} patchy", flush=True)

    vmin = float(np.percentile(density_grid, 1.0))
    vmax = float(np.percentile(density_grid, 98.0))
    if vmax <= vmin:
        vmin, vmax = float(density_grid.min()), float(density_grid.max())

    cx, cy = d_axis_info["centroid"]
    contour = d_axis_info["contour"].reshape(-1, 2)
    idx_sel = np.linspace(0, len(contour) - 1, min(N_DIRS, len(contour)), dtype=int)

    all_cells: set[tuple[int, int]] = set()
    for ci in idx_sel:
        fx, fy = int(contour[ci][0]), int(contour[ci][1])
        all_cells |= _ray_cells((H_p, W_p), cx, cy, fx, fy, cw, ch)
    all_mask = np.zeros((H_p, W_p), dtype=bool)
    for (col, row) in all_cells:
        all_mask[row, col] = True

    ex_ci = idx_sel[EXAMPLE_RAY_IDX]
    ex_endpoint = (int(contour[ex_ci][0]), int(contour[ex_ci][1]))
    ex_cells = _ray_cells((H_p, W_p), cx, cy, ex_endpoint[0], ex_endpoint[1], cw, ch)
    ex_mask = np.zeros((H_p, W_p), dtype=bool)
    for (col, row) in ex_cells:
        ex_mask[row, col] = True

    total = H_p * W_p
    any_pct = 100 * len(all_cells) / total
    ex_pct = 100 * len(ex_cells) / total
    any_dim_pct = 100 - any_pct
    ex_dim_pct = 100 - ex_pct
    print(f"  [{label}] patche dotknięte przez KTÓRYKOLWIEK z {N_DIRS} promieni: "
          f"{len(all_cells)}/{total} ({any_pct:.1f}%; WYGASZONE {any_dim_pct:.1f}%); przez "
          f"SAM promień #{EXAMPLE_RAY_IDX}: {len(ex_cells)}/{total} ({ex_pct:.1f}%; "
          f"WYGASZONE {ex_dim_pct:.1f}%)", flush=True)

    hm_smooth, smooth_lo, smooth_hi = render_density_heatmap_smooth(
        density_grid, crop_rgb, mask_cropped, d_axis_info)
    hm_full = render_density_heatmap_masked(density_grid, crop_rgb, d_axis_info, None, vmin, vmax)
    hm_any_ray = render_density_heatmap_masked(density_grid, crop_rgb, d_axis_info, all_mask, vmin, vmax)
    hm_one_ray = render_density_heatmap_masked(density_grid, crop_rgb, d_axis_info, ex_mask, vmin, vmax)

    img_a = hm_smooth   # BEZ promieni/siatki — czysty punkt odniesienia
    img_b = hm_full     # BEZ promieni (na prośbę użytkownika, 30.07) — tylko blokowa mapa
                        # patchy, ten sam punkt odniesienia co (a) ale w "surowym" stylu
                        # siatki, zamiast wygładzonym; promienie zostają w (c)/(d), gdzie
                        # są potrzebne do historii o pokryciu
    img_c = draw_rays(hm_any_ray, d_axis_info, n_dirs=N_DIRS, draw_all=True)
    img_d = draw_rays(hm_one_ray, d_axis_info, n_dirs=N_DIRS, draw_all=False,
                      single_ray_endpoint=ex_endpoint)

    fig, ax = plt.subplots(figsize=(0.55, 3.6))
    gradient = np.linspace(1, 0, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect="auto", cmap="jet", extent=[0, 1, vmin, vmax])
    ax.set_xticks([])
    ax.set_ylabel("wartość density (surowa, panele b/c/d)", fontsize=8)
    fig.tight_layout()
    colorbar_b64 = fig_to_b64(fig)

    # UWAGA: `display:flex` na kontenerze wierszy NIE renderował się side-by-side w
    # przeglądarce/viewerze użytym do sprawdzenia tego raportu (użytkownik, 30.07: "2x2"
    # wyszło jako "1x1 x2", czyli 4 wiersze po jednym obrazie) — DOKŁADNIE ten sam,
    # udokumentowany problem co przy Kroku 1 innego raportu (patrz project_state pamięci,
    # 2026-07-21: "Krok 1 was NOT actually rendering side-by-side despite correct-looking
    # display:flex HTML"). Sprawdzony tam skutecznie zamiennik: `display:inline-block;
    # vertical-align:top` NA KAŻDYM PANELU (nie na kontenerze) — stosowane tu teraz od razu,
    # zamiast ponownie ufać flexowi.
    def _panel(letter: str, title: str, img: np.ndarray, caption: str = "") -> str:
        cap_html = f'<br><span style="font-size:88%;">{caption}</span>' if caption else ""
        return (f'<div style="display:inline-block;vertical-align:top;width:460px;'
                f'margin:0 14px 14px 0;"><b>({letter}) {title}</b><br>'
                f'{img_tag(_b64_from_rgb(img, 440), style="width:440px;")}{cap_html}</div>')

    return f"""<section>
<h2>{label} <span style="font-size:65%;color:#666;">({cfg_path.name}, {run_dir.name})</span></h2>
<p style="font-size:90%;">{res['note']}. Siatka density: <b>{H_p}x{W_p} = {total} patchy</b>
(nad tym samym wyciętym fragmentem zdjęcia co druga rozdzielczość — różni się TYLKO gęstość
siatki). Patche dotknięte przez którykolwiek z {N_DIRS} promieni: <b>{len(all_cells)}/{total}
({any_pct:.1f}%)</b> — <b>{any_dim_pct:.1f}%</b> patchy ({total-len(all_cells)}) NIGDY nie
wchodzi do żadnego sygnału promienia i jest wygaszona na panelu (c). Sam promień
#{EXAMPLE_RAY_IDX}/{N_DIRS} (panel d, magenta) dotyka tylko <b>{len(ex_cells)}</b> patchy
(<b>{ex_dim_pct:.1f}%</b> wygaszone).</p>
<div>
{_panel('a', 'wygładzona mapa density (styl produkcyjny)', img_a,
        f'INTER_LINEAR, percentyl 2-98 nad maską (zakres: [{smooth_lo:.5f}, {smooth_hi:.5f}]) '
        '— BEZ promieni/siatki, punkt odniesienia')}
{_panel('b', 'blokowa mapa density (surowa, per-patch) — BEZ promieni', img_b,
        'ten sam punkt odniesienia co (a), tylko w "surowym", niewygładzonym stylu siatki '
        'patchy — promienie pojawiają się dopiero w (c)/(d), gdzie są potrzebne do '
        'historii o pokryciu')}
</div>
<div>
{_panel('c', 'wygaszone patche BEZ ŻADNEGO promienia', img_c,
        f'<b>wygaszone: {any_dim_pct:.1f}% ({total-len(all_cells)}/{total} patchy nigdy '
        'dotknięte przez żaden promień)</b>')}
{_panel('d', f'zostaje TYLKO promień #{EXAMPLE_RAY_IDX} (magenta)', img_d,
        f'<b>wygaszone: {ex_dim_pct:.1f}% ({total-len(ex_cells)}/{total} patchy poza tym '
        'jednym promieniem)</b>')}
<div style="display:inline-block;vertical-align:top;">{img_tag(colorbar_b64, style="height:300px;")}</div>
</div>
</section>"""


def main() -> None:
    print(f"Segmentacja + oś: {IMAGE_ID}", flush=True)
    cfg0 = load_merged_config(RESOLUTIONS[0]["cfg_path"], None)
    image_dir = Path(cfg0.data.image_dir)
    orig_rgb = np.array(PILImage.open(image_dir / IMAGE_ID).convert("RGB"), dtype=np.uint8)
    axis_info = detect_axis(orig_rgb, seg_params=cfg0.segmentation.as_params(),
                            nucleus_method=cfg0.segmentation.nucleus_method)
    if axis_info is None:
        raise RuntimeError("Segmentacja nie powiodła się")
    mask_arr = axis_info["mask"]
    model_input_rgb = (apply_background_mask(orig_rgb, mask_arr)
                       if cfg0.data.mask_background else orig_rgb)

    sections_html = "".join(
        process_resolution(res, orig_rgb, axis_info, mask_arr, model_input_rgb)
        for res in RESOLUTIONS
    )

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Promienie -> patche -> głosowanie ({VERSION}, {IMAGE_ID})</title>
<style>
body {{font-family:sans-serif;max-width:1500px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
h1 {{color:#1a237e;}} h2 {{color:#1a237e;}}
code {{background:#f0f0f5;padding:1px 4px;border-radius:3px;}}
</style>
</head>
<body>
<h1>Które patche naprawdę biorą udział w wykrywaniu kandydatów</h1>
<p>Otolit <code>{IMAGE_ID}</code>. Cztery obrazy (2×2) na TEJ SAMEJ mapie density: (a)
"wygładzona" mapa density — dokładnie taki styl, jaki widać w prawdziwych
kartach/raportach (percentyl 2-98 nad maską, gładkie INTER_LINEAR) — czysty punkt
odniesienia, BEZ żadnych promieni/siatki; (b) ta sama mapa, ale w "surowym", blokowym
stylu siatki patchy — też BEZ promieni, drugi punkt odniesienia (jak wygląda sygnał,
zanim w ogóle wejdzie w grę geometria promieni); (c) TERAZ dopiero pojawiają się
wszystkie {N_DIRS} promieni (jądro→kontur) w poprawionym, dobrze widocznym kolorze —
dotychczasowy ciemnoszary (110,110,110) ze <code>src/visualization.py</code> ginął na tle
żółto-zielonych patchy mapy JET (patrz zrzut ekranu w rozmowie) — i WYGASZONE
(przyciemnione, odbarwione do szarości) wszystkie patche, których ŻADEN promień nie
dotyka/nie przecina po drodze — zostają kolorowe tylko te, które faktycznie wchodzą do
sygnału jakiegokolwiek promienia (czyli mogą wpłynąć na głosowanie na kandydatów, patrz
<code>density_mechanism_v8.html</code> Krok 4a); (d) zostaje kolorowy TYLKO jeden,
przykładowy, pojedynczy promień (magenta) — pokazuje, jak WĄSKI jest zestaw patchy, które
naprawdę "widzi" jeden konkretny promień, w porównaniu do panelu (c). Pod (c) i (d) —
jawny procent WYGASZONYCH patchy w każdym przypadku. Powtórzone w dwóch rozdzielczościach
modelu (patrz sekcje niżej) na dokładnie tym samym wyciętym fragmencie zdjęcia.</p>
<p style="background:#e8f0ff;padding:8px;border-left:3px solid #2a4d8f;">
<b>Pytanie użytkownika (30.07): skoro tło jest wygaszone (alfa = 0 poza maską otolitu),
dlaczego na panelu (a) widać, jak sygnał "zapala się" tuż przy/poza konturem?</b>
Matematycznie NIC nie może być pokolorowane poza maską — <code>alpha = alpha * (maska
&gt; 0)</code> zeruje przezroczystość TWARDO, piksel po pikselu, więc żaden kolor
density nie trafia na prawdziwe tło. To, co widać, dzieje się tuż PRZY granicy maski, z
dwóch powodów naraz:
<ol style="margin:6px 0 0;">
<li><b>Artefakt brzegowy/register-token (znany z wcześniejszych sesji tego projektu)</b> —
patche siedzące na SAMEJ granicy segmentacji (sąsiadujące z tłem) dostają nienaturalnie
wysoką "surową" wartość z samej architektury ViT (dlatego backbone to
<code>dinov2_vits14_reg</code> — wariant z rejestrami, który ma TŁUMIĆ dokładnie ten
efekt, nie eliminować go do zera). To nie jest prawdziwy sygnał biologiczny, tylko
artefakt reprezentacji na styku tkanka/tło.</li>
<li><b>Sam kontur (cyjanowa linia) ma realną GRUBOŚĆ pikselową</b> (kilka px) i jest
rysowany NA WIERZCHU już wyrenderowanej mapy — połowa tej grubości leży wewnątrz maski,
połowa na zewnątrz. Kolorowy patch, który jest wciąż (ledwie) WEWNĄTRZ prawdziwej maski,
może więc optycznie wyglądać, jakby "wystawał poza" cyjanową linię, bo sama linia
zakrywa/graniczy z ostatnimi pikselami maski.</li>
</ol>
Segmentacja metodą <code>radial</code> (<code>src/otolith_axis.py::_segment_radial</code>)
celowo sięga do "wygasającego, cienkiego rąbka" otolitu (próg = ułamek jasności WŁASNEGO
promienia, nie sztywna wartość) — więc maska bywa nieco szersza niż to, co ludzkie oko
od razu nazwałoby "wyraźnie otolitem", co też przyczynia się do wrażenia sygnału "tuż przy
granicy".</p>
{sections_html}
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
