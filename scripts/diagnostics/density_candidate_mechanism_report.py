"""29.07 (follow-up) — DENSITY-ONLY deep dive: dlaczego kandydat na przyrost ląduje
DOKŁADNIE w tym, a nie innym miejscu na mapie density modelu. Klasyczna metoda jest
tu celowo pominięta (użytkownik: "metodę klasyczną w tym zestawieniu pomijamy").

Różnica względem candidate_selection_walkthrough_report.py: TAMTEN raport (sekcja G,
Krok 1) rysuje kandydatów z ``dp_walkthrough_data(grid, ...)``, gdzie ``grid`` to
mapa density z TEGO SAMEGO forward passu co głowica CORAL — 518px, BEZ przycięcia do
otolitu. To NIE jest mapa, której naprawdę używa wykrywanie kandydatów na kartach
raportu: ``run_pipeline._compute_axis_data_for_samples`` liczy dla density DRUGI,
wysokorozdzielczy forward pass (``candidates.density_image_size=728``,
``density_crop_to_otolith=true``) i TĘ mapę podaje do ``select_increments`` /
``density_peaks``. Ten skrypt używa właśnie TEJ drugiej, prawdziwej mapy — inaczej
liczby pokazane użytkownikowi nie zgadzałyby się z tym, co faktycznie ląduje na
kartach.

Pokazuje na tym samym otolicie co poprzedni raport:
  1. Sama mapa density: kształt siatki, PRAWDZIWY zakres wartości (nie [0,1] tylko
     do czytelności), siatka patchy.
  2. Wszystkie 48 promieni, surowy sygnał (przed normalizacją) — różne skale
     bezwzględne między promieniami + "schodkowy" kształt (nearest-neighbour
     próbkowanie grubej siatki).
  3. Ten sam sygnał PO normalizacji per-promień — na TYM, nie na wartościach
     bezwzględnych, algorytm szuka pików.
  4. Klastrowanie/głosowanie: mapa 2D promień×t — surowe piki KAŻDEGO z 48 promieni
     osobno, potem to samo z wykrytymi klastrami jako poziome pasma (Krok 4a),
     przestrzennie KTÓRE promienie głosują na który klaster (Krok 4b), JEDEN
     przykładowy promień krok po kroku z prawdziwymi liczbami — surowe wartości →
     normalizacja → wykrywanie piku (prominence/min_distance) → przesunięcie do
     "falling edge" (Krok 4c), zbliżenie na dokładne miejsce kropki (Krok 4d).
  5. Wszyscy zaakceptowani kandydaci (WYŁĄCZNIE density) na zdjęciu i na mapie.
  6. Eksperyment (30 kart): wymuszenie dokładnie `wiek` pików na KAŻDYM promieniu
     zamiast progu prominencji — pomaga wyraźnie, gdy CORAL trafia wiek; neutralne,
     gdy się myli. Wyniki z osobnego sweep_forced_topk_peaks.py, wklejone tu na stałe.

Klastrowanie w Kroku 4 (od v5, 29.07) używa ARC-AWARE ``_cluster_by_radius_with_arcs``
+ ``_cluster_score`` — DOKŁADNIE tego, czego naprawdę używa produkcyjny
``select_increments``. Wcześniejsze wersje (v1-v4) świadomie używały prostszego,
support-based ``_cluster_by_radius`` (sama liczba głosów, bez premii za zwartość
promieni) — żeby przykładowy promień/klaster nie zmieniał się bez wyraźnej prośby.
Przełączono na arc-aware po tym, jak użytkownik na realnym zdjęciu zaznaczył czerwonym
flamastrem wyraźnie widoczną strukturę (Wykres 1c, v4) i zapytał, czemu wygrywa inny,
gorzej wyglądający klaster — okazało się, że wskazana struktura WYGRYWA przy prawdziwym
(arc-aware) liczeniu, a poprzedni "zwycięski" klaster w tym raporcie był artefaktem
uproszczonego liczenia. Świadoma, potwierdzona przez użytkownika zmiana (nie przy okazji
czegoś innego) — patrz Krok 4a, akapit "Historia tej zmiany".

WERSJONOWANIE (od 29.07, na prośbę użytkownika): każda iteracja tego raportu jest
zapisywana pod NOWĄ nazwą pliku (``density_mechanism_v1.html``, ``_v2.html``, ...) —
poprzednie wersje zostają na dysku do porównania, nic nie jest nadpisywane. Zmienna
``VERSION`` niżej to jedyne miejsce do podbicia przy kolejnej iteracji. Każda iteracja
zmienia TYLKO fragmenty wskazane przez użytkownika, nic więcej.

Nie trenuje ani nie modyfikuje żadnego kodu w src/.

Usage: python scripts/diagnostics/density_candidate_mechanism_report.py
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
import matplotlib.cm as cm
import numpy as np
import torch
import cv2
from PIL import Image as PILImage
from scipy.signal import find_peaks

from src.report_common import fig_to_b64, img_tag, png_to_b64
from src.visualization import (render_rays_and_candidates, draw_nucleus_exclusion_zone,
                               _CONTOUR_COLOR)
from src.otolith_axis import (detect_axis, apply_background_mask, mask_bbox,
                              shift_axis_info, sample_profile_along_axis)
from src.dataset import build_transforms
from src.inference import load_model_from_checkpoint
from src.ring_extraction import (density_peaks, _cluster_by_radius_with_arcs,
                                 _cluster_score, _shift_peak_to_falling_edge)
from scripts.run_pipeline import load_merged_config

RUN_DIR = PROJECT_ROOT / "outputs" / "28.07_e9_w0.1"
CFG_PATH = PROJECT_ROOT / "configs" / "config_e9_w0.1.yaml"
CKPT = RUN_DIR / "checkpoints" / "embedded" / "best.pt"
IMAGE_ID = "2023_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg"

OUT_DIR = PROJECT_ROOT / "outputs" / "29.07_candidate_selection_walkthrough"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION = "v8"   # podbić przy KAŻDEJ kolejnej iteracji — nic nie nadpisujemy
OUT_PATH = OUT_DIR / f"density_mechanism_{VERSION}.html"
# Krok 6: obrazy + statystyki z osobnego eksperymentu na 30 kartach
# (scripts/diagnostics/sweep_forced_topk_peaks.py, uruchomiony 29.07) — NIE przeliczane
# tutaj przy każdym uruchomieniu (30 kart × 2 warianty = 60 forward passów, za wolne dla
# iteracji nad resztą raportu skupionego na JEDNYM otolicie). Liczby niżej wklejone z
# tamtego uruchomienia — jeśli sweep się zmieni, trzeba je tu ręcznie odświeżyć.
FORCED_TOPK_DIR = OUT_DIR / "forced_topk_experiment"

N_DIRS, N_SAMPLES = 48, 64


def _b64_from_rgb(arr: np.ndarray, target_w: int = 480) -> str:
    img = PILImage.fromarray(np.ascontiguousarray(arr[..., :3]).astype(np.uint8))
    if img.width > target_w:
        scale = target_w / img.width
        img = img.resize((target_w, max(1, int(round(img.height * scale)))), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_density_heatmap(grid: np.ndarray, crop_rgb: np.ndarray, axis_info: dict,
                           p_lo: float = 1.0, p_hi: float = 98.0
                           ) -> tuple[np.ndarray, float, float]:
    """Nearest-neighbour upsample (honest — NOT smoothed) of the raw grid to image
    resolution, blended over the cropped photo, plus patch-grid lines.

    Colour mapping uses a ROBUST (percentile) range, not the true global min/max —
    this grid typically has one or a few extreme outlier patches (register-token /
    edge artifacts) many times larger than the rest of the signal; a plain linear
    min-max scale crushes the entire otolith body to a single uniform colour and
    hides all real structure (found empirically while building this report — see
    the accompanying text). Returns (rgb_overlay, disp_vmin, disp_vmax) — the
    CLIPPED range actually used for colours (caller reports true min/max separately).
    """
    H, W = crop_rgb.shape[:2]
    H_p, W_p = grid.shape
    vmin = float(np.percentile(grid, p_lo))
    vmax = float(np.percentile(grid, p_hi))
    if vmax <= vmin:
        vmin, vmax = float(grid.min()), float(grid.max())
    norm = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0) if vmax > vmin else np.zeros_like(grid)
    up = cv2.resize(norm.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    uint8_map = (up * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
    rgb_map = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    blended = (0.6 * rgb_map + 0.4 * crop_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    for i in range(1, W_p):
        x = int(round(i * W / W_p))
        cv2.line(blended, (x, 0), (x, H), (60, 60, 60), 1, cv2.LINE_AA)
    for i in range(1, H_p):
        y = int(round(i * H / H_p))
        cv2.line(blended, (0, y), (W, y), (60, 60, 60), 1, cv2.LINE_AA)
    if axis_info.get("contour") is not None:
        cv2.drawContours(blended, [axis_info["contour"]], -1, (0, 230, 230), max(2, H // 300))
    return blended, vmin, vmax


def _t_to_px(t: float, cx: float, cy: float, fx: float, fy: float) -> tuple[float, float]:
    return (cx + t * (fx - cx), cy + t * (fy - cy))


def _pt_seg_dist(px, py, x0, y0, x1, y1) -> tuple[float, float]:
    """Distance (in the same units as the inputs) from point (px,py) to the segment
    (x0,y0)-(x1,y1), plus t (fraction along the segment of the closest point)."""
    dxv, dyv = x1 - x0, y1 - y0
    L2 = dxv * dxv + dyv * dyv
    tt = float(np.clip(((px - x0) * dxv + (py - y0) * dyv) / L2, 0, 1)) if L2 > 1e-9 else 0.0
    projx, projy = x0 + tt * dxv, y0 + tt * dyv
    return float(np.hypot(px - projx, py - projy)), tt


def render_ray_zoom(base_img: np.ndarray, crop_box: tuple[int, int, int, int],
                    W_p: int, H_p: int, cw: int, ch: int,
                    cx: float, cy: float, fx: float, fy: float,
                    markers: list[tuple[float, str, tuple, str]],
                    grid: np.ndarray | None = None,
                    hot_threshold: float | None = None, disp_w: int = 720) -> np.ndarray:
    """Tight, "forensic" crop around a fragment of ONE ray: real patch-grid boundaries,
    a THIN, high-contrast (MAGENTA — never appears in JET, unlike red) ray line, thin
    candidate markers (so patch-centre-vs-boundary is visible). NO printed numbers —
    tried that, unreadable at this density (user report); colour + a thin border on
    cells at/above ``hot_threshold`` carries the same information legibly.
    ``markers`` = list of ``(t, label, bgr_color, "filled"|"open")``.
    """
    x0, y0, x1, y1 = crop_box
    img = np.ascontiguousarray(base_img[y0:y1, x0:x1, :3]).copy()
    scale = max(1, int(round(disp_w / max(1, x1 - x0))))
    if scale > 1:
        img = cv2.resize(img, ((x1 - x0) * scale, (y1 - y0) * scale), interpolation=cv2.INTER_NEAREST)
    Hc, Wc = img.shape[:2]

    def to_disp(px, py):
        return int(round((px - x0) * scale)), int(round((py - y0) * scale))

    patch_w, patch_h = cw / W_p, ch / H_p
    for i in range(int(x0 / patch_w) - 1, int(x1 / patch_w) + 2):
        xg = i * patch_w
        if x0 <= xg <= x1:
            dx, _ = to_disp(xg, y0)
            cv2.line(img, (dx, 0), (dx, Hc), (130, 130, 130), 1, cv2.LINE_AA)
    for j in range(int(y0 / patch_h) - 1, int(y1 / patch_h) + 2):
        yg = j * patch_h
        if y0 <= yg <= y1:
            _, dy = to_disp(x0, yg)
            cv2.line(img, (0, dy), (Wc, dy), (130, 130, 130), 1, cv2.LINE_AA)

    if grid is not None and hot_threshold is not None:
        for j in range(max(0, int(y0 / patch_h)), min(H_p, int(y1 / patch_h) + 1)):
            for i in range(max(0, int(x0 / patch_w)), min(W_p, int(x1 / patch_w) + 1)):
                ccx, ccy = (i + 0.5) * patch_w, (j + 0.5) * patch_h
                if x0 <= ccx <= x1 and y0 <= ccy <= y1 and grid[j, i] >= hot_threshold:
                    cell_x0, _ = to_disp(i * patch_w, ccy)
                    cell_x1, _ = to_disp((i + 1) * patch_w, ccy)
                    _, cell_y0 = to_disp(ccx, j * patch_h)
                    _, cell_y1 = to_disp(ccx, (j + 1) * patch_h)
                    cv2.rectangle(img, (cell_x0 + 1, cell_y0 + 1), (cell_x1 - 1, cell_y1 - 1),
                                 (255, 230, 0), max(1, scale // 3), cv2.LINE_AA)

    p0, p1 = to_disp(cx, cy), to_disp(fx, fy)
    cv2.line(img, p0, p1, (255, 0, 255), max(1, scale // 3), cv2.LINE_AA)

    for (t, label, color, style) in markers:
        px, py = _t_to_px(t, cx, cy, fx, fy)
        dx, dy = to_disp(px, py)
        r = max(3, scale)
        if style == "filled":
            cv2.circle(img, (dx, dy), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (dx, dy), r, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (dx, dy), r, color, max(1, scale // 3), cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (min(Wc - 8, dx + r + 4), dy), cv2.FONT_HERSHEY_SIMPLEX,
                       0.42, color, 1, cv2.LINE_AA)
    return img


def render_vote_overlay(image: np.ndarray, axis_info: dict,
                        colored_memberships: list[tuple[dict, tuple]],
                        highlight_ray: int | None = None,
                        n_dirs: int = 48, inner_margin: float = 0.05) -> np.ndarray:
    """„Kto głosuje" — WSZYSTKIE ``n_dirs`` promienie narysowane od jądra, ale kolorowane
    tylko te, które należą do jednego z wybranych klastrów (2-3, żeby nie zaśmiecić
    obrazu) — reszta promieni w ogóle nie jest rysowana. Gruba linia = promień w
    ZWYCIĘSKIM CIĄGŁYM ŁUKU tego klastra (najsilniejszy dowód — sąsiadujące promienie
    zgadzają się), cienka = promień też głosuje na ten klaster, ale spoza tego zwartego
    łuku (rozrzucone, słabsze wsparcie). Prawie kopia
    ``src.visualization.render_arc_cluster_overlay``, ale z WŁASNYM, jawnym kolorem per
    klaster (nie cykliczną paletą) — żeby dało się dopasować do konwencji tego raportu
    (zielony=zwycięski, żółty=drugi klaster), a nie żeby klaster #1 zawsze wychodził
    na czerwono (co gryzie się z gorącym końcem JET, patrz reszta tego raportu).
    ``colored_memberships`` = [(membership_dict, (r,g,b)), ...].
    """
    img = np.ascontiguousarray(image[..., :3]).copy()
    H = img.shape[0]
    if axis_info is None:
        return img
    draw_nucleus_exclusion_zone(img, axis_info, inner_margin=inner_margin, n_dirs=n_dirs)
    contour = axis_info.get("contour")
    centroid = axis_info.get("centroid")
    if contour is None or centroid is None:
        return img
    cx, cy = int(centroid[0]), int(centroid[1])
    cpts = contour.reshape(-1, 2)
    idx_sel = np.linspace(0, len(cpts) - 1, min(n_dirs, len(cpts)), dtype=int)
    thin_lt = 1
    thick_lt = max(3, H // 150)
    for membership, color in colored_memberships:
        for ray_idx in membership.get("other_rays", ()):
            if 0 <= ray_idx < len(idx_sel):
                pt = cpts[idx_sel[ray_idx]]
                overlay = img.copy()
                cv2.line(overlay, (cx, cy), (int(pt[0]), int(pt[1])), color, thin_lt, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, dst=img)
        for ray_idx in membership.get("arc_rays", ()):
            if 0 <= ray_idx < len(idx_sel):
                pt = cpts[idx_sel[ray_idx]]
                cv2.line(img, (cx, cy), (int(pt[0]), int(pt[1])), color, thick_lt, cv2.LINE_AA)
    if highlight_ray is not None and 0 <= highlight_ray < len(idx_sel):
        pt = cpts[idx_sel[highlight_ray]]
        cv2.line(img, (cx, cy), (int(pt[0]), int(pt[1])), (255, 0, 255), max(1, H // 400), cv2.LINE_AA)
    cv2.drawContours(img, [contour], -1, _CONTOUR_COLOR, max(2, H // 300))
    return img


def render_patches_true_geometry(image: np.ndarray, axis_info: dict,
                                 points_colors: list[tuple[tuple[float, float], tuple]],
                                 cw: int, ch: int, W_p: int, H_p: int,
                                 n_dirs: int = 48, inner_margin: float = 0.05,
                                 highlight_ray: int | None = None) -> np.ndarray:
    """Wykres 1b/1c's companion in TRUE otolith geometry (each ray its own real length —
    the otolith isn't a circle). Fills the WHOLE PATCH CELL a peak falls in, not a dot at
    its exact pixel — matches Krok 1's "one number per patch, not per pixel" story, and
    reads far more clearly than full-length ray lines once several clusters overlap
    (user report, 29.07: the all-rays version was an unreadable starburst that didn't
    correspond legibly to what the polar chart showed). ``points_colors`` =
    ``[((x, y), (b, g, r)), ...]`` in the SAME crop pixel frame as ``axis_info``.
    ``highlight_ray`` (ORIGINAL contour-order index, not the display 0-47 one) draws that
    one ray thicker in magenta, so it's identifiable across every chart in this section
    (user report, 29.07: "zaznacz też ten promień, który rozpatrujemy")."""
    img = render_rays_and_candidates(image, axis_info, None, None,
                                     n_dirs=n_dirs, inner_margin=inner_margin)
    patch_w, patch_h = cw / W_p, ch / H_p
    for (x, y), color in points_colors:
        col = int(np.clip(x / cw * W_p, 0, W_p - 1))
        row = int(np.clip(y / ch * H_p, 0, H_p - 1))
        x0, x1 = int(col * patch_w), int((col + 1) * patch_w)
        y0, y1 = int(row * patch_h), int((row + 1) * patch_h)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)
    if highlight_ray is not None and axis_info.get("contour") is not None:
        H = img.shape[0]
        cpts = axis_info["contour"].reshape(-1, 2)
        idx_sel_local = np.linspace(0, len(cpts) - 1, min(n_dirs, len(cpts)), dtype=int)
        if 0 <= highlight_ray < len(idx_sel_local):
            cx_h, cy_h = axis_info["centroid"]
            pt = cpts[idx_sel_local[highlight_ray]]
            cv2.line(img, (int(cx_h), int(cy_h)), (int(pt[0]), int(pt[1])),
                    (255, 0, 255), max(2, H // 300), cv2.LINE_AA)
    return img


def _section_krok6_forced_topk() -> str:
    """Krok 6 — podsumowanie osobnego eksperymentu (użytkownik, 29.07): „co jeśli
    wymusimy na KAŻDYM promieniu znalezienie dokładnie `wiek` (z CORAL) pików density,
    zamiast dzisiejszego progu prominencji, który daje zmienną liczbę pików/promień?".
    Dane z ``scripts/diagnostics/sweep_forced_topk_peaks.py`` (30 kart, best+worst,
    outputs/28.07_e9_w0.1) — patrz stała FORCED_TOPK_DIR i komentarz przy niej."""
    card10_b64 = png_to_b64(FORCED_TOPK_DIR / "card10.png")
    card20_b64 = png_to_b64(FORCED_TOPK_DIR / "card20.png")
    imgs_html = ""
    if card10_b64 and card20_b64:
        imgs_html = (
            '<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;'
            'justify-content:center;">'
            f'<div style="max-width:380px;"><b>Karta #10 — wiek trafny (model=prawda=1), '
            'BITS1q</b><br>'
            f'{img_tag(card10_b64, style="width:380px;")}<br>'
            '<span style="font-size:88%;">dist_base=375px → dist_forced=165px '
            '(<b>-56%</b>)</span></div>'
            f'<div style="max-width:380px;"><b>Karta #20 — wiek NIEtrafny '
            '(model=1, prawda=6!), BIAS</b><br>'
            f'{img_tag(card20_b64, style="width:380px;")}<br>'
            '<span style="font-size:88%;">dist_base=58px → dist_forced=90px '
            '(<b>+55%, pogorszenie</b>)</span></div>'
            '</div>'
            '<p style="font-size:88%;">Cyjan = klasyczna referencja (jasność obrazu, '
            'niezależna od modelu). Czerwony X = finał dzisiejszej metody (próg '
            'prominencji). Zielony + = finał wymuszonego top-<code>wiek</code> na promień. '
            'Karta #10: baseline trafił na słaby, peryferyjny sygnał bez struktury '
            '(czerwony X nisko na zdjęciu); forced trafił bliżej jądra, bliżej '
            'referencji. Karta #20: model pomylił się drastycznie co do wieku (kazał '
            'szukać 1 pierścienia na otolicie, gdzie widać 4-5!) — wymuszony wariant, bez '
            'progu pewności, zbiegł się na silnym, ale ZŁYM (zbyt peryferyjnym) sygnale; '
            'baseline trafił lepiej tu praktycznie przez przypadek (próg prominencji '
            'odsiał konkurentów).</p>'
        )
    return f"""<section><h2>Krok 6 — eksperyment: czy pomaga wymuszenie dokładnie „wiek"
pików na KAŻDYM promieniu?</h2>
<p>Pytanie użytkownika (29.07): zamiast dzisiejszego progu prominencji (zmienna liczba
pików na promień, 0 do kilku), co jeśli KAŻDY z 48 promieni ma znaleźć dokładnie
<code>wiek</code> (z głowicy CORAL) najlepszych lokalnych maksimów, niezależnie od tego,
czy przebijają próg pewności? Sprawdzone na tej samej próbie 30 kart
(<code>outputs/28.07_e9_w0.1</code>), tą samą metryką co <code>_localization_quality</code>
(<code>mean_dist_final_to_classical_px</code>). Reszta pipeline'u (klastrowanie arc-aware
+ wybór DP) bez zmian — jedyna różnica to sposób wykrywania pików na promieniu.</p>
<h3>Wynik zbiorczy (30 kart, n=21 z ważną metryką)</h3>
<table>
<tr><th>Grupa</th><th>n kart</th><th>BASELINE (dziś)</th><th>WYMUSZONE top-wiek</th></tr>
<tr><td>Wszystkie</td><td>21</td><td>260.7px</td><td>249.1px</td></tr>
<tr><td>Wiek modelu TRAFNY (|błąd|≤1)</td><td>13</td><td colspan="2">
średnia poprawa: <b style="color:#1ec81e;">+20.8px</b> (WYMUSZONE lepsze)</td></tr>
<tr><td>Wiek modelu NIEtrafny (|błąd|≥2)</td><td>8</td><td colspan="2">
średnia poprawa: <b style="color:#b03060;">-3.1px</b> (praktycznie bez zmian)</td></tr>
<tr><td>Klasa wieku 0-6</td><td>17</td><td>260.9px</td><td><b style="color:#1ec81e;">246.1px</b></td></tr>
<tr><td>Klasa wieku 7+</td><td>4</td><td>260.0px</td><td>261.7px</td></tr>
</table>
<p><b>To dokładnie potwierdza Twoją intuicję</b> — mechanizm pomaga WYRAŹNIE, kiedy CORAL
trafia wiek (+20.8px śr. poprawy na 13 kartach), a jest praktycznie neutralny, kiedy CORAL
się myli (-3.1px na 8 kartach, szum w obie strony). W najliczniejszej, najlepiej
wytrenowanej klasie wieku 0-6 (17/21 kart z ważną metryką) — realna poprawa, ~5.7%. To
dobry znak: to NIE jest metoda, która przypadkiem naprawia złe predykcje wieku (nie ma
magii) — to metoda, która, gdy dostanie poprawny budżet, konsekwentnie znajduje LEPSZE
miejsce na osi. Dwie karty poniżej pokazują ten mechanizm bezpośrednio na zdjęciach.</p>
{imgs_html}
<p style="font-size:88%;background:#fff3cd;padding:8px;border-left:3px solid #b8860b;">
<b>Zastrzeżenia:</b> n=21 to mała próba (9/30 kart nie miało wykrytej klasycznej
referencji na żadnym wariancie); wynik per-karta ma duży rozrzut (nie każda „trafna"
karta się poprawia); to NIE jest jeszcze decyzja o wdrożeniu do <code>src/</code> — kod
eksperymentu (<code>scripts/diagnostics/sweep_forced_topk_peaks.py</code>) jest
samodzielny, nic w produkcyjnym pipeline nie zostało zmienione.</p>
</section>"""


def main() -> None:
    print("Wczytywanie configu i modelu...", flush=True)
    cfg = load_merged_config(CFG_PATH, None)
    image_dir = Path(cfg.data.image_dir)
    device = torch.device(cfg.training.device if cfg.training.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_from_checkpoint(cfg, CKPT)
    model.eval()
    device = next(model.parameters()).device

    print(f"Segmentacja + oś: {IMAGE_ID}", flush=True)
    orig_rgb = np.array(PILImage.open(image_dir / IMAGE_ID).convert("RGB"), dtype=np.uint8)
    axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                            nucleus_method=cfg.segmentation.nucleus_method)
    if axis_info is None:
        raise RuntimeError("Segmentacja nie powiodła się")
    mask_arr = axis_info["mask"]
    model_input_rgb = (apply_background_mask(orig_rgb, mask_arr)
                       if cfg.data.mask_background else orig_rgb)

    # --- Prawdziwa, WYSOKOROZDZIELCZA mapa density (dokładnie ta, której używa
    # select_increments/density_peaks na kartach raportu — NIE ta z sekcji G). ---
    print("Liczenie wysokorozdzielczej mapy density (drugi forward pass, przycięty "
          "do bboxa otolitu)...", flush=True)
    crop_x0, crop_y0, cw, ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)
    crop_rgb = model_input_rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    d_axis_info = shift_axis_info(axis_info, -crop_x0, -crop_y0)
    density_transform = build_transforms(cfg.candidates.density_image_size, "test")
    density_tensor = density_transform(PILImage.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        density_grid = model.get_density_probs(density_tensor).squeeze(0).cpu().numpy()
    H_p, W_p = density_grid.shape
    print(f"  Siatka: {H_p}x{W_p} = {H_p*W_p} patchy, wartości density w [{density_grid.min():.5f}, "
          f"{density_grid.max():.5f}], suma={density_grid.sum():.2f} (≈ wiek).", flush=True)

    min_dist = cfg.candidates.min_peak_distance
    prominence = cfg.candidates.prominence_threshold
    inner_margin = cfg.candidates.inner_margin

    dpk, dpts_crop = density_peaks(density_grid, d_axis_info, ch, cw, n_dirs=N_DIRS,
                                   n_samples=N_SAMPLES, min_distance=min_dist,
                                   prominence=prominence, inner_margin=inner_margin)
    dpts_full = [(x + crop_x0, y + crop_y0) for (x, y) in dpts_crop]
    dpk_full = [(t, s, x + crop_x0, y + crop_y0, r) for (t, s, x, y, r) in dpk]
    print(f"  {len(dpk)} kandydatów (density) na {N_DIRS} promieniach.", flush=True)

    # --- Panel 1: mapa density sama w sobie — punkt startowy, PRZED jakąkolwiek logiką
    # wykrywania kandydatów. Pokazujemy 48 promieni (żeby zapowiedzieć Krok 2/3), ale
    # BEZ kropek-kandydatów (density_pts=None/classical_pts=None poniżej) — na tym
    # etapie kandydaci jeszcze nie istnieją. ---
    print("Renderowanie Kroku 1 (mapa density)...", flush=True)
    heatmap_rgb, vmin, vmax = render_density_heatmap(density_grid, crop_rgb, d_axis_info)
    krok1_img = render_rays_and_candidates(heatmap_rgb, d_axis_info, None, None,
                                           n_dirs=N_DIRS, inner_margin=inner_margin)
    fig, ax = plt.subplots(figsize=(0.6, 4))
    gradient = np.linspace(1, 0, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect="auto", cmap="jet",
             extent=[0, 1, vmin, vmax])
    ax.set_xticks([])
    ax.set_ylabel("wartość density (surowa)", fontsize=9)
    fig.tight_layout()
    colorbar_b64 = fig_to_b64(fig)

    # --- Panel 2/3: wszystkie 48 promieni, surowe i znormalizowane ---
    print("Renderowanie Kroku 2 (wszystkie 48 promieni, surowe wartości)...", flush=True)
    cx, cy = d_axis_info["centroid"]
    contour = d_axis_info["contour"].reshape(-1, 2)
    idx_sel = np.linspace(0, len(contour) - 1, min(N_DIRS, len(contour)), dtype=int)
    cmap = cm.get_cmap("hsv")

    raw_profiles, norm_profiles, cell_counts = [], [], []
    for ray_i, ci in enumerate(idx_sel):
        cpt = (int(contour[ci][0]), int(contour[ci][1]))
        prof, _line_xy = sample_profile_along_axis(density_grid, (cx, cy), cpt, ch, cw,
                                                     n_samples=N_SAMPLES)
        raw_profiles.append(prof)
        rng = float(prof.max() - prof.min())
        norm_profiles.append((prof - prof.min()) / rng if rng > 1e-6 else None)
        xs = np.linspace(cx, cpt[0], N_SAMPLES)
        ys = np.linspace(cy, cpt[1], N_SAMPLES)
        px = np.clip((xs / max(cw, 1) * W_p).astype(np.int64), 0, W_p - 1)
        py = np.clip((ys / max(ch, 1) * H_p).astype(np.int64), 0, H_p - 1)
        cell_counts.append(len(set(zip(px.tolist(), py.tolist()))))

    t_axis = np.linspace(0, 1, N_SAMPLES)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for ray_i, prof in enumerate(raw_profiles):
        ax1.plot(t_axis, prof, color=cmap(ray_i / N_DIRS), lw=0.9, alpha=0.85)
    ax1.axvspan(0, inner_margin, color="#dc0000", alpha=0.12, zorder=0)
    ax1.set_title("Surowe wartości density (48 promieni, kolor=kierunek)", fontsize=9)
    ax1.set_xlabel("t (jądro→kontur)", fontsize=9)
    ax1.set_ylabel("wartość density (surowa)", fontsize=9)
    for ray_i, prof in enumerate(norm_profiles):
        if prof is not None:
            ax2.plot(t_axis, prof, color=cmap(ray_i / N_DIRS), lw=0.9, alpha=0.85)
    ax2.axvspan(0, inner_margin, color="#dc0000", alpha=0.12, zorder=0)
    ax2.set_title("PO normalizacji per-promień (na TYM szuka pików)", fontsize=9)
    ax2.set_xlabel("t (jądro→kontur)", fontsize=9)
    ax2.set_ylabel("wartość znorm. [0,1]", fontsize=9)
    fig.tight_layout()
    rays48_b64 = fig_to_b64(fig)
    mean_cells = float(np.mean(cell_counts))
    outlier_ray = int(np.argmax([float(p.max()) for p in raw_profiles]))
    outlier_val = float(raw_profiles[outlier_ray].max())
    typical_val = float(np.median([float(p.max()) for p in raw_profiles]))

    # --- Panel 4: JEDEN promień krok po kroku. WERSJA v5 (29.07): PRZEŁĄCZONE na
    # arc-aware klastrowanie (_cluster_by_radius_with_arcs + _cluster_score) — to jest
    # DOKŁADNIE to, czego naprawdę używa produkcyjny select_increments. Poprzednie wersje
    # (v1-v4) celowo używały prostszego, support-based _cluster_by_radius (tylko liczba
    # głosów), żeby przykładowy promień się nie zmieniał — ale użytkownik na realnym
    # zdjęciu (czerwony flamaster na Wykresie 1c, 29.07) wskazał klaster t≈0.852, który
    # wygrywa właśnie przy arc-aware (score=13.08, #1) a przegrywa przy support-based
    # (support=18<21). To jednoznacznie pokazało, że sam support-based walkthrough daje
    # MYLĄCY obraz tego, co produkcja naprawdę wybiera — więc świadoma, potwierdzona
    # zmiana (nie przypadkowa, jak poprzednio). ---
    print("Wybór przykładowego promienia i renderowanie Kroku 4 (arc-aware)...", flush=True)
    t_tol = 0.06
    clusters = _cluster_by_radius_with_arcs(dpk, t_tol=t_tol, n_dirs=N_DIRS)
    if clusters:
        best_cluster = max(clusters, key=_cluster_score)
        example = min(dpk, key=lambda p: abs(p[0] - best_cluster[0]))
    else:
        best_cluster = None
        example = max(dpk, key=lambda p: p[1]) if dpk else None

    # BUG znaleziony przez użytkownika (29.07, na podstawie Wykresu 1a — żółty klaster
    # "wyglądał" na obejmujący więcej promieni niż zielony, mimo że zielony ma wyższe
    # official support=21>18): ta funkcja liczyła przynależność NIEZALEŻNIE dla każdego
    # klastra ("czy ten pik jest w promieniu t_tol OD TEGO ŚRODKA") — gdy dwa środki
    # klastrów leżą blisko siebie (< 2×t_tol, częste przy wielu klastrach na 48
    # promieniach), ten sam pik wpadał do OBU list, więc rysowane prostokąty/łuki
    # pokazywały więcej promieni niż klaster faktycznie "wygrał". _cluster_by_radius
    # przypisuje każdy pik WYŁĄCZNIE do NAJBLIŻSZEGO środka (nie do każdego środka w
    # zasięgu) — poniższa wersja odtwarza dokładnie tę samą, wyłączną zasadę.
    _modes_arr = np.array([c[0] for c in clusters]) if clusters else np.array([])
    _peak_ts_all = np.array([p[0] for p in dpk]) if dpk else np.array([])
    _nearest_mode = (np.abs(_peak_ts_all[:, None] - _modes_arr[None, :]).argmin(axis=1)
                     if len(_modes_arr) and len(_peak_ts_all) else np.array([], dtype=int))

    def _cluster_members(mean_t: float) -> set[int]:
        if not len(_modes_arr):
            return set()
        ci = int(np.argmin(np.abs(_modes_arr - mean_t)))
        return {int(dpk[pi][4]) for pi in range(len(dpk))
               if _nearest_mode[pi] == ci and abs(dpk[pi][0] - _modes_arr[ci]) <= t_tol}

    winning = ({"mean_t": best_cluster[0], "support": best_cluster[1],
               "arc_len": best_cluster[3], "score": _cluster_score(best_cluster),
               "arc_rays": _cluster_members(best_cluster[0]), "other_rays": set()}
              if best_cluster else None)
    other = None   # ustalane niżej, po znalezieniu 2. piku naszego przykładowego promienia
    ex_ray = None

    if example is not None:
        ex_t, ex_strength, ex_x_full, ex_y_full, ex_ray = example
        ex_ci = idx_sel[ex_ray]
        ex_cpt = (int(contour[ex_ci][0]), int(contour[ex_ci][1]))
        ex_raw = raw_profiles[ex_ray]
        ex_norm = norm_profiles[ex_ray]
        idxs, props = find_peaks(ex_norm, distance=max(1, int(min_dist)),
                                 prominence=float(prominence))
        prominences = props.get("prominences", [])

        fig, (axr, axn) = plt.subplots(1, 2, figsize=(11, 3.2))
        axr.plot(t_axis, ex_raw, color="#2a4d8f", lw=1.8, drawstyle="steps-mid")
        axr.fill_between(t_axis, ex_raw.min(), ex_raw, step="mid", color="#2a4d8f", alpha=0.15)
        axr.set_title(f"Promień {ex_ray+1}/48 — SUROWE wartości ({cell_counts[ex_ray]} "
                      f"różnych komórek / {N_SAMPLES} próbek)", fontsize=9)
        axr.set_xlabel("t", fontsize=9)
        axr.set_ylabel("wartość density (surowa)", fontsize=9)

        axn.axvspan(0, inner_margin, color="#dc0000", alpha=0.12, zorder=0)
        axn.plot(t_axis, ex_norm, color="#2a78d6", lw=1.8)
        axn.fill_between(t_axis, 0, ex_norm, color="#2a78d6", alpha=0.15)
        for pi, pidx in enumerate(idxs):
            prom = prominences[pi] if pi < len(prominences) else 0.0
            t_p = pidx / (N_SAMPLES - 1)
            axn.plot([t_p], [ex_norm[pidx]], "o", color="#e01e1e", ms=7)
            axn.annotate(f"prom={prom:.3f}", (t_p, ex_norm[pidx]), textcoords="offset points",
                        xytext=(4, 6), fontsize=7, color="#e01e1e")
        # Który z wykrytych pików (idxs) faktycznie odpowiada NASZEMU kandydatowi
        # (ex_t, już PO przesunięciu) — dopasowanie po najbliższej przesuniętej t,
        # NIE zawsze idxs[0] (pierwszy wykryty pik na promieniu bywa inny niż ten,
        # który wygrał w klastrowaniu/wyborze DP).
        shift_map = {int(pidx): _shift_peak_to_falling_edge(ex_norm, int(pidx)) for pidx in idxs}
        target_idx = (min(idxs, key=lambda pidx: abs(shift_map[int(pidx)] / (N_SAMPLES - 1) - ex_t))
                     if len(idxs) else None)
        shifted_idx = shift_map.get(int(target_idx)) if target_idx is not None else None
        if shifted_idx is not None and shifted_idx != target_idx:
            t_orig = target_idx / (N_SAMPLES - 1)
            t_shift = shifted_idx / (N_SAMPLES - 1)
            axn.annotate("", xy=(t_shift, ex_norm[shifted_idx]), xytext=(t_orig, ex_norm[target_idx]),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
            axn.plot([t_shift], [ex_norm[shifted_idx]], "s", color="black", ms=8, mfc="none", mew=2)
        axn.axhline(prominence, color="#888", ls=":", lw=1.0)
        axn.set_title(f"Promień {ex_ray+1}/48 — po normalizacji + wykryte piki "
                      f"(min_distance={min_dist}, prominence≥{prominence})", fontsize=9)
        axn.set_xlabel("t", fontsize=9)
        axn.set_ylabel("wartość znorm.", fontsize=9)
        fig.tight_layout()
        example_charts_b64 = fig_to_b64(fig)

        # --- "Sądowy" zoom: DOKŁADNIE gdzie na siatce stoi kropka, i co jest obok. ---
        strongest_i = int(np.argmax(prominences)) if len(prominences) else None
        strongest_idx = int(idxs[strongest_i]) if strongest_i is not None else None
        strongest_shifted = shift_map.get(strongest_idx) if strongest_idx is not None else None

        # Klaster DRUGIEGO (własnego, zwykle silniejszego) piku tego promienia — jeśli to
        # inny klaster niż zwycięski, to jest "other" w Kroku 4a/4b (histogram + panel
        # przestrzenny), renderowanych po tym bloku.
        if (strongest_idx is not None and strongest_idx != target_idx
                and strongest_shifted is not None and clusters):
            st_t = strongest_shifted / (N_SAMPLES - 1)
            other_cluster = min(clusters, key=lambda c: abs(c[0] - st_t))
            if best_cluster is None or abs(other_cluster[0] - best_cluster[0]) > 1e-9:
                other = {"mean_t": other_cluster[0], "support": other_cluster[1],
                        "arc_len": other_cluster[3], "score": _cluster_score(other_cluster),
                        "arc_rays": _cluster_members(other_cluster[0]), "other_rays": set()}

        sel_t = (shifted_idx / (N_SAMPLES - 1)) if shifted_idx is not None else ex_t
        markers = [(sel_t, "WYBRANY (zwycieski klaster)", (30, 200, 30), "filled")]
        if strongest_idx is not None and strongest_idx != target_idx and strongest_shifted is not None:
            st_t = strongest_shifted / (N_SAMPLES - 1)
            markers.append((st_t, "najsilniejszy NA TYM promieniu", (255, 255, 0), "open"))

        fx_ray, fy_ray = ex_cpt

        # Najbliższa "gorąca" (>=p98) komórka NIE leżąca na tym promieniu — sprawdzamy,
        # czy odstający kolorystycznie fragment widziany na pełnym obrazie w ogóle jest
        # w zasięgu TEGO promienia, czy to zupełnie inny kierunek.
        p98 = float(np.percentile(density_grid, 98))
        hot_cells = np.argwhere(density_grid >= p98)
        gx0, gy0 = cx / cw * W_p, cy / ch * H_p
        gx1, gy1 = fx_ray / cw * W_p, fy_ray / ch * H_p
        hot_info = []
        for (rr, cc) in hot_cells:
            d, tt = _pt_seg_dist(cc + 0.5, rr + 0.5, gx0, gy0, gx1, gy1)
            hot_info.append((d, tt, int(rr), int(cc), float(density_grid[rr, cc])))
        hot_info.sort(key=lambda h: h[0])
        nearest_onray = hot_info[0] if hot_info else None            # ~ nasz własny silny pik
        nearest_offray = next((h for h in hot_info if h[0] > 2.0), None)

        # Kadr: od t=0.30 (pomijamy płaski odcinek bliski jądra) do t=1.05, z zapasem
        # bocznym, który mieści też najbliższą "fałszywie bliską" gorącą komórkę — żeby
        # dowód "to inny kierunek" był widoczny na TYM SAMYM obrazku, nie tylko w liczbach.
        t_lo, t_hi = 0.30, 1.05
        p_lo, p_hi = _t_to_px(t_lo, cx, cy, fx_ray, fy_ray), _t_to_px(t_hi, cx, cy, fx_ray, fy_ray)
        xs_box, ys_box = [p_lo[0], p_hi[0]], [p_lo[1], p_hi[1]]
        patch_w, patch_h = cw / W_p, ch / H_p
        pad_x, pad_y = 9 * patch_w, 9 * patch_h
        if nearest_offray is not None:
            _, _, hr, hc, _ = nearest_offray
            hx, hy = (hc + 0.5) * patch_w, (hr + 0.5) * patch_h
            xs_box.append(hx); ys_box.append(hy)
        bx0 = max(0, int(min(xs_box) - pad_x)); bx1 = min(cw, int(max(xs_box) + pad_x))
        by0 = max(0, int(min(ys_box) - pad_y)); by1 = min(ch, int(max(ys_box) + pad_y))
        crop_box = (bx0, by0, bx1, by1)

        zoom_photo = render_ray_zoom(crop_rgb, crop_box, W_p, H_p, cw, ch,
                                     cx, cy, fx_ray, fy_ray, markers)
        zoom_heatmap = render_ray_zoom(heatmap_rgb, crop_box, W_p, H_p, cw, ch,
                                       cx, cy, fx_ray, fy_ray, markers,
                                       grid=density_grid, hot_threshold=p98)
    else:
        example_charts_b64 = ""
        zoom_photo = crop_rgb
        zoom_heatmap = heatmap_rgb
        nearest_onray = nearest_offray = None

    # --- Panel 4a: mapa 2D promień×t. Wykres 1 = surowe piki, jeden słupek na promień.
    # Wykres 2 (poprzednia wersja, PASY POZIOME przez całą szerokość) ZDJĘTY — użytkownik
    # (29.07) słusznie zauważył, że zakładał, iż każdy klaster jest widoczny dookoła CAŁEGO
    # obwodu, co jest biologicznie fałszywe. Zamiast tego:
    #   Wykres 1a: klastry jako PROSTOKĄTY tylko tam, gdzie promienie faktycznie sąsiadują
    #              (przerwa w prostokącie = promień bez udziału w tym miejscu) — jeden
    #              klaster może więc narysować się jako kilka osobnych, wąskich prostokątów
    #              (promienie "w poprzek"), albo jeden szeroki (promienie "wzdłuż").
    #   Wykres 1b: to samo co Wykres 1, na kole — geometria promieni jest z natury kołowa.
    #   Wykres 1c: sąsiedztwo na kole — koło samo rozwiązuje "promień 0 sąsiaduje z 47"
    #              bez żadnego zawijania w kodzie (w przeciwieństwie do Wykresu 1a).
    # Numeracja promieni (0-47, co 7.5°) WYŁĄCZNIE w tej sekcji: promień 0 = oś pomiaru
    # (najdłuższy promień, jądro→najdalszy punkt konturu) — reszta raportu (Krok 2/3/4c)
    # nadal używa surowej kolejności konturu, nie zmienianej tutaj. ---
    from matplotlib.patches import Rectangle

    fx_axis, fy_axis = d_axis_info["far_edge"]
    axis_ray_idx = int(np.argmin([
        (int(contour[ci][0]) - fx_axis) ** 2 + (int(contour[ci][1]) - fy_axis) ** 2
        for ci in idx_sel]))

    def _disp(r: int) -> int:
        """Oryginalny indeks promienia (kolejność konturu) -> indeks wyświetlany
        (0 = oś pomiaru), TYLKO dla wykresów w tej sekcji."""
        return (r - axis_ray_idx) % N_DIRS

    def _contiguous_runs(members_disp: set[int]) -> list[tuple[int, int]]:
        """Ciągłe przebiegi (włącznie; ``end`` może przekroczyć N_DIRS-1, co koduje
        zawinięcie 47->0) indeksów WYŚWIETLANYCH w ``members_disp``."""
        if not members_disp:
            return []
        s = sorted(members_disp)
        if len(s) == N_DIRS:
            return [(0, N_DIRS - 1)]
        runs: list[list[int]] = []
        start = prev = s[0]
        for x in s[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append([start, prev])
                start = prev = x
        runs.append([start, prev])
        if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == N_DIRS - 1:
            first = runs.pop(0)
            last = runs.pop(-1)
            runs.append([last[0], first[1] + N_DIRS])
        return [(a, b) for a, b in runs]

    ray_map_raw_b64 = ray_map_clusters_lin_b64 = ""
    ray_map_polar_b64 = ray_map_polar_clusters_b64 = ""
    if dpk:
        xs_disp = [_disp(p[4]) for p in dpk]
        ys = [p[0] for p in dpk]
        ss = [p[1] for p in dpk]
        # Cieniowanie szarych (pozostałych) klastrów wg SCORE (arc-aware), nie samego
        # wsparcia — to score decyduje teraz, który klaster wygrywa, więc ciemniejszy
        # szary = wyżej w tym samym rankingu, którego szczyt jest zielony/żółty.
        max_score = max((_cluster_score(c) for c in clusters), default=1.0) or 1.0
        step_rad = 2 * np.pi / N_DIRS

        def _cluster_color(mt: float, score: float) -> tuple[str, float]:
            is_winning = winning is not None and abs(mt - winning["mean_t"]) < 1e-9
            is_other = other is not None and abs(mt - other["mean_t"]) < 1e-9
            if is_winning:
                return "#1ec81e", 0.45
            if is_other:
                return "#c8c800", 0.45
            return "#888888", 0.10 + 0.28 * (score / max_score)

        def _cluster_color_bgr(mt: float, score: float) -> tuple[int, int, int]:
            """Ta sama logika co _cluster_color, ale jako pełnokrycie (r,g,b) 0-255
            dla cv2 (render_vote_overlay na true-geometry nie ma pojęcia alpha per
            warstwa jak matplotlib — jaśniejszy = mniej pewny klaster)."""
            is_winning = winning is not None and abs(mt - winning["mean_t"]) < 1e-9
            is_other = other is not None and abs(mt - other["mean_t"]) < 1e-9
            if is_winning:
                return (30, 200, 30)
            if is_other:
                return (200, 200, 0)
            frac = score / max_score
            shade = int(200 - 90 * frac)
            return (shade, shade, shade)

        # Promień-przykład z Kroku 4c/4d (magenta wszędzie w tej sekcji), w numeracji
        # wyświetlanej (0=oś pomiaru) — żeby było widać, KTÓRY konkretnie słupek/kąt
        # rozpatrujemy niżej krok po kroku (user report, 29.07).
        ex_ray_disp = _disp(ex_ray) if ex_ray is not None else None

        # --- Wykres 1: surowe piki, jeden pionowy słupek na promień ---
        fig, ax = plt.subplots(figsize=(12, 4.2))
        for r in range(N_DIRS):
            ax.plot([r, r], [0, 1], color="#cccccc", lw=1, zorder=0)
        ax.axhspan(0, inner_margin, color="#dc0000", alpha=0.08, zorder=0)
        ax.axvline(0, color="#2a4d8f", lw=1.6, ls="--", zorder=1)
        if ex_ray_disp is not None:
            ax.axvline(ex_ray_disp, color="magenta", lw=1.8, ls="--", zorder=1,
                      label=f"rozpatrywany promień ({ex_ray_disp})")
            ax.legend(loc="upper right", fontsize=8)
        sc = ax.scatter(xs_disp, ys, c=ss, cmap="plasma", s=45, zorder=2,
                        edgecolor="black", linewidth=0.4, vmin=0, vmax=1)
        cb = fig.colorbar(sc, ax=ax, pad=0.01)
        cb.set_label("siła piku (znorm. per promień)", fontsize=8)
        ax.set_xlim(-1, N_DIRS)
        ax.set_ylim(0, 1)
        ax.set_xlabel("promień (0–47; 0 = oś pomiaru, niebieska przerywana; co 7.5°)", fontsize=9)
        ax.set_ylabel("t (0=jądro, 1=kontur)", fontsize=9)
        ax.set_title(f"Wykres 1: piki KAŻDEGO z {N_DIRS} promieni osobno — "
                     f"{len(dpk)} pików łącznie", fontsize=10)
        fig.tight_layout()
        ray_map_raw_b64 = fig_to_b64(fig)

        # --- Wykres 1a: klastry jako prostokąty, tylko nad SĄSIADUJĄCYMI promieniami ---
        fig, ax = plt.subplots(figsize=(12, 4.2))
        for (mt, sup, _mstr, _alen, _astr) in clusters:
            color, alpha = _cluster_color(mt, _cluster_score((mt, sup, _mstr, _alen, _astr)))
            members_disp = {_disp(r) for r in _cluster_members(mt)}
            for (start, end) in _contiguous_runs(members_disp):
                if end < N_DIRS:
                    ax.add_patch(Rectangle((start - 0.5, mt - t_tol), end - start + 1,
                                           2 * t_tol, facecolor=color, alpha=alpha,
                                           edgecolor=color, lw=0.8, zorder=1))
                else:
                    ax.add_patch(Rectangle((start - 0.5, mt - t_tol), N_DIRS - start,
                                           2 * t_tol, facecolor=color, alpha=alpha,
                                           edgecolor=color, lw=0.8, zorder=1))
                    ax.add_patch(Rectangle((-0.5, mt - t_tol), end - N_DIRS + 1,
                                           2 * t_tol, facecolor=color, alpha=alpha,
                                           edgecolor=color, lw=0.8, zorder=1))
        for r in range(N_DIRS):
            ax.plot([r, r], [0, 1], color="#cccccc", lw=1, zorder=0)
        ax.axhspan(0, inner_margin, color="#dc0000", alpha=0.08, zorder=0)
        ax.axvline(0, color="#2a4d8f", lw=1.6, ls="--", zorder=1)
        if ex_ray_disp is not None:
            ax.axvline(ex_ray_disp, color="magenta", lw=1.8, ls="--", zorder=1,
                      label=f"rozpatrywany promień ({ex_ray_disp})")
            ax.legend(loc="upper right", fontsize=8)
        ax.scatter(xs_disp, ys, c="#1a2740", s=26, zorder=3)
        ax.set_xlim(-1, N_DIRS)
        ax.set_ylim(0, 1)
        ax.set_xlabel("promień (0–47; 0 = oś pomiaru)", fontsize=9)
        ax.set_ylabel("t (0=jądro, 1=kontur)", fontsize=9)
        ax.set_title("Wykres 1a: klastry jako prostokąty — SZEROKI prostokąt = promienie "
                     "sąsiadują (\"wzdłuż\"); kilka WĄSKICH = rozrzucone (\"w poprzek\")",
                     fontsize=10)
        fig.tight_layout()
        ray_map_clusters_lin_b64 = fig_to_b64(fig)

        # --- Wykres 1b: to samo co Wykres 1, na kole ---
        angles = [d * step_rad for d in xs_disp]
        fig = plt.figure(figsize=(7.4, 7.4))
        axp = fig.add_subplot(projection="polar")
        axp.set_theta_zero_location("S")   # 0° w dół — spójne z osią pomiaru na zdjęciu (promień 0)
        axp.set_theta_direction(-1)
        for r in range(N_DIRS):
            axp.plot([r * step_rad, r * step_rad], [0, 1], color="#dddddd", lw=0.6, zorder=0)
        for layer_t in np.linspace(0.1, 1.0, 10):
            axp.plot(np.linspace(0, 2 * np.pi, 200), [layer_t] * 200,
                    color="#eeeeee", lw=0.5, zorder=0)
        axp.plot([0, 0], [0, 1], color="#2a4d8f", lw=1.8, ls="--", zorder=1)
        if ex_ray_disp is not None:
            th_ex = ex_ray_disp * step_rad
            axp.plot([th_ex, th_ex], [0, 1], color="magenta", lw=1.8, ls="--", zorder=1)
        sc = axp.scatter(angles, ys, c=ss, cmap="plasma", s=45, zorder=2,
                         edgecolor="black", linewidth=0.4, vmin=0, vmax=1)
        axp.set_ylim(0, 1)
        axp.set_title("Wykres 1b: to samo co Wykres 1, na kole (0°/przerywana = oś "
                      "pomiaru; jasne okręgi = linie pomocnicze co t=0.1, NIE granice "
                      "patchy — patrz tekst pod spodem)",
                      fontsize=10, pad=22)
        cb = fig.colorbar(sc, ax=axp, pad=0.1, shrink=0.7)
        cb.set_label("siła piku", fontsize=8)
        fig.tight_layout()
        ray_map_polar_b64 = fig_to_b64(fig)

        # --- Towarzysz Wykresu 1b w PRAWDZIWEJ geometrii otolitu (promienie prawdziwej
        # długości — otolit nie jest kołem). Wypełniamy CAŁY PATCH (nie kropkę w jego
        # środku) — user report 29.07: kropki za duże, a skoro to i tak jeden patch, to
        # niech będzie widać cały kwadracik. ---
        cmap_plasma = cm.get_cmap("plasma")
        peaks_points_colors = [
            ((x, y), tuple(int(255 * c) for c in cmap_plasma(float(np.clip(peak[1], 0, 1)))[:3]))
            for peak, (x, y) in zip(dpk, dpts_crop)
        ]
        peaks_true_geom = render_patches_true_geometry(
            crop_rgb, d_axis_info, peaks_points_colors, cw, ch, W_p, H_p,
            n_dirs=N_DIRS, inner_margin=inner_margin, highlight_ray=ex_ray)

        # --- Wykres 1c: sąsiedztwo na kole — zawijanie 47->0 jest naturalne (kąt) ---
        fig = plt.figure(figsize=(7.4, 7.4))
        axp = fig.add_subplot(projection="polar")
        axp.set_theta_zero_location("S")   # 0° w dół — spójne z osią pomiaru na zdjęciu (promień 0)
        axp.set_theta_direction(-1)
        for (mt, sup, _mstr, _alen, _astr) in clusters:
            color, alpha = _cluster_color(mt, _cluster_score((mt, sup, _mstr, _alen, _astr)))
            members_disp = {_disp(r) for r in _cluster_members(mt)}
            for (start, end) in _contiguous_runs(members_disp):
                th = np.linspace(start * step_rad, (end + 1) * step_rad, 40)
                axp.fill_between(th, mt - t_tol, mt + t_tol, color=color, alpha=alpha,
                                 zorder=1, linewidth=0)
        for r in range(N_DIRS):
            axp.plot([r * step_rad, r * step_rad], [0, 1], color="#dddddd", lw=0.6, zorder=0)
        for layer_t in np.linspace(0.1, 1.0, 10):
            axp.plot(np.linspace(0, 2 * np.pi, 200), [layer_t] * 200,
                    color="#eeeeee", lw=0.5, zorder=0)
        axp.plot([0, 0], [0, 1], color="#2a4d8f", lw=1.8, ls="--", zorder=1)
        if ex_ray_disp is not None:
            th_ex = ex_ray_disp * step_rad
            axp.plot([th_ex, th_ex], [0, 1], color="magenta", lw=1.8, ls="--", zorder=1)
        axp.scatter(angles, ys, c="#1a2740", s=26, zorder=3)
        axp.set_ylim(0, 1)
        axp.set_title("Wykres 1c: klastry jako łuki na kole — promień 0 i promień 47 są "
                      "tu naprawdę sąsiadami, bez żadnego sztucznego zawijania",
                      fontsize=10, pad=22)
        fig.tight_layout()
        ray_map_polar_clusters_b64 = fig_to_b64(fig)

        # --- Towarzysz Wykresu 1c w PRAWDZIWEJ geometrii. POPRZEDNIA wersja rysowała
        # PEŁNE promienie (linie środek->kontur) w kolorze klastra — z wieloma klastrami
        # naraz dawało to nieczytelny "wybuch gwiazdy" (user report, 29.07: "nie rozumiem
        # co jest narysowane"), bo linia ciągnie się przez CAŁĄ długość promienia, a
        # klaster dotyczy tylko JEDNEGO miejsca na nim. Teraz — jak w peaks_true_geom
        # wyżej — wypełniamy TYLKO patch, w którym faktycznie leży pik, kolorem KLASTRA,
        # do którego ten konkretny pik należy (dokładnie ta sama, poprawiona logika
        # wyłącznego przypisania co w _cluster_members). ---
        cluster_points_colors = []
        for pi in range(len(dpk)):
            if not len(_modes_arr):
                break
            ci = int(_nearest_mode[pi])
            if abs(dpk[pi][0] - _modes_arr[ci]) > t_tol:
                continue
            color = _cluster_color_bgr(_modes_arr[ci], _cluster_score(clusters[ci]))
            cluster_points_colors.append((dpts_crop[pi], color))
        clusters_true_geom = render_patches_true_geometry(
            crop_rgb, d_axis_info, cluster_points_colors, cw, ch, W_p, H_p,
            n_dirs=N_DIRS, inner_margin=inner_margin, highlight_ray=ex_ray)

    # --- Panel 4b: PRZESTRZENNIE — które konkretnie promienie głosują na który klaster. ---
    vote_spatial_photo = vote_spatial_heatmap = None
    if winning is not None:
        colored = [(winning, (30, 200, 30))]
        if other is not None:
            colored.append((other, (200, 200, 0)))
        vote_spatial_photo = render_vote_overlay(crop_rgb, d_axis_info, colored,
                                                 highlight_ray=ex_ray, n_dirs=N_DIRS,
                                                 inner_margin=inner_margin)
        vote_spatial_heatmap = render_vote_overlay(heatmap_rgb, d_axis_info, colored,
                                                   highlight_ray=ex_ray, n_dirs=N_DIRS,
                                                   inner_margin=inner_margin)

    # --- Panel 5: wszyscy zaakceptowani kandydaci (WYŁĄCZNIE density) ---
    # UWAGA (bug znaleziony przez użytkownika): dawniej kropki rysowano PRZEZ MUTACJĘ
    # heatmap_rgb w miejscu — a heatmap_rgb jest tym samym obiektem, do którego odwołuje
    # się Krok 1 dalej w tym samym f-stringu HTML (Python buduje f-string na końcu,
    # PO tej mutacji) — więc "surowa mapa" w Kroku 1 pokazywała finalne kandydatów,
    # zaprzeczając własnemu opisowi "punkt startowy, przed jakąkolwiek logiką". Teraz
    # kropki rysujemy na KOPII — heatmap_rgb zostaje czyste wszędzie indziej.
    print("Renderowanie Kroku 5 (wszyscy kandydaci, tylko density)...", flush=True)
    cands_on_photo = render_rays_and_candidates(orig_rgb, axis_info, dpts_full, None,
                                                n_dirs=N_DIRS, inner_margin=inner_margin)
    heatmap_with_cands = heatmap_rgb.copy()
    for (x, y) in dpts_crop:
        cv2.circle(heatmap_with_cands, (int(x), int(y)), max(4, ch // 130) + 1, (0, 0, 0), -1)
        cv2.circle(heatmap_with_cands, (int(x), int(y)), max(4, ch // 130), (244, 180, 0), -1)

    # --- Tabela liczbowa ---
    rows_html = ""
    for (t, s, x, y, r) in sorted(dpk_full, key=lambda p: p[0]):
        rows_html += (f"<tr><td>{r+1}</td><td>{t:.3f}</td><td>{s:.3f}</td>"
                     f"<td>({x},{y})</td></tr>")

    print("Składanie strony HTML...", flush=True)

    # --- Krok 4a: definicja głosowania + mapa 2D promień×t (Wykresy 1/1a/1b/1c — patrz
    # uzasadnienie w komentarzu przy renderowaniu wyżej). ---
    krok4a_html = ""
    if winning is not None:
        krok4a_html = (
            '<h3>Krok 4a — „głosowanie": co to właściwie znaczy</h3>'
            '<p>Każdy z 48 promieni znalazł swoje własne piki (Krok 2/3) — każdy pik ma swoje '
            '<code>t</code> (0=jądro, 1=kontur). Zbieramy TERAZ wszystkie te piki z WSZYSTKICH '
            f'48 promieni razem — {len(dpk)} pików w jednej puli — i pytamy: <b>ile różnych '
            'promieni ma pik w OKOLICY tego samego t?</b> To jest „głosowanie": jeden promień z '
            'pikiem przy danym t = jeden głos na to, że w tym miejscu jest prawdziwy pierścień. '
            'Logika: prawdziwy roczny przyrost to koncentryczny pierścień — powinien dawać pik '
            'z WIELU kierunków naraz. Ale <b>NIE musi być widoczny dookoła całego obwodu</b> — '
            'może być bardzo wyraźny w jednym miejscu, słabszy gdzie indziej, a jeszcze gdzie '
            'indziej nieobecny wcale.</p>'
            '<p style="background:#e8f0ff;padding:8px;border-left:3px solid #2a4d8f;">'
            '<b>Ważne — sama LICZBA głosów NIE decyduje, kto wygrywa.</b> Klaster z większą '
            'liczbą głosów, ale rozrzuconych po całym obwodzie, jest SŁABSZYM dowodem niż klaster '
            'z mniejszą liczbą głosów, ale skupionych na sąsiadujących promieniach (prawdziwy '
            'pierścień jest zwykle widoczny na spójnym kawałku obwodu, nie w przypadkowo '
            'porozrzucanych miejscach). Dlatego liczy się <b>score</b>, nie sam głosów: '
            '<code>score = 0.4 × (głosy × siła) + 0.6 × (długość najdłuższego SĄSIADUJĄCEGO '
            'odcinka promieni × jego siła)</code> — dokładnie ten wzór, którego używa prawdziwy, '
            'produkcyjny <code>select_increments</code>. Wykresy 1a i 1c niżej pokazują tę '
            'zwartość wprost, jako kształt prostokątów/łuków.</p>'
            '<p style="background:#fff0f0;padding:8px;border-left:3px solid #b03060;">'
            '<b>Historia tej zmiany (29.07):</b> we wcześniejszych wersjach tego raportu (v1–v4) '
            'wygrywał klaster t≈0.511 (21 głosów) — WYŁĄCZNIE dlatego, że miał więcej głosów niż '
            'klaster t≈0.852 (18 głosów), mimo że te 18 głosów było znacznie bardziej zwarte '
            '(dłuższy sąsiadujący odcinek). Użytkownik na realnym zdjęciu zaznaczył czerwonym '
            'flamastrem PRAWDZIWĄ, wyraźnie widoczną strukturę — okazało się, że to WŁAŚNIE klaster '
            't≈0.852: przy prawdziwym, produkcyjnym liczeniu (score, nie sam głosów) to on wygrywa '
            '(score=13.08, #1 w rankingu), a t≈0.511 spada na OSTATNIE miejsce w top 6 (score=8.74) '
            '— bo jego 21 głosów jest mocno rozrzucone po promieniach. Innymi słowy: poprzedni '
            '„zwycięski" (zielony) klaster w tym raporcie był artefaktem uproszczonego liczenia, '
            'nie tego, co naprawdę wybrałaby produkcja. Od tej wersji Krok 4 używa prawdziwego, '
            'arc-aware liczenia — dlatego przykładowy promień/klaster mógł się zmienić względem '
            'poprzednich wersji.</p>'
            '<p style="font-size:92%;"><b>Numeracja promieni w tej sekcji (tylko tu):</b> '
            '0–47, co 7.5° (360°/48). Promień <b>0 to zawsze oś pomiaru</b> — najdłuższy '
            'promień, jądro→najdalszy punkt konturu, ta sama oś, którą raport rysuje gdzie '
            'indziej jako żółtą linię. Reszta raportu (Krok 2/3, Krok 4c) używa surowej '
            'kolejności konturu — inna numeracja, celowo NIE zmieniana tam, żeby nie ruszać '
            'sekcji, o które teraz nie pytasz.</p>'
            '<h4 style="margin:1em 0 0.2em;">Wykres 1 — surowe piki, promień po promieniu</h4>'
            '<p><b>Oś X = który promień (0–47)</b>, <b>oś Y = t na tym promieniu</b>. Każdy '
            'pionowy szary słupek to jeden promień; kropki na nim to jego piki (kolor = siła). '
            'Niebieska przerywana = promień 0 (oś pomiaru). '
            '<span style="color:magenta;font-weight:bold;">Różowa/magenta przerywana</span> = '
            'promień, który rozpatrujemy krok po kroku w Kroku 4c/4d niżej — ta sama linia '
            'wraca na Wykresach 1a/1b/1c i na zdjęciach obok nich.</p>'
            f'<div style="text-align:center;">{img_tag(ray_map_raw_b64, style="width:1050px;")}</div>'
            '<h4 style="margin:1em 0 0.2em;">Wykres 1a — klastry jako prostokąty</h4>'
            '<p>Poprzednia wersja rysowała klaster jako pas na CAŁĄ szerokość — co zakładało, '
            'że pierścień jest widoczny dookoła całego obwodu. Niepoprawnie. Tu prostokąt '
            'klastra jest rysowany TYLKO nad promieniami, które faktycznie do niego należą, '
            '<b>i tylko tam, gdzie te promienie SĄSIADUJĄ ze sobą</b> — jeśli promienie '
            'głosujące na dany klaster są rozrzucone (np. promień 3, potem dopiero 19, potem '
            '31), dostajemy kilka osobnych, WĄSKICH prostokątów („promień przecina klaster w '
            'poprzek"); jeśli sąsiadują (3,4,5,6,7...), dostajemy jeden SZEROKI prostokąt '
            '(„promień biegnie wzdłuż klastra"). '
            '<span style="color:#1ec81e;font-weight:bold;">Zielony</span> = zwycięski klaster, '
            '<span style="color:#c8c800;font-weight:bold;">żółty</span> = klaster 2. piku '
            'naszego przykładowego promienia, szare = pozostałe (ciemniejszy szary = wyższy '
            'score).</p>'
            f'<div style="text-align:center;">{img_tag(ray_map_clusters_lin_b64, style="width:1050px;")}</div>'
            '<p style="font-size:92%;background:#fff3cd;padding:8px;border-left:3px solid #b8860b;">'
            '<b>Poprawka błędu (znaleziony przez użytkownika na tym właśnie wykresie):</b> Wykres 1a '
            'w poprzedniej wersji liczył przynależność do klastra NIEZALEŻNIE dla każdego klastra '
            'osobno ("czy ten pik jest w promieniu t_tol OD TEGO środka") — gdy dwa środki klastrów '
            'leżą blisko siebie, ten sam pik wpadał do OBU list naraz, więc słabszy klaster mógł na '
            'oko wyglądać na obejmujący WIĘCEJ promieni niż silniejszy. Poprawione: pik liczy się do '
            'TYLKO NAJBLIŻSZEGO klastra (dokładnie tak, jak robi to prawdziwy '
            f'<code>_cluster_by_radius_with_arcs</code>). Po poprawce: zwycięski (zielony, '
            f't≈{winning["mean_t"]:.3f}) ma {winning["support"]} głosów, długość najdłuższego '
            f'sąsiadującego odcinka {winning["arc_len"]}, score={winning["score"]:.3f}'
            + (f'; drugi (żółty, t≈{other["mean_t"]:.3f}) ma {other["support"]} głosów, '
               f'odcinek {other["arc_len"]}, score={other["score"]:.3f}' if other is not None else '') +
            ' — liczby z prostokątów/łuków wyżej i niżej ZGADZAJĄ SIĘ teraz z oficjalnym liczeniem.</p>'
            '<h4 style="margin:1em 0 0.2em;">Wykres 1b — to samo, na kole, obok PRAWDZIWEJ geometrii</h4>'
            '<p>Promienie fizycznie rozchodzą się z jądra po kole — Wykres 1 to sztucznie '
            '„rozcięty i rozłożony na płasko" widok. Po lewej: ten sam Wykres 1, ale we właściwej, '
            'kołowej geometrii (kąt = promień, 0° = oś pomiaru U DOŁU — tak samo jak na zdjęciu '
            'obok, gdzie żółta oś pomiaru też idzie w dół; odległość od środka = t). '
            '<b>Jasne okręgi to WYŁĄCZNIE linie pomocnicze co t=0.1</b> (jak linie siatki na zwykłym '
            'wykresie) — <b>NIE granice patchy.</b> Prawdziwe granice patchy różnią się między '
            'promieniami (patrz Krok 2/3: jeden promień trafia w 27 różnych komórek, inny w mniej) '
            'i nie dałoby się ich narysować jako wspólnych okręgów. Po prawej: <b>TA SAMA informacja '
            '(ten sam kolor = siła), ale na prawdziwym zdjęciu otolitu</b> — każdy promień ma tu '
            'swoją PRAWDZIWĄ długość (otolit nie jest kołem!), a każdy pik wypełnia CAŁY patch, w '
            'którym leży (nie kropkę w jego środku — to i tak jeden patch, więc niech będzie widać '
            'cały kwadracik).</p>'
            '<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;justify-content:center;">'
            f'<div>{img_tag(ray_map_polar_b64, style="height:480px;width:auto;")}</div>'
            f'<div>{img_tag(_b64_from_rgb(peaks_true_geom, 520), style="height:480px;width:auto;")}</div>'
            '</div>'
            '<h4 style="margin:1em 0 0.2em;">Wykres 1c — sąsiedztwo na kole, obok PRAWDZIWEJ geometrii</h4>'
            '<p>Ten sam problem co w Wykresie 1a (promień 0 i promień 47 SĄSIADUJĄ, mimo że w '
            'numeracji są na przeciwległych końcach) na płaskim wykresie wymaga sztucznego '
            '„zawijania" prostokąta w dwie części. Na kole ten problem znika sam. Po prawej: '
            '<b>dokładnie to, co po lewej, przeniesione na zdjęcie</b> — dla każdego piku patch, w '
            'którym leży, wypełniony kolorem klastra, do którego TEN KONKRETNY pik należy '
            f'(zielony=zwycięski t≈{winning["mean_t"]:.3f}'
            + (f', żółty=drugi klaster t≈{other["mean_t"]:.3f}' if other is not None else '')
            + ', szary=pozostałe klastry). '
            'Poprzednia wersja rysowała tu PEŁNE promienie (linia od jądra do konturu) w kolorze '
            'klastra — z wieloma klastrami naraz wychodził nieczytelny "wybuch gwiazdy", bo linia '
            'ciągnie się przez CAŁĄ długość promienia, a klaster dotyczy tylko jednego miejsca na '
            'nim. Teraz widać wprost, gdzie na zdjęciu leżą kolorowe patche.</p>'
            '<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;justify-content:center;">'
            f'<div>{img_tag(ray_map_polar_clusters_b64, style="height:480px;width:auto;")}</div>'
            f'<div>{img_tag(_b64_from_rgb(clusters_true_geom, 520), style="height:480px;width:auto;")}</div>'
            '</div>'
            '<h4 style="margin:1.4em 0 0.2em;color:#b03060;">Otwarte pytanie (jeszcze nie '
            'rozwiązane w kodzie): czy t jest porównywalne MIĘDZY promieniami?</h4>'
            '<p>Cała metoda (klastrowanie po <code>t</code>, wszystkie wykresy wyżej) milcząco '
            'zakłada, że ten sam <code>t</code> na różnych promieniach = ten sam rok wzrostu — '
            'inaczej mówiąc, że otolit rośnie IZOTROPOWO (tak samo szybko we wszystkich kierunkach '
            'od jądra). <code>t</code> jest już znormalizowane do DŁUGOŚCI każdego promienia z '
            'osobna (promień o długości 400px i promień o długości 250px oba mają t∈[0,1]) — to '
            'usuwa różnice w SAMYM KSZTAŁCIE otolitu. Ale nie usuwa różnic w TEMPIE PRZYROSTU: '
            'jeśli otolit rośnie szybciej w kierunku osi pomiaru (dlatego zresztą technicy wybierają '
            'właśnie ten kierunek — tam pierścienie są najbardziej rozsunięte i czytelne) niż w '
            'kierunku przeciwnym, to ten sam biologiczny rok może wypaść na np. t=0.8 na jednym '
            'promieniu i t=0.7 na przeciwległym — NIE dlatego, że coś jest źle zmierzone, tylko '
            'dlatego, że wzrost faktycznie jest anizotropowy. Obecne klastrowanie (tu i w '
            'produkcyjnym <code>select_increments</code>) tego nie koryguje — porównuje surowe t '
            'między wszystkimi promieniami tak, jakby wzrost był jednakowy we wszystkich '
            'kierunkach. Nie mam gotowej poprawki do zaproponowania na już — to wymagałoby albo '
            'kalibracji każdego promienia względem osi pomiaru (np. po pozycjach pierścieni '
            'znalezionych klasycznie), albo innej definicji „odległości" niż liniowy ułamek '
            'promienia. Zostawiam to jako świadomie otwarty punkt — daj znać, czy chcesz, żebym to '
            'zbadał jako osobny, następny krok, czy na razie zostawiamy jako udokumentowane '
            'ograniczenie.</p>'
        )

    # --- Krok 4b: przestrzennie, KTO głosuje ---
    krok4b_html = ""
    if vote_spatial_photo is not None:
        krok4b_html = (
            '<h3>Krok 4b — przestrzennie: KTÓRE konkretnie promienie głosują</h3>'
            '<p>Ten sam wynik, ale narysowany na otolicie zamiast na wykresie — żeby było widać '
            'GDZIE geograficznie leżą głosujące promienie, nie tylko ich liczbę. '
            f'<span style="color:#1ec81e;font-weight:bold;">Zielone</span> promienie głosują na '
            f'zwycięski klaster (t≈{winning["mean_t"]:.3f}, {winning["support"]} głosów)'
            + (f'; <span style="color:#c8c800;font-weight:bold;">żółte</span> głosują na drugi '
               f'klaster (t≈{other["mean_t"]:.3f}, {other["support"]} głosów) — ten, do którego '
               'należy DRUGI pik naszego przykładowego promienia.'
               if other is not None else '.') +
            ' Różowa/magenta linia = nasz przykładowy promień z Kroku 4c/4d niżej.</p>'
            '<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;">'
            f'<div><b>Na zdjęciu</b><br>{img_tag(_b64_from_rgb(vote_spatial_photo, 480), style="width:480px;")}</div>'
            f'<div><b>Na mapie density</b><br>{img_tag(_b64_from_rgb(vote_spatial_heatmap, 480), style="width:480px;")}</div>'
            '</div>'
        )

    example_block = ""
    if example is not None:
        target_prom = (prominences[list(idxs).index(target_idx)]
                       if target_idx is not None and len(idxs) else 0.0)
        max_prom = max(prominences) if len(prominences) else 0.0
        weak_note = ""
        if len(idxs) > 1 and target_prom < max_prom - 1e-9 and other is not None:
            weak_note = (
                f' Na TYM promieniu jest drugi, znacznie silniejszy pik samodzielnie '
                f'(prominencja {max_prom:.3f} vs {target_prom:.3f} u wybranego) — ale silny SAM '
                f'W SOBIE nie wystarczy: ten silniejszy pik należy do klastra t≈{other["mean_t"]:.3f} '
                f'({other["support"]} głosów, odcinek {other["arc_len"]}, score={other["score"]:.3f}), '
                f'podczas gdy słabszy pik trafia w klaster t≈{winning["mean_t"]:.3f} '
                f'({winning["support"]} głosów, odcinek {winning["arc_len"]}, '
                f'score={winning["score"]:.3f}). Wygrywa wyższy SCORE (głosy ważone zwartością '
                'odcinka), nie siła pojedynczego piku ani sama liczba głosów.'
            )
        forensic_note = ""
        if nearest_onray is not None:
            d0, r0, c0, v0 = nearest_onray[0], nearest_onray[2], nearest_onray[3], nearest_onray[4]
            forensic_note += (
                f'Najbliższa komórka „gorąca" (≥98. percentyl całej mapy) do tego promienia jest '
                f'w odległości <b>{d0:.2f} szerokości patcha</b> od jego linii — to praktycznie '
                'nasz własny, silniejszy pik (żółty pierścień). ')
        if nearest_offray is not None:
            d1 = nearest_offray[0]
            forensic_note += (
                f'Kolejna taka komórka jest w odległości <b>{d1:.1f} szerokości patcha</b> — '
                'wyraźnie inny kierunek, nie ten promień (widoczna jako żółta ramka w rogu kadru '
                'niżej, żeby to było widać na oko, nie tylko w liczbach).'
            )
        example_block = (
            f'{krok4a_html}{krok4b_html}'
            '<h3>Krok 4c — nasz przykładowy promień ma piki w DWÓCH klastrach</h3>'
            f'<p>Promień <b>{ex_ray+1}/48</b> (różowa/magenta linia wyżej) — wybrany, bo ma pik '
            'należący do zwycięskiego klastra. Surowe wartości density wzdłuż tego promienia '
            f'mieszczą się w [{ex_raw.min():.5f}, {ex_raw.max():.5f}] — próbkujemy 64 razy, ale '
            f'trafiamy tylko <b>{cell_counts[ex_ray]} różnych komórek siatki</b> (stąd schodkowy '
            'kształt lewego wykresu). Po normalizacji tego JEDNEGO promienia szukamy lokalnych '
            f'maksimów (<code>prominence≥{prominence}</code>, <code>min_distance={min_dist}</code>) '
            '— czerwone kropki na prawym wykresie. Kwadrat = pozycja PO przesunięciu na "falling '
            'edge" (granica jasna→ciemna, konwencja odczytu przyrostów — nie sam szczyt piku).'
            f'{weak_note}</p>'
            f'<div style="text-align:center;margin-top:8px;">{img_tag(example_charts_b64, style="width:900px;")}</div>'
            '<h3 style="margin-top:1.4em;">Krok 4d — zbliżenie: gdzie DOKŁADNIE stoi kropka</h3>'
            '<p style="font-size:92%;">Poprzednia wersja tego raportu rysowała promień i kropkę na '
            '<b>czerwono</b> — tym samym kolorem co gorący koniec mapy JET, więc na małym podglądzie '
            'wizualnie zlewały się z tłem. Tu: cienka <b>magenta</b> (kolor spoza palety JET), '
            'cienkie znaczniki, realne granice patchy, BEZ liczb w komórkach (nieczytelne przy tej '
            'gęstości — kolor + żółta ramka na „gorących" komórkach wystarczą).</p>'
            '<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;">'
            f'<div><b>Ten fragment promienia na zdjęciu</b><br>{img_tag(_b64_from_rgb(zoom_photo, 640), style="width:640px;")}</div>'
            f'<div><b>Ten sam fragment na mapie density</b><br>'
            f'{img_tag(_b64_from_rgb(zoom_heatmap, 640), style="width:640px;")}</div>'
            '</div>'
            '<p style="font-size:92%;margin-top:6px;"><span style="color:#1ec81e;">●</span> zielona '
            'kropka = kandydat pokazany w tym raporcie (należy do zwycięskiego klastra) &nbsp;'
            '<span style="color:#c8c800;">○</span> żółty pierścień = najsilniejszy pojedynczy pik NA '
            'TYM promieniu (jeśli inny niż zielony) &nbsp; cienka żółta ramka wokół komórki = ta '
            'komórka jest „gorąca" (≥98. percentyl CAŁEJ mapy), niekoniecznie leży na tym promieniu.</p>'
            f'<p>{forensic_note}</p>'
        )

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Mechanizm wyboru kandydatów — TYLKO density ({VERSION}, {IMAGE_ID})</title>
<style>
body {{font-family:sans-serif;max-width:1300px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
h1 {{color:#1a237e;}} h2 {{color:#1a237e;}} h3 {{color:#283593;}}
table {{border-collapse:collapse;margin:0.6em 0;}}
td,th {{padding:4px 10px;border:1px solid #ddd;font-size:90%;text-align:right;}}
th {{background:#f0f0f5;}}
code {{background:#f0f0f5;padding:1px 4px;border-radius:3px;}}
</style>
</head>
<body>
<h1>Skąd biorą się kandydaci na przyrosty — WYŁĄCZNIE mapa density</h1>
<p>Otolit <code>{IMAGE_ID}</code>. Ten raport celowo pomija metodę klasyczną (jasność obrazu) —
pokazuje wyłącznie, jak z mapy density modelu (głowica <code>density_head</code>, Kierunek B,
suma≈wiek) powstają konkretne piksele-kandydaci. Użyto PRAWDZIWEJ, wysokorozdzielczej mapy
({H_p}×{W_p} patchy, przycięta do bboxa otolitu, {cfg.candidates.density_image_size}px) —
dokładnie tej, której realnie używa wykrywanie kandydatów na kartach raportu (nie uproszczonej
518px z sekcji G poprzedniego raportu).</p>

<section><h2>Krok 1 — sama mapa density</h2>
<p>Jedna liczba na kwadracik patcha 14×14&nbsp;px. PRAWDZIWY zakres wartości na tym zdjęciu to
<b>[{density_grid.min():.5f}, {density_grid.max():.5f}]</b> — ale to globalne maksimum pochodzi
z <b>jednego skrajnego patcha</b> (promień {outlier_ray+1}/48, blisko krawędzi/konturu, wartość
{outlier_val:.5f}) — kilkadziesiąt razy większego niż typowy „mocny" patch gdzie indziej
(mediana szczytu promienia ≈{typical_val:.5f}). Malowanie kolorem wprost na tym zakresie zalałoby
CAŁY otolit jednym kolorem (sprawdzone — tak właśnie wygląda, patrz Krok 2/3 niżej), więc poniższa
mapa używa kolorów przyciętych do percentyla 1–98 (<b>[{vmin:.5f}, {vmax:.5f}]</b> — pasek z prawej),
żeby była widoczna WEWNĘTRZNA struktura. To jest jednocześnie żywy przykład, DLACZEGO wykrywanie
pików normalizuje KAŻDY promień osobno (Krok 2/3) zamiast porównywać surowe wartości globalnie —
jeden odstający patch inaczej zdominowałby cały obraz. Suma wszystkich {H_p*W_p} komórek = <b>{density_grid.sum():.2f}</b>
— głowica density jest trenowana tak, by ta suma zbiegała do przewidywanego wieku (count-consistency).
Poniżej: TYLKO mapa + {N_DIRS} promieni pomiarowych (jądro→kontur, szare linie) — to jest nasz punkt
startowy. Żadnych kandydatów jeszcze tu nie ma — ci pojawiają się dopiero w Kroku 2/3.</p>
<div style="display:flex;gap:12px;align-items:flex-start;">
<div>{img_tag(_b64_from_rgb(krok1_img, 560), style="width:560px;")}</div>
<div>{img_tag(colorbar_b64, style="height:340px;")}</div>
</div>
</section>

<section><h2>Krok 2/3 — wszystkie 48 promieni: surowe vs znormalizowane</h2>
<p>Dla każdego z 48 kierunków (jądro→kontur) próbkujemy mapę 64&nbsp;razy (nearest-neighbour —
każda próbka bierze wartość NAJBLIŻSZEJ komórki siatki, bez wygładzania). Średnio próbki te
trafiają w tylko <b>{mean_cells:.1f} różnych komórek</b> na promień (z 64 próbek) — stąd
schodkowy kształt lewego wykresu. Zwróć uwagę, że na lewym wykresie widać JEDEN promień
(nr {outlier_ray+1}/48) strzelający do {outlier_val:.4f}, podczas gdy wszystkie pozostałe 47
są przy nim wizualnie płaskie przy zerze — to ten sam odstający patch z Kroku 1, w praktyce.
<b>Lewy</b> wykres = wartości surowe: widać, że skala
BEZWZGLĘDNA jest bardzo różna między promieniami (niektóre kierunki mają dużo wyższe density
niż inne). <b>Prawy</b> wykres = to samo PO normalizacji każdego promienia OSOBNO do [0,1] —
to na TYCH krzywych, nie na surowych wartościach, wykrywane są piki. Konsekwencja: pik jest
zawsze <i>lokalnym</i>, względnym maksimum SWOJEGO WŁASNEGO promienia — nie porównaniem
wartości bezwzględnych między promieniami ani z resztą obrazu.</p>
<div style="text-align:center;">{img_tag(rays48_b64, style="width:1000px;")}</div>
</section>

<section>
{example_block}
<h3 style="margin-top:1.4em;">Dlaczego dokładnie TU, a nie gdzie indziej — podsumowanie mechanizmu</h3>
<ol>
<li><b>Nearest-neighbour próbkowanie:</b> 64 punkty na promieniu, każdy bierze wartość
najbliższej komórki siatki {H_p}×{W_p} — stąd „schodki", nie gładka krzywa.</li>
<li><b>Normalizacja PER PROMIEŃ (nie globalna):</b> każdy promień jest przeskalowany do [0,1]
niezależnie. Pik to lokalne maksimum WZGLĘDEM WŁASNEGO zakresu tego promienia — słaby,
płaski promień i mocny, wyraźny promień są traktowane tak samo, jeśli oba mają JAKIŚ wyraźny
lokalny garb.</li>
<li><b>Dwa progi na znormalizowanym sygnale:</b> odległość ≥ <code>min_distance={min_dist}</code>
próbek od poprzedniego zaakceptowanego piku NA TYM SAMYM promieniu, i
<code>prominence≥{prominence}</code> — pik musi wystawać co najmniej tyle ponad sąsiednie
doliny (w skali [0,1] tego promienia).</li>
<li><b>Okno pozycji:</b> surowa pozycja piku musi mieścić się w
<code>t∈[{inner_margin}, {1-0.08:.2f}]</code> (czerwona strefa na wykresach = wykluczona
strefa jądra).</li>
<li><b>Przesunięcie na „falling edge":</b> zgłoszony piksel to NIE szczyt piku — algorytm idzie
w przód aż do najbliższego dołka, potem cofa się do punktu w połowie wysokości między szczytem
a dołkiem (konwencja odczytu przyrostów: granica jasna→ciemna, nie sam szczyt jasności).</li>
<li><b>Głosowanie promieni (Krok 4a/4b):</b> pik, który przetrwał wszystkie progi wyżej, i tak
NIE trafia automatycznie do finalnych przyrostów — musi jeszcze wygrać z innymi kandydatami na tym
samym otolicie. Wygrywa klaster (grupa pików z różnych promieni przy podobnym t) z NAJWYŻSZYM
SCORE — nie samą liczbą głosów, tylko głosami ZWAŻONYMI zwartością: klaster, którego głosujące
promienie SĄSIADUJĄ ze sobą, wygrywa z klastrem o większej liczbie głosów, ale rozrzuconych po
całym obwodzie (patrz Krok 4a — realny przykład na tym otolicie). To dlatego kropka może wylądować
na promieniu, który SAM W SOBIE ma tam słaby sygnał — liczy się to, czy INNE, SĄSIADUJĄCE promienie
widzą coś podobnego w tym samym miejscu, nie siła jednego piku z osobna.</li>
</ol>
</section>

<section><h2>Krok 5 — wszyscy zaakceptowani kandydaci (tylko density)</h2>
<p><b>{len(dpk)}</b> kandydatów z {N_DIRS} promieni. Poniżej — na zdjęciu, na mapie density,
i tabela ze WSZYSTKIMI (promień, t, siła piku, piksel).</p>
<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;">
<div><b>Na zdjęciu (przycięte do bboxa)</b><br>{img_tag(_b64_from_rgb(cands_on_photo, 480), style="width:480px;")}</div>
<div><b>Na mapie density</b><br>{img_tag(_b64_from_rgb(heatmap_with_cands, 480), style="width:480px;")}</div>
</div>
<table><tr><th>promień</th><th>t</th><th>siła (znorm.)</th><th>piksel (pełny obraz)</th></tr>
{rows_html}
</table>
</section>
{_section_krok6_forced_topk()}
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
