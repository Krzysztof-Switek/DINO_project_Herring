"""31.07 (cd. 5) — Test WSZYSTKICH metod wykrywania fragmentów przyrostów, na 2 otolitach.

Kontynuacja v1-v7 (`ring_shortest_path_report.py`, `ring_fragment_fusion_report.py`,
`ring_fragment_detection_report.py`) — TE ustalenia NIE są tu powtarzane. Ten raport zestawia
WSZYSTKIE metody wypróbowane do tej pory PLUS nowe, które NIE wymagają adnotacji (bez INBD, bez
uczonych detektorów krawędzi — te potrzebują treningu na ręcznie zaznaczonych granicach, których
nie mamy):

Już znane (v1-v7):
  1. Gradient promieniowy (tani odpowiednik filtru grzbietowego)
  2. Filtr Frangiego (prawdziwy detektor linii/rurek, Frangi i in. 1998)
  3. Circular shortest path (najkrótsza ZAMKNIĘTA ścieżka na grafie kosztów)
  4. Pewne otwarte fragmenty łuku (bez wymuszania zamknięcia)
  5. RANSAC — składanie fragmentów w pierścień, odrzucanie odstających

Nowe w tej sesji (bez adnotacji):
  6. CLAHE + mapa density + Frangi — czy wzmocnienie kontrastu odblokuje coś z rzadkiego
     sygnału density
  7. Bank filtrów Gabora (wieloorientacyjny detektor krawędzi/tekstury — klasyka z biometrii
     odcisków palców)
  8. Transformata Hougha (klasyczne, deterministyczne "głosowanie" na okręgi)
  9. Aktywne kontury / "snake" (krzywa doklejająca się do krawędzi, Kass/Witkin/Terzopoulos 1987)
  10. Łączenie łańcuchów ("spider web", Rodin & Troadec / CS-TRD) — okazuje się, że TEN mechanizm
      już częściowo istnieje w produkcyjnym kodzie projektu
      (`_cluster_by_radius_with_arcs`/`assign_rays_to_clusters`/`render_arc_cluster_overlay`) —
      tu tylko pokazany wprost, nie wdrożony od nowa.

src/ CAŁKOWICIE niedotknięte (poza odczytem/importem istniejących funkcji).

Usage: PYTHONIOENCODING=utf-8 python scripts/diagnostics/all_methods_comparison_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from skimage.filters import gabor
from skimage.segmentation import active_contour

from src.report_common import fig_to_b64, img_tag
from src.visualization import _CONTOUR_COLOR, render_arc_cluster_overlay
from src.otolith_axis import detect_axis, apply_background_mask, mask_bbox, shift_axis_info
from src.dataset import build_transforms, decode_age_ordinal
from src.inference import load_model_from_checkpoint
from src.ring_extraction import classical_increments, _cluster_by_radius_with_arcs, assign_rays_to_clusters
from scripts.run_pipeline import load_merged_config

from scripts.diagnostics.ring_shortest_path_report import (
    ray_cast_R_theta, polar_unwrap, apply_margins, frangi_cost_field,
    find_confident_open_arcs, render_open_arcs_unwrap, render_open_arcs_on_photo,
    extract_k_rings_ordered, ring_path_to_xy, render_unwrap, render_rings_on_photo,
    _b64_from_rgb, MODEL_VARIANTS, N_ANGLE, N_RADIUS, WINDOW, JUMP_PENALTY,
    INNER_MARGIN, EDGE_MARGIN, ARC_STRIDE_DEG, WIDTH_CEILING_WEIGHT, WIDTH_DECAY_WEIGHT,
    MIN_GAP_FRAC, _RING_COLORS,
)
from scripts.diagnostics.ring_fragment_fusion_report import ransac_fit_ring, render_fusion_unwrap, render_fusion_on_photo

OTOLITHS = [
    {"image_id": "2023_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg",
     "label": "Otolit A: FishIndex95, BITS1q, wiek etykiety=4"},
    {"image_id": "2022_BITS4q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex19_Single2_Right.jpg",
     "label": "Otolit B: FishIndex19 (GlebiaGdanska), BITS4q, wiek etykiety=4"},
]

FUSION_ARC_TOP_K = 12
FUSION_ARC_MIN_GAP_DEG = 12
FUSION_ARC_MIN_GAP_T = 0.06

OUT_DIR = PROJECT_ROOT / "outputs" / "31.07_ring_shortest_path"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION = "v8"
OUT_PATH = OUT_DIR / f"report_{VERSION}_all_methods.html"


def _normalize(field: np.ndarray) -> np.ndarray:
    rng = float(field.max() - field.min())
    return (field - field.min()) / rng if rng > 1e-6 else field * 0.0


# ---------------------------------------------------------------------------
# Nowe pola/metody (6-9)
# ---------------------------------------------------------------------------

def clahe_density_frangi_field(density_grid: np.ndarray, out_size: int = 256) -> np.ndarray:
    """(6) CLAHE (wyrównanie histogramu lokalnego kontrastu) na mapie density, potem
    powiększenie i filtr Frangiego — sprawdza, czy wzmocnienie kontrastu odblokuje coś z
    rzadkiego sygnału density (którego surowa wersja jest ~99% pusta, patrz
    ``outputs/31.07_radial_Analiza_wnioski.md``)."""
    d = _normalize(density_grid.astype(np.float32))
    d_u8 = (d * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    d_enh = clahe.apply(d_u8).astype(np.float32) / 255.0
    d_big = cv2.resize(d_enh, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    ridge_b = frangi_cost_field(d_big * 255.0)
    return ridge_b.astype(np.float32)


def gabor_cost_field(gray_crop: np.ndarray, frequencies=(0.06, 0.12),
                     n_theta: int = 8) -> np.ndarray:
    """(7) Bank filtrów Gabora — wiele orientacji i częstości naraz, bierzemy maksimum
    modułu odpowiedzi. Klasyczna metoda z biometrii (wzmacnianie linii papilarnych
    odcisku palca) — inny matematyczny "przepis" niż Frangi, ale ten sam cel: znajdź
    lokalne, wydłużone struktury liniowe."""
    g = gray_crop.astype(np.float64)
    rng = g.max() - g.min()
    g = (g - g.min()) / rng if rng > 1e-6 else g * 0.0
    best = np.zeros_like(g)
    for freq in frequencies:
        for i in range(n_theta):
            theta = i * np.pi / n_theta
            real, imag = gabor(g, frequency=freq, theta=theta)
            mag = np.hypot(real, imag)
            best = np.maximum(best, mag)
    return best.astype(np.float32)


def hough_circles_on_field(field_u8: np.ndarray, min_r: int, max_r: int) -> np.ndarray | None:
    """(8) Klasyczna transformata Hougha na okręgi (``cv2.HoughCircles``) — każdy piksel
    krawędzi "głosuje" na wszystkie okręgi, które mogłyby przez niego przechodzić;
    okrąg z największą liczbą głosów wygrywa. Zakłada, że szukany kształt jest
    KOŁEM — otolit śledzia nim nie jest (ma "ogon"), więc to świadomie ostry test
    tego założenia, nie tylko metoda do wypróbowania."""
    circles = cv2.HoughCircles(field_u8, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r // 2,
                               param1=60, param2=32, minRadius=min_r, maxRadius=max_r)
    return circles


def run_snake(field_cartesian: np.ndarray, cx: float, cy: float, R_theta: np.ndarray,
             angles: np.ndarray, t_init: float, n_points: int = 150) -> np.ndarray:
    """(9) Aktywny kontur / "snake" (Kass, Witkin, Terzopoulos 1987) — startuje jako
    okrąg przy promieniu ``t_init`` i pozwala mu się "przykleić" do najbliższego
    silnego miejsca w polu ``field_cartesian`` (tu: odpowiedź Frangiego), krok po
    kroku ciągnąc się w stronę jasnych pikseli (``w_line=1``)."""
    theta_pts = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    theta_wrapped = ((theta_pts + np.pi) % (2 * np.pi)) - np.pi
    R_interp = np.interp(theta_wrapped, angles, R_theta, period=2 * np.pi)
    r0 = t_init * R_interp
    rows = cy + r0 * np.sin(theta_pts)
    cols = cx + r0 * np.cos(theta_pts)
    init = np.stack([rows, cols], axis=1)
    snake = active_contour(field_cartesian, init, alpha=0.02, beta=8.0, w_line=1.0,
                          w_edge=0.0, gamma=0.005, max_num_iter=250, boundary_condition="periodic")
    return snake  # (n_points, 2) w kolejności (row=y, col=x)


def render_snake_on_photo(crop_rgb: np.ndarray, axis_info: dict,
                          snakes: list[tuple[np.ndarray, tuple]]) -> np.ndarray:
    out = np.ascontiguousarray(crop_rgb[..., :3]).copy()
    H = out.shape[0]
    if axis_info.get("contour") is not None:
        cv2.drawContours(out, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    for snake, color in snakes:
        pts = np.stack([snake[:, 1], snake[:, 0]], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=max(2, H // 400), lineType=cv2.LINE_AA)
    return out


def render_hough_on_photo(crop_rgb: np.ndarray, axis_info: dict, circles: np.ndarray | None,
                          max_draw: int = 6) -> np.ndarray:
    out = np.ascontiguousarray(crop_rgb[..., :3]).copy()
    H = out.shape[0]
    if axis_info.get("contour") is not None:
        cv2.drawContours(out, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    if circles is not None:
        # cv2.HoughCircles zwraca kandydatów w przybliżonej kolejności malejącej pewności —
        # rysujemy tylko najlepszych max_draw, żeby obraz był czytelny (surowa liczba
        # znalezionych kandydatów i tak podana w podpisie/tekście).
        for i, c in enumerate(circles[0][:max_draw]):
            x, y, r = c
            color = _RING_COLORS[i % len(_RING_COLORS)]
            cv2.circle(out, (int(x), int(y)), int(r), color, max(2, H // 400), cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Przetwarzanie jednego otolitu — liczy WSZYSTKIE pola raz, zwraca dict wyników
# ---------------------------------------------------------------------------

def process_otolith(model, cfg, device, image_id: str) -> dict:
    print(f"\n=== {image_id} ===", flush=True)
    image_dir = Path(cfg.data.image_dir)
    orig_rgb = np.array(PILImage.open(image_dir / image_id).convert("RGB"), dtype=np.uint8)
    axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                            nucleus_method=cfg.segmentation.nucleus_method)
    if axis_info is None:
        raise RuntimeError(f"Segmentacja nie powiodła się: {image_id}")
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
    print(f"  Wiek (CORAL): {predicted_age}", flush=True)

    density_transform = build_transforms(cfg.candidates.density_image_size, "test")
    density_tensor = density_transform(PILImage.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        density_grid = model.get_density_probs(density_tensor).squeeze(0).cpu().numpy()

    gray_crop = crop_rgb.mean(axis=2).astype(np.float32)
    R_theta, angles = ray_cast_R_theta(mask_cropped, cx, cy, N_ANGLE)

    print("  Pola: gradient, Frangi, Gabor, CLAHE+density+Frangi...", flush=True)
    polar_classical = _normalize(polar_unwrap(gray_crop, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))
    polar_grad = _normalize(np.abs(np.gradient(polar_classical, axis=0)))
    frangi_cart = frangi_cost_field(gray_crop)
    polar_frangi = _normalize(polar_unwrap(frangi_cart, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))
    gabor_cart = gabor_cost_field(gray_crop)
    polar_gabor = _normalize(polar_unwrap(gabor_cart, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))
    clahe_dens_cart = clahe_density_frangi_field(density_grid)
    polar_clahe_dens = _normalize(polar_unwrap(clahe_dens_cart, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))

    return dict(image_id=image_id, orig_rgb=orig_rgb, axis_info=axis_info, mask_arr=mask_arr,
               crop_rgb=crop_rgb, mask_cropped=mask_cropped, d_axis_info=d_axis_info,
               cx=cx, cy=cy, cw=cw, ch=ch, predicted_age=predicted_age,
               density_grid=density_grid, gray_crop=gray_crop, R_theta=R_theta, angles=angles,
               polar_grad=polar_grad, polar_frangi=polar_frangi, polar_gabor=polar_gabor,
               polar_clahe_dens=polar_clahe_dens, frangi_cart=frangi_cart, gabor_cart=gabor_cart)


# ---------------------------------------------------------------------------
# Renderowanie sekcji per metoda
# ---------------------------------------------------------------------------

def section_confident_arcs(name: str, field_key: str, description: str, tuning: str,
                           data_by_oto: list[dict]) -> str:
    imgs = []
    for d in data_by_oto:
        polar_norm = d[field_key]
        cost = apply_margins(1.0 - polar_norm, N_RADIUS, INNER_MARGIN, EDGE_MARGIN)
        arcs = find_confident_open_arcs(cost, 60, ARC_STRIDE_DEG, WINDOW, JUMP_PENALTY,
                                        FUSION_ARC_TOP_K, FUSION_ARC_MIN_GAP_DEG, FUSION_ARC_MIN_GAP_T)
        ts = sorted(round(a["mean_t"], 3) for a in arcs)
        print(f"  [{name}] {d['image_id'][:40]}...: {len(arcs)} fragmentów, t={ts}", flush=True)
        photo = render_open_arcs_on_photo(d["crop_rgb"], d["d_axis_info"], arcs, d["R_theta"], d["angles"], N_RADIUS)
        imgs.append(f"""<div style="display:inline-block;vertical-align:top;width:32%;margin:0 1% 10px 0;text-align:center;">
<b>{d['image_id'][:45]}</b><br>{len(arcs)} fragmentów, t={ts}<br>
{img_tag(_b64_from_rgb(photo, 360), style="width:360px;")}</div>""")
    return f"""<section>
<h2>{name}</h2>
<p style="font-size:90%;">{description}</p>
<p style="font-size:88%;color:#555;"><b>Sugestia strojenia:</b> {tuning}</p>
{"".join(imgs)}
</section>"""


def section_circular_shortest_path(data_by_oto: list[dict]) -> str:
    imgs = []
    for d in data_by_oto:
        cost = apply_margins(1.0 - d["polar_frangi"], N_RADIUS, INNER_MARGIN, EDGE_MARGIN)
        paths = extract_k_rings_ordered(cost, d["predicted_age"], WINDOW, JUMP_PENALTY, MIN_GAP_FRAC,
                                        WIDTH_CEILING_WEIGHT, WIDTH_DECAY_WEIGHT)
        rings_xy = [ring_path_to_xy(p, d["R_theta"], d["angles"], d["cx"], d["cy"], N_RADIUS) for p in paths]
        photo = render_rings_on_photo(d["crop_rgb"], d["d_axis_info"], rings_xy)
        ts = sorted(round(float(np.mean(p)) / (N_RADIUS - 1), 3) for p in paths)
        print(f"  [circular shortest path] {d['image_id'][:40]}...: {len(paths)} pierścieni, t={ts}", flush=True)
        imgs.append(f"""<div style="display:inline-block;vertical-align:top;width:32%;margin:0 1% 10px 0;text-align:center;">
<b>{d['image_id'][:45]}</b><br>K={d['predicted_age']}, znaleziono {len(paths)}, t={ts}<br>
{img_tag(_b64_from_rgb(photo, 360), style="width:360px;")}</div>""")
    return f"""<section>
<h2>3. Circular shortest path (najkrótsza ZAMKNIĘTA ścieżka), pole=Frangi + prior szerokości</h2>
<p style="font-size:90%;">Jeden ciągły graf kosztów z całego pola (bez binowania), szuka
GLOBALNIE najtańszej zamkniętej pętli — powtórzone K={"wiek"} razy (prior szerokości: pierwszy
przyrost najszerszy, kolejne nie szersze — patrz v5). Skąd: rodzina metod z segmentacji warstw
siatkówki OCT (Li i in. 2006) i liczenia słojów drzew (CS-TRD, 2023-2025).</p>
<p style="font-size:88%;color:#555;"><b>Sugestia strojenia:</b> <code>JUMP_PENALTY</code> (kara za
skok promienia) i <code>first_ring_max_t</code> — oba nieskalibrowane, dobrane "na oko".</p>
{"".join(imgs)}
</section>"""


def section_ransac(data_by_oto: list[dict]) -> str:
    imgs = []
    for d in data_by_oto:
        cost = apply_margins(1.0 - d["polar_frangi"], N_RADIUS, INNER_MARGIN, EDGE_MARGIN)
        arcs = find_confident_open_arcs(cost, 60, ARC_STRIDE_DEG, WINDOW, JUMP_PENALTY,
                                        FUSION_ARC_TOP_K, FUSION_ARC_MIN_GAP_DEG, FUSION_ARC_MIN_GAP_T)
        fit = ransac_fit_ring(arcs, d["angles"], N_RADIUS)
        n_in = len(fit["inlier_frag_idx"]) if fit else 0
        print(f"  [RANSAC] {d['image_id'][:40]}...: {n_in}/{len(arcs)} inlierów", flush=True)
        photo = render_fusion_on_photo(d["crop_rgb"], d["d_axis_info"], arcs, fit, d["R_theta"], d["angles"], d["cx"], d["cy"])
        imgs.append(f"""<div style="display:inline-block;vertical-align:top;width:32%;margin:0 1% 10px 0;text-align:center;">
<b>{d['image_id'][:45]}</b><br>{n_in}/{len(arcs)} fragmentów jako inliery<br>
{img_tag(_b64_from_rgb(photo, 360), style="width:360px;")}</div>""")
    return f"""<section>
<h2>5. RANSAC — składanie fragmentów w pierścień, odrzucanie odstających</h2>
<p style="font-size:90%;">Dopasowanie gładkiego modelu t(&theta;) (szereg Fouriera rzędu 2) do
zbioru pewnych fragmentów, odporne na fragmenty spoza tego samego pierścienia. Skąd: Fischler
&amp; Bolles 1981 — klasyka widzenia komputerowego (np. granica tęczówki/źrenicy częściowo
zasłoniętej powiekami). Odłożone na razie na prośbę użytkownika — pokazane tu tylko jako
zestawienie, nie jako kierunek do kontynuowania.</p>
<p style="font-size:88%;color:#555;"><b>Sugestia strojenia:</b> tolerancja inliera (dziś 0,035) i
rząd szeregu Fouriera (dziś 2) — wyższy rząd = więcej swobody kształtu, ale ryzyko przeuczenia na
małej liczbie fragmentów.</p>
{"".join(imgs)}
</section>"""


def section_hough(data_by_oto: list[dict]) -> str:
    imgs = []
    for d in data_by_oto:
        H, W = d["frangi_cart"].shape[:2]
        field_u8 = (np.clip(d["frangi_cart"], 0, 1) * 255).astype(np.uint8)
        field_u8 = cv2.GaussianBlur(field_u8, (5, 5), 0)
        min_r = int(0.15 * min(H, W))
        max_r = int(0.48 * min(H, W))
        circles = hough_circles_on_field(field_u8, min_r, max_r)
        n_found = 0 if circles is None else circles.shape[1]
        print(f"  [Hough] {d['image_id'][:40]}...: {n_found} okręgów", flush=True)
        photo = render_hough_on_photo(d["crop_rgb"], d["d_axis_info"], circles)
        imgs.append(f"""<div style="display:inline-block;vertical-align:top;width:32%;margin:0 1% 10px 0;text-align:center;">
<b>{d['image_id'][:45]}</b><br>{n_found} okręgów znalezionych<br>
{img_tag(_b64_from_rgb(photo, 360), style="width:360px;")}</div>""")
    return f"""<section>
<h2>8. Transformata Hougha (klasyczne "głosowanie" na okręgi)</h2>
<p style="font-size:90%;">Każdy piksel krawędzi (tu: odpowiedzi Frangiego) "głosuje" na
wszystkie możliwe okręgi przechodzące przezeń; okrąg z największą liczbą głosów wygrywa.
Bardzo stara (Hough 1962), niezawodna metoda — ALE zakłada, że szukany kształt jest KOŁEM.
Otolit śledzia nie jest kołowy (ma "ogon"/wcięcie) — to świadomie ostry test tego założenia.
Rysowane jest tylko najlepszych 6 kandydatów (dla czytelności) — surowa liczba znaleziona przez
OpenCV podana w podpisie.</p>
<p style="font-size:88%;color:#555;"><b>Wynik nieoptymistyczny:</b> setki (100+) kandydatów o
podobnej ocenie — brak jednego wyraźnego zwycięzcy, bo ŻADEN okrąg nie pasuje dobrze do
nie-kołowego kształtu otolitu. To sam w sobie wynik: potwierdza, że czysty Hough-na-okręgi w
pikselach x/y nie jest dobrym dopasowaniem do tego problemu.
<b>Sugestia strojenia:</b> <code>param2</code> (próg głosów) i zakres promienia; ALTERNATYWA:
liczyć Hougha w przestrzeni (kąt, t) zamiast pikseli x/y (szukanie poziomych linii zamiast
okręgów) — nie wypróbowane w tym raporcie, prawdopodobnie sensowniejsze.</p>
{"".join(imgs)}
</section>"""


def section_snake(data_by_oto: list[dict]) -> str:
    imgs = []
    t_inits = [0.3, 0.6, 0.85]
    colors = [(230, 30, 30), (30, 170, 230), (230, 170, 30)]
    for d in data_by_oto:
        snakes = []
        for t0, color in zip(t_inits, colors):
            try:
                snake = run_snake(d["frangi_cart"], d["cx"], d["cy"], d["R_theta"], d["angles"], t0)
                snakes.append((snake, color))
            except Exception as e:
                print(f"  [snake] {d['image_id'][:30]} t0={t0}: BŁĄD {e}", flush=True)
        print(f"  [snake] {d['image_id'][:40]}...: {len(snakes)}/{len(t_inits)} zbieżnych", flush=True)
        photo = render_snake_on_photo(d["crop_rgb"], d["d_axis_info"], snakes)
        imgs.append(f"""<div style="display:inline-block;vertical-align:top;width:32%;margin:0 1% 10px 0;text-align:center;">
<b>{d['image_id'][:45]}</b><br>starty t&#8320;={t_inits}<br>
{img_tag(_b64_from_rgb(photo, 360), style="width:360px;")}</div>""")
    return f"""<section>
<h2>9. Aktywne kontury / "snake"</h2>
<p style="font-size:90%;">Startuje jako okrąg przy zadanym promieniu i "przykleja się" krok po
kroku do najbliższego silnego miejsca w polu Frangiego — jak rozciągnięta gumka opadająca na
najbliższą krawędź. Skąd: Kass, Witkin, Terzopoulos, 1987 — klasyka segmentacji obrazu
medycznego (np. kontur serca/narządu w USG/MRI). Trzy kolory = trzy różne promienie startowe
(0,3 / 0,6 / 0,85) na tym samym zdjęciu — czy zbiegają do różnych, sensownych pierścieni, czy
wszystkie do tego samego miejsca?</p>
<p style="font-size:88%;color:#555;"><b>Sugestia strojenia:</b> <code>alpha</code>/<code>beta</code>
(sztywność/gładkość konturu) i liczba punktów — zbyt niska sztywność pozwala konturowi
"pozwijać się" w szum zamiast trzymać kształt pierścienia.</p>
{"".join(imgs)}
</section>"""


def section_chain_linking(data_by_oto: list[dict]) -> str:
    imgs = []
    for d in data_by_oto:
        result = classical_increments(d["gray_crop"], d["d_axis_info"], n_dirs=48, n_samples=64)
        peaks = result["peaks"]
        clusters = _cluster_by_radius_with_arcs(peaks, t_tol=0.06, n_dirs=48)
        memberships = assign_rays_to_clusters(peaks, clusters, t_tol=0.06, n_dirs=48)
        print(f"  [chain-linking] {d['image_id'][:40]}...: {len(clusters)} klastrów/łańcuchów", flush=True)
        photo = render_arc_cluster_overlay(d["crop_rgb"], d["d_axis_info"], memberships, n_dirs=48)
        imgs.append(f"""<div style="display:inline-block;vertical-align:top;width:32%;margin:0 1% 10px 0;text-align:center;">
<b>{d['image_id'][:45]}</b><br>{len(clusters)} łańcuchów (kolor=łańcuch, grube=rdzeń, cienkie=reszta)<br>
{img_tag(_b64_from_rgb(photo, 360), style="width:360px;")}</div>""")
    return f"""<section>
<h2>10. Łączenie łańcuchów krawędzi ("spider web", Rodin &amp; Troadec / CS-TRD)</h2>
<p style="font-size:90%;">Zamiast dopasowywać jeden gładki model (RANSAC), łączy PIKI z
SĄSIEDNICH promieni w łańcuchy (jeśli leżą blisko siebie po promieniu i po kącie) — mocna,
zwarta łukowa reszta wygrywa z rozproszonym szumem o tym samym wsparciu. Skąd: Rodin &amp;
Troadec 1997 (odczyt wieku ryb wprost!) i nowoczesny CS-TRD (słoje drzew, 2023). <b>Ciekawostka:</b>
ten mechanizm JUŻ ISTNIEJE w produkcyjnym kodzie tego projektu
(<code>src/ring_extraction.py::_cluster_by_radius_with_arcs</code>,
<code>assign_rays_to_clusters</code>, <code>src/visualization.py::render_arc_cluster_overlay</code>)
— tu tylko pokazany wprost, nie zaimplementowany od nowa.</p>
<p style="font-size:88%;color:#555;"><b>Wynik do dostrojenia:</b> na tych 2 zdjęciach powstało
~11 osobnych, drobnych klastrów zamiast kilku wyraźnych łańcuchów — sygnał klasyczny (jasność) na
tych konkretnych zdjęciach jest widocznie bardziej poszarpany niż na przykładach, na których
`t_tol=0,06` było wcześniej kalibrowane. <b>Sugestia strojenia:</b> większe <code>t_tol</code>
(tolerancja promieniowa łączenia w klaster) i/lub większe <code>max_gap</code> (ile pominiętych
promieni w łańcuchu jeszcze tolerujemy) — warte ponownego sprawdzenia na tym konkretnym
zestawieniu, nie tylko na przykładach z 20.07/29.07.</p>
{"".join(imgs)}
</section>"""


def section_raw_field(name_num: str, field_key: str, description: str, tuning: str,
                      data_by_oto: list[dict]) -> str:
    imgs = []
    for d in data_by_oto:
        b64 = render_unwrap(d[field_key], [], f"{name_num}: {d['image_id'][:35]}")
        imgs.append(f"""<div style="display:inline-block;vertical-align:top;width:49%;margin:0 0.5% 10px 0;text-align:center;">
<b>{d['image_id'][:45]}</b><br>{img_tag(b64, style="width:100%;")}</div>""")
    return f"""<section>
<h2>{name_num}</h2>
<p style="font-size:90%;">{description}</p>
<p style="font-size:88%;color:#555;"><b>Sugestia strojenia:</b> {tuning}</p>
{"".join(imgs)}
</section>"""


def main() -> None:
    base = MODEL_VARIANTS[0]
    cfg = load_merged_config(base["cfg_path"], None)
    device = torch.device(cfg.training.device if cfg.training.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Wczytywanie modelu ({base['name']})...", flush=True)
    model = load_model_from_checkpoint(cfg, base["run_dir"] / "checkpoints" / "embedded" / "best.pt")
    model.eval()
    device = next(model.parameters()).device

    data_by_oto = [process_otolith(model, cfg, device, o["image_id"]) for o in OTOLITHS]

    print("\nGeneruję sekcje raportu...", flush=True)
    sections = []
    sections.append(section_raw_field(
        "1. Gradient promieniowy — surowe pole (wejście do metody 4)",
        "polar_grad",
        "Dla każdego promienia patrzymy, jak szybko zmienia się jasność w miarę oddalania się "
        "od jądra. Tani, bezzależnościowy odpowiednik filtru grzbietowego. Skąd: fundament "
        "klasycznej detekcji krawędzi (Sobel/Canny), przetwarzanie obrazu od lat 60-70.",
        "brak nowych parametrów (surowa pochodna) — jedyna \"regulacja\" to wygładzenie przed "
        "różniczkowaniem, nie wypróbowane.",
        data_by_oto))
    sections.append(section_raw_field(
        "2. Filtr Frangiego — surowe pole (wejście do metod 3-6, RANSAC, snake)",
        "polar_frangi",
        "Sprawdza w każdym pikselu, jak bardzo lokalne otoczenie wygląda na fragment cienkiej, "
        "krzywoliniowej linii (macierz Hessego, wiele skal naraz). Skąd: Frangi i in. 1998 — "
        "wykrywanie naczyń krwionośnych na zdjęciach medycznych (ten sam problem geometryczny).",
        "zakres skal <code>sigmas=range(2,10,2)</code> — dobrany pod grubość widocznych pasm na "
        "tych zdjęciach, nie skalibrowany systematycznie.",
        data_by_oto))
    sections.append(section_confident_arcs(
        "4. Pewne otwarte fragmenty łuku — na polu Frangiego (bez wymuszania zamknięcia)",
        "polar_frangi",
        "Zamiast całej zamkniętej pętli, szuka najtańszych OTWARTYCH odcinków 60&deg; w wielu "
        "miejscach, zatrzymuje tylko najpewniejsze (różne promienie i kąty, nie duplikaty).",
        "długość fragmentu (dziś 60&deg;) i liczba zwracanych fragmentów (dziś 12) — dłuższe "
        "fragmenty = mocniejszy dowód ciągłości, ale mniej lokalnej elastyczności.",
        data_by_oto))
    sections.append(section_circular_shortest_path(data_by_oto))
    sections.append(section_ransac(data_by_oto))
    sections.append(section_raw_field(
        "6. CLAHE + mapa density + Frangi — surowe pole",
        "polar_clahe_dens",
        "Wyrównanie lokalnego kontrastu (CLAHE) na mapie density modelu, potem powiększenie i "
        "filtr Frangiego — próba \"wyciągnięcia\" czegokolwiek z rzadkiego sygnału density "
        "(surowa wersja jest ~99% pusta, patrz <code>31.07_radial_Analiza_wnioski.md</code>).",
        "<code>clipLimit</code>/<code>tileGridSize</code> CLAHE (dziś 2,0 / 4&times;4) — "
        "zbyt agresywne wyrównanie może wzmocnić szum tak samo jak sygnał.",
        data_by_oto))
    sections.append(section_raw_field(
        "7. Bank filtrów Gabora — surowe pole",
        "polar_gabor",
        "Wiele filtrów wykrywających linie pod RÓŻNYMI kątami osobno (nie jedna, uśredniona "
        "wielo-skalowa odpowiedź jak Frangi) — bierzemy maksimum. Skąd: klasyka biometrii, "
        "wzmacnianie linii papilarnych odcisku palca.",
        "częstości (dziś 0,06/0,12) i liczba orientacji (dziś 8) — więcej orientacji = gładszy "
        "wynik, ale wolniej.",
        data_by_oto))
    sections.append(section_hough(data_by_oto))
    sections.append(section_snake(data_by_oto))
    sections.append(section_chain_linking(data_by_oto))

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Test wszystkich metod wykrywania fragmentów ({VERSION})</title>
<style>
body {{font-family:sans-serif;max-width:1150px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
h1 {{color:#1a237e;}} h2 {{color:#1a237e;font-size:110%;}}
code {{background:#f0f0f5;padding:1px 4px;border-radius:3px;}}
</style>
</head>
<body>
<h1>Test wszystkich metod wykrywania fragmentów przyrostów (2 otolity)</h1>
<p>Kontynuacja v1-v7 (patrz <code>outputs/31.07_ring_shortest_path/report_v5.html</code>,
<code>report_v6_fragment_fusion.html</code>, <code>report_v7_fragment_detection.html</code> i
pamięć projektu) — zestawienie WSZYSTKICH wypróbowanych do tej pory metod plus nowych, które NIE
wymagają adnotacji (bez INBD, bez uczonych detektorów krawędzi). Dwa otolity: <b>{OTOLITHS[0]['label']}</b>
oraz <b>{OTOLITHS[1]['label']}</b>. Nic w <code>src/</code> nie zostało zmienione (metoda 10
pokazuje kod JUŻ ISTNIEJĄCY w produkcji, nie nowy).</p>
{"".join(sections)}
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
