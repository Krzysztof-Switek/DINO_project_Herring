"""31.07 — Prototyp Fazy A: circular/closed shortest-path (graph-based ring tracing).

Kontekst: E9 (prior koncentryczności w stracie treningowej) zamknięty tego samego dnia z
wynikiem NULL na wszystkich trzech testowanych wagach — patrz
``outputs/31.07_radial_Analiza_wnioski.md``. Przegląd literatury
(``plans and summaries/30.07_wykrywanie_przyrostów_LIRERATURA.md`` §2.1/§4) wskazał jako
następny krok śledzenie pierścienia jako najkrótszej ZAMKNIĘTEJ ścieżki na grafie — metoda
POST-HOC (bez treningu), która naprawia to, co dwie już wypróbowane metody wykrywania
kandydatów (48 dosłownych linii, i sektorowa agregacja kątowa z 30.07, odrzucona -15%) mają
wspólnego: obie TNĄ obwód na niezależne kierunki, decydują "czy tu jest pik" OSOBNO w każdym,
dopiero POTEM sklejają w pierścień. Tu kolejność jest odwrócona: jeden graf kosztów z CAŁEGO
ciągłego pola (interpolacja dwuliniowa, bez binowania/MAX), a gładkość pierścienia wymuszona
PRZEZ SAMĄ optymalizację (kara za skok promienia), nie przez osobny post-hoc krok punktujący.

Ten skrypt to Faza A (jeden otolit, wizualny prototyp) — src/ CAŁKOWICIE niedotknięte, zgodnie
z ustaloną w projekcie zasadą "dowód w diagnostyce najpierw, decyzja o promocji osobno" (patrz
sweep_forced_topk_peaks.py, polar_sector_coverage_report.py). Faza B (walidacja na 30 kartach,
ten sam protokół co poprzednie sweepy) to osobny, następny krok — NIE wykonywany tutaj.

Model/checkpoint: ``configs/config.yaml`` + ``outputs/22.07_reg/checkpoints/embedded/best.pt``
— produkcyjna baza BEZ wagi E9 (E9 dał zerowy efekt, nie ma sensu dziedziczyć go w nowym
eksperymencie, patrz uzasadnienie w ``outputs/31.07_radial_Analiza_wnioski.md``).

Cztery niezależne pola wejściowe testowane tym samym mechanizmem grafu (ten sam kod, inny sygnał
wejściowy) — żeby odróżnić "czy DZIAŁA SAM MECHANIZM" od "czy pomaga na dzisiejszym, wiadomo że
bardzo rzadkim, sygnale density":
  1. density   — mapa gęstości modelu (to, o co pytał przegląd literatury wprost), sprawdzona na
     DWÓCH rozdzielczościach (518px produkcyjna / 966px hires966) — czy więcej detalu pomaga
     mechanizmowi, mimo że sama uwaga jest bardziej rozproszona (28.07_966_comparison_TO_DO.md),
  2. classical (jasność) — surowa jasność zdjęcia,
  3. classical (gradient promieniowy) — tani, bezzależnościowy odpowiednik filtru grzbietowego,
  4. classical (Frangi) — PRAWDZIWY filtr grzbietowy (`skimage.filters.frangi`), zainstalowany
     specjalnie do tego eksperymentu.

DODATKOWO (odpowiedź na pytanie użytkownika "czy koniecznie musimy szukać całego pierścienia"):
przyrosty bywają widoczne tylko na CZĘŚCI obwodu — wymuszanie PEŁNEJ zamkniętej pętli przez
słabo widoczne fragmenty może psuć wynik nawet tam, gdzie sygnał jest mocny. Sekcja
"pewne fragmenty łuków" szuka NAJTAŃSZYCH OTWARTYCH (nie zamkniętych) fragmentów o ustalonej
długości kątowej, bez wymuszania zamknięcia — dokładnie ten pomysł, który w literaturze
(patrz przegląd niżej: RANSAC-fitting z fragmentów, Rodin & Troadec, CS-TRD "spider web", INBD)
jest już dobrze ugruntowaną rodziną metod.

Usage: PYTHONIOENCODING=utf-8 python scripts/diagnostics/ring_shortest_path_report.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from skimage.filters import frangi

from src.report_common import fig_to_b64, img_tag
from src.visualization import _CONTOUR_COLOR, _AXIS_COLOR
from src.otolith_axis import (detect_axis, apply_background_mask, mask_bbox, shift_axis_info,
                              sample_profile_along_axis)
from src.dataset import build_transforms, decode_age_ordinal
from src.inference import load_model_from_checkpoint
from src.candidates import find_candidate_peaks
from src.ring_extraction import select_increments
from scripts.run_pipeline import load_merged_config

# 31.07 cd. — użytkownik zwrócił uwagę na realny efekt biologiczny: ryby złowione w rejsach
# BITS1q/BITS2q (I/II kwartał) mają, wg dotychczasowej (niezweryfikowanej etykietowo,
# [[quarter_age_convention_issue]]) hipotezy, jeszcze NIEUFORMOWANY w pełni tegoroczny
# pierścień w chwili połowu — więc REALNA liczba w pełni uformowanych, czytelnych przyrostów
# to `wiek - 1`, a "wiek"-ty przyrost jest dopiero FORMUJĄCY SIĘ tuż przy brzegu (nie licz go
# jak pełny pierścień). Trzy otolity z TEJ SAMEJ lokalizacji/FishIndex (różne lata/kwartały)
# wybrane do eksploracji: młoda ryba (wiek 1, Q1), średnia (wiek 2, Q4 — kontrola, BEZ
# korekty kwartału) i ta z poprzednich sesji (wiek 4, Q1).
OTOLITHS = [
    {"image_id": "2023_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg",
     "label": "FishIndex95 / 2023 BITS1q / etykieta wieku=4 (Q1 &rarr; oczekiwane: 3 pełne + 1 formujący się na brzegu)",
     "show_density": True, "arc_lens": [60, 120]},
    {"image_id": "2023_BITS4q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg",
     "label": "FishIndex95 / 2023 BITS4q / etykieta wieku=2 (Q4 &rarr; kontrola: BEZ korekty, pierścień roczny już uformowany)",
     "show_density": False, "arc_lens": [60]},
    {"image_id": "2024_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg",
     "label": "FishIndex95 / 2024 BITS1q / etykieta wieku=1 (Q1, młoda ryba &rarr; oczekiwane: 0 pełnych + 1 formujący się)",
     "show_density": False, "arc_lens": [60]},
]


def _quarter_of(image_id: str) -> str | None:
    m = re.search(r"BITS([1-4])q", image_id)
    return f"BITS{m.group(1)}q" if m else None


def _quarter_adjustment(image_id: str) -> int:
    """1 = odejmij jeden przyrost od K (Q1/Q2 — tegoroczny pierścień jeszcze się formuje),
    0 = bez korekty (Q3/Q4 — pierścień roczny już w pełni uformowany do dnia połowu)."""
    return 1 if _quarter_of(image_id) in ("BITS1q", "BITS2q") else 0


MODEL_VARIANTS = [
    {"name": "baseline (518px, 22.07_reg)", "slug": "baseline",
     "cfg_path": PROJECT_ROOT / "configs" / "config.yaml",
     "run_dir": PROJECT_ROOT / "outputs" / "22.07_reg"},
    {"name": "hires966 (966px, 23.07_hires966)", "slug": "hires966",
     "cfg_path": PROJECT_ROOT / "configs" / "config_hires966.yaml",
     "run_dir": PROJECT_ROOT / "outputs" / "23.07_hires966"},
]

OUT_DIR = PROJECT_ROOT / "outputs" / "31.07_ring_shortest_path"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION = "v5"
OUT_PATH = OUT_DIR / f"report_{VERSION}.html"

# --- Parametry grafu / DP (rząd wielkości "do wypróbowania i obejrzenia", nie skalibrowane
# na danych — jawnie oznaczone, ten sam status co inner_margin=0.20 gdzie indziej w projekcie) ---
N_ANGLE = 360          # kolumny rozwinięcia biegunowego (rozdzielczość kątowa, 1 stopień/kolumna)
N_RADIUS = 100         # wiersze (rozdzielczość promieniowa t=0..1)
WINDOW = 4             # maks. skok promienia (w binach wierszy) między sąsiednimi kolumnami kąta
JUMP_PENALTY = 0.4     # waga kary za skok promienia (na jednostkę delta^2) względem kosztu sygnału
BLOCK_HALF = 4         # promień "blokady" (w binach) wokół już znalezionego pierścienia, dla K>1
INNER_MARGIN = 0.20    # jak w produkcji (configs/config.yaml candidates.inner_margin)
EDGE_MARGIN = 0.08     # jak w produkcji (domyślne edge_margin w src/ring_extraction.py)
BIG_COST = 1.0e6       # koszt "zakazany" (margines jądra/krawędzi, zablokowany pierścień)

# --- Prior szerokości (31.07 cd.) — te same liczby co produkcyjny config.yaml, świadomie
# powielone, nie wymyślone od nowa (patrz extract_k_rings_ordered). ---
MIN_GAP_FRAC = 0.02
WIDTH_CEILING_WEIGHT = 3.0
WIDTH_DECAY_WEIGHT = 1.0

# --- Parametry ekstrakcji PEWNYCH OTWARTYCH FRAGMENTÓW łuku (bonus, patrz docstring modułu) ---
ARC_LEN_DEG = 60       # długość kątowa jednego testowanego fragmentu (stopnie)
ARC_STRIDE_DEG = 6     # co ile stopni próbujemy nowy start fragmentu
ARC_TOP_K = 6          # ile najpewniejszych (najtańszych/stopień) fragmentów pokazać
ARC_MIN_GAP_DEG = 20   # min. odstęp kątowy między środkami wybranych fragmentów (NMS)

_RING_COLORS = [
    (230, 30, 30), (30, 170, 230), (230, 170, 30), (160, 30, 200),
    (30, 200, 120), (230, 30, 150), (150, 150, 30), (30, 120, 230),
    (200, 90, 30), (90, 200, 200), (150, 90, 220), (230, 230, 30),
    (30, 60, 150), (150, 30, 30), (30, 150, 90), (200, 140, 200),
]


def _b64_from_rgb(arr: np.ndarray, target_w: int = 520) -> str:
    img = PILImage.fromarray(np.ascontiguousarray(arr[..., :3]).astype(np.uint8))
    if img.width > target_w:
        scale = target_w / img.width
        img = img.resize((target_w, max(1, int(round(img.height * scale)))), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Geometria biegunowa: R(theta) + rozwinięcie CIĄGŁE (bez binowania) przez cv2.remap
# ---------------------------------------------------------------------------

def ray_cast_R_theta(mask: np.ndarray, cx: float, cy: float, n_angle: int) -> tuple[np.ndarray, np.ndarray]:
    """Promień otolitu w każdym z n_angle kierunków — ta sama metoda co
    ``otolith_axis.compute_polar_grid`` (rzucanie promieni po masce), powielona tu lokalnie
    (świadomie, prototyp nie zmienia src/ — ten sam wzorzec co ``_bin_idx_grid`` w
    ``polar_sector_coverage_report.py``). Zwraca (R_theta, angles), oba (n_angle,)."""
    mask_bin = np.asarray(mask) > 0
    H, W = mask_bin.shape[:2]
    angles = np.linspace(-np.pi, np.pi, n_angle, endpoint=False)
    r_max = float(np.hypot(max(cx, W - cx), max(cy, H - cy))) + 1.0
    radii = np.arange(1.0, max(r_max, 2.0))
    R_theta = np.zeros(n_angle, dtype=np.float32)
    for i, a in enumerate(angles):
        xs = np.clip((cx + radii * np.cos(a)).astype(np.int64), 0, W - 1)
        ys = np.clip((cy + radii * np.sin(a)).astype(np.int64), 0, H - 1)
        inside = np.nonzero(mask_bin[ys, xs])[0]
        R_theta[i] = float(radii[inside.max()]) if inside.size else 0.0
    return R_theta, angles


def polar_unwrap(field: np.ndarray, field_w_px: float, field_h_px: float,
                 R_theta: np.ndarray, angles: np.ndarray, cx: float, cy: float,
                 n_radius: int) -> np.ndarray:
    """Ciągłe (dwuliniowo interpolowane) rozwinięcie biegunowe pola 2D ``field``.

    ``field`` może być w DOWOLNEJ rozdzielczości (patch grid density_head ALBO surowy
    grayscale crop) — ``field_w_px``/``field_h_px`` to rozmiar oryginalnego obszaru
    (w pikselach), na który ``field`` jest rozciągnięte, żeby poprawnie przeliczyć pixel
    -> indeks tablicy ``field``. W ODRÓŻNIENIU od agregacji sektorowej (30.07, MAX per bin)
    tu KAŻDY punkt (kąt, promień) dostaje WŁASNĄ, ciągłą, interpolowaną wartość — zero
    binowania, zero utraty informacji z definicji (zgodnie z planem Fazy A).

    Zwraca (n_radius, n_angle) float32 — wiersz 0 = jądro (t=0), wiersz n_radius-1 = własny
    brzeg otolitu w danym kierunku (t=1); kolumna = kąt.
    """
    n_angle = len(angles)
    Hf, Wf = field.shape[:2]
    t_vals = np.linspace(0.0, 1.0, n_radius, dtype=np.float32)
    radius_px = t_vals[:, None] * R_theta[None, :]                     # (n_radius, n_angle)
    x = cx + radius_px * np.cos(angles)[None, :]
    y = cy + radius_px * np.sin(angles)[None, :]
    map_x = (x * (Wf / field_w_px)).astype(np.float32)
    map_y = (y * (Hf / field_h_px)).astype(np.float32)
    return cv2.remap(field.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Graf kosztów + najkrótsza ZAMKNIĘTA ścieżka (DP, zawinięcie 0/360 przez próbę KAŻDEGO
# startowego promienia r0, ten sam duch co _dp_select_t gdzie indziej w projekcie)
# ---------------------------------------------------------------------------

def _shift(arr: np.ndarray, delta: int, axis: int) -> np.ndarray:
    """arr przesunięty o `delta` wzdłuż `axis`, brzegi wypełnione +inf (bez zawijania —
    to promień, NIE kąt; zawijanie kąta obsługuje sama pętla po kolumnach)."""
    out = np.full_like(arr, np.inf)
    n = arr.shape[axis]
    if delta >= 0:
        src = [slice(None)] * arr.ndim; src[axis] = slice(0, n - delta)
        dst = [slice(None)] * arr.ndim; dst[axis] = slice(delta, n)
    else:
        src = [slice(None)] * arr.ndim; src[axis] = slice(-delta, n)
        dst = [slice(None)] * arr.ndim; dst[axis] = slice(0, n + delta)
    out[tuple(dst)] = arr[tuple(src)]
    return out


def find_best_ring(cost: np.ndarray, window: int, jump_penalty: float) -> tuple[np.ndarray, float]:
    """Najtańsza ZAMKNIĘTA ścieżka przez ``cost`` (n_radius, n_angle) — jedna pełna pętla
    0°..360°, dokładnie jeden wiersz (promień) na kolumnę, sąsiednie kolumny ograniczone do
    promieni w oknie ``+-window`` (gładkość), z kwadratową karą za skok. Rozwiązuje
    zawinięcie kąta próbując KAŻDY możliwy startowy promień r0 w kolumnie 0 (n_radius
    niezależnych przebiegów DP, zwektoryzowane razem jako macierz (r0 x r_biezacy)), potem
    dobiera najtańsze DOMKNIĘCIE pętli (powrót z ostatniej kolumny do r0). Druga, 1-wymiarowa
    faza (ustalone r0) odtwarza rzeczywistą ścieżkę przez wskaźniki wsteczne.

    Zwraca (path, total_cost): ``path`` (n_angle,) int — wybrany wiersz (promień) na każdą
    kolumnę (kąt); ``total_cost`` — koszt tej pętli.
    """
    n_radius, n_angle = cost.shape
    idx = np.arange(n_radius)

    # --- Faza 1: wektoryzowana po WSZYSTKICH startowych promieniach r0 naraz. ---
    dp = np.full((n_radius, n_radius), np.inf, dtype=np.float64)     # dp[r0, r_biezacy]
    dp[idx, idx] = cost[idx, 0]
    for a in range(1, n_angle):
        best = np.full_like(dp, np.inf)
        for delta in range(-window, window + 1):
            cand = _shift(dp, delta, axis=1) + jump_penalty * (delta ** 2)
            best = np.minimum(best, cand)
        dp = best + cost[np.newaxis, :, a]

    closing = np.full(n_radius, np.inf)
    for delta in range(-window, window + 1):
        r_last = idx + delta
        valid = (r_last >= 0) & (r_last < n_radius)
        vals = np.full(n_radius, np.inf)
        vals[valid] = dp[idx[valid], r_last[valid]] + jump_penalty * (delta ** 2)
        closing = np.minimum(closing, vals)
    best_r0 = int(np.argmin(closing))
    total_cost = float(closing[best_r0])

    # --- Faza 2: 1D DP z jawnymi wskaźnikami wstecznymi, ustalone r0=best_r0, do odtworzenia ścieżki. ---
    dp1 = np.full(n_radius, np.inf)
    dp1[best_r0] = cost[best_r0, 0]
    backptr = np.zeros((n_angle, n_radius), dtype=np.int64)
    for a in range(1, n_angle):
        new_dp1 = np.full(n_radius, np.inf)
        choice = np.full(n_radius, -1, dtype=np.int64)
        for delta in range(-window, window + 1):
            r_prev = idx - delta
            valid = (r_prev >= 0) & (r_prev < n_radius)
            cand = np.full(n_radius, np.inf)
            cand[valid] = dp1[r_prev[valid]] + jump_penalty * (delta ** 2)
            better = cand < new_dp1
            new_dp1 = np.where(better, cand, new_dp1)
            choice = np.where(better, r_prev, choice)
        backptr[a] = choice
        dp1 = new_dp1 + cost[:, a]

    best_r_last, best_val = best_r0, np.inf
    for delta in range(-window, window + 1):
        r_last = best_r0 + delta
        if 0 <= r_last < n_radius and dp1[r_last] + jump_penalty * (delta ** 2) < best_val:
            best_val = dp1[r_last] + jump_penalty * (delta ** 2)
            best_r_last = r_last

    path = np.zeros(n_angle, dtype=np.int64)
    path[n_angle - 1] = best_r_last
    for a in range(n_angle - 1, 0, -1):
        path[a - 1] = backptr[a, path[a]]
    path[0] = best_r0
    return path, total_cost


def block_ring(cost: np.ndarray, path: np.ndarray, block_half: int) -> np.ndarray:
    """Kopia ``cost`` z pasem +-block_half wokół ``path`` ustawionym na BIG_COST — żeby
    kolejne wywołanie find_best_ring nie znalazło tego samego pierścienia ponownie."""
    out = cost.copy()
    n_radius, n_angle = cost.shape
    for a in range(n_angle):
        lo = max(0, int(path[a]) - block_half)
        hi = min(n_radius, int(path[a]) + block_half + 1)
        out[lo:hi, a] = BIG_COST
    return out


def extract_k_rings(cost: np.ndarray, k: int, window: int, jump_penalty: float,
                    block_half: int) -> list[np.ndarray]:
    """Zachłanna ekstrakcja K pierścieni: znajdź najtańszy, zablokuj pas wokół niego,
    powtórz. (Faza C w planie literaturowym — wspólna, jednoczesna optymalizacja K
    nieprzecinających się ścieżek — zostaje jako rozszerzenie NA PRZYSZŁOŚĆ, jeśli to
    podejście zachłanne pokaże obiecujący sygnał.)"""
    cur = cost.copy()
    paths = []
    for _ in range(max(0, k)):
        path, _ = find_best_ring(cur, window, jump_penalty)
        paths.append(path)
        cur = block_ring(cur, path, block_half)
    # Posortuj wewnątrz-na-zewnątrz (średni promień rosnąco) — czytelniejsza kolejność wyświetlania.
    paths.sort(key=lambda p: float(np.mean(p)))
    return paths


def extract_k_rings_ordered(cost: np.ndarray, k: int, window: int, jump_penalty: float,
                            min_gap_frac: float = 0.02,
                            width_ceiling_weight: float = 3.0,
                            width_decay_weight: float = 1.0,
                            first_ring_max_t: float = 0.55) -> list[np.ndarray]:
    """Ekstrakcja K pierścieni OD JĄDRA NA ZEWNĄTRZ (nie: globalnie najtańszy pierwszy jak
    ``extract_k_rings``) — mirror biologicznego priora z produkcyjnego `_dp_select_t`
    (`src/ring_extraction.py`, `configs/config.yaml::candidates.width_decay_weight/
    width_ceiling_weight`): pierwszy przyrost od PRAWDZIWEGO jądra jest zwykle najszerszy,
    kolejne nie powinny go przekraczać i ogólnie się zwężają (potwierdzone w literaturze
    rybackiej — przyrosty gęstnieją przy brzegu wraz z wiekiem, malejące tempo wzrostu).

    Każdy kolejny pierścień szukany WYŁĄCZNIE poza poprzednim (twardy zakaz cofania się,
    ``min_gap_frac``), z miękką karą kwadratową za odstęp WIĘKSZY niż: (a) odstęp
    pierwszego pierścienia od jądra (``width_ceiling_weight`` — "sufit"), (b) odstęp
    poprzedniego kroku (``width_decay_weight`` — "malejąco"). Wagi 0 wyłączają odpowiedni
    człon (domyślne 3.0/1.0 = te same liczby co produkcyjny config, świadomie powielone,
    nie wymyślone od nowa).

    ``first_ring_max_t`` (NOWE, naprawia realny błąd znaleziony 31.07 cd.): bez tego pierwsze
    wywołanie ``find_best_ring`` szuka GLOBALNIE najtańszego pierścienia GDZIEKOLWIEK — jeśli
    to wychodzi blisko brzegu (co się zdarzało na density/gradiencie), "sufit szerokości"
    zostaje ustalony na tej podstawie i nie zostaje już miejsca na ŻADEN kolejny pierścień
    (twardy zakaz cofania + margines krawędzi razem zamykają cały pozostały promień). Twardo
    ogranicza WYŁĄCZNIE pierwsze wywołanie do t &le; ``first_ring_max_t`` — pierwszy przyrost
    "od prawdziwego jądra" nie powinien i tak siedzieć w zewnętrznej połowie promienia.
    """
    n_radius, n_angle = cost.shape
    t_axis = np.linspace(0.0, 1.0, n_radius)
    cur = cost.copy()
    paths: list[np.ndarray] = []
    prev_t = 0.0
    prev_gap: float | None = None
    ceiling_gap: float | None = None
    for ring_idx in range(max(0, k)):
        step_cost = cur.copy()
        forbidden = t_axis <= (prev_t + min_gap_frac)
        if ring_idx == 0 and first_ring_max_t is not None:
            forbidden = forbidden | (t_axis > first_ring_max_t)
        step_cost[forbidden, :] = BIG_COST
        # Jeśli po zakazie cofania nie zostało już ŻADNE dozwolone miejsce (np. poprzedni
        # pierścień wylądował tuż pod marginesem krawędzi) — zatrzymaj się, NIE zwracaj
        # bezsensownego "najtańszego z samych BIG_COST" pierścienia (byłby to czysty artefakt
        # remisów argmin, nie realny wynik).
        if bool(np.all(step_cost >= BIG_COST)):
            break
        gap = np.maximum(t_axis - prev_t, 0.0)
        if ceiling_gap is not None and width_ceiling_weight > 0:
            step_cost = step_cost + (width_ceiling_weight * np.maximum(0.0, gap - ceiling_gap) ** 2)[:, None]
        if prev_gap is not None and width_decay_weight > 0:
            step_cost = step_cost + (width_decay_weight * np.maximum(0.0, gap - prev_gap) ** 2)[:, None]

        path, path_cost = find_best_ring(step_cost, window, jump_penalty)
        if path_cost >= BIG_COST:
            break
        t_mean = float(np.mean(path)) / max(1, n_radius - 1)
        this_gap = t_mean - prev_t
        if ceiling_gap is None:
            ceiling_gap = this_gap
        prev_gap = this_gap
        prev_t = t_mean
        paths.append(path)
        cur = block_ring(cur, path, BLOCK_HALF)
    return paths


def ring_path_to_xy(path: np.ndarray, R_theta: np.ndarray, angles: np.ndarray,
                    cx: float, cy: float, n_radius: int) -> np.ndarray:
    """Ścieżka (indeksy promienia per kolumna kąta) -> (n_angle, 2) współrzędne pikselowe
    na oryginalnym (przyciętym) zdjęciu, przez odwrotną transformatę biegunową."""
    t = path.astype(np.float32) / max(1, n_radius - 1)
    radius_px = t * R_theta
    x = cx + radius_px * np.cos(angles)
    y = cy + radius_px * np.sin(angles)
    return np.stack([x, y], axis=1)


def apply_margins(cost: np.ndarray, n_radius: int, inner_margin: float, edge_margin: float) -> np.ndarray:
    """Zakazuje (BIG_COST) wiersze promienia t<inner_margin lub t>1-edge_margin — te same
    marginesy co produkcyjne candidates.inner_margin / domyślne edge_margin, żeby wynik był
    porównywalny ze STARĄ metodą, nie tylko koncepcyjnie podobny."""
    out = cost.copy()
    t = np.linspace(0.0, 1.0, n_radius)
    bad = (t < inner_margin) | (t > 1.0 - edge_margin)
    out[bad, :] = BIG_COST
    return out


def frangi_cost_field(gray_crop: np.ndarray, sigmas=range(2, 10, 2)) -> np.ndarray:
    """PRAWDZIWY filtr grzbietowy Frangiego (``skimage.filters.frangi``) na surowym,
    kartezjańskim (nie rozwiniętym biegunowo) obrazie w skali szarości — dokładnie
    zalecenie z przeglądu literatury (``30.07_wykrywanie_przyrostów_LIRERATURA.md`` §2.4):
    "liczy jak bardzo to wygląda na fragment krzywej w KAŻDYM pikselu... pełne pokrycie z
    definicji". Polaryzacja przyrostu (jasny czy ciemny pierścień) nie jest tu zakładana z
    góry — liczymy OBA warianty (``black_ridges`` True/False) i bierzemy maksimum, żeby nie
    zgadywać polaryzacji tego konkretnego zdjęcia."""
    g = gray_crop.astype(np.float64)
    rng = float(g.max() - g.min())
    g = (g - g.min()) / rng if rng > 1e-6 else g * 0.0
    ridge_bright = frangi(g, sigmas=sigmas, black_ridges=False)
    ridge_dark = frangi(g, sigmas=sigmas, black_ridges=True)
    return np.maximum(ridge_bright, ridge_dark).astype(np.float32)


# ---------------------------------------------------------------------------
# Bonus: PEWNE OTWARTE FRAGMENTY łuku, bez wymuszania zamknięcia pętli — odpowiedź na
# pytanie użytkownika "czy koniecznie musimy szukać całego pierścienia". Przyrosty bywają
# widoczne tylko na CZĘŚCI obwodu; wymuszanie pełnego zamknięcia (find_best_ring) zmusza
# ścieżkę, by przeszła też przez odcinki, gdzie sygnału praktycznie nie ma, co może
# degradować wynik nawet w miejscach gdzie sygnał jest mocny. Tu: dla wielu możliwych
# startów kątowych, znajdź najtańszy OTWARTY fragment o stałej długości kątowej (bez
# kary za zamknięcie) — im niższy średni koszt/kolumnę, tym "pewniejszy" fragment.
# ---------------------------------------------------------------------------

def _best_open_path_from_start(cost: np.ndarray, start_col: int, arc_len: int,
                               window: int, jump_penalty: float) -> tuple[np.ndarray, float]:
    """Najtańsza OTWARTA (bez zamknięcia) ścieżka o długości ``arc_len`` kolumn, zaczynająca
    się w kolumnie ``start_col`` (zawijana modulo n_angle) — ten sam DP co ``find_best_ring``,
    ale bez fazy domykania pętli."""
    n_radius, n_angle = cost.shape
    idx = np.arange(n_radius)
    cols = [(start_col + i) % n_angle for i in range(arc_len)]

    dp1 = cost[:, cols[0]].astype(np.float64).copy()
    backptr = np.zeros((arc_len, n_radius), dtype=np.int64)
    for i in range(1, arc_len):
        new_dp1 = np.full(n_radius, np.inf)
        choice = np.full(n_radius, -1, dtype=np.int64)
        for delta in range(-window, window + 1):
            r_prev = idx - delta
            valid = (r_prev >= 0) & (r_prev < n_radius)
            cand = np.full(n_radius, np.inf)
            cand[valid] = dp1[r_prev[valid]] + jump_penalty * (delta ** 2)
            better = cand < new_dp1
            new_dp1 = np.where(better, cand, new_dp1)
            choice = np.where(better, r_prev, choice)
        backptr[i] = choice
        dp1 = new_dp1 + cost[:, cols[i]]

    r_end = int(np.argmin(dp1))
    total_cost = float(dp1[r_end])
    path = np.zeros(arc_len, dtype=np.int64)
    path[arc_len - 1] = r_end
    for i in range(arc_len - 1, 0, -1):
        path[i - 1] = backptr[i, path[i]]
    return path, total_cost


def find_confident_open_arcs(cost: np.ndarray, arc_len_deg: int, stride_deg: int,
                             window: int, jump_penalty: float, top_k: int,
                             min_gap_deg: int, min_gap_t: float = 0.0) -> list[dict]:
    """Skanuje wiele startów kątowych (co ``stride_deg``), dla każdego znajduje najtańszy
    OTWARTY fragment długości ``arc_len_deg``, ranguje po ŚREDNIM koszcie/kolumnę (niższy =
    pewniejszy), i zwraca ``top_k`` najlepszych, nienachodzących się (NMS po środku kątowym,
    min. odstęp ``min_gap_deg`` ORAZ, jeśli ``min_gap_t>0``, po średnim promieniu fragmentu —
    bez tego drugiego warunku "top K" bywał zdominowany przez prawie-duplikaty tego samego
    miejsca przesunięte o kilka stopni, maskując inny, osobny pewny obszar przy innym
    promieniu — patrz 31.07 cd., "co jest powyżej łuku")."""
    n_radius, n_angle = cost.shape
    arc_len = max(2, int(round(arc_len_deg * n_angle / 360.0)))
    stride = max(1, int(round(stride_deg * n_angle / 360.0)))
    min_gap = min_gap_deg * n_angle / 360.0

    candidates = []
    for start_col in range(0, n_angle, stride):
        path, total_cost = _best_open_path_from_start(cost, start_col, arc_len, window, jump_penalty)
        candidates.append({
            "start_col": start_col, "arc_len": arc_len, "path": path,
            "avg_cost": total_cost / arc_len,
            "center_col": (start_col + arc_len / 2.0) % n_angle,
            "mean_t": float(np.mean(path)) / max(1, n_radius - 1),
        })
    candidates.sort(key=lambda c: c["avg_cost"])

    chosen: list[dict] = []
    for cand in candidates:
        if len(chosen) >= top_k:
            break
        c1 = cand["center_col"]
        too_close = False
        for prev in chosen:
            c2 = prev["center_col"]
            ang_dist = min(abs(c1 - c2), n_angle - abs(c1 - c2))
            if ang_dist < min_gap and (min_gap_t <= 0.0 or abs(cand["mean_t"] - prev["mean_t"]) < min_gap_t):
                too_close = True
                break
        if not too_close:
            chosen.append(cand)
    return chosen


# ---------------------------------------------------------------------------
# Referencja klasyczna na OSI (ta sama metodologia co mean_dist_final_to_classical_px
# w scripts/run_pipeline.py::_localization_quality — replikowana lokalnie, świadomie,
# ten sam wzorzec co reszta diagnostyk tej sesji).
# ---------------------------------------------------------------------------

def classical_axis_points(gray_crop: np.ndarray, cx: float, cy: float, fx: float, fy: float,
                          min_dist: int, prominence: float, n_samples: int = 50) -> list[tuple[float, float]]:
    profile, line_xy = sample_profile_along_axis(gray_crop, (cx, cy), (fx, fy),
                                                  gray_crop.shape[0], gray_crop.shape[1], n_samples)
    rng = float(profile.max() - profile.min())
    prof = (profile - profile.min()) / rng if rng > 1e-6 else profile
    peak_idx = find_candidate_peaks(prof, min_dist, prominence)
    return [(float(line_xy[i][0]), float(line_xy[i][1])) for i in peak_idx if 0 <= i < len(line_xy)]


def mean_dist(finals: list, classical: list) -> float | None:
    if not finals or not classical:
        return None
    fa = np.asarray(finals, dtype=np.float32)
    ca = np.asarray(classical, dtype=np.float32)
    dists = np.sqrt(((fa[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
    return round(float(dists.min(axis=1).mean()), 2)


# ---------------------------------------------------------------------------
# Wizualizacje
# ---------------------------------------------------------------------------

def render_unwrap(polar_field: np.ndarray, paths: list[np.ndarray], title: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(polar_field, aspect="auto", cmap="jet", origin="lower",
                   extent=[0, 360, 0, 1])
    n_radius = polar_field.shape[0]
    for i, path in enumerate(paths):
        t = path.astype(np.float32) / max(1, n_radius - 1)
        deg = np.linspace(0, 360, len(path), endpoint=False)
        color = np.array(_RING_COLORS[i % len(_RING_COLORS)]) / 255.0
        ax.plot(np.r_[deg, 360], np.r_[t, t[0]], color=color, linewidth=1.6)
    ax.set_xlabel("kąt θ (stopnie)")
    ax.set_ylabel("znormalizowany promień t (0=jądro, 1=brzeg)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return fig_to_b64(fig)


def render_rings_on_photo(crop_rgb: np.ndarray, axis_info: dict, rings_xy: list[np.ndarray]) -> np.ndarray:
    out = np.ascontiguousarray(crop_rgb[..., :3]).copy()
    H = out.shape[0]
    if axis_info.get("contour") is not None:
        cv2.drawContours(out, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    for i, xy in enumerate(rings_xy):
        color = _RING_COLORS[i % len(_RING_COLORS)]
        pts = xy.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=max(2, H // 500),
                     lineType=cv2.LINE_AA)
    return out


def render_comparison(crop_rgb: np.ndarray, axis_info: dict, old_pts: list,
                      rings_xy: list[np.ndarray], label_new: str) -> np.ndarray:
    out = np.ascontiguousarray(crop_rgb[..., :3]).copy()
    H = out.shape[0]
    if axis_info.get("contour") is not None:
        cv2.drawContours(out, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    cv2.line(out, (int(cx), int(cy)), (int(fx), int(fy)), _AXIS_COLOR, max(1, H // 400))
    for i, xy in enumerate(rings_xy):
        color = _RING_COLORS[i % len(_RING_COLORS)]
        pts = xy.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=max(2, H // 500),
                     lineType=cv2.LINE_AA)
    r = max(4, H // 110)
    for (x, y) in old_pts:
        cv2.circle(out, (int(x), int(y)), r, (230, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), r, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def render_open_arcs_unwrap(polar_field: np.ndarray, arcs: list[dict], n_angle: int, title: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(polar_field, aspect="auto", cmap="jet", origin="lower", extent=[0, 360, 0, 1])
    n_radius = polar_field.shape[0]
    for i, arc in enumerate(arcs):
        t = arc["path"].astype(np.float32) / max(1, n_radius - 1)
        deg = np.array([(arc["start_col"] + j) % n_angle for j in range(arc["arc_len"])]) * 360.0 / n_angle
        order = np.argsort(deg) if deg[0] > deg[-1] else np.arange(len(deg))  # unikaj linii-zszywki przy zawinięciu
        color = np.array(_RING_COLORS[i % len(_RING_COLORS)]) / 255.0
        # rysuj bez łączenia przez zawinięcie 0/360
        breaks = np.where(np.diff(deg) < 0)[0]
        if len(breaks) == 0:
            ax.plot(deg, t, color=color, linewidth=2.2)
        else:
            b = breaks[0] + 1
            ax.plot(deg[:b], t[:b], color=color, linewidth=2.2)
            ax.plot(deg[b:], t[b:], color=color, linewidth=2.2)
        ax.text(float(arc["center_col"]) * 360.0 / n_angle, float(t[len(t)//2]) + 0.03,
               f"{i+1}", color=color, fontsize=9, fontweight="bold", ha="center")
    ax.set_xlabel("kąt θ (stopnie)")
    ax.set_ylabel("znormalizowany promień t (0=jądro, 1=brzeg)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return fig_to_b64(fig)


def render_open_arcs_on_photo(crop_rgb: np.ndarray, axis_info: dict, arcs: list[dict],
                              R_theta: np.ndarray, angles: np.ndarray, n_radius: int) -> np.ndarray:
    out = np.ascontiguousarray(crop_rgb[..., :3]).copy()
    H = out.shape[0]
    cx, cy = axis_info["centroid"]
    if axis_info.get("contour") is not None:
        cv2.drawContours(out, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    n_angle = len(angles)
    for i, arc in enumerate(arcs):
        cols = np.array([(arc["start_col"] + j) % n_angle for j in range(arc["arc_len"])])
        sub_R = R_theta[cols]
        sub_angles = angles[cols]
        t = arc["path"].astype(np.float32) / max(1, n_radius - 1)
        radius_px = t * sub_R
        x = cx + radius_px * np.cos(sub_angles)
        y = cy + radius_px * np.sin(sub_angles)
        xy = np.stack([x, y], axis=1).astype(np.int32).reshape(-1, 1, 2)
        color = _RING_COLORS[i % len(_RING_COLORS)]
        cv2.polylines(out, [xy], isClosed=False, color=color, thickness=max(3, H // 350), lineType=cv2.LINE_AA)
        mx, my = int(x[len(x) // 2]), int(y[len(y) // 2])
        cv2.putText(out, str(i + 1), (mx + 4, my - 4), cv2.FONT_HERSHEY_SIMPLEX,
                   max(0.5, H / 900), (0, 0, 0), max(2, H // 250), cv2.LINE_AA)
        cv2.putText(out, str(i + 1), (mx + 4, my - 4), cv2.FONT_HERSHEY_SIMPLEX,
                   max(0.5, H / 900), color, max(1, H // 500), cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _render_field_section(field_name: str, polar_norm: np.ndarray, k_rings: int, k_label: str,
                          R_theta: np.ndarray, angles: np.ndarray, cx: float, cy: float,
                          fx: float, fy: float, crop_rgb: np.ndarray, d_axis_info: dict,
                          old_final_pts: list, classical_ref_pts: list, caveat: str) -> str:
    print(f"[{field_name}] graf kosztów (K={k_rings}, {k_label})...", flush=True)
    cost = 1.0 - polar_norm
    cost = apply_margins(cost, N_RADIUS, INNER_MARGIN, EDGE_MARGIN)

    paths = extract_k_rings_ordered(cost, k_rings, WINDOW, JUMP_PENALTY, MIN_GAP_FRAC,
                                    WIDTH_CEILING_WEIGHT, WIDTH_DECAY_WEIGHT)
    if not paths:
        note = ("Zerowe K (młoda ryba, Q1/Q2 &rarr; oczekiwany brak jeszcze pełnych przyrostów,"
                " tylko formujący się na brzegu, nieliczony tutaj)." if k_rings == 0 else
                "Mechanizm zatrzymał się wcześniej niż K — brak miejsca na kolejny pierścień po "
                "zastosowaniu twardego zakazu cofania się + marginesu krawędzi (poprzedni "
                "pierścień wylądował zbyt blisko brzegu).")
        print(f"[{field_name}] K={k_rings} ({k_label}): 0 pierścieni — {note}", flush=True)
        return f"""<section>
<h2>Pole wejściowe: <code>{field_name}</code> &mdash; K={k_rings} ({k_label})</h2>
<p style="font-size:90%;">{note}</p>
{caveat}
</section>"""
    rings_xy = [ring_path_to_xy(p, R_theta, angles, cx, cy, N_RADIUS) for p in paths]

    axis_angle = float(np.arctan2(fy - cy, fx - cx))
    a_idx = int(np.argmin(np.abs(np.angle(np.exp(1j * (angles - axis_angle))))))
    new_axis_pts = [(float(xy[a_idx][0]), float(xy[a_idx][1])) for xy in rings_xy]

    d_new = mean_dist(new_axis_pts, classical_ref_pts)
    d_old = mean_dist(old_final_pts, classical_ref_pts)

    # --- Liczbowa kontrola rozstawu promieniowego: teraz z WYMUSZONYM priorem "maleje/sufit"
    # (extract_k_rings_ordered) — sprawdzamy, czy w ogóle udało się go spełnić (może nie, jeśli
    # sygnał gdzie indziej jest dużo silniejszy niż kara).
    ring_t = [float(np.mean(p)) / max(1, N_RADIUS - 1) for p in paths]   # już posortowane (od jądra)
    gaps = [round(ring_t[0], 4)] + [round(ring_t[i + 1] - ring_t[i], 4) for i in range(len(ring_t) - 1)]
    frac_flat = float((polar_norm < 0.05).mean())
    print(f"  [{field_name}] pierścieni: {len(paths)} (K={k_rings}, {k_label}); t={[round(x,3) for x in ring_t]} "
         f"gaps(od jądra)={gaps}; pole<0.05 na {frac_flat*100:.1f}% powierzchni; "
         f"mean_dist do klasyki-na-osi: NOWA(shortest-path)={d_new} STARA(48 linii)={d_old}",
         flush=True)

    unwrap_b64 = render_unwrap(polar_norm, paths, f"Rozwinięcie biegunowe: {field_name}")
    rings_photo = render_rings_on_photo(crop_rgb, d_axis_info, rings_xy)
    cmp_photo = render_comparison(crop_rgb, d_axis_info, old_final_pts, rings_xy, field_name)

    gap_trend = ""
    if len(gaps) >= 3:
        inner_gaps = gaps[1:]  # pomijamy pierwszy (jądro->pierścień 1), patrzymy na kolejne
        if all(inner_gaps[i] >= inner_gaps[i + 1] for i in range(len(inner_gaps) - 1)):
            gap_trend = ("<b style=\"color:#0a7a1e;\">odstępy MALEJĄ w stronę brzegu</b> — zgodne "
                        "z biologicznym oczekiwaniem (wolniejszy wzrost, ciaśniejsze przyrosty w "
                        "starszym wieku).")
        elif all(inner_gaps[i] <= inner_gaps[i + 1] for i in range(len(inner_gaps) - 1)):
            gap_trend = ("<b style=\"color:#a00;\">odstępy ROSNĄ w stronę brzegu</b> — "
                        "PRZECIWNIE do biologicznego oczekiwania (prior szerokości nie wystarczył, "
                        "żeby to wymusić — sygnał musiał być silniejszy niż kara).")
        else:
            gap_trend = "odstępy nie są ściśle monotoniczne, ale prior nie pozwolił im ROSNĄĆ ponad sufit."

    return f"""<section>
<h2>Pole wejściowe: <code>{field_name}</code> &mdash; K={k_rings} ({k_label})</h2>
<p style="font-size:90%;">Znaleziono <b>{len(paths)}</b> pierścieni, metodą sekwencyjną OD JĄDRA
NA ZEWNĄTRZ (<code>extract_k_rings_ordered</code>, 31.07 cd.) — teraz z priorem szerokości
(sufit=pierwszy odstęp, malejąco = ten sam mechanizm co produkcyjne <code>width_ceiling_weight=
{WIDTH_CEILING_WEIGHT}</code>/<code>width_decay_weight={WIDTH_DECAY_WEIGHT}</code>).
Bonus-metryka (JEDEN otolit, nie substytut Fazy B na 30 kartach): odległość do klasycznej
referencji NA OSI (ta sama metodologia co produkcyjne <code>mean_dist_final_to_classical_px</code>)
&mdash; <b>NOWA (shortest-path) = {d_new}px</b> vs <b>STARA (48 linii, produkcyjna) = {d_old}px</b>.</p>
<p style="font-size:90%;background:#f7f7f7;padding:8px;border-radius:4px;">
<b>Rozstaw promieniowy (t, jądro&rarr;brzeg):</b> {', '.join(f'{x:.3f}' for x in ring_t)}<br>
<b>Odstępy (jądro&rarr;1, 1&rarr;2, ...):</b> {gaps} — {gap_trend}<br>
<b>Pole kosztu prawie puste (&lt;0.05) na {frac_flat*100:.1f}% powierzchni</b> — im wyższy ten
procent, tym MNIEJ realnego sygnału stało za wyborem pozycji.</p>
{caveat}
<h3>Rozwinięcie biegunowe (wejście do grafu) + znalezione pierścienie</h3>
{img_tag(unwrap_b64, style="width:100%;max-width:820px;")}
<h3>Pierścienie na zdjęciu (odwrotna transformata biegunowa)</h3>
<div style="text-align:center;">{img_tag(_b64_from_rgb(rings_photo, 480), style="width:480px;")}</div>
<h3>Porównanie: STARA metoda (czerwone kropki, 48 linii) vs NOWA (kolorowe krzywe, shortest-path)</h3>
<div style="text-align:center;">{img_tag(_b64_from_rgb(cmp_photo, 480), style="width:480px;")}</div>
</section>"""


def process_otolith(meta: dict, model, cfg, device, model_input_size: int) -> str:
    image_id = meta["image_id"]
    print(f"\n=== Otolit: {image_id} ({meta['label']}) ===", flush=True)
    image_dir = Path(cfg.data.image_dir)
    orig_rgb = np.array(PILImage.open(image_dir / image_id).convert("RGB"), dtype=np.uint8)
    axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                            nucleus_method=cfg.segmentation.nucleus_method)
    if axis_info is None:
        print(f"  Segmentacja nie powiodła się dla {image_id} — pomijam.", flush=True)
        return ""
    mask_arr = axis_info["mask"]
    model_input_rgb = (apply_background_mask(orig_rgb, mask_arr)
                       if cfg.data.mask_background else orig_rgb)

    crop_x0, crop_y0, cw, ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)
    crop_rgb = model_input_rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    mask_cropped = mask_arr[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    d_axis_info = shift_axis_info(axis_info, -crop_x0, -crop_y0)
    cx, cy = d_axis_info["centroid"]
    fx, fy = d_axis_info["far_edge"]

    gray_crop = crop_rgb.mean(axis=2).astype(np.float32)
    classical_ref_pts = classical_axis_points(gray_crop, cx, cy, fx, fy, min_dist=1, prominence=0.02)
    R_theta, angles = ray_cast_R_theta(mask_cropped, cx, cy, N_ANGLE)

    def _normalize(field: np.ndarray) -> np.ndarray:
        rng = float(field.max() - field.min())
        return (field - field.min()) / rng if rng > 1e-6 else field * 0.0

    age_transform = build_transforms(model_input_size, "test")
    age_tensor = age_transform(PILImage.fromarray(model_input_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        coral_logits = model(age_tensor)["coral_logits"]
    predicted_age = int(decode_age_ordinal(coral_logits).item())

    density_transform = build_transforms(cfg.candidates.density_image_size, "test")
    density_tensor = density_transform(PILImage.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        density_grid = model.get_density_probs(density_tensor).squeeze(0).cpu().numpy()

    old_result = select_increments(
        density_grid, d_axis_info, predicted_age, ch, cw,
        min_distance=cfg.candidates.min_peak_distance, prominence=cfg.candidates.prominence_threshold,
        inner_margin=cfg.candidates.inner_margin,
        width_decay_weight=cfg.candidates.width_decay_weight,
        width_ceiling_weight=cfg.candidates.width_ceiling_weight)
    old_final_pts = old_result["final_axis_pts"]

    q_adj = _quarter_adjustment(image_id)
    k_variants = [(predicted_age, f"K=wiek={predicted_age}")]
    if q_adj > 0 and predicted_age - q_adj >= 0 and predicted_age - q_adj != predicted_age:
        k_variants.append((predicted_age - q_adj,
                           f"K=wiek-{q_adj}={predicted_age - q_adj} (kwartał {_quarter_of(image_id)}: "
                           f"tegoroczny przyrost jeszcze formujący się na brzegu, nieliczony jako pełny)"))
    print(f"  Wiek (CORAL): {predicted_age}; warianty K: {[k for k, _ in k_variants]}", flush=True)

    polar_classical = _normalize(polar_unwrap(gray_crop, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))
    polar_classical_grad = _normalize(np.abs(np.gradient(polar_classical, axis=0)))
    print("  Liczenie filtru Frangiego (skimage)...", flush=True)
    frangi_field = frangi_cost_field(gray_crop)
    polar_frangi = _normalize(polar_unwrap(frangi_field, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))
    polar_density = _normalize(polar_unwrap(density_grid, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))

    grad_caveat = (
        "<p style=\"font-size:90%;color:#0a7a1e;\">Tani, bezzależnościowy odpowiednik filtru "
        "grzbietowego (gradient promieniowy |d intensywność/dt|) — premiuje PRZEJŚCIA, nie samą "
        "jasność.</p>")
    frangi_caveat = (
        "<p style=\"font-size:90%;color:#0a7a1e;\"><b>Prawdziwy filtr Frangiego</b> "
        "(<code>skimage.filters.frangi</code>, oba warianty polaryzacji, maksimum).</p>")
    density_caveat = (
        "<p style=\"font-size:90%;color:#a05a00;\"><b>Uwaga:</b> produkcyjna mapa density jest "
        "zwykle ekstremalnie rzadka (patrz <code>outputs/31.07_radial_Analiza_wnioski.md</code>) — "
        "jeśli rozwinięcie niżej jest w większości puste, znalezione pierścienie prawdopodobnie "
        "stłoczyły się przy dozwolonej krawędzi/marginesie, nie przez realny sygnał.</p>")

    field_sections = []
    fields: list[tuple[str, np.ndarray, str]] = [
        ("classical (gradient promieniowy)", polar_classical_grad, grad_caveat),
        ("classical (Frangi)", polar_frangi, frangi_caveat),
    ]
    if meta.get("show_density", False):
        fields.append(("density (produkcyjny model)", polar_density, density_caveat))
    for field_name, polar_norm, caveat in fields:
        for k_rings, k_label in k_variants:
            field_sections.append(_render_field_section(
                field_name, polar_norm, k_rings, k_label, R_theta, angles, cx, cy, fx, fy,
                crop_rgb, d_axis_info, old_final_pts, classical_ref_pts, caveat))

    print("  Szukanie pewnych otwartych fragmentów łuku...", flush=True)
    arc_sections = []
    arc_fields = fields if meta.get("show_density", False) is False else fields[:2]
    for arc_len_deg in meta.get("arc_lens", [ARC_LEN_DEG]):
        for arc_field_name, arc_polar, _ in arc_fields:
            if "density" in arc_field_name:
                continue
            arc_cost = apply_margins(1.0 - arc_polar, N_RADIUS, INNER_MARGIN, EDGE_MARGIN)
            arcs = find_confident_open_arcs(arc_cost, arc_len_deg, ARC_STRIDE_DEG, WINDOW,
                                            JUMP_PENALTY, ARC_TOP_K, ARC_MIN_GAP_DEG)
            print(f"    [{arc_field_name}, dlugosc={arc_len_deg}deg] {len(arcs)} fragmentow, avg_cost="
                 f"{[round(a['avg_cost'], 4) for a in arcs]}", flush=True)
            unwrap_b64 = render_open_arcs_unwrap(arc_polar, arcs, N_ANGLE,
                                                f"Pewne otwarte fragmenty ({arc_len_deg}&deg;): {arc_field_name}")
            photo = render_open_arcs_on_photo(crop_rgb, d_axis_info, arcs, R_theta, angles, N_RADIUS)
            arc_sections.append(f"""<section>
<h2>Pewne otwarte fragmenty łuku ({arc_len_deg}&deg;) &mdash; <code>{arc_field_name}</code></h2>
<p style="font-size:90%;">Najtańsze OTWARTE fragmenty o długości {arc_len_deg}&deg;, rangowane po
średnim koszcie/kolumnę (pewności) — top {len(arcs)} nienachodzących się, ponumerowanych od
najpewniejszego.</p>
{img_tag(unwrap_b64, style="width:100%;max-width:820px;")}
<div style="text-align:center;">{img_tag(_b64_from_rgb(photo, 480), style="width:480px;")}</div>
</section>""")

    return f"""<h1 style="border-top:4px solid #1a237e;padding-top:0.6em;">Otolit: {image_id}</h1>
<p style="font-size:95%;background:#eef2ff;padding:8px;border-radius:4px;"><b>{meta['label']}</b></p>
{"".join(field_sections)}
{"".join(arc_sections)}"""


def main() -> None:
    base_variant = MODEL_VARIANTS[0]
    cfg = load_merged_config(base_variant["cfg_path"], None)
    device = torch.device(cfg.training.device if cfg.training.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = base_variant["run_dir"] / "checkpoints" / "embedded" / "best.pt"
    print(f"Wczytywanie modelu ({base_variant['name']})...", flush=True)
    model = load_model_from_checkpoint(cfg, ckpt)
    model.eval()
    device = next(model.parameters()).device

    otolith_sections = [
        process_otolith(meta, model, cfg, device, cfg.data.image_size)
        for meta in OTOLITHS
    ]

    literature_section = """<section>
<h2>Przegląd literatury: czy "wykryj pewne fragmenty, dopasuj/wybierz w postprodukcji" to
znana rodzina metod?</h2>
<p style="font-size:92%;">Tak — i to dobrze ugruntowana. Cztery niezależne, zbieżne źródła:</p>
<ul style="font-size:92%;">
<li><b>RANSAC-fitting elipsy/okręgu z fragmentów łuku</b> (klasyczna literatura widzenia
    komputerowego, dojrzała od lat 2000., stosowana m.in. do wykrywania granicy tęczówki/źrenicy
    &mdash; problem strukturalnie identyczny: okrąg czI[psi]sto zasłonięty powiekami, widoczny
    tylko fragmentami) &mdash; podejście hierarchiczne: pikselowe krawędzie &rarr; odcinki &rarr;
    łączenie w łuki wg bliskości/krzywizny &rarr; RANSAC dopasowuje elipsę/okrąg do zebranych
    fragmentów, odporny na okluzję/brakujące dane.</li>
<li><b>Rodin &amp; Troadec (1997)</b>, już cytowane w tym projekcie
    (<code>20.07_detekcja przyrostów_TO_DO.md</code>, wcześniej odłożone jako "Opcja B") &mdash;
    rekonstrukcja pierścienia jako grafu przez ŁĄCZENIE węzłów wykrytych w układzie biegunowym,
    z interpolacją przez słabo widoczne przerwy na podstawie już potwierdzonej geometrii.</li>
<li><b>CS-TRD / DeepCS-TRD "spider web"</b> (2023-2025, słoje drzew &mdash; ten sam problem
    geometrycznie) &mdash; wykrywa łańcuchy krawędzi z RÓŻNYCH promieni, łączy je w pierścień
    iteracyjną procedurą (łańcuchy nieprzecinające się + silna krawędź + warunek gładkości).</li>
<li><b>INBD (Iterative Next Boundary Detection, CVPR 2023)</b> &mdash; nowe znalezisko tej sesji:
    sieć (U-Net, transformata biegunowa) rosnąca WARSTWA PO WARSTWIE od rdzenia na zewnątrz,
    wykrywając KOLEJNĄ granicę pierścienia w każdym kroku &mdash; wbudowane pojęcie chronologii
    wzrostu. Innego rzędu zmiana niż reszta (wymaga treningu/adnotacji granic), ale koncepcyjnie
    bliska: nie wymusza jednego globalnego kształtu, tylko rośnie lokalnie, krok po kroku.</li>
</ul>
<p style="font-size:92%;"><b>Wniosek:</b> pomysł użytkownika (wykryj pewne fragmenty łuku, dopiero
w postprodukcji zdecyduj, które złożyć w pierścień) pokrywa się dokładnie z tym, co te cztery
niezależne źródła już robią. Sekcja "pewne otwarte fragmenty łuku" wyżej to pierwszy, minimalny
krok w tym kierunku (te same graf/DP co reszta tego prototypu, tylko bez wymuszania zamknięcia) —
naturalne rozszerzenie: łączenie/wybór fragmentów w pełne pierścienie (jak RANSAC/Rodin&amp;Troadec/
CS-TRD) to osobny, kolejny krok, NIE zaimplementowany w tym skrypcie.</p>
<p style="font-size:92%;"><b>Dodatkowo (31.07 cd.), z literatury RYBACKIEJ (nie widzenia
komputerowego)</b> — dokładnie potwierdza obserwacje użytkownika o efekcie brzegu i stłoczeniu:
"marginal increment analysis"/"edge type" (klasyczna metodologia odczytu wieku, patrz przeglądowy
<a href="https://uni.hi.is/scampana/files/2016/01/campana-2001-1.pdf">Campana 2001, Journal of
Fish Biology</a>) formalnie uwzględnia, że przyrost przy brzegu może być tylko CZĘŚCIOWO
uformowany zależnie od sezonu połowu — dokładnie mechanizm opisany przez użytkownika dla
BITS1q/BITS2q. Niezależnie potwierdzone: "increments are normally spaced further apart near the
core... and are more crowded toward the edge as growth declines with age, which is a typical
pattern for most fish species" oraz "in fish greater than age 6 or 7, annuli may be difficult or
impossible to follow... because of crowding" — dokładnie zgodne z decyzją użytkownika, żeby nie
przejmować się wynikami dla wieku 7+ w tym eksperymencie.</p>
</section>"""

    body = "".join(otolith_sections) + literature_section
    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Prototyp: circular shortest-path ({VERSION})</title>
<style>
body {{font-family:sans-serif;max-width:1000px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
h1 {{color:#1a237e;}} h2 {{color:#1a237e;}}
code {{background:#f0f0f5;padding:1px 4px;border-radius:3px;}}
</style>
</head>
<body>
<h1 style="border-top:none;">Prototyp Fazy A (cd.): circular/closed shortest-path + prior
szerokości + korekta kwartału połowu</h1>
<p>Dodane względem poprzedniej iteracji: (1) <b>prior szerokości</b>
(<code>extract_k_rings_ordered</code> — sekwencyjnie od jądra na zewnątrz, kara za odstęp
przekraczający sufit pierwszego przyrostu i/lub poprzedni odstęp, te same wagi co produkcyjny
<code>width_ceiling_weight={WIDTH_CEILING_WEIGHT}</code>/<code>width_decay_weight={WIDTH_DECAY_WEIGHT}</code>);
(2) <b>korekta liczby K o kwartał połowu</b> (BITS1q/BITS2q &rarr; K=wiek&minus;1, bo tegoroczny
przyrost jeszcze się formuje przy brzegu); (3) <b>trzy otolity</b> tej samej lokalizacji/ryby w
różnym wieku i kwartale, żeby sprawdzić, czy wzorzec (jeden pierścień u młodych, kilka stłoczonych
u starszych) faktycznie się pojawia; (4) <b>strojenie długości fragmentu łuku</b> (60&deg; vs
120&deg;) na otolicie, który użytkownik ocenił jako najbardziej przekonujący. Parametry grafu
(<code>N_RADIUS={N_RADIUS}, N_ANGLE={N_ANGLE}, WINDOW={WINDOW}, JUMP_PENALTY={JUMP_PENALTY}</code>)
niezmienione, wciąż nieskalibrowane. Faza B (30 kart) to osobny, następny krok — nie wykonana
tutaj. Nic w <code>src/</code> nie zostało zmienione.</p>
{body}
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()