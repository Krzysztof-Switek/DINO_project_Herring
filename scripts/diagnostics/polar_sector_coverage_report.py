"""30.07 — Prototyp: sektorowa agregacja KĄTOWA zamiast próbkowania 48 dosłownych linii.

Kontekst (patrz ``ray_patch_voting_report.py``, ten sam otolit): dzisiejsze wykrywanie
kandydatów (``src/ring_extraction.py::select_increments``/``_all_ray_peaks``) próbkuje
64 punkty (nearest-neighbour) wzdłuż każdej z 48 linii jądro→kontur — zmierzone pokrycie
to tylko **~29-37% patchy** w siatce density. Reszta patchy NIGDY nie wchodzi do żadnego
sygnału używanego przez wykrywanie — z definicji ogranicza to jakość lokalizacji
(użytkownik, 30.07).

Ten skrypt to DZIAŁAJĄCY PROTOTYP naprawy, jako samodzielna diagnostyka — NIC w ``src/``
nie jest zmieniane, zgodnie z ustaloną w tym projekcie zasadą: najpierw dowód w
diagnostyce, decyzja o promocji do produkcji osobno (tak jak
``sweep_forced_topk_peaks.py`` wcześniej).

## Projekt algorytmu

Zamiast próbkować linię, KAŻDY patch w siatce (H_p, W_p) dostaje własny znormalizowany
promień ``t = r/R(theta)`` i sektor kątowy (0..47) — dokładnie to, co już liczy
``otolith_axis.compute_polar_grid`` (dziś używane WYŁĄCZNIE przez trening, stratę E9
``density_concentricity_loss``, NIGDY przez wykrywanie kandydatów przy inferencji).
Zamiast 64 próbek na linii, każdy z 48 sektorów dostaje gęsty profil 1D zbudowany przez
AGREGACJĘ MAX (nie średnią — uśrednianie rozmywa pojedynczy silny patch, MAX zachowuje
sygnał piku) wartości density WSZYSTKICH patchy trafiających w dany bin (sektor, t_bin).
Puste biny (rzadkie patche blisko konturu) wypełniane najbliższym niepustym binem WZDŁUŻ
t w tym samym sektorze — ten sam rodzaj łagodnego przybliżenia co dzisiejszy "schodkowy"
efekt, ale nad KOMPLETNYM zbiorem patchy, nie próbką 64 punktów.

Świadoma DODATKOWA różnica (nie tylko naprawa pokrycia): dzisiejsze 48 "promieni" to
punkty równomiernie rozłożone PO INDEKSIE KONTURU (na nie-kołowym, "ogoniastym" otolicie
to NIE jest to samo co równo po kącie). ``compute_polar_grid`` dzieli równo PO KĄCIE —
dokładnie ta sama parametryzacja, której już używa wytrenowana strata E9. Przejście na
sektory kątowe to więc też zysk SPÓJNOŚCI trening/inferencja, nie tylko pokrycia.

Gotowe profile (już znormalizowane per-sektor do [0,1]) wchodzą do ISTNIEJĄCEGO,
przetestowanego kodu BEZ ZMIAN: ``find_peaks`` + ``_shift_peak_to_falling_edge`` +
``_cluster_by_radius_with_arcs`` + ``_cluster_score`` + ``_dp_select_t`` +
``_project_to_axis`` — WSZYSTKIE zaimportowane wprost z ``src.ring_extraction``. Zmienia
się WYŁĄCZNIE sposób budowy profilu wejściowego. Pozycja piksela wybranego piku = miejsce
RZECZYWISTEGO patcha, który dał wartość MAX w zwycięskim binie (prostsze i bardziej
wierne niż interpolacja wzdłuż wyidealizowanej linii).

Nie trenuje ani nie modyfikuje żadnego kodu w src/.

Usage: PYTHONIOENCODING=utf-8 python scripts/diagnostics/polar_sector_coverage_report.py
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
import torch
import cv2
from PIL import Image as PILImage
from scipy.signal import find_peaks

from src.report_common import fig_to_b64, img_tag
from src.visualization import _CONTOUR_COLOR, _AXIS_COLOR
from src.otolith_axis import (detect_axis, apply_background_mask, mask_bbox, shift_axis_info,
                              compute_polar_grid)
from src.dataset import build_transforms, decode_age_ordinal
from src.inference import load_model_from_checkpoint
from src.ring_extraction import (select_increments, _shift_peak_to_falling_edge,
                                 _cluster_by_radius_with_arcs, _cluster_score,
                                 _dp_select_t, _project_to_axis)
from scripts.run_pipeline import load_merged_config

IMAGE_ID = "2023_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg"

OUT_DIR = PROJECT_ROOT / "outputs" / "29.07_candidate_selection_walkthrough"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION = "v2"
OUT_PATH = OUT_DIR / f"polar_sector_coverage_{VERSION}.html"

N_DIRS, N_SAMPLES = 48, 64        # N_SAMPLES: tylko dla odtworzenia STAREJ metody (porownanie)
N_T_BINS_CANDIDATES = [8, 12, 16, 20, 24, 30, 40, 52, 64]
EMPTY_FRAC_TARGET = 0.20          # wybierz NAJWIEKSZE n_t_bins z pustych-binow <= 20% (im
                                  # wiecej binow, tym lepsza rozdzielczosc polozenia pierscienia
                                  # -- wiec maksymalizujemy n_t_bins pod warunkiem okupacji, nie
                                  # minimalizujemy pustych binow za wszelka cene)

RESOLUTIONS = [
    {"label": "Niska rozdzielczość", "slug": "low",
     "cfg_path": PROJECT_ROOT / "configs" / "config_e9_w0.1.yaml",
     "run_dir": PROJECT_ROOT / "outputs" / "28.07_e9_w0.1"},
    {"label": "Wysoka rozdzielczość", "slug": "high",
     "cfg_path": PROJECT_ROOT / "configs" / "config_hires966.yaml",
     "run_dir": PROJECT_ROOT / "outputs" / "23.07_hires966"},
]

_OLD_COLOR = (230, 30, 30)      # czerwony — finalne przyrosty, STARA metoda (linie)
_NEW_COLOR = (30, 200, 30)      # zielony — finalne przyrosty, NOWA metoda (sektory)


def _b64_from_rgb(arr: np.ndarray, target_w: int = 480) -> str:
    import base64, io
    img = PILImage.fromarray(np.ascontiguousarray(arr[..., :3]).astype(np.uint8))
    if img.width > target_w:
        scale = target_w / img.width
        img = img.resize((target_w, max(1, int(round(img.height * scale)))), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _old_ray_cells(H_p: int, W_p: int, cx: float, cy: float, contour: np.ndarray,
                   cw: int, ch: int, n_dirs: int = N_DIRS,
                   n_samples: int = N_SAMPLES) -> set[tuple[int, int]]:
    """Patche dotknięte przez STARĄ metodę (48 dosłownych linii, nearest-neighbour) —
    ten sam rachunek co ``ray_patch_voting_report.py::_ray_cells``, tu dla porównania."""
    idx_sel = np.linspace(0, len(contour) - 1, min(n_dirs, len(contour)), dtype=int)
    cells: set[tuple[int, int]] = set()
    for ci in idx_sel:
        fx, fy = float(contour[ci][0]), float(contour[ci][1])
        xs = np.linspace(cx, fx, n_samples)
        ys = np.linspace(cy, fy, n_samples)
        px = np.clip((xs / max(cw, 1) * W_p).astype(np.int64), 0, W_p - 1)
        py = np.clip((ys / max(ch, 1) * H_p).astype(np.int64), 0, H_p - 1)
        cells |= set(zip(px.tolist(), py.tolist()))
    return cells


def _bin_idx_grid(H_p: int, W_p: int, H: int, W: int, cx: float, cy: float,
                  n_dirs: int) -> np.ndarray:
    """Sektor kątowy (0..n_dirs-1) każdego patcha — dokładnie te same ~5 linii, które
    ``compute_polar_grid`` liczy WEWNĘTRZNIE (``src/otolith_axis.py:610-624``), ale nie
    zwraca — powielone tu (świadomie, prototyp nie zmienia ``src/``)."""
    py = (np.arange(H_p, dtype=np.float32) + 0.5) * (H / H_p)
    px = (np.arange(W_p, dtype=np.float32) + 0.5) * (W / W_p)
    grid_x, grid_y = np.meshgrid(px, py)
    dx = grid_x - cx
    dy = grid_y - cy
    theta = np.arctan2(dy, dx)
    return np.clip(((theta + np.pi) / (2.0 * np.pi) * n_dirs).astype(np.int64), 0, n_dirs - 1)


def build_sector_profiles(density_grid: np.ndarray, t_grid: np.ndarray, valid_grid: np.ndarray,
                          bin_idx: np.ndarray, n_dirs: int, n_t_bins: int):
    """Buduje gęsty profil 1D na SEKTOR przez agregację MAX wszystkich patchy trafiających
    w każdy bin (sektor, t_bin) — zamiast 64 próbek na dosłownej linii. Zwraca
    ``(profiles, pos_row, pos_col, empty_frac)``: ``profiles[s]`` to znormalizowany
    [0,1] profil (lub None, jeśli płaski/pusty sektor); ``pos_row``/``pos_col`` (n_dirs,
    n_t_bins) to współrzędne PATCHA, który dał wartość max w tym binie (PO wypełnieniu
    pustych binów najbliższym niepustym wzdłuż t); ``empty_frac`` = ułamek (sektor,
    t_bin) par PRZED wypełnieniem, które nie dostały ŻADNEGO patcha — miara, czy
    ``n_t_bins`` jest za drobne względem gęstości patchy."""
    max_val = np.full((n_dirs, n_t_bins), -np.inf, dtype=np.float64)
    has_val = np.zeros((n_dirs, n_t_bins), dtype=bool)
    pos_row = np.full((n_dirs, n_t_bins), -1, dtype=np.int64)
    pos_col = np.full((n_dirs, n_t_bins), -1, dtype=np.int64)

    rows, cols = np.nonzero(valid_grid)
    for r, c in zip(rows.tolist(), cols.tolist()):
        t = float(t_grid[r, c])
        if t < 0.0:
            continue
        tb = min(int(t * n_t_bins), n_t_bins - 1)
        s = int(bin_idx[r, c])
        v = float(density_grid[r, c])
        if v > max_val[s, tb]:
            max_val[s, tb] = v
            has_val[s, tb] = True
            pos_row[s, tb] = r
            pos_col[s, tb] = c

    empty_frac = float(1.0 - has_val.mean())

    filled = max_val.copy()
    for s in range(n_dirs):
        idxs = np.nonzero(has_val[s])[0]
        if idxs.size == 0:
            continue
        for tb in range(n_t_bins):
            if not has_val[s, tb]:
                nearest = idxs[np.argmin(np.abs(idxs - tb))]
                filled[s, tb] = max_val[s, nearest]
                pos_row[s, tb] = pos_row[s, nearest]
                pos_col[s, tb] = pos_col[s, nearest]

    profiles: list[np.ndarray | None] = []
    for s in range(n_dirs):
        p = filled[s]
        rng = float(p.max() - p.min())
        profiles.append(((p - p.min()) / rng).astype(np.float32) if rng > 1e-6 else None)
    return profiles, pos_row, pos_col, empty_frac


def sector_peaks(profiles, pos_row, pos_col, cw: int, ch: int, H_p: int, W_p: int,
                 n_t_bins: int, min_distance: int, prominence: float,
                 inner_margin: float, edge_margin: float = 0.08):
    """Analogon ``_all_ray_peaks`` (``src/ring_extraction.py:240-283``), ale na profilach
    zbudowanych przez ``build_sector_profiles`` zamiast próbki linii — reszta (find_peaks,
    okno t, przesunięcie na falling-edge) IDENTYCZNA logika, ta sama funkcja
    ``_shift_peak_to_falling_edge`` zaimportowana wprost z ``src.ring_extraction``."""
    peaks: list[tuple[float, float, int, int, int]] = []
    candidate_pts: list[tuple[int, int]] = []
    patch_w, patch_h = cw / W_p, ch / H_p
    for s, p in enumerate(profiles):
        if p is None:
            continue
        idxs, _ = find_peaks(p, distance=max(1, int(min_distance)), prominence=float(prominence))
        for idx in idxs:
            t_orig = idx / max(1, n_t_bins - 1)
            if t_orig < inner_margin or t_orig > 1.0 - edge_margin:
                continue
            edge_idx = _shift_peak_to_falling_edge(p, int(idx))
            t = edge_idx / max(1, n_t_bins - 1)
            r, c = int(pos_row[s, edge_idx]), int(pos_col[s, edge_idx])
            x, y = int((c + 0.5) * patch_w), int((r + 0.5) * patch_h)
            peaks.append((t, float(p[idx]), x, y, s))
            candidate_pts.append((x, y))
    return peaks, candidate_pts


def render_coverage_panel(crop_rgb: np.ndarray, axis_info: dict, keep_mask: np.ndarray,
                          H_p: int, W_p: int, dim_gray_factor: float = 0.30) -> np.ndarray:
    """Patche w ``keep_mask`` (H_p,W_p bool) podświetlone na zielono (na spodzie zdjęcia),
    reszta przyciemniona/odbarwiona — czysty wizualny procent pokrycia, bez kolorowania
    density (to porównanie ZASIĘGU, nie sygnału)."""
    H, W = crop_rgb.shape[:2]
    out = crop_rgb.astype(np.float32).copy()
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dimmed = np.stack([gray, gray, gray], axis=-1) * dim_gray_factor
    up_keep = cv2.resize(keep_mask.astype(np.uint8), (W, H),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    tint = np.zeros_like(out); tint[..., 1] = 255
    lit = 0.55 * tint + 0.45 * out
    out = np.where(up_keep[..., None], lit, dimmed).clip(0, 255).astype(np.uint8)
    for i in range(1, W_p):
        x = int(round(i * W / W_p))
        cv2.line(out, (x, 0), (x, H), (60, 60, 60), 1, cv2.LINE_AA)
    for i in range(1, H_p):
        y = int(round(i * H / H_p))
        cv2.line(out, (0, y), (W, y), (60, 60, 60), 1, cv2.LINE_AA)
    if axis_info.get("contour") is not None:
        cv2.drawContours(out, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    return out


def render_final_comparison(crop_rgb: np.ndarray, axis_info: dict,
                            old_pts: list, new_pts: list) -> np.ndarray:
    out = np.ascontiguousarray(crop_rgb[..., :3]).copy()
    H = out.shape[0]
    if axis_info.get("contour") is not None:
        cv2.drawContours(out, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    cv2.line(out, (int(cx), int(cy)), (int(fx), int(fy)), _AXIS_COLOR, max(1, H // 400))
    r = max(4, H // 110)
    for (x, y) in old_pts:
        cv2.circle(out, (int(x), int(y)), r, _OLD_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), r, (0, 0, 0), 1, cv2.LINE_AA)
    for (x, y) in new_pts:
        cv2.circle(out, (int(x), int(y)), max(2, r // 2), _NEW_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), max(2, r // 2), (0, 0, 0), 1, cv2.LINE_AA)
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

    # --- Wiek (CORAL) — na pełnym, niewyciętym obrazie, cfg.data.image_size, tak jak trening. ---
    age_transform = build_transforms(cfg.data.image_size, "test")
    age_tensor = age_transform(PILImage.fromarray(model_input_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        coral_logits = model(age_tensor)["coral_logits"]
    predicted_age = int(decode_age_ordinal(coral_logits).item())
    print(f"  [{label}] przewidziany wiek (CORAL): {predicted_age}", flush=True)

    crop_x0, crop_y0, cw, ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)
    crop_rgb = model_input_rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    mask_cropped = mask_arr[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    d_axis_info = shift_axis_info(axis_info, -crop_x0, -crop_y0)
    density_transform = build_transforms(cfg.candidates.density_image_size, "test")
    density_tensor = density_transform(PILImage.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        density_grid = model.get_density_probs(density_tensor).squeeze(0).cpu().numpy()
    H_p, W_p = density_grid.shape
    total = H_p * W_p
    print(f"  [{label}] siatka: {H_p}x{W_p} = {total} patchy", flush=True)

    min_distance = cfg.candidates.min_peak_distance
    prominence = cfg.candidates.prominence_threshold
    inner_margin = cfg.candidates.inner_margin
    width_decay_weight = cfg.candidates.width_decay_weight
    width_ceiling_weight = cfg.candidates.width_ceiling_weight

    # --- STARA metoda: 48 dosłownych linii, produkcyjny select_increments NIEZMIENIONY. ---
    cx, cy = d_axis_info["centroid"]
    contour = d_axis_info["contour"].reshape(-1, 2)
    old_cells = _old_ray_cells(H_p, W_p, cx, cy, contour, cw, ch)
    old_mask = np.zeros((H_p, W_p), dtype=bool)
    for (col, row) in old_cells:
        old_mask[row, col] = True
    old_pct = 100 * len(old_cells) / total
    old_result = select_increments(
        density_grid, d_axis_info, predicted_age, ch, cw,
        min_distance=min_distance, prominence=prominence, inner_margin=inner_margin,
        width_decay_weight=width_decay_weight, width_ceiling_weight=width_ceiling_weight)
    old_final_pts = old_result["final_axis_pts"]

    # --- NOWA metoda: compute_polar_grid + agregacja sektorowa MAX. ---
    t_grid, valid_grid, _theta_grid = compute_polar_grid(mask_cropped, (cx, cy), H_p, W_p, n_angle_bins=N_DIRS)
    bin_idx = _bin_idx_grid(H_p, W_p, ch, cw, cx, cy, N_DIRS)
    n_valid = int(valid_grid.sum())
    new_pct = 100 * n_valid / total

    sweep_rows = []
    for cand in N_T_BINS_CANDIDATES:
        _, _, _, ef = build_sector_profiles(density_grid, t_grid, valid_grid, bin_idx, N_DIRS, cand)
        sweep_rows.append((cand, ef))
    qualifying = [c for c, ef in sweep_rows if ef <= EMPTY_FRAC_TARGET]
    n_t_bins = max(qualifying) if qualifying else min(sweep_rows, key=lambda row: row[1])[0]
    print(f"  [{label}] sweep n_t_bins (ułamek pustych binów PRZED wypełnieniem): "
          + ", ".join(f"{c}:{ef*100:.1f}%" for c, ef in sweep_rows)
          + f" -> wybrano n_t_bins={n_t_bins}", flush=True)

    profiles, pos_row, pos_col, empty_frac = build_sector_profiles(
        density_grid, t_grid, valid_grid, bin_idx, N_DIRS, n_t_bins)
    new_peaks, new_candidate_pts = sector_peaks(
        profiles, pos_row, pos_col, cw, ch, H_p, W_p, n_t_bins,
        min_distance=min_distance, prominence=prominence, inner_margin=inner_margin)
    new_clusters = _cluster_by_radius_with_arcs(new_peaks, t_tol=0.06, n_dirs=N_DIRS)
    new_cands = [(c[0], _cluster_score(c)) for c in new_clusters]
    new_chosen_t = _dp_select_t(new_cands, max(0, predicted_age), min_gap=0.04,
                                spread_weight=1.5, width_decay_weight=width_decay_weight,
                                width_ceiling_weight=width_ceiling_weight)
    new_final_pts = _project_to_axis(new_chosen_t, d_axis_info)

    print(f"  [{label}] pokrycie: STARA {len(old_cells)}/{total} ({old_pct:.1f}%) vs "
          f"NOWA {n_valid}/{total} ({new_pct:.1f}%) — kandydatów: STARA "
          f"{len(old_result['candidate_pts'])} / NOWA {len(new_candidate_pts)}; "
          f"finalnych (wiek={predicted_age}): STARA {len(old_final_pts)} / "
          f"NOWA {len(new_final_pts)}", flush=True)

    cov_old = render_coverage_panel(crop_rgb, d_axis_info, old_mask, H_p, W_p)
    cov_new = render_coverage_panel(crop_rgb, d_axis_info, valid_grid, H_p, W_p)
    final_cmp = render_final_comparison(crop_rgb, d_axis_info, old_final_pts, new_final_pts)

    sweep_table = "".join(
        f"<tr><td>{c}</td><td>{ef*100:.1f}%</td>"
        f"{'<td>&larr; wybrane</td>' if c == n_t_bins else '<td></td>'}</tr>"
        for c, ef in sweep_rows)

    return f"""<section>
<h2>{label} <span style="font-size:65%;color:#666;">({cfg_path.name}, {run_dir.name})</span></h2>
<p style="font-size:90%;">Siatka density: <b>{H_p}x{W_p} = {total} patchy</b>. Przewidziany
wiek (CORAL): <b>{predicted_age}</b>.</p>
<table><tr><th>n_t_bins (kandydat)</th><th>puste biny PRZED wypełnieniem</th><th></th></tr>
{sweep_table}</table>
<p style="font-size:90%;">Wybrane <b>n_t_bins={n_t_bins}</b> (NAJWIĘKSZE z ułamkiem pustych
&le;{EMPTY_FRAC_TARGET*100:.0f}% — maksymalizujemy rozdzielczość położenia pierścienia pod
warunkiem, że biny wciąż są w większości zapełnione realnymi patchami, nie tylko
wypełniaczem).</p>
<h3>Pokrycie patchy: STARA (linie) vs NOWA (sektory)</h3>
<p style="font-size:90%;"><b>{new_pct:.1f}%</b> ({n_valid}/{total}) to TEORETYCZNE MAKSIMUM
możliwe do wykorzystania na tym zdjęciu — to WSZYSTKIE patche leżące wewnątrz
zsegmentowanego otolitu (reszta bboxa to tło/róg kadru, np. wcięcie "widełek" ogona —
nigdy nie powinno wchodzić do sygnału). <b>STARA metoda</b> (48 dosłownych linii, 64
próbki/linia) dociera do <b>{len(old_cells)}/{total} ({old_pct:.1f}%)</b> — czyli tylko
<b>{100*len(old_cells)/max(1,n_valid):.0f}%</b> z tego, co w ogóle możliwe. <b>NOWA
metoda</b> (sektor kątowy + bin t, <code>compute_polar_grid</code>) dociera do
<b>DOKŁADNIE {n_valid}/{total} ({new_pct:.1f}%)</b> — czyli 100% teoretycznego maksimum:
KAŻDY patch w masce wpływa na jakiś sektor, żaden nie jest pomijany z definicji jak przy
próbkowaniu linii.</p>
<div>
<div style="display:inline-block;vertical-align:top;width:460px;margin:0 14px 14px 0;">
<b>STARA — pokrycie {old_pct:.1f}%</b><br>{img_tag(_b64_from_rgb(cov_old, 440), style="width:440px;")}</div>
<div style="display:inline-block;vertical-align:top;width:460px;margin:0 14px 14px 0;">
<b>NOWA — pokrycie {new_pct:.1f}%</b><br>{img_tag(_b64_from_rgb(cov_new, 440), style="width:440px;")}</div>
</div>
<h3 style="margin-top:1.2em;">Finalne kandydaty (N={predicted_age}): STARA vs NOWA</h3>
<p style="font-size:90%;"><span style="color:#e01e1e;font-weight:bold;">&#9679; czerwone</span>
= STARA metoda (produkcyjny <code>select_increments</code>, niezmieniony) —
<b>{len(old_final_pts)}</b> punktów. <span style="color:#1ec81e;font-weight:bold;">&#9679;
zielone (mniejsze)</span> = NOWA metoda (ta sama klasteryzacja/DP, inny profil wejściowy) —
<b>{len(new_final_pts)}</b> punktów. Oba rzutowane na tę samą oś pomiaru
(<code>_project_to_axis</code>), do bezpośredniego porównania.</p>
<div style="text-align:center;">{img_tag(_b64_from_rgb(final_cmp, 480), style="width:480px;")}</div>
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
<title>Prototyp: pokrycie 100% patchy (sektory kątowe) ({VERSION}, {IMAGE_ID})</title>
<style>
body {{font-family:sans-serif;max-width:1300px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
h1 {{color:#1a237e;}} h2 {{color:#1a237e;}}
table {{border-collapse:collapse;margin:0.6em 0;}}
td,th {{padding:4px 10px;border:1px solid #ddd;font-size:90%;text-align:center;}}
th {{background:#f0f0f5;}}
code {{background:#f0f0f5;padding:1px 4px;border-radius:3px;}}
</style>
</head>
<body>
<h1>Prototyp: wykrywanie kandydatów BEZ utraty informacji</h1>
<p>Otolit <code>{IMAGE_ID}</code>. Dzisiejsza metoda (48 dosłownych linii jądro→kontur,
64 próbki nearest-neighbour/linia) dotyka tylko ~29-37% patchy w siatce density — reszta
NIGDY nie wchodzi do żadnego sygnału wykrywania kandydatów. Tu: prototyp zastąpienia
próbkowania linii AGREGACJĄ SEKTOROWĄ — każdy patch w otolicie dostaje własny sektor
kątowy (0-47) i znormalizowany promień <code>t</code> (przez już istniejącą, ale dziś
używaną WYŁĄCZNIE do treningu funkcję <code>otolith_axis.compute_polar_grid</code>), a
profil każdego sektora budowany jest przez MAX ze wszystkich patchy trafiających w dany
bin — żaden patch w masce nie jest pomijany z definicji. Reszta mechanizmu (wykrywanie
pików, przesunięcie na falling-edge, klasteryzacja arc-aware, wybór DP dokładnie N=wiek)
to DOKŁADNIE ten sam, niezmieniony, zaimportowany kod z <code>src/ring_extraction.py</code>
— zmienia się WYŁĄCZNIE sposób budowy profilu wejściowego. Nic w <code>src/</code> nie
zostało zmienione — to walidacja przed ewentualną promocją do produkcji.</p>
{sections_html}
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
