"""31.07 (cd. 7) — Liczbowe porównanie WSZYSTKICH metod na tej samej 30-kartowej próbie.

Kontynuacja v1-v8 (`ring_shortest_path_report.py`, `ring_fragment_fusion_report.py`,
`ring_fragment_detection_report.py`, `all_methods_comparison_report.py`) — tamte raporty są
wizualne/jakościowe na 1-2 zdjęciach; TEN skrypt liczy `mean_dist_final_to_classical_px`
(ta sama metodologia co w całym projekcie — odległość finalnych punktów metody do klasycznej
referencji NA OSI, uśredniona po kartach) na TEJ SAMEJ 30-kartowej próbie (15
najlepszych+15 najgorszych wg błędu wieku), którą projekt używa od 21.07 do wszystkich
poprzednich porównań (E9, agregacja sektorowa, hires966, itd.) — więc wynik jest wprost
porównywalny z całą resztą historii projektu.

Metody PRODUKCYJNE (classical/dp/density/consensus) NIE są tu przeliczane od nowa — cytowane
wprost z `outputs/28.07_e9_w0.1/localization_quality.json` (dokładnie ta sama 30-kartowa
próba, ten sam checkpoint bez E9 na wieku — E9 nie miało żadnego wpływu na wiek, patrz
`outputs/31.07_radial_Analiza_wnioski.md`).

Nowe metody przeliczone na 30 kartach:
  - gradient / Frangi — najkrótsza ZAMKNIĘTA ścieżka (K=wiek, prior szerokości)
  - gradient / Frangi / Gabor / CLAHE+density+Frangi — pewne otwarte fragmenty (top 8, rzut
    na oś)
  - Frangi — RANSAC (dopasowana krzywa, wartość na osi)
  - Hough (najlepsze K=wiek okręgów, przybliżone jako punkty na osi)
  - snake (jeden kanoniczny start t&#8320;=0,5, punkt na osi)

src/ CAŁKOWICIE niedotknięte.

Usage: PYTHONIOENCODING=utf-8 python scripts/diagnostics/all_methods_30card_sweep.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import torch
from PIL import Image as PILImage

from src.otolith_axis import detect_axis, apply_background_mask, mask_bbox, shift_axis_info
from src.dataset import build_transforms, decode_age_ordinal
from src.inference import load_model_from_checkpoint
from scripts.run_pipeline import load_merged_config

from scripts.diagnostics.ring_shortest_path_report import (
    ray_cast_R_theta, polar_unwrap, apply_margins, frangi_cost_field,
    find_confident_open_arcs, extract_k_rings_ordered, classical_axis_points, mean_dist,
    MODEL_VARIANTS, N_ANGLE, N_RADIUS, WINDOW, JUMP_PENALTY, INNER_MARGIN, EDGE_MARGIN,
    ARC_STRIDE_DEG, ARC_MIN_GAP_DEG, WIDTH_CEILING_WEIGHT, WIDTH_DECAY_WEIGHT, MIN_GAP_FRAC,
)
from scripts.diagnostics.ring_fragment_fusion_report import ransac_fit_ring
from scripts.diagnostics.all_methods_comparison_report import (
    gabor_cost_field, clahe_density_frangi_field, hough_circles_on_field, run_snake,
)

# 30 kart — ta sama próba co outputs/28.07_e9_w0.1/localization_quality.json (i cała reszta
# historii projektu, 21.07 w przód).
CARDS = [
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex32_Single2_Right.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex32_Single1_Left.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex27_Single2_Right.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex23_Single1_LowQuality.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex20_Single2_Left.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex110_Single2_Left.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex107_Single2_Left.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex107_Single1_Right.jpg",
    "2023_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg",
    "2023_BITS1q_HER_ZatokaGdanska_Embedded_Sharpest_FishIndex115_Single1_Left.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex92_Single2_Right.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex92_Single1_Left.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex89_Single2_Broken.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex89_Single1_Right.jpg",
    "2023_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex68_Single2_Right.jpg",
    "2023_BITS4q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex145_Single1_Left.jpg",
    "2024_BITS1q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex90_Single1_Right.jpg",
    "2023_BITS4q_HER_KolobrzeskoDarlowskie_Embedded_Sharpest_FishIndex65_Single1_Right.jpg",
    "2024_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex63_Single1_Right.jpg",
    "2022_BIAS_HER_ZatokaGdanska_Embedded_Sharpest_FishIndex91_Single1_Left.jpg",
    "2024_BITS1q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex63_Single2_Left.jpg",
    "2024_BITS1q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex38_Single2_Right.jpg",
    "2023_BITS1q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex49_Single2_Right.jpg",
    "2024_BITS1q_HER_BornholmskieS_Embedded_Sharpest_FishIndex33_Single2_Left.jpg",
    "2024_BITS1q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex90_Single2_Left.jpg",
    "2024_BITS1q_HER_BornholmskieS_Embedded_Sharpest_FishIndex33_Single3_Right.jpg",
    "2024_BITS1q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex38_Single1_Left.jpg",
    "2023_BITS4q_HER_KolobrzeskoDarlowskie_Embedded_Sharpest_FishIndex59_Single1_Right.jpg",
    "2024_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex104_Single1_Left.jpg",
    "2023_BITS4q_HER_LawicaSrodkowa_Embedded_Sharpest_FishIndex53_Single2_Left.jpg",
]

# Cytowane wprost z outputs/28.07_e9_w0.1/localization_quality.json — TA SAMA próba, NIE
# przeliczane od nowa (metody produkcyjne, niezmienione).
REFERENCE_ROWS = [
    ("classical (produkcyjny, arc-aware+DP na sygnale klasycznym)", 206.82),
    ("consensus (produkcyjny, density+classical bez DP)", 236.44),
    ("dp (produkcyjny, fuzja classical+density, arc-aware+DP)", 229.17),
    ("density (produkcyjny, surowy sygnał modelu)", 260.71),
]

GABOR_FREQS_FAST = (0.08,)
GABOR_N_THETA_FAST = 4
SNAKE_T0 = 0.5

OUT_PATH = PROJECT_ROOT / "outputs" / "31.07_ring_shortest_path" / "30card_sweep_all_methods.json"
OUT_MD = PROJECT_ROOT / "outputs" / "31.07_ring_shortest_path" / "30card_sweep_all_methods.md"


def _normalize(field: np.ndarray) -> np.ndarray:
    rng = float(field.max() - field.min())
    return (field - field.min()) / rng if rng > 1e-6 else field * 0.0


def _angular_dist(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def axis_col_index(angles: np.ndarray, axis_angle: float) -> int:
    return int(np.argmin(_angular_dist(angles, axis_angle)))


def closed_ring_axis_pts(polar_norm, k, R_theta, angles, cx, cy, a_idx):
    cost = apply_margins(1.0 - polar_norm, N_RADIUS, INNER_MARGIN, EDGE_MARGIN)
    paths = extract_k_rings_ordered(cost, k, WINDOW, JUMP_PENALTY, MIN_GAP_FRAC,
                                    WIDTH_CEILING_WEIGHT, WIDTH_DECAY_WEIGHT)
    pts = []
    for p in paths:
        t = float(p[a_idx]) / max(1, N_RADIUS - 1)
        r = t * R_theta[a_idx]
        pts.append((cx + r * np.cos(angles[a_idx]), cy + r * np.sin(angles[a_idx])))
    return pts


def confident_arc_axis_pts(polar_norm, top_k, R_theta, angles, cx, cy, a_idx):
    cost = apply_margins(1.0 - polar_norm, N_RADIUS, INNER_MARGIN, EDGE_MARGIN)
    arcs = find_confident_open_arcs(cost, 60, ARC_STRIDE_DEG, WINDOW, JUMP_PENALTY,
                                    top_k, ARC_MIN_GAP_DEG, 0.06)
    pts = []
    for a in arcs:
        r = a["mean_t"] * R_theta[a_idx]
        pts.append((cx + r * np.cos(angles[a_idx]), cy + r * np.sin(angles[a_idx])))
    return pts, arcs


def hough_axis_pts(field_cart, cx, cy, axis_angle, k):
    H, W = field_cart.shape[:2]
    field_u8 = (np.clip(field_cart, 0, 1) * 255).astype(np.uint8)
    field_u8 = cv2.GaussianBlur(field_u8, (5, 5), 0)
    min_r = int(0.15 * min(H, W))
    max_r = int(0.48 * min(H, W))
    circles = hough_circles_on_field(field_u8, min_r, max_r)
    if circles is None:
        return []
    pts = []
    for c in circles[0][:max(1, k)]:
        _x0, _y0, r = c
        pts.append((cx + r * np.cos(axis_angle), cy + r * np.sin(axis_angle)))
    return pts


def snake_axis_pt(field_cart, cx, cy, R_theta, angles, axis_angle, t0=SNAKE_T0, n_points=120):
    snake = run_snake(field_cart, cx, cy, R_theta, angles, t0, n_points)
    theta_pts = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    axis_theta = axis_angle % (2 * np.pi)
    idx = int(np.argmin(_angular_dist(theta_pts, axis_theta)))
    row, col = snake[idx]
    return [(col, row)]


def process_card(model, cfg, device, image_id: str) -> dict:
    image_dir = Path(cfg.data.image_dir)
    orig_rgb = np.array(PILImage.open(image_dir / image_id).convert("RGB"), dtype=np.uint8)
    axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                            nucleus_method=cfg.segmentation.nucleus_method)
    if axis_info is None:
        return {}
    mask_arr = axis_info["mask"]
    model_input_rgb = (apply_background_mask(orig_rgb, mask_arr)
                       if cfg.data.mask_background else orig_rgb)
    crop_x0, crop_y0, cw, ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)
    crop_rgb = model_input_rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    mask_cropped = mask_arr[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    d_axis_info = shift_axis_info(axis_info, -crop_x0, -crop_y0)
    cx, cy = d_axis_info["centroid"]
    fx, fy = d_axis_info["far_edge"]
    axis_angle = float(np.arctan2(fy - cy, fx - cx))

    age_transform = build_transforms(cfg.data.image_size, "test")
    age_tensor = age_transform(PILImage.fromarray(model_input_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        coral_logits = model(age_tensor)["coral_logits"]
    predicted_age = max(1, int(decode_age_ordinal(coral_logits).item()))

    density_transform = build_transforms(cfg.candidates.density_image_size, "test")
    density_tensor = density_transform(PILImage.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        density_grid = model.get_density_probs(density_tensor).squeeze(0).cpu().numpy()

    gray_crop = crop_rgb.mean(axis=2).astype(np.float32)
    R_theta, angles = ray_cast_R_theta(mask_cropped, cx, cy, N_ANGLE)
    a_idx = axis_col_index(angles, axis_angle)

    # WAŻNE: te same progi co produkcyjna referencja w scripts/run_pipeline.py
    # (cfg.candidates.min_peak_distance/prominence_threshold), NIE dowolne liczby — inaczej
    # zbiór punktów referencyjnych jest gęstszy/bardziej permisywny niż ten, względem którego
    # liczone są cytowane wartości classical/dp/density/consensus (206-260px), i porównanie
    # jest nieuczciwe (sztucznie zaniża mean_dist nowych metod, patrz handoff 31.07).
    classical_ref_pts = classical_axis_points(
        gray_crop, cx, cy, fx, fy,
        min_dist=cfg.candidates.min_peak_distance,
        prominence=cfg.candidates.prominence_threshold)

    polar_classical = _normalize(polar_unwrap(gray_crop, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))
    polar_grad = _normalize(np.abs(np.gradient(polar_classical, axis=0)))
    frangi_cart = frangi_cost_field(gray_crop)
    polar_frangi = _normalize(polar_unwrap(frangi_cart, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))
    gabor_cart = gabor_cost_field(gray_crop, frequencies=GABOR_FREQS_FAST, n_theta=GABOR_N_THETA_FAST)
    polar_gabor = _normalize(polar_unwrap(gabor_cart, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))
    clahe_dens_cart = clahe_density_frangi_field(density_grid)
    polar_clahe = _normalize(polar_unwrap(clahe_dens_cart, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))

    results: dict[str, float | None] = {}

    def _safe(name, fn):
        try:
            pts = fn()
            results[name] = mean_dist(pts, classical_ref_pts)
        except Exception as e:
            print(f"    [BŁĄD {name}] {e}", flush=True)
            results[name] = None

    _safe("gradient — closed shortest-path", lambda: closed_ring_axis_pts(
        polar_grad, predicted_age, R_theta, angles, cx, cy, a_idx))
    _safe("Frangi — closed shortest-path", lambda: closed_ring_axis_pts(
        polar_frangi, predicted_age, R_theta, angles, cx, cy, a_idx))

    grad_arcs_pts, grad_arcs = confident_arc_axis_pts(polar_grad, 8, R_theta, angles, cx, cy, a_idx)
    results["gradient — pewne otwarte fragmenty"] = mean_dist(grad_arcs_pts, classical_ref_pts)

    frangi_arcs_pts, frangi_arcs = confident_arc_axis_pts(polar_frangi, 8, R_theta, angles, cx, cy, a_idx)
    results["Frangi — pewne otwarte fragmenty"] = mean_dist(frangi_arcs_pts, classical_ref_pts)

    gabor_arcs_pts, _ = confident_arc_axis_pts(polar_gabor, 8, R_theta, angles, cx, cy, a_idx)
    results["Gabor — pewne otwarte fragmenty"] = mean_dist(gabor_arcs_pts, classical_ref_pts)

    clahe_arcs_pts, _ = confident_arc_axis_pts(polar_clahe, 8, R_theta, angles, cx, cy, a_idx)
    results["CLAHE+density+Frangi — pewne otwarte fragmenty"] = mean_dist(clahe_arcs_pts, classical_ref_pts)

    def _ransac():
        fit = ransac_fit_ring(frangi_arcs, angles, N_RADIUS)
        if fit is None:
            return []
        t = float(np.clip(fit["predict"](np.array([axis_angle]))[0], 0.0, 1.0))
        r = t * R_theta[a_idx]
        return [(cx + r * np.cos(angles[a_idx]), cy + r * np.sin(angles[a_idx]))]
    _safe("Frangi — RANSAC", _ransac)

    _safe("Hough", lambda: hough_axis_pts(frangi_cart, cx, cy, axis_angle, predicted_age))
    _safe("snake (t0=0,5)", lambda: snake_axis_pt(frangi_cart, cx, cy, R_theta, angles, axis_angle))

    results["_predicted_age"] = predicted_age
    return results


def main() -> None:
    base = MODEL_VARIANTS[0]
    cfg = load_merged_config(base["cfg_path"], None)
    device = torch.device(cfg.training.device if cfg.training.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Wczytywanie modelu ({base['name']})...", flush=True)
    model = load_model_from_checkpoint(cfg, base["run_dir"] / "checkpoints" / "embedded" / "best.pt")
    model.eval()
    device = next(model.parameters()).device

    all_results: dict[str, list[float]] = {}
    per_card: list[dict] = []
    t_start = time.time()
    for i, image_id in enumerate(CARDS):
        t0 = time.time()
        print(f"\n[{i+1}/{len(CARDS)}] {image_id}", flush=True)
        try:
            res = process_card(model, cfg, device, image_id)
        except Exception as e:
            print(f"  BŁĄD KARTY: {e}", flush=True)
            res = {}
        if not res:
            continue
        per_card.append({"image_id": image_id, **res})
        for k, v in res.items():
            if k.startswith("_"):
                continue
            all_results.setdefault(k, []).append(v)
        print(f"  ({time.time()-t0:.1f}s) " + ", ".join(
            f"{k}={v}" for k, v in res.items() if not k.startswith("_")), flush=True)

    print(f"\nCałość: {time.time()-t_start:.1f}s\n", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Metoda':55s} {'n':>4s} {'mean_dist_px':>14s}", flush=True)
    print("-" * 70, flush=True)
    summary_rows = []
    for name, ref_val in REFERENCE_ROWS:
        summary_rows.append((name, 30, ref_val))
        print(f"{name:55s} {'30*':>4s} {ref_val:>14.2f}", flush=True)
    for name, vals in all_results.items():
        vals_clean = [v for v in vals if v is not None]
        m = round(float(np.mean(vals_clean)), 2) if vals_clean else None
        summary_rows.append((name, len(vals_clean), m))
        print(f"{name:55s} {len(vals_clean):>4d} {str(m):>14s}", flush=True)
    print("=" * 70, flush=True)
    print("(*) cytowane z outputs/28.07_e9_w0.1/localization_quality.json, ta sama próba,", flush=True)
    print("    NIE przeliczane od nowa.", flush=True)

    import json
    OUT_PATH.write_text(json.dumps({"summary": summary_rows, "per_card": per_card}, indent=2), encoding="utf-8")

    lines = ["# Liczbowe porównanie wszystkich metod — 30 kart\n",
            "Ta sama 30-kartowa próba co cała historia projektu (`outputs/28.07_e9_w0.1/",
            "localization_quality.json`). `classical`/`consensus`/`dp`/`density` cytowane, nie",
            "przeliczane. Reszta policzona w tym skrypcie.\n",
            "| Metoda | n kart | mean_dist (px) |",
            "|---|---|---|"]
    for name, n, m in summary_rows:
        lines.append(f"| {name} | {n} | {m} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}\nZapisano: {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
