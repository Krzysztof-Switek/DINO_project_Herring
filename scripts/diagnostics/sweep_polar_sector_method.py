"""30.07: eksperyment ilościowy (30 kart) — czy AGREGACJA SEKTOROWA (compute_polar_grid,
patrz polar_sector_coverage_report.py) naprawdę poprawia lokalizację przyrostów względem
dzisiejszego próbkowania 48 dosłownych linii, czy tylko "wygląda ładniej" na jednym
otolicie?

Metodologia SKOPIOWANA z sweep_forced_topk_peaks.py (ta sama próba 30 kart — best+worst
z outputs/28.07_e9_w0.1, ta sama metryka mean_dist do klasycznej referencji jednoosiowej).
Różnica: TU porównujemy DWA sposoby budowy profilu na promień/sektor:

  BASELINE: src.ring_extraction.density_peaks — dzisiejsze 48 dosłownych linii,
            64 próbki/linia (nearest-neighbour).
  NOWA:     lokalna reimplementacja (NIE dotyka src/) — sektor kątowy + bin t dla
            KAŻDEGO patcha w masce (otolith_axis.compute_polar_grid), profil na sektor
            = MAX ze wszystkich patchy trafiających w dany bin. Reszta pipeline'u
            (klastrowanie arc-aware + wybór DP) BEZ ZMIAN — dokładnie te same funkcje
            zaimportowane z src.ring_extraction.

n_t_bins ustalone RAZ na n_t_bins=12 (nie sweep'owane per karta) — ta wartość została już
zmierzona i wybrana dla DOKŁADNIE tej samej siatki 52x52 (density_image_size=728,
config_e9_w0.1) w polar_sector_coverage_report.py (sweep pustych binów: 12 = największe
n_t_bins z <20% pustych binów PRZED wypełnieniem, na przykładowym otolicie). Siatka
(H_p, W_p) jest identyczna dla każdej karty w tym biegu (density_transform zawsze
przeskalowuje crop do tego samego density_image_size), więc dobór jest ważny dla
wszystkich 30 kart, nie tylko tej jednej.

Usage: PYTHONIOENCODING=utf-8 python scripts/diagnostics/sweep_polar_sector_method.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import cv2
from PIL import Image as PILImage
from scipy.signal import find_peaks

from scripts.run_pipeline import load_merged_config
from src.visualization import select_top_k_samples, _CONTOUR_COLOR, _AXIS_COLOR
from src.candidates import find_candidate_peaks
from src.otolith_axis import (detect_axis, apply_background_mask, mask_bbox,
                              shift_axis_info, sample_profile_along_axis, compute_polar_grid)
from src.inference import load_model_from_checkpoint
from src.dataset import build_transforms
from src.ring_extraction import (density_peaks, _shift_peak_to_falling_edge,
                                 _cluster_by_radius_with_arcs, _cluster_score,
                                 _dp_select_t, _project_to_axis)

RUN_DIR = PROJECT_ROOT / "outputs" / "28.07_e9_w0.1"
CFG_PATH = PROJECT_ROOT / "configs" / "config_e9_w0.1.yaml"
CKPT = RUN_DIR / "checkpoints" / "embedded" / "best.pt"
PRED_CSV = RUN_DIR / "emb_on_emb" / "predictions.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "29.07_candidate_selection_walkthrough" / "polar_sector_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DIRS, N_SAMPLES = 48, 64
N_T_BINS = 12          # patrz uzasadnienie w module docstring
T_TOL = 0.06
DP_MIN_GAP = 0.04
DP_SPREAD_WEIGHT = 1.5


def _old_ray_cells(H_p, W_p, cx, cy, contour, cw, ch, n_dirs=N_DIRS, n_samples=N_SAMPLES):
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


def _bin_idx_grid(H_p, W_p, H, W, cx, cy, n_dirs):
    py = (np.arange(H_p, dtype=np.float32) + 0.5) * (H / H_p)
    px = (np.arange(W_p, dtype=np.float32) + 0.5) * (W / W_p)
    grid_x, grid_y = np.meshgrid(px, py)
    dx = grid_x - cx
    dy = grid_y - cy
    theta = np.arctan2(dy, dx)
    return np.clip(((theta + np.pi) / (2.0 * np.pi) * n_dirs).astype(np.int64), 0, n_dirs - 1)


def build_sector_profiles(density_grid, t_grid, valid_grid, bin_idx, n_dirs, n_t_bins):
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
    profiles = []
    for s in range(n_dirs):
        p = filled[s]
        rng = float(p.max() - p.min())
        profiles.append(((p - p.min()) / rng).astype(np.float32) if rng > 1e-6 else None)
    return profiles, pos_row, pos_col


def sector_peaks(profiles, pos_row, pos_col, cw, ch, H_p, W_p, n_t_bins,
                 min_distance, prominence, inner_margin, edge_margin=0.08):
    peaks, candidate_pts = [], []
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


def _select_from_peaks(peaks, age, axis_info, width_decay_weight=1.0, width_ceiling_weight=3.0):
    clusters = _cluster_by_radius_with_arcs(peaks, t_tol=T_TOL, n_dirs=N_DIRS)
    cands = [(c[0], _cluster_score(c)) for c in clusters]
    chosen_t = _dp_select_t(cands, max(0, int(age)), DP_MIN_GAP, DP_SPREAD_WEIGHT,
                            width_decay_weight, width_ceiling_weight)
    return _project_to_axis(chosen_t, axis_info), chosen_t


def mean_dist(finals, ref):
    if not finals or not ref:
        return None
    fa = np.asarray(finals, dtype=np.float32)
    ca = np.asarray(ref, dtype=np.float32)
    d = np.sqrt(((fa[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
    return float(d.min(axis=1).mean())


def render_card_overlay(crop_rgb, axis_info, old_pts, new_pts) -> np.ndarray:
    out = np.ascontiguousarray(crop_rgb[..., :3]).copy()
    H = out.shape[0]
    if axis_info.get("contour") is not None:
        cv2.drawContours(out, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    cv2.line(out, (int(cx), int(cy)), (int(fx), int(fy)), _AXIS_COLOR, max(1, H // 400))
    r = max(4, H // 110)
    for (x, y) in old_pts:
        cv2.circle(out, (int(x), int(y)), r, (230, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), r, (0, 0, 0), 1, cv2.LINE_AA)
    for (x, y) in new_pts:
        cv2.circle(out, (int(x), int(y)), max(2, r // 2), (30, 200, 30), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), max(2, r // 2), (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    print("Wczytywanie configu i modelu...", flush=True)
    cfg = load_merged_config(CFG_PATH, None)
    image_dir = Path(cfg.data.image_dir)
    model = load_model_from_checkpoint(cfg, CKPT)
    model.eval()
    device = next(model.parameters()).device
    density_transform = build_transforms(cfg.candidates.density_image_size, "test")
    min_dist = cfg.candidates.min_peak_distance
    prominence = cfg.candidates.prominence_threshold
    inner_margin = cfg.candidates.inner_margin
    width_decay_weight = cfg.candidates.width_decay_weight
    width_ceiling_weight = cfg.candidates.width_ceiling_weight

    best, worst = select_top_k_samples(
        PRED_CSV, cfg.inference.increment_samples.top_k_best,
        cfg.inference.increment_samples.top_k_worst)
    samples = list(best) + list(worst)
    print(f"  {len(samples)} kart (best+worst) do przetworzenia.", flush=True)

    rows = []
    overlays: dict[str, np.ndarray] = {}
    for i, row in enumerate(samples):
        iid = str(row["image_id"])
        img_path = image_dir / iid
        if not img_path.exists():
            continue
        print(f"  [{i+1}/{len(samples)}] {iid[:55]}", flush=True)
        orig_rgb = np.array(PILImage.open(img_path).convert("RGB"), dtype=np.uint8)
        H, W = orig_rgb.shape[:2]
        axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                                nucleus_method=cfg.segmentation.nucleus_method)
        if axis_info is None:
            continue
        mask_arr = axis_info["mask"]
        model_input_rgb = (apply_background_mask(orig_rgb, mask_arr)
                           if cfg.data.mask_background else orig_rgb)

        crop_x0, crop_y0, cw, ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)
        crop_rgb = model_input_rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
        mask_cropped = mask_arr[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
        d_axis_info = shift_axis_info(axis_info, -crop_x0, -crop_y0)
        density_tensor = density_transform(PILImage.fromarray(crop_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            density_grid = model.get_density_probs(density_tensor).squeeze(0).cpu().numpy()
        H_p, W_p = density_grid.shape
        total = H_p * W_p

        age = int(row.get("predicted_age", 0))
        true_age = int(row.get("age", -1))
        abs_err = abs(age - true_age)
        cx, cy = d_axis_info["centroid"]
        contour = d_axis_info["contour"].reshape(-1, 2)

        # Klasyczna referencja jednoosiowa (ta sama, której używa _localization_quality).
        gray = orig_rgb.mean(axis=2)
        prof_1d, line_xy = sample_profile_along_axis(
            gray, axis_info["centroid"], axis_info["far_edge"], H, W, n_samples=50)
        prof_1d = np.asarray(prof_1d, dtype=np.float32)
        rng = float(prof_1d.max() - prof_1d.min())
        if rng > 1e-6:
            prof_1d = (prof_1d - prof_1d.min()) / rng
        classical_ref = []
        for pi in find_candidate_peaks(prof_1d, min_dist, prominence):
            pi = int(pi)
            if 0 <= pi < len(line_xy):
                classical_ref.append((int(line_xy[pi][0]), int(line_xy[pi][1])))

        # --- BASELINE: dzisiejsze 48 dosłownych linii ---
        old_cells = _old_ray_cells(H_p, W_p, cx, cy, contour, cw, ch)
        dpk_old, _ = density_peaks(density_grid, d_axis_info, ch, cw, n_dirs=N_DIRS,
                                   n_samples=N_SAMPLES, min_distance=min_dist,
                                   prominence=prominence, inner_margin=inner_margin)
        finals_old, chosen_old = _select_from_peaks(
            dpk_old, age, axis_info, width_decay_weight, width_ceiling_weight)

        # --- NOWA: sektor kątowy + bin t (compute_polar_grid), agregacja MAX ---
        t_grid, valid_grid, _theta_grid = compute_polar_grid(mask_cropped, (cx, cy), H_p, W_p,
                                                n_angle_bins=N_DIRS)
        bin_idx = _bin_idx_grid(H_p, W_p, ch, cw, cx, cy, N_DIRS)
        n_valid = int(valid_grid.sum())
        profiles, pos_row, pos_col = build_sector_profiles(
            density_grid, t_grid, valid_grid, bin_idx, N_DIRS, N_T_BINS)
        dpk_new, _ = sector_peaks(profiles, pos_row, pos_col, cw, ch, H_p, W_p, N_T_BINS,
                                  min_dist, prominence, inner_margin)
        finals_new, chosen_new = _select_from_peaks(
            dpk_new, age, axis_info, width_decay_weight, width_ceiling_weight)

        rows.append({
            "iid": iid, "age": age, "true_age": true_age, "abs_err": abs_err,
            "cov_old_pct": 100 * len(old_cells) / total, "cov_new_pct": 100 * n_valid / total,
            "n_peaks_old": len(dpk_old), "n_peaks_new": len(dpk_new),
            "n_final_old": len(chosen_old), "n_final_new": len(chosen_new),
            "dist_old": mean_dist(finals_old, classical_ref),
            "dist_new": mean_dist(finals_new, classical_ref),
            "changed": chosen_old != chosen_new,
            "finals_old": finals_old, "finals_new": finals_new,
        })
        overlays[iid] = render_card_overlay(crop_rgb, d_axis_info, finals_old, finals_new)

    print("\n=== Wyniki per karta ===", flush=True)
    print(f"{'karta':45s} {'wiek':>4s} {'praw':>4s} {'err':>3s} {'cov_S':>6s} {'cov_N':>6s} "
         f"{'n_fin_S':>8s} {'n_fin_N':>8s} {'dist_S':>8s} {'dist_N':>8s}", flush=True)
    for r in rows:
        ds = f"{r['dist_old']:.0f}" if r["dist_old"] is not None else "None"
        dn = f"{r['dist_new']:.0f}" if r["dist_new"] is not None else "None"
        print(f"{r['iid'][:45]:45s} {r['age']:>4d} {r['true_age']:>4d} {r['abs_err']:>3d} "
             f"{r['cov_old_pct']:>5.1f}% {r['cov_new_pct']:>5.1f}% {r['n_final_old']:>8d} "
             f"{r['n_final_new']:>8d} {ds:>8s} {dn:>8s}", flush=True)

    print("\n=== Podsumowanie ===", flush=True)
    dists_o = [r["dist_old"] for r in rows if r["dist_old"] is not None]
    dists_n = [r["dist_new"] for r in rows if r["dist_new"] is not None]
    n_changed = sum(1 for r in rows if r["changed"])
    shortfall_o = sum(1 for r in rows if r["n_final_old"] < r["age"])
    shortfall_n = sum(1 for r in rows if r["n_final_new"] < r["age"])
    print(f"Kart: {len(rows)}")
    print(f"Śr. pokrycie patchy — STARA: {np.mean([r['cov_old_pct'] for r in rows]):.1f}%  "
         f"NOWA: {np.mean([r['cov_new_pct'] for r in rows]):.1f}%")
    print(f"mean_dist do klasycznej referencji — STARA: {np.mean(dists_o):.2f}px (n={len(dists_o)})  "
         f"NOWA: {np.mean(dists_n):.2f}px (n={len(dists_n)})")
    print(f"Kart, gdzie finalny wybór się ZMIENIŁ: {n_changed}/{len(rows)} "
         f"({n_changed/len(rows):.1%})" if rows else "brak kart")
    print(f"Kart z niedoborem (mniej finałów niż wiek) — STARA: {shortfall_o}/{len(rows)}  "
         f"NOWA: {shortfall_n}/{len(rows)}")

    def _improvement(r):
        if r["dist_old"] is None or r["dist_new"] is None:
            return None
        return r["dist_old"] - r["dist_new"]

    print("\n=== Poprawa (dist_old - dist_new) wg trafności wieku ===")
    for label, pred in (("trafny wiek (|błąd|<=1)", lambda r: r["abs_err"] <= 1),
                        ("nietrafny wiek (|błąd|>=2)", lambda r: r["abs_err"] >= 2)):
        sub = [r for r in rows if pred(r)]
        imps = [v for v in (_improvement(r) for r in sub) if v is not None]
        print(f"  {label}: {len(sub)} kart, {len(imps)} z ważną metryką, "
             f"śr. poprawa = {np.mean(imps):+.1f}px" if imps else f"  {label}: brak ważnych kart")

    print("\n=== Wg klasy wieku ===")
    for label, pred in (("wiek 0-6", lambda r: 0 <= r["age"] <= 6),
                        ("wiek 7+", lambda r: r["age"] >= 7)):
        sub = [r for r in rows if pred(r)]
        do_sub = [r["dist_old"] for r in sub if r["dist_old"] is not None]
        dn_sub = [r["dist_new"] for r in sub if r["dist_new"] is not None]
        if do_sub:
            print(f"  {label}: {len(sub)} kart — STARA {np.mean(do_sub):.1f}px, "
                 f"NOWA {np.mean(dn_sub):.1f}px (n={len(dn_sub)})")
        else:
            print(f"  {label}: {len(sub)} kart, brak ważnej metryki")

    # --- Zapisz 2 przykładowe karty (największa poprawa + mediana poprawy) do embeddingu
    # w głównym raporcie (ta sama konwencja co Krok 6 w density_candidate_mechanism_report.py:
    # liczby/obrazy z tego skryptu wklejane tam NA STAŁE, nie przeliczane przy każdym uruchomieniu). ---
    valid_rows = [r for r in rows if r["dist_old"] is not None and r["dist_new"] is not None]
    if valid_rows:
        best_improve = max(valid_rows, key=lambda r: r["dist_old"] - r["dist_new"])
        sorted_by_imp = sorted(valid_rows, key=lambda r: r["dist_old"] - r["dist_new"])
        median_row = sorted_by_imp[len(sorted_by_imp) // 2]
        for tag, r in (("best_improvement", best_improve), ("median", median_row)):
            out_path = OUT_DIR / f"card_{tag}.png"
            PILImage.fromarray(overlays[r["iid"]]).save(out_path)
            print(f"Zapisano przykładową kartę ({tag}): {out_path} — "
                 f"dist_old={r['dist_old']:.0f}px, dist_new={r['dist_new']:.0f}px, "
                 f"wiek={r['age']} (prawda={r['true_age']})", flush=True)

    print("""
=== Interpretacja ===
- dist_new wyraźnie NIŻSZY niż dist_old -> pełne pokrycie patchy (sektory) daje lepszą,
  mniej szumną pulę kandydatów niż rzadkie próbkowanie linii.
  dist_new WYŻSZY -> dodatkowe patche wstrzykują więcej szumu niż realnego sygnału.
  Podobne -> zmiana neutralna na tej próbie.
- Kart ZMIENIONYCH blisko 0% -> w praktyce nic się nie zmienia mimo innej metody budowy
  profilu (arc-aware clustering + DP i tak zbiega do tych samych klastrów).
""")


if __name__ == "__main__":
    main()
