"""29.07: eksperyment ilościowy — czy WYMUSZENIE dokładnie `wiek` (z głowicy CORAL)
pików density NA KAŻDYM promieniu osobno (zamiast dzisiejszego progu prominencji,
który daje zmienną liczbę pików per promień) poprawia lokalizację przyrostów?

Pytanie użytkownika (29.07): "co jeśli wymuszalibyśmy na każdym promieniu znalezienie
pików w ilości odpowiadającej wiekowi przekazanemu z głowicy CORAL?"

Metodologia skopiowana z sweep_classical_concentricity_weight.py (ten sam projekt, ta
sama próba 30 kart, ta sama metryka mean_dist do klasycznej referencji jednoosiowej —
_localization_quality używa dokładnie tego samego porównania). Różnica: TU porównujemy
DWA warianty wykrywania pików density na TYCH SAMYCH kartach, nie sweep jednego wagowego
parametru:

  BASELINE: src.ring_extraction.density_peaks — dzisiejszy próg prominencji,
            zmienna liczba pików per promień (0 do kilku).
  FORCED:   lokalna reimplementacja (NIE dotyka src/) — każdy promień: wszystkie lokalne
            maksima (tylko odstęp min_distance, BEZ progu prominencji), posortowane wg
            prominencji, brane top-min(wiek, znalezione). Reszta pipeline'u (klastrowanie
            arc-aware + wybór DP) BEZ ZMIAN — dokładnie te same funkcje.

Prawdziwa, wysokorozdzielcza mapa density (density_image_size + crop-do-bboxa) — jak w
całej reszcie tej sesji, nie uproszczona wersja z podstawowego forward passu.

Usage: python scripts/diagnostics/sweep_forced_topk_peaks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image as PILImage
from scipy.signal import find_peaks, peak_prominences

from scripts.run_pipeline import load_merged_config
from src.visualization import select_top_k_samples
from src.candidates import find_candidate_peaks
from src.otolith_axis import (detect_axis, apply_background_mask, mask_bbox,
                              shift_axis_info, sample_profile_along_axis)
from src.inference import load_model_from_checkpoint
from src.dataset import build_transforms
from src.ring_extraction import (density_peaks, _all_ray_profiles, _shift_peak_to_falling_edge,
                                 _cluster_by_radius_with_arcs, _cluster_score,
                                 _dp_select_t, _project_to_axis)

RUN_DIR = PROJECT_ROOT / "outputs" / "28.07_e9_w0.1"   # ostatni bieg 518px z tej sesji
CFG_PATH = PROJECT_ROOT / "configs" / "config_e9_w0.1.yaml"
CKPT = RUN_DIR / "checkpoints" / "embedded" / "best.pt"
PRED_CSV = RUN_DIR / "emb_on_emb" / "predictions.csv"

N_DIRS, N_SAMPLES = 48, 64
T_TOL = 0.06
DP_MIN_GAP = 0.04
DP_SPREAD_WEIGHT = 1.5


def _all_ray_peaks_forced_topk(density_grid, axis_info, image_h, image_w, k: int,
                               *, n_dirs: int = N_DIRS, n_samples: int = N_SAMPLES,
                               min_distance: int = 3, inner_margin: float = 0.05,
                               edge_margin: float = 0.08):
    """Jak ring_extraction._all_ray_peaks, ale BEZ progu prominencji: każdy promień
    zwraca top-min(k, znalezione) lokalnych maksimów wg prominencji (odstęp
    min_distance nadal wymuszony). Samodzielna reimplementacja — src/ nietknięte."""
    profiles, line_xys, _ = _all_ray_profiles(density_grid, axis_info, image_h, image_w,
                                              n_dirs=n_dirs, n_samples=n_samples)
    peaks, candidate_pts = [], []
    for ray_idx, (pn, line_xy) in enumerate(zip(profiles, line_xys)):
        if pn is None or k <= 0:
            continue
        idxs, _ = find_peaks(pn, distance=max(1, int(min_distance)))
        if len(idxs) == 0:
            continue
        proms = peak_prominences(pn, idxs)[0]
        order = np.argsort(-proms)
        top = idxs[order[:k]]
        for idx in top:
            t_orig = idx / max(1, n_samples - 1)
            if t_orig < inner_margin or t_orig > 1.0 - edge_margin:
                continue
            edge_idx = _shift_peak_to_falling_edge(pn, int(idx))
            t = edge_idx / max(1, n_samples - 1)
            x, y = int(line_xy[edge_idx][0]), int(line_xy[edge_idx][1])
            peaks.append((t, float(pn[idx]), x, y, ray_idx))
            candidate_pts.append((x, y))
    return peaks, candidate_pts


def _select_from_peaks(peaks, age: int, axis_info,
                       width_decay_weight: float = 1.0, width_ceiling_weight: float = 3.0):
    """Klastrowanie arc-aware + wybór DP dokładnie `age` finałów — WSPÓLNE dla obu
    wariantów (jedyna różnica to `peaks` na wejściu)."""
    clusters = _cluster_by_radius_with_arcs(peaks, t_tol=T_TOL, n_dirs=N_DIRS)
    cands = [(c[0], _cluster_score(c)) for c in clusters]
    chosen_t = _dp_select_t(cands, max(0, int(age)), DP_MIN_GAP, DP_SPREAD_WEIGHT,
                            width_decay_weight, width_ceiling_weight)
    return _project_to_axis(chosen_t, axis_info), chosen_t, len(clusters)


def mean_dist(finals, ref):
    if not finals or not ref:
        return None
    fa = np.asarray(finals, dtype=np.float32)
    ca = np.asarray(ref, dtype=np.float32)
    d = np.sqrt(((fa[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
    return float(d.min(axis=1).mean())


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

        # Prawdziwa, wysokorozdzielcza mapa density (crop-do-bboxa + drugi forward pass) —
        # dokładnie ta, której używa select_increments na kartach raportu.
        crop_x0, crop_y0, cw, ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)
        crop_rgb = model_input_rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
        d_axis_info = shift_axis_info(axis_info, -crop_x0, -crop_y0)
        density_tensor = density_transform(PILImage.fromarray(crop_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            density_grid = model.get_density_probs(density_tensor).squeeze(0).cpu().numpy()

        age = int(row.get("predicted_age", 0))
        true_age = int(row.get("age", -1))
        abs_err = abs(age - true_age)

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

        # `t` (promień znormalizowany) jest niezależny od układu (crop to czyste
        # przesunięcie osi, nie przeskalowanie) — _select_from_peaks projektuje na
        # PEŁNY axis_info, więc finals_* są od razu w pikselach całego zdjęcia, bez
        # dodatkowego przesuwania o crop_x0/crop_y0.
        # --- BASELINE: dzisiejsze wykrywanie (próg prominencji, zmienna liczba/promień) ---
        dpk_base, _ = density_peaks(density_grid, d_axis_info, ch, cw, n_dirs=N_DIRS,
                                    n_samples=N_SAMPLES, min_distance=min_dist,
                                    prominence=prominence, inner_margin=inner_margin)
        finals_base, chosen_base, n_clusters_base = _select_from_peaks(
            dpk_base, age, axis_info, width_decay_weight, width_ceiling_weight)

        # --- FORCED: top-`age` pików na KAŻDYM promieniu, reszta pipeline'u identyczna ---
        dpk_forced, _ = _all_ray_peaks_forced_topk(
            density_grid, d_axis_info, ch, cw, age, n_dirs=N_DIRS, n_samples=N_SAMPLES,
            min_distance=min_dist, inner_margin=inner_margin)
        finals_forced, chosen_forced, n_clusters_forced = _select_from_peaks(
            dpk_forced, age, axis_info, width_decay_weight, width_ceiling_weight)

        rows.append({
            "iid": iid, "age": age, "true_age": true_age, "abs_err": abs_err,
            "n_peaks_base": len(dpk_base), "n_peaks_forced": len(dpk_forced),
            "n_clusters_base": n_clusters_base, "n_clusters_forced": n_clusters_forced,
            "n_final_base": len(chosen_base), "n_final_forced": len(chosen_forced),
            "dist_base": mean_dist(finals_base, classical_ref),
            "dist_forced": mean_dist(finals_forced, classical_ref),
            "changed": chosen_base != chosen_forced,
        })

    print("\n=== Wyniki per karta ===", flush=True)
    print(f"{'karta':45s} {'wiek':>4s} {'praw':>4s} {'err':>3s} {'n_pik_B':>8s} {'n_pik_F':>8s} "
         f"{'n_fin_B':>8s} {'n_fin_F':>8s} {'dist_B':>8s} {'dist_F':>8s}", flush=True)
    for r in rows:
        db = f"{r['dist_base']:.0f}" if r["dist_base"] is not None else "None"
        df = f"{r['dist_forced']:.0f}" if r["dist_forced"] is not None else "None"
        print(f"{r['iid'][:45]:45s} {r['age']:>4d} {r['true_age']:>4d} {r['abs_err']:>3d} "
             f"{r['n_peaks_base']:>8d} {r['n_peaks_forced']:>8d} {r['n_final_base']:>8d} "
             f"{r['n_final_forced']:>8d} {db:>8s} {df:>8s}", flush=True)

    print("\n=== Podsumowanie ===", flush=True)
    dists_b = [r["dist_base"] for r in rows if r["dist_base"] is not None]
    dists_f = [r["dist_forced"] for r in rows if r["dist_forced"] is not None]
    n_changed = sum(1 for r in rows if r["changed"])
    shortfall_b = sum(1 for r in rows if r["n_final_base"] < r["age"])
    shortfall_f = sum(1 for r in rows if r["n_final_forced"] < r["age"])
    print(f"Kart: {len(rows)}")
    print(f"mean_dist do klasycznej referencji — BASELINE: {np.mean(dists_b):.2f}px "
         f"(n={len(dists_b)})  FORCED: {np.mean(dists_f):.2f}px (n={len(dists_f)})")
    print(f"Śr. liczba pików (pula kandydatów) — BASELINE: "
         f"{np.mean([r['n_peaks_base'] for r in rows]):.1f}  FORCED: "
         f"{np.mean([r['n_peaks_forced'] for r in rows]):.1f}")
    print(f"Śr. liczba klastrów — BASELINE: "
         f"{np.mean([r['n_clusters_base'] for r in rows]):.1f}  FORCED: "
         f"{np.mean([r['n_clusters_forced'] for r in rows]):.1f}")
    print(f"Kart, gdzie finalny wybór (final_t) się ZMIENIŁ: {n_changed}/{len(rows)} "
         f"({n_changed/len(rows):.1%})" if rows else "brak kart")
    print(f"Kart z niedoborem (mniej finałów niż wiek) — BASELINE: {shortfall_b}/{len(rows)}  "
         f"FORCED: {shortfall_f}/{len(rows)}")

    # --- Korelacja z trafnością wieku (użytkownik, 29.07): czy mechanizm pomaga BARDZIEJ
    # na kartach, gdzie CORAL trafił wiek? I osobno: jak wygląda w samej klasie wieku 0-6
    # (najliczniejszej, najlepiej wytrenowanej). ---
    def _improvement(r):
        if r["dist_base"] is None or r["dist_forced"] is None:
            return None
        return r["dist_base"] - r["dist_forced"]   # dodatnie = FORCED lepszy

    print("\n=== Poprawa (dist_base - dist_forced) wg trafności wieku ===")
    for label, pred in (("trafny wiek (|błąd|<=1)", lambda r: r["abs_err"] <= 1),
                        ("nietrafny wiek (|błąd|>=2)", lambda r: r["abs_err"] >= 2)):
        sub = [r for r in rows if pred(r)]
        imps = [_improvement(r) for r in sub]
        imps = [v for v in imps if v is not None]
        print(f"  {label}: {len(sub)} kart, {len(imps)} z ważną metryką, "
             f"śr. poprawa = {np.mean(imps):+.1f}px" if imps else f"  {label}: brak ważnych kart")

    print("\n=== Wg klasy wieku ===")
    for label, pred in (("wiek 0-6", lambda r: 0 <= r["age"] <= 6),
                        ("wiek 7+", lambda r: r["age"] >= 7)):
        sub = [r for r in rows if pred(r)]
        db_sub = [r["dist_base"] for r in sub if r["dist_base"] is not None]
        df_sub = [r["dist_forced"] for r in sub if r["dist_forced"] is not None]
        if db_sub:
            print(f"  {label}: {len(sub)} kart — BASELINE {np.mean(db_sub):.1f}px, "
                 f"FORCED {np.mean(df_sub):.1f}px (n={len(df_sub)})")
        else:
            print(f"  {label}: {len(sub)} kart, brak ważnej metryki")

    print("""
=== Interpretacja ===
- dist_F wyraźnie NIŻSZY niż dist_B -> wymuszenie top-K per promień pomaga (więcej
  materiału do głosowania, arc-aware scoring i tak odsiewa szum).
  dist_F wyraźnie WYŻSZY -> wymuszanie pików na "cichych" promieniach wstrzykuje szum,
  który psuje wynik (ryzyko, o którym mówiliśmy).
  Podobne -> zmiana neutralna na tej próbie (jak E9 w=0.1) — nie warto komplikować pipeline'u.
- frac zmienionych kart blisko 0% -> zmiana w praktyce nic nie zmienia (nawet jeśli
  metryka się rusza, dzieje się to na niewielu kartach).
- n_peaks_forced powinno być ~n_dirs*wiek (mniej, jeśli inner/edge_margin coś odetnie) —
  dużo więcej niż n_peaks_base -> potwierdza, że rzeczywiście "wstrzykujemy" dodatkowe
  kandydatury, których dziś nie ma.
""")


if __name__ == "__main__":
    main()
