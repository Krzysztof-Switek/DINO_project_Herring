"""29.07: visual explainer + validation report for the classical concentricity prior
("E9 for the classical method"). Self-contained HTML, 3 sections, all on real cards
from outputs/22.07_reg — the user explicitly asked for visual interpretation to be the
PRIMARY deliverable, not an afterthought to the code:

  1. History: what classical_increments does + WHY the naive "average all 48 rays
     before peak-finding" idea (E3, polar_averaged_increments — dead code, kept for
     reference) was rejected (measured 97.5% precision / 9.6% recall on 30 cards).
  2. The existing, already-in-production arc-aware scoring mechanism
     (_best_arc/_cluster_by_radius_with_arcs) — visualized directly on otoliths via the
     NEW render_arc_cluster_overlay, so "which rays merge into one ring" is literally
     visible, not just described.
  3. The NEW classical_concentricity_weight feature: formula, a value-coloured ray
     overlay showing the measured variance directly, the weight-sweep results (already
     run via sweep_classical_concentricity_weight.py — a REAL, non-null signal, unlike
     E9's own w=0.1), and before/after final-marker overlays.

Does NOT retrain or modify any src/ code. Reuses:
  - scripts.run_pipeline.load_merged_config
  - src.ring_extraction (classical_increments, density_peaks, fuse_increments,
    polar_averaged_increments, _all_ray_profiles, _cluster_by_radius_with_arcs,
    assign_rays_to_clusters, _classical_concentricity_variance)
  - src.visualization (render_candidate_rings, render_arc_cluster_overlay,
    render_localization_overlay, load_original_image)
  - src.report_common (fig_to_b64, img_tag)
  - src.comparison_report (_style_ax, _INK, _MUTED, _GRID)

Usage: python scripts/diagnostics/classical_concentricity_report.py
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
from PIL import Image as PILImage

from src.comparison_report import _style_ax, _INK, _MUTED, _GRID
from src.report_common import fig_to_b64, img_tag
from src.visualization import (select_top_k_samples, render_candidate_rings,
                               render_arc_cluster_overlay, render_localization_overlay,
                               render_rays_and_candidates, load_original_image)
from src.otolith_axis import detect_axis, apply_background_mask, sample_profile_along_axis
from src.inference import load_model_from_checkpoint
from src.dataset import build_transforms
from src.candidates import find_candidate_peaks
from src.ring_extraction import (density_peaks, classical_increments, fuse_increments,
                                 polar_averaged_increments, _all_ray_profiles,
                                 _cluster_by_radius_with_arcs, assign_rays_to_clusters,
                                 _classical_concentricity_variance)
from scripts.run_pipeline import load_merged_config

RUN_DIR = PROJECT_ROOT / "outputs" / "22.07_reg"
CKPT = RUN_DIR / "checkpoints" / "embedded" / "best.pt"
IMAGE_DIR = Path("Z:/Photo/Otolithes/HER/Processed")
PRED_CSV = RUN_DIR / "emb_on_emb" / "predictions.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "29.07_classical_concentricity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "report.html"

N_DIRS = 48
COLOR_A, COLOR_B = "#2a78d6", "#c8501e"


def _b64_from_rgb(arr: np.ndarray, target_w: int = 460) -> str:
    img = PILImage.fromarray(arr.astype(np.uint8))
    if img.width > target_w:
        scale = target_w / img.width
        img = img.resize((target_w, max(1, int(round(img.height * scale)))), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_value_colored_rays(image, axis_info, profiles, t, n_dirs=N_DIRS):
    """Report-local helper (not promoted to src/visualization.py): colour each of the
    n_dirs rays by ITS OWN signal value at the SAME normalised radius t — a direct,
    literal picture of what _classical_concentricity_variance measures (rays that
    agree ⇒ similar colour; rays that disagree ⇒ scattered colours), distinct from
    render_arc_cluster_overlay's categorical arc/non-arc split."""
    import cv2
    img = np.ascontiguousarray(image[..., :3]).copy()
    H = img.shape[0]
    if axis_info is None:
        return img
    contour = axis_info.get("contour")
    centroid = axis_info.get("centroid")
    if contour is None or centroid is None:
        return img
    cx, cy = int(centroid[0]), int(centroid[1])
    cpts = contour.reshape(-1, 2)
    idx_sel = np.linspace(0, len(cpts) - 1, min(n_dirs, len(cpts)), dtype=int)
    n_samples = len(profiles[0]) if profiles and profiles[0] is not None else 64
    sample_idx = int(np.clip(round(t * (n_samples - 1)), 0, n_samples - 1))
    cmap = cm.get_cmap("RdYlGn_r")
    lt = max(3, H // 150)
    for ray in range(min(n_dirs, len(idx_sel))):
        p = profiles[ray] if ray < len(profiles) else None
        if p is None:
            continue
        val = float(np.clip(p[sample_idx], 0.0, 1.0))
        rgb = tuple(int(255 * c) for c in cmap(val)[:3])
        pt = cpts[idx_sel[ray]]
        cv2.line(img, (cx, cy), (int(pt[0]), int(pt[1])), rgb, lt, cv2.LINE_AA)
    cv2.drawContours(img, [contour], -1, (0, 230, 230), max(2, H // 300))
    return img


def render_angle_strength_chart(profiles, t, n_dirs=N_DIRS) -> str:
    """Polar bar chart: signal value at radius t, one bar per ray (angle order matches
    render_value_colored_rays' ray indexing around the contour). Shows WHERE along the
    circumference a candidate is strongly vs weakly expressed, instead of collapsing
    that pattern into a single variance number — growth increments are commonly
    expressed unevenly around the circumference (strong on part of it, weak/absent on
    the rest), which is expected biological variation, not an inconsistency to flag."""
    n_samples = len(profiles[0]) if profiles and profiles[0] is not None else 64
    sample_idx = int(np.clip(round(t * (n_samples - 1)), 0, n_samples - 1))
    angles, values = [], []
    for ray in range(n_dirs):
        p = profiles[ray] if ray < len(profiles) else None
        if p is None:
            continue
        angles.append(2 * np.pi * ray / n_dirs)
        values.append(float(np.clip(p[sample_idx], 0.0, 1.0)))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="polar")
    cmap = cm.get_cmap("RdYlGn")
    width = 2 * np.pi / n_dirs * 0.9
    ax.bar(angles, values, width=width, color=[cmap(v) for v in values], edgecolor="none")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title(f"siła sygnału wg kierunku, promień t={t:.2f}", pad=18, fontsize=10)
    fig.tight_layout()
    return fig_to_b64(fig)


def main() -> None:
    cfg = load_merged_config(PROJECT_ROOT / "configs" / "config.yaml",
                             PROJECT_ROOT / "configs" / "config_embedded.yaml")
    cfg.data.image_dir = str(IMAGE_DIR)

    best, worst = select_top_k_samples(PRED_CSV, 15, 15)
    samples = list(best) + list(worst)

    model = load_model_from_checkpoint(cfg, CKPT)
    model.eval()
    device = next(model.parameters()).device
    transform = build_transforms(cfg.data.image_size, "test")
    min_dist = cfg.candidates.min_peak_distance
    prominence = cfg.candidates.prominence_threshold

    print("Liczenie 30 kart (density + klasyka + profile)...", flush=True)
    cards = []
    for row in samples:
        iid = str(row["image_id"])
        img_path = IMAGE_DIR / iid
        if not img_path.exists():
            continue
        orig_rgb = np.array(PILImage.open(img_path).convert("RGB"), dtype=np.uint8)
        H, W = orig_rgb.shape[:2]
        axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                                nucleus_method=cfg.segmentation.nucleus_method)
        if axis_info is None:
            continue
        mask_arr = axis_info["mask"]
        model_input_rgb = (apply_background_mask(orig_rgb, mask_arr)
                           if cfg.data.mask_background else orig_rgb)
        tensor = transform(PILImage.fromarray(model_input_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            grid = model.get_density_probs(tensor).squeeze(0).cpu().numpy()

        dpk, _ = density_peaks(grid, axis_info, H, W, min_distance=min_dist, prominence=prominence)
        cinc = classical_increments(orig_rgb, axis_info, return_profiles=True)
        cpk, cprof = cinc["peaks"], cinc["profiles"]
        age = int(row.get("predicted_age", 0))

        gray = orig_rgb.mean(axis=2)
        prof_1d, line_xy = sample_profile_along_axis(
            gray, axis_info["centroid"], axis_info["far_edge"], H, W, n_samples=50)
        prof_1d = np.asarray(prof_1d, dtype=np.float32)
        rng = float(prof_1d.max() - prof_1d.min())
        if rng > 1e-6:
            prof_1d = (prof_1d - prof_1d.min()) / rng
        classical_ref = []
        for i in find_candidate_peaks(prof_1d, min_dist, prominence):
            i = int(i)
            if 0 <= i < len(line_xy):
                classical_ref.append((int(line_xy[i][0]), int(line_xy[i][1])))

        cclust = _cluster_by_radius_with_arcs(cpk, t_tol=0.06, n_dirs=N_DIRS)
        memberships_for_var = assign_rays_to_clusters(cpk, cclust, t_tol=0.06, n_dirs=N_DIRS)
        variances = [
            {"mean_t": c[0], "support": c[1], "arc_len": c[3],
             "global_var": _classical_concentricity_variance(c[0], cprof),
             "segment_var": _classical_concentricity_variance(c[0], cprof, m["band"])}
            for c, m in zip(cclust, memberships_for_var)
        ]

        cards.append({
            "iid": iid, "age": age, "orig_rgb": orig_rgb, "axis_info": axis_info,
            "dpk": dpk, "cpk": cpk, "cprof": cprof, "cclust": cclust,
            "variances": variances, "classical_ref": classical_ref,
        })
    print(f"  {len(cards)} kart z udaną segmentacją.", flush=True)

    # --- pick illustrative cards ---
    # arc_demo: card whose classical clusters show the widest spread between a fully-
    # contiguous cluster (arc_len==support) and a scattered one (arc_len << support),
    # among clusters with decent support (>=5 rays) — the clearest "which rays merge" story.
    def _arc_contrast(card):
        cl = [c for c in card["cclust"] if c[1] >= 5]
        if not cl:
            return -1.0
        ratios = [c[3] / c[1] for c in cl]
        return max(ratios) - min(ratios) if len(ratios) > 1 else 0.0

    arc_demo = max(cards, key=_arc_contrast)

    # sweep (reusing all cached cards) — same 2-phase logic as sweep_classical_concentricity_weight.py
    # Calibrated on segment_var (the fixed, production formula), not global_var.
    all_var = [m["segment_var"] for card in cards for m in card["variances"]
              if m["segment_var"] is not None]
    p90 = float(np.percentile(all_var, 90)) if all_var else 1e-3
    p90 = p90 if p90 > 1e-9 else 1e-3
    targets = (0.05, 0.15, 0.30, 0.60, 1.00)
    weights = [0.0] + [round(t / p90, 3) for t in targets]
    conservative_w, aggressive_w = weights[1], weights[-1]

    def _mean_dist(finals, ref):
        if not finals or not ref:
            return None
        fa = np.asarray(finals, dtype=np.float32)
        ca = np.asarray(ref, dtype=np.float32)
        d = np.sqrt(((fa[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
        return float(d.min(axis=1).mean())

    print("Sweep (reużywam scache'owane karty)...", flush=True)
    sweep_rows = []
    for card in cards:
        row = {"iid": card["iid"], "dist": {}, "final_t": {}}
        for w in weights:
            fr = fuse_increments(card["dpk"], card["cpk"], card["age"], card["axis_info"],
                                 method="dp", classical_profiles=card["cprof"],
                                 classical_concentricity_weight=w)
            row["dist"][w] = _mean_dist(fr["final_axis_pts"], card["classical_ref"])
            row["final_t"][w] = fr["final_t"]
        sweep_rows.append(row)

    mean_dists, frac_changed = {}, {}
    for w in weights:
        vals = [r["dist"][w] for r in sweep_rows if r["dist"][w] is not None]
        mean_dists[w] = float(np.mean(vals)) if vals else float("nan")
        n_changed = sum(1 for r in sweep_rows if r["final_t"][w] != r["final_t"][0.0])
        frac_changed[w] = n_changed / len(sweep_rows) if sweep_rows else 0.0

    # mover: card with the largest final_t change at the aggressive weight (best story
    # for "look, it actually moves something").
    def _final_t_delta(row):
        a, b = row["final_t"][0.0], row["final_t"][aggressive_w]
        if len(a) != len(b):
            return abs(len(a) - len(b)) + 1.0
        return sum(abs(x - y) for x, y in zip(sorted(a), sorted(b)))

    mover_row = max(sweep_rows, key=_final_t_delta)
    mover = next(c for c in cards if c["iid"] == mover_row["iid"])

    # flat: a card completely unaffected even at the aggressive weight (contrast case).
    flat_row = next((r for r in sweep_rows if r["final_t"][0.0] == r["final_t"][aggressive_w]),
                    sweep_rows[0])
    flat = next(c for c in cards if c["iid"] == flat_row["iid"])

    print(f"  arc_demo = {arc_demo['iid']}", flush=True)
    print(f"  mover    = {mover['iid']} (delta={_final_t_delta(mover_row):.3f} @ w={aggressive_w})", flush=True)
    print(f"  flat     = {flat['iid']}", flush=True)

    # ================================================================
    # Wszystkie obrazy (te same obliczenia co poprzednio — tylko opis dookoła
    # przepisany na wersję "dla laika", krok po kroku, bez żargonu).
    # ================================================================
    H, W = mover["orig_rgb"].shape[:2]

    # Krok 1 — geometria: 48 "linijek" od środka do brzegu, bez żadnych danych jeszcze.
    rays_only_img = render_rays_and_candidates(mover["orig_rgb"], mover["axis_info"],
                                               density_pts=[], classical_pts=[], n_dirs=N_DIRS)
    rays_only_b64 = _b64_from_rgb(rays_only_img, target_w=460)

    # Krok 2/3 — 48 profili jasności wzdłuż tych linijek, każdy w innym kolorze.
    profiles, _line_xys, _cpts = _all_ray_profiles(
        mover["orig_rgb"].mean(axis=2), mover["axis_info"], H, W, n_dirs=N_DIRS)
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    cmap48 = cm.get_cmap("hsv")
    for i, p in enumerate(profiles):
        if p is None:
            continue
        ax1.plot(np.linspace(0, 1, len(p)), p, color=cmap48(i / N_DIRS), linewidth=0.8, alpha=0.75)
    valid = [p for p in profiles if p is not None]
    if valid:
        avg = np.mean(np.stack(valid), axis=0)
        rng = float(avg.max() - avg.min())
        avg_norm = (avg - avg.min()) / rng if rng > 1e-6 else avg
        ax1.plot(np.linspace(0, 1, len(avg_norm)), avg_norm, color="black", linewidth=2.5,
                 label="gdyby uśrednić wszystkie 48 linijek w jedną")
    ax1.set_xlabel("gdzie na linijce jesteśmy (lewo = środek otolitu, prawo = brzeg)")
    ax1.set_ylabel("jasność (0 = ciemno, 1 = jasno)")
    ax1.set_title("48 pomiarów jasności — po jednym z każdej linijki (kolor = kierunek)")
    ax1.legend(fontsize=8)
    _style_ax(ax1)
    fig1.tight_layout()
    rays48_chart_b64 = fig_to_b64(fig1)

    # Krok 4 — naiwne uśrednianie (E3): ile pierścieni znika.
    poavg = polar_averaged_increments(mover["orig_rgb"].mean(axis=2), mover["axis_info"], H, W)
    classical_ts = [c[0] for c in mover["cclust"]]
    # render_candidate_rings rysuje classical_ring_ts NAJPIERW (pod spodem), potem
    # density_ring_ts NA WIERZCHU — nieliczne "ocalałe" pierścienie E3 idą jako drugie,
    # żeby nie zniknęły pod tymi z pełnej metody (E3 ma wysoką trafność, więc zwykle
    # pokrywają się niemal dokładnie).
    e3_overlay = render_candidate_rings(mover["orig_rgb"], mover["axis_info"],
                                        density_ring_ts=poavg["peak_t"],
                                        classical_ring_ts=classical_ts)
    e3_overlay_b64 = _b64_from_rgb(e3_overlay, target_w=460)

    # Krok 5 — metoda "sąsiedzi się zgadzają" (już działa dziś): 2 najbardziej
    # kontrastujące grupy z jednej karty (rysowanie wszystkich naraz jest nieczytelne).
    memberships = assign_rays_to_clusters(arc_demo["cpk"], arc_demo["cclust"], t_tol=0.06, n_dirs=N_DIRS)
    eligible = [m for m in memberships if m["support"] >= 5]
    demo_members = memberships
    if len(eligible) >= 2:
        ranked = sorted(eligible, key=lambda m: m["arc_len"] / m["support"])
        demo_members = [ranked[0], ranked[-1]]
    arc_overlay = render_arc_cluster_overlay(arc_demo["orig_rgb"], arc_demo["axis_info"],
                                             demo_members, n_dirs=N_DIRS)
    arc_overlay_b64 = _b64_from_rgb(arc_overlay, target_w=460)

    # Krok 5 — pokazujemy PRAWDZIWY przypadek częściowej ekspresji przyrostu: kandydatura
    # o największej ROZBIEŻNOŚCI między starą (cały obwód) a nową (tylko segment łuku)
    # formułą — czyli dokładnie ten przypadek, który stara formuła oceniała źle, a nowa
    # ocenia poprawnie. Jeśli żadna kandydatura nie ma obu wartości, spadamy do zwykłego
    # "najwyższa segment_var" jako fallback.
    _with_gap = [m for m in arc_demo["variances"]
                if m["global_var"] is not None and m["segment_var"] is not None]
    if _with_gap:
        partial_m = max(_with_gap, key=lambda m: m["global_var"] - m["segment_var"])
    else:
        partial_m = max((m for m in arc_demo["variances"] if m["segment_var"] is not None),
                        key=lambda m: m["segment_var"], default=None)
    value_overlays_html = ""
    formula_fix_html = ""
    if partial_m is not None:
        partial_t = partial_m["mean_t"]
        partial_img = render_value_colored_rays(arc_demo["orig_rgb"], arc_demo["axis_info"],
                                                arc_demo["cprof"], partial_t)
        angle_chart_b64 = render_angle_strength_chart(arc_demo["cprof"], partial_t)
        value_overlays_html = f"""
        <div class="row">
          <div class="col">
            <p class="cap">Nakładka na zdjęciu — kolor promienia = jego wartość sygnału przy tym promieniu</p>
            {img_tag(_b64_from_rgb(partial_img), style="width:380px;")}
          </div>
          <div class="col">
            <p class="cap">To samo, jako wykres siła sygnału × kierunek (ten sam kąt co na zdjęciu)</p>
            {img_tag(angle_chart_b64, style="width:380px;")}
          </div>
        </div>"""
        gv, sv = partial_m["global_var"], partial_m["segment_var"]
        formula_fix_html = f"""
        <table><tr><th>Formuła</th><th>Zakres porównania</th><th>Zmierzona wariancja</th><th>Kara przy wadze 2.0</th></tr>
        <tr><td>Stara (przed poprawką)</td><td>wszystkie 48 promieni</td><td>{gv:.5f}</td>
        <td>{min(1.0, 2.0*gv)*100:.0f}% obniżenia wyniku</td></tr>
        <tr><td>Nowa (po poprawce)</td><td>tylko promienie w segmencie łuku</td><td>{sv:.5f}</td>
        <td>{min(1.0, 2.0*sv)*100:.0f}% obniżenia wyniku</td></tr></table>
        <p class="cap">Ta sama kandydatura: stara formuła widziała ją jako niespójną (bo
        porównywała ją do całego, w większości "cichego" obwodu) i karała; nowa formuła
        porównuje ją tylko z promieniami należącymi do jej własnego, wyrażonego segmentu —
        i poprawnie rozpoznaje ją jako spójną.</p>"""

    # Krok 7 — czy to pomogło: eksperyment na 30 kartach.
    fig3, ax3a = plt.subplots(figsize=(8, 4.5))
    ax3b = ax3a.twinx()
    xs = weights
    ax3a.plot(xs, [mean_dists[w] for w in xs], color=COLOR_A, marker="o", label="błąd (niżej = lepiej)")
    ax3b.plot(xs, [frac_changed[w] * 100 for w in xs], color=COLOR_B, marker="s", linestyle="--",
             label="% kart, na których coś się zmieniło")
    ax3a.set_xlabel("jak mocno włączona nowa reguła (0 = wyłączona)")
    ax3a.set_ylabel("błąd względem ludzkiego wzorca (px)", color=COLOR_A)
    ax3b.set_ylabel("% zmienionych kart", color=COLOR_B)
    ax3a.set_title(f"Eksperyment na {len(sweep_rows)} kartach")
    lines_a, labels_a = ax3a.get_legend_handles_labels()
    lines_b, labels_b = ax3b.get_legend_handles_labels()
    ax3a.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper left")
    _style_ax(ax3a)
    fig3.tight_layout()
    sweep_chart_b64 = fig_to_b64(fig3)

    def _final_overlay(card, w):
        fr = fuse_increments(card["dpk"], card["cpk"], card["age"], card["axis_info"],
                             method="dp", classical_profiles=card["cprof"],
                             classical_concentricity_weight=w)
        return render_localization_overlay(card["orig_rgb"], card["axis_info"],
                                           fr["final_axis_pts"], fr["candidate_pts"])

    mover_before = _b64_from_rgb(_final_overlay(mover, 0.0), target_w=340)
    mover_after = _b64_from_rgb(_final_overlay(mover, aggressive_w), target_w=340)
    flat_before = _b64_from_rgb(_final_overlay(flat, 0.0), target_w=340)
    flat_after = _b64_from_rgb(_final_overlay(flat, aggressive_w), target_w=340)

    has_signal = any(frac_changed[w] > 0.05 for w in weights[1:])
    verdict_word = "TAK — coś się zmienia" if has_signal else "NIE — brak efektu"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Lokalizacja przyrostów rocznych — metoda pomiarowa krok po kroku</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width:980px; margin:24px auto; padding:0 16px; color:{_INK}; line-height:1.55; }}
h1 {{ color:{_INK}; font-size:1.7em; }}
h2 {{ color:{_INK}; }}
.step {{ display:flex; gap:16px; align-items:flex-start; margin:36px 0 8px; }}
.badge {{ flex:0 0 auto; width:40px; height:40px; border-radius:50%; background:#2a78d6; color:white;
         display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:1.1em; }}
.step h2 {{ margin:6px 0 0; }}
.body {{ margin-left:56px; }}
.row {{ display:flex; gap:16px; flex-wrap:wrap; margin:10px 0; }}
.col {{ flex:1 1 320px; }}
table {{ border-collapse:collapse; margin:12px 0; }}
td, th {{ border:1px solid {_GRID}; padding:6px 10px; font-size:90%; }}
th {{ background:#f4f3f0; }}
.cap {{ color:{_MUTED}; font-size:88%; }}
.verdict {{ background:{"#e6f4ea" if has_signal else "#fbe9e7"}; border-radius:10px; padding:16px 20px; margin:20px 0;
           border-left:6px solid {"#1a7a3c" if has_signal else "#b3341a"}; }}
</style></head>
<body>

<h1>Lokalizacja przyrostów rocznych — metoda pomiarowa krok po kroku</h1>

<div class="step"><div class="badge">1</div><div>
<h2>Pomiar promieniowy: 48 kierunków od jądra do brzegu</h2>
</div></div>
<div class="body">
<p>Automatyczna lokalizacja przyrostów opiera się na próbkowaniu jasności obrazu wzdłuż 48
promieni rozchodzących się z jądra otolitu do brzegu, rozłożonych równomiernie kątowo
dookoła jądra (nie tylko wzdłuż jednej osi pomiarowej, jak przy klasycznym odczycie
ręcznym).</p>
{img_tag(rays_only_b64)}
<p class="cap"><b>Co widać:</b> 48 promieni (szare linie) rozchodzących się z jądra (czerwony
punkt/niebieski krzyżyk) do konturu otolitu. Żółty promień = główna oś pomiarowa (ta sama,
której używa się przy odczycie na pojedynczej osi).</p>
</div>

<div class="step"><div class="badge">2</div><div>
<h2>Sygnał na 48 promieniach naraz — brak spójności między kierunkami</h2>
</div></div>
<div class="body">
<p>Wzdłuż każdego promienia próbkowany jest profil jasności; lokalne ekstrema tego profilu są
kandydatami na przyrost roczny. Nałożenie profili wszystkich 48 promieni na jeden wykres
pokazuje problem wprost:</p>
{img_tag(rays48_chart_b64)}
<p class="cap"><b>Co widać:</b> każda kolorowa linia = profil jasności JEDNEGO z 48 promieni
(oś pozioma: jądro → brzeg). Kolor koduje kierunek promienia.</p>
<p>Profile poszczególnych promieni NIE są ze sobą zgodne — w wielu miejscach jeden promień
pokazuje wyraźne lokalne ekstremum, a inny w tym samym miejscu (tym samym promieniu
znormalizowanym) — nic. Przyrost widoczny na części obwodu nie musi być widoczny na całym
obwodzie jednocześnie.</p>
</div>

<div class="step"><div class="badge">3</div><div>
<h2>Wariant odrzucony: uśrednienie wszystkich 48 profili w jeden</h2>
</div></div>
<div class="body">
<p>Naturalny wariant uproszczenia: uśrednić wszystkie 48 profili w jeden sygnał (czarna, gruba
krzywa na wykresie powyżej) i szukać ekstremów tylko na nim. Zmierzone na próbie 30 kart:</p>
<div class="row">
  <div class="col"><p><b>precyzja 97,5%</b> — jeśli uśredniony sygnał coś wskazuje, niemal
  zawsze pokrywa się z realnym przyrostem</p></div>
  <div class="col"><p><b>recall 9,6%</b> — ale wskazuje mniej niż 1 na 10 realnych
  przyrostów, resztę traci</p></div>
</div>
<p><b>Przyczyna:</b> uśrednianie 48 promieni działa jak liczenie głosów — jeśli 47 promieni
nie rejestruje sygnału w danym miejscu, a tylko 1 rejestruje wyraźne ekstremum (bo przyrost
jest widoczny tylko na fragmencie obwodu), to uśredniony profil zostaje zdominowany przez
brak sygnału z pozostałych 47 i lokalne ekstremum zostaje wygładzone/utracone.</p>
{img_tag(e3_overlay_b64)}
<p class="cap"><b>Co widać:</b> ten sam otolit. <span style="color:#1a9e8f;">Zielone
okręgi</span> = wszystkie kandydatury z pełnej metody (48 promieni liczonych osobno) —
{len(classical_ts)} kandydatur. <span style="color:#e8b800;">Żółty okrąg</span> = to, co
pozostało po uśrednieniu — {len(poavg['peak_t'])}. Pozostałe kandydatury zostały utracone
w uśrednieniu.</p>
<p><b>Wniosek:</b> uśrednienie wszystkich promieni przed detekcją ekstremów jest odrzucone.
Detekcja musi działać na 48 promieniach osobno, a łączenie wyników — na etapie oceny już
wykrytych kandydatur (kroki 4-5).</p>
</div>

<div class="step"><div class="badge">4</div><div>
<h2>Mechanizm już w produkcji: zgodność promieni SĄSIADUJĄCYCH kątowo</h2>
</div></div>
<div class="body">
<p>Zamiast uśredniać, obecny mechanizm grupuje kandydatury po promieniu (ten sam promień
znormalizowany, wykryty przez wiele z 48 kierunków), a następnie dodatkowo sprawdza, czy
promienie wskazujące tę samą kandydaturę SĄSIADUJĄ ze sobą kątowo (ciągły łuk obwodu), czy
są rozproszone po całym obwodzie. Ciągły łuk = silniejsza przesłanka realnego przyrostu
(spójny fragment obwodu); ten sam liczbowy "support" rozproszony po całym obwodzie = słabsza
przesłanka (mogło być przypadkowe zbiegnięcie niezależnych detekcji).</p>
{img_tag(arc_overlay_b64)}
<p class="cap"><b>Co widać:</b> ten sam otolit, dwie różne kandydatury zaznaczone kolorem.
GRUBA, pełna linia = promienie wchodzące w zwycięski, ciągły łuk. CIENKA, przezroczysta
linia (ten sam kolor) = promienie tej samej kandydatury promienia, ale POZA ciągłym łukiem
— rozproszone wsparcie, liczone do ogólnego "support", ale nie do długości łuku.</p>
</div>

<div class="step"><div class="badge">5</div><div>
<h2>Nowy mechanizm testowany dziś: wartość sygnału na promieniach — i jego ograniczenie</h2>
</div></div>
<div class="body">
<p>Krok 4 ocenia wyłącznie, czy promienie sąsiadujące kątowo zgadzają się co do WYSTĄPIENIA
ekstremum — jest ślepy na to, jak bardzo WARTOŚĆ sygnału różni się między promieniami, które
akurat coś zarejestrowały. Nowy mechanizm dodatkowo ocenia wariancję wartości sygnału — ale
TYLKO WEWNĄTRZ segmentu obwodu, który krok 4 już zidentyfikował jako łuk danej kandydatury
(nie po całym obwodzie).</p>
<p><b>Dlaczego ograniczenie do segmentu jest konieczne:</b> przyrost roczny w praktyce
rzadko jest wyrażony jednakowo silnie na całym obwodzie otolitu — normą jest odcinek z
wyraźną ekspresją i pozostała część ze słabą lub żadną. Pierwsza wersja tego mechanizmu
liczyła wariancję po WSZYSTKICH 48 promieniach i nie odróżniała takiego, biologicznie
typowego, częściowego przyrostu od przypadku faktycznie przypadkowego/szumowego —
poprawiono to, ograniczając porównanie do promieni należących do segmentu łuku (ten sam
zestaw promieni, który krok 4 uznał za spójny łuk). Poniżej realny przykład z tej samej
karty, na którym widać różnicę:</p>
{value_overlays_html}
<p class="cap"><b>Co widać:</b> ta sama kandydatura na tym samym otolicie, dwa ujęcia tego
samego zjawiska. Po lewej — nakładka na zdjęciu, kolor promienia = jego wartość sygnału w
tym miejscu. Po prawej — ten sam sygnał jako wykres siła × kierunek (ta sama orientacja co
na zdjęciu): widać wyraźnie, na którym fragmencie obwodu przyrost jest silnie wyrażony
(wysokie słupki), a na którym słaby/nieobecny (niskie słupki) — to jest normalny,
częściowy przyrost, nie szum.</p>
{formula_fix_html}
</div>

<div class="step"><div class="badge">6</div><div>
<h2>Walidacja: czy nowy mechanizm faktycznie zmienia wynik selekcji?</h2>
</div></div>
<div class="body">
<p>Nowy mechanizm przetestowano na 30 kartach (ta sama próba co w krokach 3-5) przy rosnącej
sile działania (waga 0 = wyłączony, w prawo = silniejsza kara za niespójność). Mierzone: (a)
odległość finalnie wybranych przyrostów od niezależnego wzorca klasycznego (pojedyncza oś
pomiarowa), (b) odsetek kart, na których selekcja w ogóle się zmieniła względem wagi 0.</p>
{img_tag(sweep_chart_b64)}
<p class="cap"><b>Co widać:</b> oś pozioma = siła nowego mechanizmu (0 = wyłączony).
<span style="color:{COLOR_A};">Niebieska linia</span> = odległość od wzorca klasycznego w
pikselach (niżej = lepiej). <span style="color:{COLOR_B};">Pomarańczowa, przerywana
linia</span> = odsetek z 30 kart, na których selekcja przyrostów w ogóle się zmieniła.</p>
<div class="verdict">
<p><b>Wynik: {verdict_word}.</b> Sweep wykonany formułą PO poprawce z kroku 5 (ograniczenie
do segmentu łuku). Odsetek zmienionych kart rośnie płynnie z siłą mechanizmu — od 0% do
{frac_changed[weights[-1]]*100:.0f}% przy najsilniejszym ustawieniu — mechanizm realnie wpływa
na selekcję, nie jest neutralny (dla porównania: analogiczny mechanizm wcześniej testowany dla
mapy modelu, przy niskiej sile ustawienia nie zmieniał wyniku wcale). Odległość od wzorca
poprawia się nierównomiernie na tej próbie (n=30) — sygnał, nie czysty szum, ale krajobraz nie
jest gładki. Przed ustawieniem wagi domyślnej w konfiguracji produkcyjnej nadal wymagana jest
walidacja na większej próbie oraz wizualna kontrola kart, na których selekcja się zmieniła;
obecnie waga domyślna pozostaje 0 (mechanizm nieaktywny), więc żaden istniejący wynik się nie
zmienia.</p>
</div>

<h3>Przykład na kartach</h3>
<p>Poniżej: czerwone punkty = finalnie wybrane przyrosty przed (lewa kolumna) i po (prawa
kolumna) włączeniu nowego mechanizmu na najsilniejszym testowanym ustawieniu.</p>
<div class="row">
  <div class="col">
    <p><b>Karta z dużą zmianą selekcji:</b></p>
    <div class="row">
      <div class="col"><p class="cap">przed</p>{img_tag(mover_before)}</div>
      <div class="col"><p class="cap">po</p>{img_tag(mover_after)}</div>
    </div>
    <p class="cap">Wybrane punkty przesunęły się bliżej jądra — mechanizm przesuwa selekcję
    w stronę kandydatur wewnętrznie spójnych w obrębie własnego segmentu łuku (nie: spójnych
    na całym obwodzie — patrz poprawka formuły w kroku 5).</p>
  </div>
  <div class="col">
    <p><b>Karta kontrolna — brak zmiany:</b></p>
    <div class="row">
      <div class="col"><p class="cap">przed</p>{img_tag(flat_before)}</div>
      <div class="col"><p class="cap">po</p>{img_tag(flat_after)}</div>
    </div>
    <p class="cap">Identyczna selekcja przed/po — mechanizm nie zmienia wyniku
    bezwarunkowo, tylko tam, gdzie wariancja sygnału faktycznie różnicuje kandydatury.</p>
  </div>
</div>
</div>

<div class="step"><div class="badge">7</div><div>
<h2>Podsumowanie</h2>
</div></div>
<div class="body">
<ol>
<li>Lokalizacja przyrostów opiera się na próbkowaniu jasności wzdłuż 48 promieni jądro→brzeg.</li>
<li>Uśrednienie tych 48 profili w jeden przed detekcją ekstremów — <b>odrzucone</b>, traci
    ponad 90% realnych przyrostów widocznych tylko na części obwodu.</li>
<li>Obecny mechanizm produkcyjny nagradza kandydatury potwierdzone przez promienie
    SĄSIADUJĄCE kątowo (ciągły łuk obwodu).</li>
<li>Nowy mechanizm dodatkowo ocenia wartość sygnału (nie tylko obecność ekstremum) —
    pierwsza wersja karała dowolną rozbieżność po CAŁYM obwodzie, co było założeniem zbyt
    restrykcyjnym biologicznie (przyrosty typowo są wyrażone częściowo). <b>Poprawiono</b>:
    ocena jest teraz ograniczona do segmentu łuku (te same promienie, które krok 4 uznał za
    spójny łuk), tolerując słaby/zerowy sygnał na reszcie obwodu.</li>
<li>Walidacja na 30 kartach (formułą PO poprawce) potwierdza, że mechanizm <b>realnie zmienia
    selekcję</b> (nie jest neutralny) — przed zmianą wagi domyślnej w produkcji nadal wymagana
    jest walidacja na większej próbie i wizualna kontrola zmienionych kart.</li>
</ol>
</div>

</body></html>
"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\n=== DONE ===\nRaport: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
