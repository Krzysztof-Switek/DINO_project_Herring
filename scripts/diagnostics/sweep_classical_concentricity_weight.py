"""29.07: two-phase sweep of classical_concentricity_weight (E9-for-classical) on the
SAME 30-card sample (15 best + 15 worst) used throughout this project's post-hoc
diagnostics. Modeled on scripts/diagnostics/sweep_spread_weight.py's structure — no
retraining, one forward pass per card, everything else pure Python/numpy.

Phase 1 (calibration): measure the REAL distribution of classical concentricity
variance on this dataset's classical clusters, so sweep weights are chosen from data
instead of guessed blind (unlike E9's own 0.1/1.0/10.0 bracket, which cost three
~13h server trainings before finding out w=0.1 had no effect at all).

Phase 2 (sweep): for each weight, measure mean_dist to the single-axis classical
reference (same metric _localization_quality uses) AND the fraction of cards whose
final_t changes AT ALL vs weight=0 — the hard, metric-independent signal that was
missing from the first pass at evaluating E9's own w=0.1 (a metric can look flat while
individual card selections are still moving, or vice versa). Also runs the w=0.0 pass
twice to confirm determinism — this script holds ONE fixed checkpoint constant (no
retraining, no RNG in the classical path), so any drift here would be a bug in the new
code, NOT the ~2-4% cross-CHECKPOINT noise floor documented in
plans and summaries/29.07_session_handoff.md (that noise floor is about different
TRAINED checkpoints producing different predicted_age; it does not apply when sweeping
one fixed checkpoint's already-computed peaks).

Usage: python scripts/diagnostics/sweep_classical_concentricity_weight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image as PILImage

from scripts.run_pipeline import load_merged_config
from src.visualization import select_top_k_samples
from src.candidates import find_candidate_peaks
from src.otolith_axis import (detect_axis, apply_background_mask,
                              sample_profile_along_axis)
from src.inference import load_model_from_checkpoint
from src.dataset import build_transforms
from src.ring_extraction import (density_peaks, classical_increments, fuse_increments,
                                 _cluster_by_radius_with_arcs,
                                 _classical_concentricity_variance)

RUN_DIR = PROJECT_ROOT / "outputs" / "22.07_reg"    # current stable baseline (not E9)
CKPT = RUN_DIR / "checkpoints" / "embedded" / "best.pt"
IMAGE_DIR = Path("Z:/Photo/Otolithes/HER/Processed")
PRED_CSV = RUN_DIR / "emb_on_emb" / "predictions.csv"

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

# ---------------------------------------------------------------------------
# Phase 1: load every card once, compute peaks/profiles/clusters, pool variances.
# ---------------------------------------------------------------------------
cards = []       # per-card cached data for phase 2
all_variances = []

for row in samples:
    iid = str(row["image_id"])
    img_path = IMAGE_DIR / iid
    if not img_path.exists():
        continue
    img_pil = PILImage.open(img_path).convert("RGB")
    orig_rgb = np.array(img_pil, dtype=np.uint8)
    H, W = orig_rgb.shape[:2]

    axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                            nucleus_method=cfg.segmentation.nucleus_method)
    if axis_info is None:
        continue
    mask_arr = axis_info["mask"]
    model_input_rgb = apply_background_mask(orig_rgb, mask_arr) if cfg.data.mask_background else orig_rgb

    tensor = transform(PILImage.fromarray(model_input_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        grid = model.get_density_probs(tensor).squeeze(0).cpu().numpy()

    dpk, _ = density_peaks(grid, axis_info, H, W, min_distance=min_dist, prominence=prominence)
    cinc = classical_increments(orig_rgb, axis_info, return_profiles=True)
    cpk, cprof = cinc["peaks"], cinc["profiles"]
    age = int(row.get("predicted_age", 0))

    # Single-axis classical reference (matches _localization_quality's classical_pts).
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

    cclust = _cluster_by_radius_with_arcs(cpk, t_tol=0.06, n_dirs=48)
    for c in cclust:
        v = _classical_concentricity_variance(c[0], cprof)
        if v is not None:
            all_variances.append(v)

    cards.append({"iid": iid, "age": age, "axis_info": axis_info, "dpk": dpk,
                  "cpk": cpk, "cprof": cprof, "classical_ref": classical_ref})

print(f"=== Phase 1: {len(cards)} cards, {len(all_variances)} classical clusters ===")
if all_variances:
    v = np.asarray(all_variances)
    percentiles = {p: float(np.percentile(v, p)) for p in (50, 75, 90, 95, 99)}
    print("Classical concentricity variance percentiles (bounded theoretical max 0.25):")
    for p, val in percentiles.items():
        print(f"  p{p}: {val:.5f}")
    p90 = percentiles[90] if percentiles[90] > 1e-9 else 1e-3
else:
    print("No classical clusters found at all — cannot calibrate weights from data.")
    p90 = 1e-3

WEIGHTS = [0.0] + [round(target / p90, 3) for target in (0.05, 0.15, 0.30, 0.60, 1.00)]
print(f"\nDerived sweep weights (targeting {0.05, 0.15, 0.30, 0.60, 1.00} score loss at p90 variance): {WEIGHTS}")


# ---------------------------------------------------------------------------
# Phase 2: sweep, using ONLY cached per-card data (no further model forward passes).
# ---------------------------------------------------------------------------
def mean_dist(finals, ref):
    if not finals or not ref:
        return None
    fa = np.asarray(finals, dtype=np.float32)
    ca = np.asarray(ref, dtype=np.float32)
    d = np.sqrt(((fa[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
    return float(d.min(axis=1).mean())


per_card_results = []   # list of dicts: iid -> {w: final_t}
for card in cards:
    row_result = {"iid": card["iid"], "age": card["age"], "dists": {}, "final_t": {}}
    for w in WEIGHTS:
        fr = fuse_increments(card["dpk"], card["cpk"], card["age"], card["axis_info"],
                             method="dp", classical_profiles=card["cprof"],
                             classical_concentricity_weight=w)
        row_result["dists"][w] = mean_dist(fr["final_axis_pts"], card["classical_ref"])
        row_result["final_t"][w] = fr["final_t"]
    per_card_results.append(row_result)
    print(f"{card['iid'][:45]:45s} age={card['age']:2d} " + " ".join(
        f"w={w}:{row_result['dists'][w]:.0f}" if row_result['dists'][w] is not None else f"w={w}:None"
        for w in WEIGHTS))

# Determinism check: rerun w=0.0 and compare bit-for-bit.
determinism_ok = True
for card in cards:
    fr_a = fuse_increments(card["dpk"], card["cpk"], card["age"], card["axis_info"],
                           method="dp", classical_profiles=card["cprof"],
                           classical_concentricity_weight=0.0)
    fr_b = fuse_increments(card["dpk"], card["cpk"], card["age"], card["axis_info"],
                           method="dp", classical_profiles=card["cprof"],
                           classical_concentricity_weight=0.0)
    if fr_a["final_t"] != fr_b["final_t"]:
        determinism_ok = False
        print(f"  [!] non-determinism at w=0.0 on {card['iid']}")

print(f"\n=== Determinism check (w=0.0, two passes, same fixed checkpoint): "
      f"{'OK — bit-identical' if determinism_ok else 'FAILED — see above'} ===")

print("\n=== Aggregate summary ===")
print(f"{'weight':>10s}  {'mean_dist':>10s}  {'n':>4s}  {'frac_changed_vs_w0':>18s}")
baseline_final_t = {r["iid"]: r["final_t"][0.0] for r in per_card_results}
for w in WEIGHTS:
    vals = [r["dists"][w] for r in per_card_results if r["dists"][w] is not None]
    n_changed = sum(
        1 for r in per_card_results if r["final_t"][w] != baseline_final_t[r["iid"]]
    )
    frac_changed = n_changed / len(per_card_results) if per_card_results else 0.0
    mean_val = np.mean(vals) if vals else float("nan")
    print(f"{w:>10.4f}  {mean_val:>10.2f}  {len(vals):>4d}  {frac_changed:>17.1%}")

print("""
=== Interpretation ===
- frac_changed_vs_w0 near 0 at every weight tested -> the term has no practical effect
  here (an E9-w0.1-style null result); may mean classical clusters on these cards are
  already highly concentric by construction (little variance to penalise) - itself an
  informative finding, not a bug.
- mean_dist improving then plateauing/worsening as weight grows -> pick the weight at
  the elbow; before promoting it to configs/config.yaml, visually spot-check a few
  cards where final_t changed (render_localization_overlay / render_arc_cluster_overlay,
  weight=0 vs the candidate weight) - see scripts/diagnostics/classical_concentricity_report.py.
""")
