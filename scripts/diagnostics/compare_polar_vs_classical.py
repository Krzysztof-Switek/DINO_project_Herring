"""E3 diagnostic (21.07): compare the new polar_averaged_increments (single angularly-
averaged profile) against the existing per-ray classical_increments (many independent
per-ray peaks, clustered by radius) on the same 15-best+15-worst card sample used
throughout this session's sweeps. No model/checkpoint needed -- both candidate sources
only need the raw image + axis_info, so this is much cheaper than the DP sweeps.

Metrics per card:
  n_classical_clusters      -- all radius-clusters from per-ray classical peaks (incl. weak/noisy)
  n_classical_well_supported-- clusters seen on >=15% of the 48 rays (proxy for "real, broad ring")
  n_polar_peaks             -- peaks in the single averaged profile
  precision_vs_well_sup     -- fraction of polar peaks within t_tol of a well-supported classical cluster
  recall_vs_well_sup        -- fraction of well-supported classical clusters matched by a polar peak
  precision_vs_all          -- fraction of polar peaks within t_tol of ANY classical cluster (incl. weak)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image as PILImage

from scripts.run_pipeline import load_merged_config
from src.visualization import select_top_k_samples
from src.otolith_axis import detect_axis
from src.ring_extraction import classical_increments, polar_averaged_increments, _cluster_by_radius

RUN_DIR = PROJECT_ROOT / "outputs" / "20.07_reg"
IMAGE_DIR = Path("Z:/Photo/Otolithes/HER/Processed")
PRED_CSV = RUN_DIR / "emb_on_emb" / "predictions.csv"

cfg = load_merged_config(PROJECT_ROOT / "configs" / "config.yaml",
                         PROJECT_ROOT / "configs" / "config_embedded.yaml")
cfg.data.image_dir = str(IMAGE_DIR)

best, worst = select_top_k_samples(PRED_CSV, 15, 15)
samples = list(best) + list(worst)

N_DIRS = 48
T_TOL = 0.06
SUPPORT_THRESH = max(1, int(round(0.15 * N_DIRS)))   # ~7 rays


def match_frac(query_ts, ref_ts, t_tol=T_TOL):
    """Fraction of query_ts that has some ref_t within t_tol."""
    if not query_ts:
        return None
    if not ref_ts:
        return 0.0
    ref = np.asarray(ref_ts, dtype=np.float64)
    hits = 0
    for t in query_ts:
        if np.min(np.abs(ref - t)) <= t_tol:
            hits += 1
    return hits / len(query_ts)


rows = []
n_skipped = 0

for row in samples:
    iid = str(row["image_id"])
    img_path = IMAGE_DIR / iid
    if not img_path.exists():
        n_skipped += 1
        continue
    img_pil = PILImage.open(img_path).convert("RGB")
    orig_rgb = np.array(img_pil, dtype=np.uint8)
    H, W = orig_rgb.shape[:2]

    axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                            nucleus_method=cfg.segmentation.nucleus_method)
    if axis_info is None:
        n_skipped += 1
        continue

    cres = classical_increments(orig_rgb, axis_info, n_dirs=N_DIRS, t_tol=T_TOL)
    cclusters = cres["clusters"]
    well_sup = [c for c in cclusters if c[1] >= SUPPORT_THRESH]

    pres = polar_averaged_increments(orig_rgb, axis_info, H, W, n_dirs=N_DIRS)
    ppeaks = pres["peak_t"]

    all_ts = [c[0] for c in cclusters]
    ws_ts = [c[0] for c in well_sup]

    rows.append({
        "iid": iid,
        "n_classical_clusters": len(cclusters),
        "n_well_supported": len(well_sup),
        "n_polar_peaks": len(ppeaks),
        "precision_vs_well_sup": match_frac(ppeaks, ws_ts),
        "recall_vs_well_sup": match_frac(ws_ts, ppeaks),
        "precision_vs_all": match_frac(ppeaks, all_ts),
    })

print(f"\nCards analysed: {len(rows)}  (skipped: {n_skipped})\n")
hdr = f"{'image_id':45s} {'n_clust':>7s} {'n_wsup':>6s} {'n_polar':>7s} {'prec_ws':>7s} {'rec_ws':>6s} {'prec_all':>8s}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    def fmt(v):
        return f"{v:.2f}" if v is not None else "  n/a"
    print(f"{r['iid']:45s} {r['n_classical_clusters']:7d} {r['n_well_supported']:6d} "
          f"{r['n_polar_peaks']:7d} {fmt(r['precision_vs_well_sup']):>7s} "
          f"{fmt(r['recall_vs_well_sup']):>6s} {fmt(r['precision_vs_all']):>8s}")

print()


def agg(key):
    vals = [r[key] for r in rows if r[key] is not None]
    return (np.mean(vals), np.median(vals), len(vals)) if vals else (None, None, 0)


for key in ["n_classical_clusters", "n_well_supported", "n_polar_peaks",
           "precision_vs_well_sup", "recall_vs_well_sup", "precision_vs_all"]:
    m, med, n = agg(key)
    if m is None:
        print(f"{key:28s} n/a")
    else:
        print(f"{key:28s} mean={m:6.3f}  median={med:6.3f}  n={n}")
