"""E12 diagnostic (21.07): does the "one t per ring, same for all 48 rays" assumption
(self-similar/isometric growth) actually hold on our otoliths, or does elongation /
nucleus-estimate uncertainty break it? PURE MEASUREMENT -- no pipeline code changes.

For each card in the same 30-card (15 best + 15 worst) sample used by the other 21.07
sweeps:
  1. elongation      = minAreaRect aspect ratio of the segmented contour (long/short side)
  2. nucleus_offset  = normalised pixel distance between the GEOMETRIC mask centroid and
                       the INTENSITY-weighted centroid (our two competing nucleus estimators,
                       resolve_centroid(method="geometric") vs (method="intensity")) -- a
                       proxy for how uncertain the ray-origin placement is, since there is
                       no manually-annotated ground-truth nucleus to compare against.
  3. Pick the SINGLE strongest/most-confident classical ring cluster (max support*strength,
     support>=5 rays) from classical per-ray peaks, then compute the STANDARD DEVIATION of
     that cluster's individual per-ray peak positions (t) -- i.e. how much even the most
     confidently-detected ring wobbles in radius across rays.

Then correlate std(t) against elongation and nucleus_offset, and report what fraction of
cards already have std(t) > T_TOL (0.06, the clustering tolerance used everywhere in the
pipeline) -- if that fraction is large, radial jitter is NOT just clustering noise, it's a
real geometric effect, and E13 (allometric/elliptical correction) would be justified.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image as PILImage

from scripts.run_pipeline import load_merged_config
from src.visualization import select_top_k_samples
from src.otolith_axis import segment_otolith, resolve_centroid, _largest_contour
from src.ring_extraction import _all_ray_peaks, _cluster_by_radius

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
MIN_SUPPORT = 5

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

    mask = segment_otolith(orig_rgb, **cfg.segmentation.as_params())
    if mask is None:
        n_skipped += 1
        continue
    contour = _largest_contour(mask)
    if contour is None or len(contour) < 5:
        n_skipped += 1
        continue

    geo_c = resolve_centroid(orig_rgb, mask, "geometric")
    int_c = resolve_centroid(orig_rgb, mask, "intensity")
    if geo_c is None or int_c is None:
        n_skipped += 1
        continue

    (rw, rh) = cv2.minAreaRect(contour)[1]
    if rw <= 0 or rh <= 0:
        n_skipped += 1
        continue
    elongation = max(rw, rh) / min(rw, rh)
    diag_px = float(np.hypot(rw, rh))

    nucleus_offset_px = float(np.hypot(geo_c[0] - int_c[0], geo_c[1] - int_c[1]))
    nucleus_offset_norm = nucleus_offset_px / diag_px if diag_px > 1e-6 else None

    nucleus_pt = int_c if cfg.segmentation.nucleus_method == "intensity" else geo_c
    axis_info_for_rays = {"contour": contour, "centroid": nucleus_pt}
    gray = orig_rgb.mean(axis=2).astype(np.float32)
    peaks, _cpts = _all_ray_peaks(
        gray, axis_info_for_rays, H, W, n_dirs=N_DIRS,
        min_distance=1, prominence=0.02)
    clusters = _cluster_by_radius(peaks, T_TOL)
    qualifying = [c for c in clusters if c[1] >= MIN_SUPPORT]
    pool = qualifying if qualifying else clusters
    if not pool:
        n_skipped += 1
        continue
    best_cluster = max(pool, key=lambda c: c[1] * c[2])
    mean_t, support, _mean_strength = best_cluster

    member_ts = [p[0] for p in peaks if abs(p[0] - mean_t) <= T_TOL]
    if len(member_ts) < 2:
        n_skipped += 1
        continue
    std_t = float(np.std(member_ts))
    range_t = float(max(member_ts) - min(member_ts))

    rows.append({
        "iid": iid, "elongation": elongation, "nucleus_offset_norm": nucleus_offset_norm,
        "support": len(member_ts), "std_t": std_t, "range_t": range_t,
    })

print(f"\nCards analysed: {len(rows)}  (skipped: {n_skipped})\n")
hdr = f"{'image_id':45s} {'elong':>6s} {'nuc_off':>8s} {'supp':>5s} {'std_t':>7s} {'range_t':>8s}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['iid']:45s} {r['elongation']:6.2f} {r['nucleus_offset_norm']:8.4f} "
          f"{r['support']:5d} {r['std_t']:7.4f} {r['range_t']:8.4f}")

elong = np.asarray([r["elongation"] for r in rows])
nuc = np.asarray([r["nucleus_offset_norm"] for r in rows])
std_t = np.asarray([r["std_t"] for r in rows])

r_elong = float(np.corrcoef(elong, std_t)[0, 1]) if len(rows) > 2 else float("nan")
r_nuc = float(np.corrcoef(nuc, std_t)[0, 1]) if len(rows) > 2 else float("nan")

n_exceed = int(np.sum(std_t > T_TOL))
print(f"\nmean std_t   = {std_t.mean():.4f}   median = {np.median(std_t):.4f}   (T_TOL={T_TOL})")
print(f"cards with std_t > T_TOL: {n_exceed}/{len(rows)} ({100*n_exceed/len(rows):.0f}%)")
print(f"corr(elongation, std_t)        r = {r_elong:+.3f}")
print(f"corr(nucleus_offset, std_t)    r = {r_nuc:+.3f}")
print(f"elongation range: {elong.min():.2f} - {elong.max():.2f}  (mean {elong.mean():.2f})")
print(f"nucleus_offset_norm range: {nuc.min():.4f} - {nuc.max():.4f}  (mean {nuc.mean():.4f})")
