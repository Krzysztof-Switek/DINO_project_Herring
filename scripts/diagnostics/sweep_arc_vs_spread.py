"""Isolate the two candidate causes of dp's mean_dist regression: arc-aware scoring
(_merge_clusters, 0.4/0.6 weights) vs spread_weight in _dp_select_t. Computes, on the SAME
10-card sample: (a) pre-arc formula (support*strength only) + spread_weight=0 — i.e. the
dp behaviour as it was BEFORE either 20.07/21.07 change, (b) current arc formula +
spread_weight=0, (c) current arc formula + spread_weight=1.5 (today's default)."""
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
from src.otolith_axis import detect_axis, apply_background_mask, sample_profile_along_axis
from src.inference import load_model_from_checkpoint
from src.dataset import build_transforms
from src.ring_extraction import (density_peaks, classical_increments, _cluster_by_radius,
                                 _dp_select_t, _project_to_axis)

RUN_DIR = PROJECT_ROOT / "outputs" / "20.07_reg"
CKPT = RUN_DIR / "checkpoints" / "embedded" / "best.pt"
IMAGE_DIR = Path("Z:/Photo/Otolithes/HER/Processed")
PRED_CSV = RUN_DIR / "emb_on_emb" / "predictions.csv"

cfg = load_merged_config(PROJECT_ROOT / "configs" / "config.yaml",
                         PROJECT_ROOT / "configs" / "config_embedded.yaml")
cfg.data.image_dir = str(IMAGE_DIR)

best, worst = select_top_k_samples(PRED_CSV, 15,
                                   15)
samples = list(best) + list(worst)

model = load_model_from_checkpoint(cfg, CKPT)
model.eval()
device = next(model.parameters()).device
transform = build_transforms(cfg.data.image_size, "test")
min_dist = cfg.candidates.min_peak_distance
prominence = cfg.candidates.prominence_threshold
T_TOL = 0.06


def merge_pre_arc(density_pks, classical_pks, t_tol=T_TOL):
    """The _merge_clusters formula as it was BEFORE 20.07's arc-aware change: score =
    support*mean_strength, no arc term."""
    dclust = _cluster_by_radius(density_pks, t_tol)
    cclust = _cluster_by_radius(classical_pks, t_tol)
    merged, used = [], [False] * len(cclust)
    for (dt, ds, dstr) in dclust:
        score = ds * dstr
        t = dt
        best_i, best_d = -1, t_tol
        for i, c in enumerate(cclust):
            if not used[i] and abs(c[0] - dt) <= best_d:
                best_i, best_d = i, abs(c[0] - dt)
        if best_i >= 0:
            c = cclust[best_i]
            used[best_i] = True
            score += c[1] * c[2]
            t = 0.5 * (dt + c[0])
        merged.append((t, score))
    for i, c in enumerate(cclust):
        if not used[i]:
            merged.append((c[0], c[1] * c[2]))
    return merged


def mean_dist(finals, ref):
    if not finals or not ref:
        return None
    fa = np.asarray(finals, dtype=np.float32)
    ca = np.asarray(ref, dtype=np.float32)
    d = np.sqrt(((fa[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
    return float(d.min(axis=1).mean())


from src.ring_extraction import _merge_clusters as merge_with_arc

results = {"A_pre_arc_w0": [], "B_arc_w0": [], "C_arc_w1.5": []}

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
    cpk = classical_increments(orig_rgb, axis_info)["peaks"]
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

    k = max(0, age)

    merged_a = merge_pre_arc(dpk, cpk)
    chosen_a = _dp_select_t(merged_a, k, 0.04, spread_weight=0.0)
    results["A_pre_arc_w0"].append(mean_dist(_project_to_axis(chosen_a, axis_info), classical_ref))

    merged_bc = merge_with_arc(dpk, cpk, T_TOL, 48)
    cands_bc = [(t, s) for (t, s, _src) in merged_bc]
    chosen_b = _dp_select_t(cands_bc, k, 0.04, spread_weight=0.0)
    results["B_arc_w0"].append(mean_dist(_project_to_axis(chosen_b, axis_info), classical_ref))

    chosen_c = _dp_select_t(cands_bc, k, 0.04, spread_weight=1.5)
    results["C_arc_w1.5"].append(mean_dist(_project_to_axis(chosen_c, axis_info), classical_ref))

print()
for label, vals in results.items():
    valid = [v for v in vals if v is not None]
    print(f"{label:16s} mean={np.mean(valid):7.2f}  n={len(valid)}  raw={[round(v,1) if v is not None else None for v in vals]}")
