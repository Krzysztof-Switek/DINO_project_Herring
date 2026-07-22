"""Fast (no report render) sweep of dp_spread_weight over the SAME 10 cards used in
outputs/20.07_reg (5 best + 5 worst by |pred-true| age error) — measures mean_dist to the
single-axis classical reference, exactly like scripts.run_pipeline._localization_quality,
for each candidate weight, so we validate on the FULL sample instead of one example."""
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
from src.ring_extraction import density_peaks, classical_increments, fuse_increments

RUN_DIR = PROJECT_ROOT / "outputs" / "20.07_reg"
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

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
per_card = []  # list of dicts: iid -> {w: dist}

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

    # Single-axis classical reference (matches _localization_quality's classical_pts:
    # grayscale profile along the SAME axis, normalised to [0,1], THEN peak-found —
    # not the density profile).
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

    def mean_dist(finals, ref):
        if not finals or not ref:
            return None
        fa = np.asarray(finals, dtype=np.float32)
        ca = np.asarray(ref, dtype=np.float32)
        d = np.sqrt(((fa[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
        return float(d.min(axis=1).mean())

    row_result = {"iid": iid, "age": age}
    for w in WEIGHTS:
        fr = fuse_increments(dpk, cpk, age, axis_info, method="dp", dp_spread_weight=w)
        row_result[w] = mean_dist(fr["final_axis_pts"], classical_ref)
    per_card.append(row_result)
    print(f"{iid[:45]:45s} age={age:2d} " + " ".join(
        f"w={w}:{row_result[w]:.0f}" if row_result[w] is not None else f"w={w}:None"
        for w in WEIGHTS))

print()
print("=== mean over cards (ignoring None) ===")
for w in WEIGHTS:
    vals = [r[w] for r in per_card if r[w] is not None]
    print(f"spread_weight={w:4.2f}  mean_dist={np.mean(vals):7.2f}  n={len(vals)}")
