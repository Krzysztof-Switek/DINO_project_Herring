"""30.09 — earlier localization methods re-scored with the current ZEGAR pairing rules.

The earlier reference numbers (Run N 14.9 px / 62 %, strip "Track A + maska" 18.7 px / 52.7 %) were
computed with a fixed pairing tolerance of 10 % of the axis length (`outputs/09.09_masked_wizualizacje/
paired_metric_all_runs.py`). That tolerance is wider than the gap between outer rings, so it rewards any
point in the ring band near the edge. This script re-scores the same methods with the same rules as
`expert_annotation_eval_wedge.py`, and with a no-image control using the same number of points.

  Run N          density head on the whole-otolith square grid (outputs/11.08_radial_attention, best.pt);
                 points = the pipeline's own increments, their number = the age predicted by the model
                 (read from the saved per-image points of the 09.09 run — no re-inference).
  Track A+maska  strip along the reading axis (outputs/08.09_strip_a_masked, best.pt), re-run here;
                 scored with the ring count given (as it was) and with its own count (round of the
                 density sum).

    python scripts/diagnostics/zegar_reference_runs.py --tolerance axis10          # must reproduce 09.09
    python scripts/diagnostics/zegar_reference_runs.py --tolerance half_spacing
Writes experiments/density_head_lab/zegar_reference_runs_<tolerance>.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "diagnostics"))

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

import expert_annotation_eval as ev
from expert_annotation_eval_wedge import (N_SAMPLES_AXIS, geometric_t, paired_eval, ring_tolerances,
                                          summarize)
from train_zegar_localization_head import decode_topk_peaks
from scripts.run_pipeline import load_merged_config
from src.dataset import build_transforms
from src.inference import load_model_from_checkpoint
from src.otolith_axis import detect_axis
from src.strip_extraction import extract_strip

RUNN_PICKLE = PROJECT_ROOT / "outputs/09.09_masked_wizualizacje/paired_eval/runn_detail_for_cards.pkl"
STRIP_CFG = PROJECT_ROOT / "configs/config_strip_a_masked.yaml"
STRIP_CKPT = PROJECT_ROOT / "outputs/08.09_strip_a_masked/checkpoints/embedded/best.pt"
FRACTION = {"axis10": None, "half_spacing": 0.5, "quarter_spacing": 0.25}


def tol_for(gt_t, max_gap_t, tolerance):
    frac = FRACTION[tolerance]
    return ring_tolerances(gt_t, max_gap_t, frac) if frac else None


def run_n(tolerance: str, max_gap_t: float) -> dict:
    detail = pickle.loads(RUNN_PICKLE.read_bytes())
    rows, ctrl = [], []
    for sample, d in sorted(detail.items()):
        gt_t, t_model, L = d["gt_t"], d["t_model"], d["length_px"]
        tol = tol_for(gt_t, max_gap_t, tolerance)
        rows.append(paired_eval(t_model, gt_t, L, max_gap_t, tol))
        ctrl.append(paired_eval(geometric_t(len(t_model)), gt_t, L, max_gap_t, tol))
    return {"Run N · liczba = przewidziany wiek": summarize(rows),
            "Run N · kontrola (ta sama liczba, reguła)": summarize(ctrl)}


def strip(tolerance: str) -> dict:
    ann = ev.load_expert_annotations()
    cfg = load_merged_config(STRIP_CFG, None)
    seg = cfg.segmentation.as_params()
    L_px, W_px = cfg.data.strip_length_px, cfg.data.strip_width_px
    h_p, w_p = W_px // cfg.data.patch_size, L_px // cfg.data.patch_size
    max_gap_t = cfg.candidates.min_peak_distance / N_SAMPLES_AXIS
    tf = build_transforms(L_px, split="test", strip=True)
    model = load_model_from_checkpoint(cfg, STRIP_CKPT)
    model.eval()
    out = {k: [] for k in ("oracle", "oracle_ctrl", "count", "count_ctrl")}
    for sample in sorted(ann["Sample"].unique()):
        sub = ann[ann.Sample == sample]
        raw = np.array(PILImage.open(ev.IMAGE_DIR / f"{sample}.jpg").convert("RGB"), dtype=np.uint8)
        cropped, x0, y0, _ = ev.resolve_and_crop_target_otolith(raw, (float(sub.x.mean()), float(sub.y.mean())), seg)
        if cropped is None:
            continue
        axis_info = detect_axis(cropped, seg_params=seg, nucleus_method=cfg.segmentation.nucleus_method,
                                axis_method=cfg.segmentation.axis_method)
        if axis_info is None:
            continue
        t_kk = [ev.point_to_axis_t(x, y, axis_info) for x, y in
                zip(sub[sub.annotator == "KK"].x - x0, sub[sub.annotator == "KK"].y - y0)]
        t_ss = [ev.point_to_axis_t(x, y, axis_info) for x, y in
                zip(sub[sub.annotator == "SS"].x - x0, sub[sub.annotator == "SS"].y - y0)]
        pairs, _, _ = ev.match_t_values(t_kk, t_ss, max_gap_t)
        gt_t = [(t_kk[i] + t_ss[j]) / 2.0 for i, j in pairs]
        if not gt_t:
            continue
        strip_rgb, M = extract_strip(cropped, axis_info["mask"], axis_info["centroid"], axis_info["far_edge"],
                                     axis_info["length_px"], L_px, W_px)
        with torch.no_grad():
            grid = model.get_density_probs(tf(PILImage.fromarray(strip_rgb)).unsqueeze(0),
                                           patch_grid=(h_p, w_p))[0].numpy()
        M_inv = cv2.invertAffineTransform(M)

        def to_t(peaks):
            res = []
            for r, c in peaks:
                x, y = (M_inv @ np.array([(c + 0.5) * cfg.data.patch_size, (r + 0.5) * cfg.data.patch_size, 1.0]))[:2]
                res.append(ev.point_to_axis_t(x, y, axis_info))
            return res

        L = axis_info["length_px"]
        tol = tol_for(gt_t, max_gap_t, tolerance)
        t_or = to_t(decode_topk_peaks(grid, len(gt_t), min_dist_patches=1.5))
        out["oracle"].append(paired_eval(t_or, gt_t, L, max_gap_t, tol))
        out["oracle_ctrl"].append(paired_eval(geometric_t(len(gt_t)), gt_t, L, max_gap_t, tol))
        k_own = int(round(float(grid.sum())))
        t_ct = to_t(decode_topk_peaks(grid, k_own, min_dist_patches=1.5)) if k_own > 0 else []
        out["count"].append(paired_eval(t_ct, gt_t, L, max_gap_t, tol))
        out["count_ctrl"].append(paired_eval(geometric_t(len(t_ct)), gt_t, L, max_gap_t, tol))
        print(f"  {sample}", flush=True)
    return {"Pasek (Track A + maska) · liczba podana": summarize(out["oracle"]),
            "Pasek · reguła bez zdjęcia (liczba podana)": summarize(out["oracle_ctrl"]),
            "Pasek · liczba = suma mapy": summarize(out["count"]),
            "Pasek · kontrola (ta sama liczba, reguła)": summarize(out["count_ctrl"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", choices=list(FRACTION), default="half_spacing")
    args = ap.parse_args()
    max_gap_t = 5 / N_SAMPLES_AXIS
    res = run_n(args.tolerance, max_gap_t) | strip(args.tolerance)
    for name, s in res.items():
        px = s["pooled_mean_over_all_pairs_px"]
        print(f"{name:48s} {px if px is None else round(px, 1)!s:>6} px  pokrycie {s['pairing_coverage']:.3f}  "
              f"trafność {s['precision'] if s['precision'] is None else round(s['precision'], 3)}  "
              f"wskazań {s['total_model_points']}/{s['total_gt_rings']}")
    path = PROJECT_ROOT / "experiments" / "density_head_lab" / f"zegar_reference_runs_{args.tolerance}.json"
    path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"Zapisano: {path}")


if __name__ == "__main__":
    main()
