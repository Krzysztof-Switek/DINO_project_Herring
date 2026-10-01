"""30.09 — does a lab head know where it is right?

Two controls with the head's own number of points k: the spacing rule (gaps shrinking 20 % outward)
and a population prior — the k most frequent ring positions on the OTHER 41 ZEGAR images (leave-one-
out), chosen with the same minimum spacing. Beating the prior means reading the individual otolith,
not only knowing that rings usually sit near the edge.
 Precision of its 1st, 2nd, 3rd … most confident
point on each ZEGAR image, against the same-rank point of a no-image control, plus a bootstrap
interval (over images) for the head-minus-control precision in the free-count mode.

    python scripts/diagnostics/zegar_confidence_ranks.py --arms A5 --tolerance half_spacing
Writes experiments/density_head_lab/zegar_confidence_<tolerance>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "diagnostics"))

import numpy as np
import torch
from PIL import Image as PILImage

import expert_annotation_eval as ev
from expert_annotation_eval_wedge import (N_SAMPLES_AXIS, collect_lab_heads, decode_peaks, geometric_t,
                                          lab_head_map, load_lab_backbone, load_lab_head,
                                          match_t_values_tol, prepare_sample, ring_tolerances)
from scripts.run_pipeline import load_merged_config
from src.dataset import build_transforms
from src.wedge_extraction import (extract_polar_wedge_band, extract_polar_wedge_band_validity,
                                  wedge_band_geometries_from_axis_info, wedge_band_polar_coords)

FRACTION = {"half_spacing": 0.5, "quarter_spacing": 0.25}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A5")
    ap.add_argument("--tolerance", choices=list(FRACTION), default="half_spacing")
    ap.add_argument("--max-rank", type=int, default=4)
    ap.add_argument("--lab-dir", default="experiments/density_head_lab/raw")
    args = ap.parse_args()
    lab_dir = PROJECT_ROOT / args.lab_dir
    heads = [load_lab_head(p) for p in collect_lab_heads(lab_dir, args.arms.split(","), True)]
    cfg = load_merged_config(PROJECT_ROOT / "configs/config_wedge_b.yaml", None)
    seg = cfg.segmentation.as_params()
    max_gap_t = cfg.candidates.min_peak_distance / N_SAMPLES_AXIS
    ps = cfg.data.patch_size
    bw = [n * ps for n in cfg.data.wedge_band_n_angle_patches]
    bh = [n * ps for n in cfg.data.wedge_band_n_radius_patches]
    tf = build_transforms(1, split="test", wedge=True)
    bb = load_lab_backbone(cfg, heads[0]["backbone_from"], heads[0]["checkpoint_sha"], None)
    ann = ev.load_expert_annotations()
    # pass 1: expert consensus positions of every image (for the leave-one-out population prior)
    gt_all = {}
    for sample in sorted(ann["Sample"].unique()):
        prep = prepare_sample(ann, sample, cfg, seg, max_gap_t)
        if not isinstance(prep, str):
            gt_all[sample] = prep[2]
    grid = np.linspace(0, 1, 201)

    def prior_points(sample, k):
        others = np.concatenate([np.asarray(v) for s_, v in gt_all.items() if s_ != sample])
        dens = np.exp(-0.5 * ((grid[:, None] - others[None, :]) / 0.02) ** 2).sum(1)
        from src.peak_decoding import decode_topk_peaks_real_t
        return [float(grid[i]) for i in decode_topk_peaks_real_t(dens, grid, k)]

    rank_hits = {h["name"]: np.zeros((args.max_rank, 2)) for h in heads}      # [hits, n] per rank
    ctrl_rank = np.zeros((args.max_rank, 2))
    per_image = {h["name"]: [] for h in heads}                                # (hits, n, ctrl_hits)
    for sample in sorted(ann["Sample"].unique()):
        prep = prepare_sample(ann, sample, cfg, seg, max_gap_t)
        if isinstance(prep, str):
            continue
        cropped, axis_info, gt_t, _kk, _ss = prep
        tol = ring_tolerances(gt_t, max_gap_t, FRACTION[args.tolerance])
        mask = axis_info["mask"]
        bands = wedge_band_geometries_from_axis_info(mask, axis_info, cfg.data.wedge_delta_theta_deg,
                                                     cfg.data.wedge_band_edges_t, bw, bh)
        imgs, bt, bth, btis, locs = [], [], [], [], []
        for band in bands:
            arr = extract_polar_wedge_band(cropped, mask, band)
            imgs.append(tf(PILImage.fromarray(arr)).unsqueeze(0))
            tg, thg = wedge_band_polar_coords(band, ps)
            bt.append(torch.from_numpy(tg.reshape(1, -1).copy()).float())
            bth.append(torch.from_numpy(thg.reshape(1, -1).copy()).float())
            btis.append(torch.from_numpy(extract_polar_wedge_band_validity(mask, band, ps).reshape(1, -1).copy()).float())
            locs.extend((band, r, c) for r in range(tg.shape[0]) for c in range(tg.shape[1]))
        with torch.no_grad():
            tok = torch.cat([bb.forward_features(im)["x_norm_patchtokens"] for im in imgs], 1)

        def hits_of(pts):
            pairs, _, _ = match_t_values_tol(pts, gt_t, tol)
            hit = np.zeros(len(pts), bool)
            for i, _j in pairs:
                hit[i] = True
            return hit

        # control ranked by the order the spacing rule lays points from the edge inward
        ctrl = geometric_t(args.max_rank)[::-1]
        ch = hits_of(ctrl)
        for r in range(min(args.max_rank, len(ctrl))):
            ctrl_rank[r] += (ch[r], 1)
        for h in heads:
            dens, dec_t, valid, to_axis_t = lab_head_map(h, tok, bt, bth, btis, locs, bands, axis_info, ps, True)
            idx = decode_peaks(dens, dec_t, valid, "count", len(gt_t))
            idx = sorted(idx, key=lambda i: -dens[i])                     # most confident first
            pts = [to_axis_t(i) for i in idx]
            hit = hits_of(pts)
            for r in range(min(args.max_rank, len(pts))):
                rank_hits[h["name"]][r] += (hit[r], 1)
            c_hit = hits_of(geometric_t(len(pts)))
            p_hit = hits_of(prior_points(sample, len(pts))) if pts else np.zeros(0, bool)
            per_image[h["name"]].append((int(hit.sum()), len(pts), int(c_hit.sum()), int(p_hit.sum())))
        print(f"  {sample}", flush=True)

    rng = np.random.default_rng(0)
    out = {"tolerance": args.tolerance, "control_by_rank": [round(h / n, 3) if n else None for h, n in ctrl_rank]}
    for name in rank_hits:
        arr = np.array(per_image[name], float)                            # hits, n, ctrl_hits
        diffs, diffs_p = [], []
        for _ in range(2000):
            s = arr[rng.integers(0, len(arr), len(arr))]
            if s[:, 1].sum():
                diffs.append((s[:, 0].sum() - s[:, 2].sum()) / s[:, 1].sum())
                diffs_p.append((s[:, 0].sum() - s[:, 3].sum()) / s[:, 1].sum())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        plo, phi = np.percentile(diffs_p, [2.5, 97.5])
        out[name] = {"precision_by_rank": [round(h / n, 3) if n else None for h, n in rank_hits[name]],
                     "n_by_rank": [int(n) for _h, n in rank_hits[name]],
                     "precision": round(arr[:, 0].sum() / arr[:, 1].sum(), 3),
                     "control_precision": round(arr[:, 2].sum() / arr[:, 1].sum(), 3),
                     "diff_ci95": [round(lo, 3), round(hi, 3)],
                     "prior_precision": round(arr[:, 3].sum() / arr[:, 1].sum(), 3),
                     "diff_prior_ci95": [round(plo, 3), round(phi, 3)]}
        o = out[name]
        print(f"{name}: trafność {o['precision']}  reguła {o['control_precision']} {o['diff_ci95']}  "
              f"prior populacyjny {o['prior_precision']} {o['diff_prior_ci95']}  wg rangi {o['precision_by_rank']}")
    print("kontrola wg rangi:", out["control_by_rank"])
    path = lab_dir / f"zegar_confidence_{args.tolerance}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Zapisano:", path)


if __name__ == "__main__":
    main()
