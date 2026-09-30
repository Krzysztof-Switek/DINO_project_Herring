"""30.09 — ZEGAR floor for the wedge-band localization metric: what does the metric give with NO model?

With the ring count k taken from the expert consensus (as in every ZEGAR number of this project),
the decoder picks k positions with a minimum spacing in t. Part of any score may therefore come
from k and the spacing alone. Two floors, scored with the same pairing as
`expert_annotation_eval_wedge.py` (`prepare_sample`, `paired_eval`, `summarize`):

  even        k positions evenly spaced along the axis, t = (i + 0.5) / k — no image at all;
  geometric   k positions with spacing shrinking outward (each gap 0.8× the previous), a crude
              growth-slows-with-age prior — still no image;
  random_head an untrained lab head (A3 layout, prior bias) on real band tokens, 5 seeds,
              decoded like a trained one — the model's structure without any learning.

    python scripts/diagnostics/zegar_baselines_wedge.py
Writes experiments/density_head_lab/zegar_baselines.json
"""
from __future__ import annotations

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
from expert_annotation_eval_wedge import (N_SAMPLES_AXIS, geometric_t, paired_eval, ring_tolerances, prepare_sample, summarize,
                                          lab_head_t_model)
from scripts.run_pipeline import load_merged_config
from src.dataset import build_transforms
from src.wedge_extraction import (extract_polar_wedge_band, extract_polar_wedge_band_validity,
                                  wedge_band_geometries_from_axis_info, wedge_band_polar_coords)


def even_t(k: int) -> list[float]:
    return [(i + 0.5) / k for i in range(k)]


TOL = "axis10"


def main() -> None:
    global TOL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", choices=["axis10", "half_spacing", "quarter_spacing"], default="axis10")
    TOL = ap.parse_args().tolerance
    from density_head_lab import ARMS, build_head, set_prior
    from cache_wedge_band_tokens import load_backbone

    cfg = load_merged_config(PROJECT_ROOT / "configs" / "config_wedge_b.yaml", None)
    seg = cfg.segmentation.as_params()
    max_gap_t = cfg.candidates.min_peak_distance / N_SAMPLES_AXIS
    ps = cfg.data.patch_size
    bw = [n * ps for n in cfg.data.wedge_band_n_angle_patches]
    bh = [n * ps for n in cfg.data.wedge_band_n_radius_patches]
    shapes = list(zip(cfg.data.wedge_band_n_radius_patches, cfg.data.wedge_band_n_angle_patches))
    tf = build_transforms(1, split="test", wedge=True)
    backbone, _ = load_backbone(cfg, "raw")
    backbone.eval()
    heads = []
    for s in range(5):
        torch.manual_seed(s)
        h = build_head(ARMS["A3"], 384, shapes).eval()
        set_prior(h, 3.4, 5852)
        heads.append({"name": f"random_seed{s}", "arm": ARMS["A3"], "head": h})

    ann = ev.load_expert_annotations()
    rows = {"even": [], "geometric": []} | {h["name"]: [] for h in heads} \
        | {h["name"] + "_os": [] for h in heads}
    for i, sample in enumerate(sorted(ann["Sample"].unique()), 1):
        prep = prepare_sample(ann, sample, cfg, seg, max_gap_t)
        if isinstance(prep, str):
            continue
        cropped, axis_info, gt_t, _t_kk, _t_ss = prep
        L, k = axis_info["length_px"], len(gt_t)
        frac = {"half_spacing": 0.5, "quarter_spacing": 0.25}.get(TOL)
        tol = ring_tolerances(gt_t, max_gap_t, frac) if frac else None
        rows["even"].append(paired_eval(even_t(k), gt_t, L, max_gap_t, tol))
        rows["geometric"].append(paired_eval(geometric_t(k), gt_t, L, max_gap_t, tol))
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
            btis.append(torch.from_numpy(extract_polar_wedge_band_validity(mask, band, ps)
                                         .reshape(1, -1).copy()).float())
            locs.extend((band, r, c) for r in range(tg.shape[0]) for c in range(tg.shape[1]))
        with torch.no_grad():
            tok = torch.cat([backbone.forward_features(im)["x_norm_patchtokens"] for im in imgs], 1)
        for h in heads:
            for suffix, on_axis in (("", False), ("_os", True)):
                t_model = lab_head_t_model(h, tok, bt, bth, btis, locs, bands, k, axis_info, ps,
                                           radial_on_axis=on_axis)
                rows[h["name"] + suffix].append(paired_eval(t_model, gt_t, L, max_gap_t, tol))
        print(f"  [{i}] {sample}", flush=True)

    out = {name: summarize(r) for name, r in rows.items()}
    for name, s in out.items():
        print(f"{name:22s} {s['pooled_mean_over_all_pairs_px']:6.2f} px   pokrycie {s['pairing_coverage']:.3f}")
    path = PROJECT_ROOT / "experiments" / "density_head_lab" / (
        "zegar_baselines.json" if TOL == "axis10" else f"zegar_baselines_{TOL}.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Zapisano: {path}")


if __name__ == "__main__":
    main()
