"""01.10 (plan 30.09 §2.4) — the whole ZEGAR evaluation of lab heads in one pass.

One backbone forward per image, every head decoded twice:
  sposób 1 (oracle)  k = the experts' ring count → coverage, distance (px);
  sposób 2 (count)   k = the head's own count     → precision, coverage, count error,
                                                     precision by confidence rank;
against the two no-image controls with the same k:
  kontrola 1  spacing rule (gaps shrinking 0.8× outward, `geometric_t`);
  kontrola 2  population prior — the k most frequent ring positions on the OTHER 41 images
              (leave-one-out, same minimum spacing as the head's decoder);
plus
  inner third   share of the head's points with axis t < 1/3 (experts ≈ 0.5 %) — flags heads
                that light up nucleus texture;
  between-image variability  Spearman correlation, over images, of the head's outermost /
                innermost point with the experts' outermost / innermost ring. A head that only
                knows where rings usually lie gives ≈ 0 for the outermost ring; the population
                prior's own value is reported as the reference.
Bootstrap over images (2000 draws, seed 0) for every head − control difference.

Verdict per arm (plan §2.3): H1 when precision − prior has CI95 > 0 in ≥ 80 % of the seeds (4/5),
the arm matured and the count-mode coverage is not more than 5 pt below the base arm; negative
when that CI includes or is below 0 in ≥ 2 seeds.

    python scripts/diagnostics/zegar_report.py --lab-dir experiments/density_head_lab/raw --arms A5,B1,B2
Writes <lab-dir>/zegar_report_<tolerance>[_<tag>].json, .csv (one row per head) and
zegar_report_<tolerance>[_<tag>]/<head>.csv (one row per image); --limit adds _limit<N>.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "diagnostics"))

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from scipy.stats import spearmanr

import expert_annotation_eval as ev
from expert_annotation_eval_wedge import (N_SAMPLES_AXIS, collect_lab_heads, decode_peaks, geometric_t,
                                          lab_head_map, load_lab_backbone, load_lab_head,
                                          match_t_values_tol, prepare_sample, ring_tolerances)
from scripts.run_pipeline import load_merged_config
from src.dataset import build_transforms
from src.peak_decoding import decode_topk_peaks_real_t
from src.wedge_extraction import (extract_polar_wedge_band, extract_polar_wedge_band_validity,
                                  wedge_band_geometries_from_axis_info, wedge_band_polar_coords)

FRACTION = {"half_spacing": 0.5, "quarter_spacing": 0.25}
INNER = 1.0 / 3.0
PRIOR_GRID = np.linspace(0, 1, 201)
PRIOR_SIGMA = 0.02            # same smoothing as zegar_confidence_ranks.py
N_BOOT = 2000


def population_prior_points(gt_all: dict, sample: str, k: int) -> list:
    """k most frequent ring positions on all OTHER images (leave-one-out)."""
    if k <= 0:
        return []
    others = np.concatenate([np.asarray(v) for s, v in gt_all.items() if s != sample])
    dens = np.exp(-0.5 * ((PRIOR_GRID[:, None] - others[None, :]) / PRIOR_SIGMA) ** 2).sum(1)
    return [float(PRIOR_GRID[i]) for i in decode_topk_peaks_real_t(dens, PRIOR_GRID, k)]


def hits(pts: list, gt_t: list, tol: list) -> np.ndarray:
    pairs, _, _ = match_t_values_tol(pts, gt_t, tol)
    h = np.zeros(len(pts), bool)
    for i, _j in pairs:
        h[i] = True
    return h


def paired_px(pts: list, gt_t: list, tol: list, length_px: float) -> list:
    pairs, _, _ = match_t_values_tol(pts, gt_t, tol)
    return [abs(pts[i] - gt_t[j]) * length_px for i, j in pairs]


def boot_ci(num_a: np.ndarray, num_b: np.ndarray, den: np.ndarray, rng) -> list:
    """CI95 of (Σa − Σb) / Σden with images resampled."""
    n = len(den)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        s = den[i].sum()
        if s:
            d.append((num_a[i].sum() - num_b[i].sum()) / s)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return round(float(spearmanr(a[ok], b[ok]).statistic), 3)


def band_inputs(cropped, axis_info, cfg, backbones: dict):
    """Band canvases of one image → (bands, t, θ, tissue, patch locations, {backbone key: tokens})."""
    ps = cfg.data.patch_size
    bw = [n * ps for n in cfg.data.wedge_band_n_angle_patches]
    bh = [n * ps for n in cfg.data.wedge_band_n_radius_patches]
    tf = build_transforms(1, split="test", wedge=True)
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
        btis.append(torch.from_numpy(
            extract_polar_wedge_band_validity(mask, band, ps).reshape(1, -1).copy()).float())
        locs.extend((band, r, c) for r in range(tg.shape[0]) for c in range(tg.shape[1]))
    with torch.no_grad():
        tokens = {key: torch.cat([bb.forward_features(im)["x_norm_patchtokens"] for im in imgs], 1)
                  for key, bb in backbones.items()}
    return bands, bt, bth, btis, locs, tokens


def head_summary(df: pd.DataFrame, ranks: np.ndarray, rng) -> dict:
    """Aggregate one head's per-image table."""
    g = lambda c: df[c].to_numpy(float)
    n_gt, n_pts = g("n_gt"), g("count_n")
    out = {
        "images": len(df), "gt_rings": int(n_gt.sum()), "count_points": int(n_pts.sum()),
        # sposób 1
        "oracle_coverage": round(g("oracle_hits").sum() / n_gt.sum(), 3),
        "oracle_rule_coverage": round(g("oracle_rule_hits").sum() / n_gt.sum(), 3),
        "oracle_prior_coverage": round(g("oracle_prior_hits").sum() / n_gt.sum(), 3),
        "oracle_px": round(float(np.mean(np.concatenate(df["oracle_px"].to_list()))), 2)
        if df["oracle_px"].map(len).sum() else None,
        "oracle_minus_rule_ci95": boot_ci(g("oracle_hits"), g("oracle_rule_hits"), n_gt, rng),
        "oracle_minus_prior_ci95": boot_ci(g("oracle_hits"), g("oracle_prior_hits"), n_gt, rng),
        # sposób 2
        "precision": round(g("count_hits").sum() / max(n_pts.sum(), 1), 3),
        "coverage": round(g("count_hits").sum() / n_gt.sum(), 3),
        "rule_precision": round(g("count_rule_hits").sum() / max(n_pts.sum(), 1), 3),
        "prior_precision": round(g("count_prior_hits").sum() / max(n_pts.sum(), 1), 3),
        "precision_minus_rule_ci95": boot_ci(g("count_hits"), g("count_rule_hits"), n_pts, rng),
        "precision_minus_prior_ci95": boot_ci(g("count_hits"), g("count_prior_hits"), n_pts, rng),
        "count_mae": round(float(np.mean(np.abs(n_pts - n_gt))), 3),
        "count_px": round(float(np.mean(np.concatenate(df["count_px"].to_list()))), 2)
        if df["count_px"].map(len).sum() else None,
        "precision_by_rank": [round(h / n, 3) if n else None for h, n in ranks],
        # diagnostics
        "inner_third_share": round(g("count_inner").sum() / max(n_pts.sum(), 1), 3),
        "inner_third_images": int((g("count_inner") > 0).sum()),
        "corr_outer": spearman(g("model_outer"), g("gt_outer")),
        "corr_inner": spearman(g("model_inner"), g("gt_inner")),
        "sd_outer": round(float(np.nanstd(g("model_outer"))), 3),
    }
    return out


def arm_verdicts(table: pd.DataFrame, matured: dict, base_arm: str | None) -> list[dict]:
    rows = []
    base_cov = (table[table.arm == base_arm]["coverage"].median()
                if base_arm and (table.arm == base_arm).any() else None)
    for arm, sub in table.groupby("arm", sort=False):
        lo = sub["precision_minus_prior_ci95"].map(lambda c: c[0])
        n = len(sub)
        n_pos = int((lo > 0).sum())
        n_mat = int(sum(matured.get(h, False) for h in sub["head"]))
        cov = float(sub["coverage"].median())
        cov_ok = base_cov is None or arm == base_arm or cov >= base_cov - 0.05
        if n_pos >= math.ceil(0.8 * n) and n_mat == n and cov_ok:
            verdict = "H1 potwierdzona"
        elif n - n_pos >= 2:
            verdict = "negatywny"
        else:
            verdict = "niejednoznaczny"
        rows.append({"arm": arm, "seeds": n, "matured": n_mat,
                     "precision": f"{sub['precision'].min():.2f}–{sub['precision'].max():.2f}",
                     "prior_precision": f"{sub['prior_precision'].min():.2f}–{sub['prior_precision'].max():.2f}",
                     "ci_prior_gt0": f"{n_pos}/{n}",
                     "rule_precision": f"{sub['rule_precision'].min():.2f}–{sub['rule_precision'].max():.2f}",
                     "oracle_coverage": f"{sub['oracle_coverage'].min():.2f}–{sub['oracle_coverage'].max():.2f}",
                     "coverage_median": round(cov, 3),
                     "inner_third": f"{sub['inner_third_share'].min():.2f}–{sub['inner_third_share'].max():.2f}",
                     "corr_outer": f"{sub['corr_outer'].min()}…{sub['corr_outer'].max()}",
                     "verdict": verdict})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab-dir", default="experiments/density_head_lab/raw")
    ap.add_argument("--arms", default="A5")
    ap.add_argument("--base", default="A5", help="arm the coverage criterion compares with")
    ap.add_argument("--tolerance", choices=list(FRACTION), default="half_spacing")
    ap.add_argument("--max-rank", type=int, default=4)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config_wedge_b.yaml"))
    ap.add_argument("--backbone-from", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke)")
    ap.add_argument("--tag", default=None, help="suffix of the output name, e.g. E1")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    lab_dir = Path(args.lab_dir) if Path(args.lab_dir).is_absolute() else PROJECT_ROOT / args.lab_dir
    paths = collect_lab_heads(lab_dir, [a.strip() for a in args.arms.split(",")], False)
    if not paths:
        sys.exit(f"Brak głowic {args.arms} w {lab_dir}")
    heads = [load_lab_head(p) for p in paths]
    matured = {}
    for h in heads:
        s = h["path"].parent / "summary.json"
        matured[h["name"]] = bool(s.exists() and json.loads(s.read_text(encoding="utf-8")).get("matured"))
    print(f"ZEGAR report — {len(heads)} głowic: {[h['name'] for h in heads]}  tolerancja {args.tolerance}",
          flush=True)

    cfg = load_merged_config(Path(args.config), None)
    seg = cfg.segmentation.as_params()
    max_gap_t = cfg.candidates.min_peak_distance / N_SAMPLES_AXIS
    ps = cfg.data.patch_size
    backbones = {}
    for h in heads:
        key = (h["backbone_from"], h["checkpoint_sha"])
        if key not in backbones:
            backbones[key] = load_lab_backbone(cfg, h["backbone_from"], h["checkpoint_sha"],
                                               args.backbone_from)

    ann = ev.load_expert_annotations()
    samples = sorted(ann["Sample"].unique())
    preps = {}
    for s in samples:
        p = prepare_sample(ann, s, cfg, seg, max_gap_t)
        if not isinstance(p, str):
            preps[s] = p
    gt_all = {s: p[2] for s, p in preps.items()}            # prior uses all images, LOO per image
    if args.limit:
        samples = samples[: args.limit]
    frac = FRACTION[args.tolerance]

    per_image = {h["name"]: [] for h in heads}
    ranks = {h["name"]: np.zeros((args.max_rank, 2)) for h in heads}
    prior_ref = []
    for n_img, sample in enumerate(samples, 1):
        if sample not in preps:
            print(f"  [{n_img}/{len(samples)}] {sample}: pominięte", flush=True)
            continue
        cropped, axis_info, gt_t, _kk, _ss = preps[sample]
        tol = ring_tolerances(gt_t, max_gap_t, frac)
        L = axis_info["length_px"]
        bands, bt, bth, btis, locs, tokens = band_inputs(cropped, axis_info, cfg, backbones)
        k_gt = len(gt_t)
        prior_k_gt = population_prior_points(gt_all, sample, k_gt)
        prior_ref.append((max(prior_k_gt), min(prior_k_gt), max(gt_t), min(gt_t)))
        o_rule, o_prior = hits(geometric_t(k_gt), gt_t, tol), hits(prior_k_gt, gt_t, tol)
        for h in heads:
            dens, dec_t, valid, to_axis_t = lab_head_map(
                h, tokens[(h["backbone_from"], h["checkpoint_sha"])], bt, bth, btis, locs, bands,
                axis_info, ps, radial_on_axis=True)
            o_pts = [to_axis_t(i) for i in decode_peaks(dens, dec_t, valid, "oracle", k_gt)]
            c_idx = sorted(decode_peaks(dens, dec_t, valid, "count", k_gt), key=lambda i: -dens[i])
            c_pts = [to_axis_t(i) for i in c_idx]
            c_hit = hits(c_pts, gt_t, tol)
            for r in range(min(args.max_rank, len(c_pts))):
                ranks[h["name"]][r] += (c_hit[r], 1)
            k = len(c_pts)
            per_image[h["name"]].append({
                "Sample": sample, "n_gt": k_gt,
                "oracle_hits": int(hits(o_pts, gt_t, tol).sum()),
                "oracle_rule_hits": int(o_rule.sum()), "oracle_prior_hits": int(o_prior.sum()),
                "oracle_px": paired_px(o_pts, gt_t, tol, L),
                "count_n": k, "count_hits": int(c_hit.sum()),
                "count_rule_hits": int(hits(geometric_t(k), gt_t, tol).sum()),
                "count_prior_hits": int(hits(population_prior_points(gt_all, sample, k), gt_t, tol).sum()),
                "count_px": paired_px(c_pts, gt_t, tol, L),
                "count_inner": int(sum(t < INNER for t in c_pts)),
                "model_outer": max(o_pts) if o_pts else np.nan,
                "model_inner": min(o_pts) if o_pts else np.nan,
                "gt_outer": max(gt_t), "gt_inner": min(gt_t),
            })
        print(f"  [{n_img}/{len(samples)}] {sample}", flush=True)

    rng = np.random.default_rng(0)
    # a smoke run (--limit) never overwrites a full report; --tag keeps several full reports apart
    stem = f"zegar_report_{args.tolerance}" + (f"_{args.tag}" if args.tag else "")         + (f"_limit{args.limit}" if args.limit else "")
    out_dir = lab_dir / stem
    out_dir.mkdir(exist_ok=True)
    rows = []
    for h in heads:
        df = pd.DataFrame(per_image[h["name"]])
        df.drop(columns=["oracle_px", "count_px"]).to_csv(out_dir / f"{h['name']}.csv", index=False)
        rows.append({"head": h["name"], "arm": h["arm"].name, "matured": matured[h["name"]],
                     **head_summary(df, ranks[h["name"]], rng)})
    table = pd.DataFrame(rows)
    pr = np.array(prior_ref, float)
    reference = {"experts_inner_third_share": round(float(np.mean(
                     [t < INNER for v in gt_all.values() for t in v])), 3),
                 "prior_corr_outer": spearman(pr[:, 0], pr[:, 2]),
                 "prior_corr_inner": spearman(pr[:, 1], pr[:, 3])}
    verdicts = arm_verdicts(table, matured, args.base)
    table.to_csv(lab_dir / f"{stem}.csv", index=False)
    (lab_dir / f"{stem}.json").write_text(json.dumps(
        {"tolerance": args.tolerance, "arms": verdicts, "heads": rows, "reference": reference},
        indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 250)
    cols = ["head", "matured", "oracle_coverage", "oracle_rule_coverage", "precision", "rule_precision",
            "prior_precision", "precision_minus_prior_ci95", "coverage", "count_points",
            "inner_third_share", "corr_outer", "corr_inner"]
    print("\n" + table[cols].to_string(index=False))
    print("\nOdniesienie:", reference)
    print("\n" + pd.DataFrame(verdicts).to_string(index=False))
    print(f"\nZapisano: {lab_dir / f'{stem}.json'}")


if __name__ == "__main__":
    main()
