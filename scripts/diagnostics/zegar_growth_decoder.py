"""01.10 — E4 (plan 30.09 §3): an explicit growth pattern in the decoder, no training.

The lab decoder takes the k brightest rows with a minimum spacing. E4 asks whether the head's row
map carries ring information that this greedy choice wastes: a dynamic-programming decoder picks
the k rows (ordered nucleus → edge) maximising

    Σ log p(row)  −  λ · Σ ( log(g_{j+1} / g_j) − log 0.8 )² / (2σ²)

where g_j are consecutive gaps in the row's own t — the same "each gap 0.8× the previous" rhythm
as the spacing-rule control, with a log-normal tolerance (Troadec 1991; graph-based ring
detection with an a-priori growth pattern). Parameters are fixed BEFORE looking at ZEGAR and never
tuned on it: ratio 0.8, σ = 0.35, λ = 1, minimum gap = the greedy decoder's DEFAULT_MIN_GAP_T.

Same k as before: oracle (experts' count, sposób 1) and count (the head's own Σp, sposób 2).
Controls with the same decoder and the same k (the critical ones, plan §3 E4):
  flat   a constant map — the growth pattern alone (ties broken towards the nucleus);
  prior  the population prior (expert rings of the OTHER 41 images) as the map — rhythm plus
         "where rings usually are", no image.
Criterion: precision(DP on head) − precision(DP on control) with bootstrap CI95 > 0.
Also reported: DP vs the greedy decoder on the same head map.

Row heads only (A5, B1, B2, B6 …).

    python scripts/diagnostics/zegar_growth_decoder.py --arms A5,B1,B2
Writes <lab-dir>/zegar_growth_decoder_<tolerance>[_<tag>].{json,csv}.
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

import expert_annotation_eval as ev
from expert_annotation_eval_wedge import (N_SAMPLES_AXIS, collect_lab_heads, decode_peaks, lab_head_map,
                                          load_lab_backbone, load_lab_head, prepare_sample,
                                          ring_tolerances)
from scripts.run_pipeline import load_merged_config
from src.peak_decoding import DEFAULT_MIN_GAP_T
from zegar_report import FRACTION, PRIOR_SIGMA, band_inputs, boot_ci, hits

RATIO = 0.8
SIGMA = 0.35
LAMBDA = 1.0
P_FLOOR = 1e-4


def growth_dp(p: np.ndarray, t: np.ndarray, k: int, ratio: float = RATIO, sigma: float = SIGMA,
              lam: float = LAMBDA, min_gap: float = DEFAULT_MIN_GAP_T) -> list[int]:
    """Indices of the k rows maximising Σ log p − λ·rhythm penalty (see module docstring).

    Exact DP over (previous, last) pairs: O(k·n³) with n ≈ 44 rows, vectorised per step.
    Returns indices in the order of increasing t; fewer than k if no k rows fit the minimum gap.
    """
    n = len(p)
    if k <= 0 or n == 0:
        return []
    order = np.argsort(t, kind="stable")
    ts = np.asarray(t, float)[order]
    u = np.log(np.clip(np.asarray(p, float)[order], P_FLOOR, 1.0))
    if k == 1:
        return [int(order[int(np.argmax(u))])]
    gap = ts[None, :] - ts[:, None]                           # gap[a, b] = t_b − t_a
    ok = gap >= min_gap - 1e-12
    neg = -np.inf
    score = np.where(ok, u[:, None] + u[None, :], neg)        # sequences of 2 ending (a, b)
    if k == 2:
        if not np.isfinite(score).any():
            return [int(order[int(np.argmax(u))])]
        a, b = np.unravel_index(int(np.argmax(score)), score.shape)
        return [int(order[a]), int(order[b])]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(gap[None, :, :] / gap[:, :, None])        # [a, b, c] = log(g_bc / g_ab)
    pen = lam * (lr - math.log(ratio)) ** 2 / (2 * sigma ** 2)
    valid3 = ok[:, :, None] & ok[None, :, :]
    pen = np.where(valid3, pen, np.inf)
    back = []
    for _ in range(k - 2):
        cand = score[:, :, None] - pen                        # [a, b, c]
        arg = np.argmax(cand, axis=0)                         # best a for (b, c)
        best = np.take_along_axis(cand, arg[None], axis=0)[0] + u[None, :]
        back.append(arg)
        score = np.where(np.isfinite(best), best, neg)
    if not np.isfinite(score).any():                          # k rows do not fit: fall back
        return growth_dp(p, t, k - 1, ratio, sigma, lam, min_gap)
    b, c = np.unravel_index(int(np.argmax(score)), score.shape)
    seq = [c, b]
    for arg in reversed(back):
        a = arg[seq[-1], seq[-2]]
        seq.append(a)
    return [int(order[i]) for i in reversed(seq)]


def prior_map(gt_all: dict, sample: str, axis_t: np.ndarray) -> np.ndarray:
    """Population-prior 'probability' at each row's axis t (leave-one-out), scaled to max 0.9."""
    others = np.concatenate([np.asarray(v) for s, v in gt_all.items() if s != sample])
    d = np.exp(-0.5 * ((axis_t[:, None] - others[None, :]) / PRIOR_SIGMA) ** 2).sum(1)
    return 0.9 * d / max(d.max(), 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab-dir", default="experiments/density_head_lab/raw")
    ap.add_argument("--arms", default="A5")
    ap.add_argument("--tolerance", choices=list(FRACTION), default="half_spacing")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config_wedge_b.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    lab_dir = Path(args.lab_dir) if Path(args.lab_dir).is_absolute() else PROJECT_ROOT / args.lab_dir
    heads = [load_lab_head(p) for p in collect_lab_heads(lab_dir, [a.strip() for a in args.arms.split(",")], False)]
    heads = [h for h in heads if h["arm"].head.startswith("rows")]
    if not heads:
        sys.exit("Brak głowic wierszowych.")
    print(f"E4 — dekoder ze wzorcem wzrostu, {len(heads)} głowic: {[h['name'] for h in heads]}", flush=True)
    cfg = load_merged_config(Path(args.config), None)
    seg = cfg.segmentation.as_params()
    max_gap_t = cfg.candidates.min_peak_distance / N_SAMPLES_AXIS
    ps = cfg.data.patch_size
    backbones = {}
    for h in heads:
        key = (h["backbone_from"], h["checkpoint_sha"])
        if key not in backbones:
            backbones[key] = load_lab_backbone(cfg, h["backbone_from"], h["checkpoint_sha"], None)
    ann = ev.load_expert_annotations()
    samples = sorted(ann["Sample"].unique())
    preps = {s: p for s in samples if not isinstance(p := prepare_sample(ann, s, cfg, seg, max_gap_t), str)}
    gt_all = {s: p[2] for s, p in preps.items()}
    if args.limit:
        samples = samples[: args.limit]

    rows = []
    for n_img, sample in enumerate(samples, 1):
        if sample not in preps:
            continue
        cropped, axis_info, gt_t, _kk, _ss = preps[sample]
        tol = ring_tolerances(gt_t, max_gap_t, FRACTION[args.tolerance])
        bands, bt, bth, btis, locs, tokens = band_inputs(cropped, axis_info, cfg, backbones)
        for h in heads:
            dens, dec_t, valid, to_axis_t = lab_head_map(
                h, tokens[(h["backbone_from"], h["checkpoint_sha"])], bt, bth, btis, locs, bands,
                axis_info, ps, radial_on_axis=True)
            axis_t = np.array([to_axis_t(i) for i in range(len(dens))])
            maps = {"head": dens, "flat": np.full_like(dens, 0.5), "prior": prior_map(gt_all, sample, axis_t)}
            k_count = int(round(float((dens * valid).sum())))
            rec = {"head": h["name"], "arm": h["arm"].name, "Sample": sample, "n_gt": len(gt_t)}
            for mode, k in (("oracle", len(gt_t)), ("count", k_count)):
                g_idx = decode_peaks(dens, dec_t, valid, mode, len(gt_t))
                rec[f"{mode}_n"] = len(g_idx)
                rec[f"{mode}_greedy_hits"] = int(hits([axis_t[i] for i in g_idx], gt_t, tol).sum())
                for name, m in maps.items():
                    idx = growth_dp(m, dec_t, len(g_idx))
                    rec[f"{mode}_dp_{name}_hits"] = int(hits([axis_t[i] for i in idx], gt_t, tol).sum())
                    rec[f"{mode}_dp_{name}_n"] = len(idx)
            rows.append(rec)
        print(f"  [{n_img}/{len(samples)}] {sample}", flush=True)

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    out = []
    for name, sub in df.groupby("head", sort=False):
        r = {"head": name, "arm": sub["arm"].iloc[0]}
        for mode in ("oracle", "count"):
            den = sub[f"{mode}_n"].to_numpy(float) if mode == "count" else sub["n_gt"].to_numpy(float)
            g = lambda c: sub[c].to_numpy(float)
            r[f"{mode}_greedy"] = round(g(f"{mode}_greedy_hits").sum() / den.sum(), 3)
            for m in ("head", "flat", "prior"):
                r[f"{mode}_dp_{m}"] = round(g(f"{mode}_dp_{m}_hits").sum() / den.sum(), 3)
            r[f"{mode}_dp_minus_flat_ci95"] = boot_ci(g(f"{mode}_dp_head_hits"), g(f"{mode}_dp_flat_hits"), den, rng)
            r[f"{mode}_dp_minus_prior_ci95"] = boot_ci(g(f"{mode}_dp_head_hits"), g(f"{mode}_dp_prior_hits"), den, rng)
            r[f"{mode}_dp_minus_greedy_ci95"] = boot_ci(g(f"{mode}_dp_head_hits"), g(f"{mode}_greedy_hits"), den, rng)
        out.append(r)
    table = pd.DataFrame(out)
    stem = f"zegar_growth_decoder_{args.tolerance}" + (f"_{args.tag}" if args.tag else "") \
        + (f"_limit{args.limit}" if args.limit else "")
    table.to_csv(lab_dir / f"{stem}.csv", index=False)
    df.to_csv(lab_dir / f"{stem}_per_image.csv", index=False)
    (lab_dir / f"{stem}.json").write_text(json.dumps(
        {"tolerance": args.tolerance, "params": {"ratio": RATIO, "sigma": SIGMA, "lambda": LAMBDA,
                                                 "min_gap": DEFAULT_MIN_GAP_T},
         "heads": out}, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print("\n" + table.to_string(index=False))
    print(f"\nZapisano: {lab_dir / f'{stem}.json'}")


if __name__ == "__main__":
    main()
