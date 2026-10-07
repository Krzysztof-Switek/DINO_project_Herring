"""07.10 — E11 (plan 30.09 §7): is the edge of the good B6 seeds position, not ring appearance?

Two controls on saved row heads, no training:

  A  row-space population prior. Expert rings of the OTHER 41 images are converted to the contour-normalised
     radius of the canvas rows (each image through its own row → axis mapping), the k most frequent positions
     are taken there and mapped back onto THIS image's reading axis through its own mapping. It is a "head"
     that knows only where its rows lie — the geometry the 2.2 population prior (built on the axis) lacks.
  B  description swap. The head reads the DINOv2 tokens of another image (donors: the next 3 images in the
     list, cyclically) and its points are scored on the recipient's experts, in the recipient's geometry.
     If precision does not drop, the head's output does not depend on the image content.
  C  (E12) count-conditioned row-space prior: as A, but only from the other images whose expert ring count n
     satisfies |n − k| ≤ 1 (widened by 1 until ≥ 5 images). It knows where the rows lie AND how many rings
     there are, not what they look like.

k: oracle (experts' count; same k for own and donor tokens — binding for B) and count (the head's own Σp;
binding for A). Reported next to the axis population prior of zegar_report.py.
Verdict for the named seeds (default B6 0, 2, 4): H-pasmo confirmed when head − A has CI95 ∋ 0 or < 0 in ≥ 2/3
AND own − donor has CI95 ∋ 0 in ≥ 2/3; rejected when both CI95 > 0 in ≥ 2/3; otherwise inconclusive.

    python scripts/diagnostics/zegar_band_position_test.py --arms B6
Writes <lab-dir>/zegar_band_position_<tolerance>[_<tag>].{json,csv}.
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

import expert_annotation_eval as ev
from expert_annotation_eval_wedge import (N_SAMPLES_AXIS, collect_lab_heads, decode_peaks, lab_head_map,
                                          load_lab_backbone, load_lab_head, prepare_sample, ring_tolerances)
from scripts.run_pipeline import load_merged_config
from src.peak_decoding import decode_topk_peaks_real_t
from zegar_report import (FRACTION, PRIOR_GRID, PRIOR_SIGMA, band_inputs, boot_ci, hits,
                          population_prior_points)

N_DONORS = 3
VALID_ROW = 0.5


def row_mapping(dec_t: np.ndarray, valid: np.ndarray, axis_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(contour t, axis t) of the image's tissue rows, sorted, axis t made non-decreasing."""
    rows = np.flatnonzero(valid >= VALID_ROW)
    rows = rows[np.argsort(dec_t[rows], kind="stable")]
    ct, at = dec_t[rows].astype(float), np.maximum.accumulate(axis_t[rows].astype(float))
    return ct, at


def row_prior_points(maps: dict, gt_axis: dict, sample: str, k: int) -> list:
    """k most frequent contour-t ring positions on the other images, mapped onto this image's axis."""
    if k <= 0:
        return []
    others = np.concatenate([np.interp(gt_axis[s], maps[s][1], maps[s][0]) for s in maps if s != sample])
    dens = np.exp(-0.5 * ((PRIOR_GRID[:, None] - others[None, :]) / PRIOR_SIGMA) ** 2).sum(1)
    peaks_ct = [float(PRIOR_GRID[i]) for i in decode_topk_peaks_real_t(dens, PRIOR_GRID, k)]
    ct, at = maps[sample]
    return [float(np.interp(c, ct, at)) for c in peaks_ct]


def row_prior_points_k(maps: dict, gt_axis: dict, sample: str, k: int, window: int = 1,
                       min_images: int = 5) -> list:
    """Row-space prior from the other images with a similar ring count (|n − k| ≤ window, widened)."""
    if k <= 0:
        return []
    others = [s for s in maps if s != sample]
    w = window
    while True:
        pool = [s for s in others if abs(len(gt_axis[s]) - k) <= w]
        if len(pool) >= min(min_images, len(others)):
            break
        w += 1
    sub = {s: maps[s] for s in pool} | {sample: maps[sample]}
    return row_prior_points(sub, {s: gt_axis[s] for s in sub}, sample, k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab-dir", default="experiments/density_head_lab/raw")
    ap.add_argument("--arms", default="B6")
    ap.add_argument("--verdict-heads", default="B6_seed0,B6_seed2,B6_seed4")
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
    keys = {(h["backbone_from"], h["checkpoint_sha"]) for h in heads}
    if len(keys) != 1:
        sys.exit("Głowice z różnych cache — uruchom osobno.")
    key = keys.pop()
    print(f"E11 — {len(heads)} głowic: {[h['name'] for h in heads]}", flush=True)
    cfg = load_merged_config(Path(args.config), None)
    seg = cfg.segmentation.as_params()
    max_gap_t = cfg.candidates.min_peak_distance / N_SAMPLES_AXIS
    ps = cfg.data.patch_size
    backbones = {key: load_lab_backbone(cfg, key[0], key[1], None)}
    ann = ev.load_expert_annotations()
    samples = sorted(ann["Sample"].unique())
    if args.limit:
        samples = samples[: args.limit]

    # pass 1: tokens and geometry of every image (one backbone forward each)
    data, ref = {}, heads[0]
    for n, s in enumerate(samples, 1):
        p = prepare_sample(ann, s, cfg, seg, max_gap_t)
        if isinstance(p, str):
            continue
        cropped, axis_info, gt_t, _kk, _ss = p
        bands, bt, bth, btis, locs, tokens = band_inputs(cropped, axis_info, cfg, backbones)
        _d, dec_t, valid, to_axis_t = lab_head_map(ref, tokens[key], bt, bth, btis, locs, bands, axis_info, ps,
                                                   radial_on_axis=True)
        axis_t = np.array([to_axis_t(i) for i in range(len(dec_t))])
        data[s] = {"gt": gt_t, "tok": tokens[key], "bt": bt, "bth": bth, "btis": btis, "locs": locs,
                   "bands": bands, "axis_info": axis_info, "dec_t": dec_t, "valid": valid, "axis_t": axis_t,
                   "tol": ring_tolerances(gt_t, max_gap_t, FRACTION[args.tolerance])}
        print(f"  [{n}/{len(samples)}] {s}", flush=True)
    names = list(data)
    maps = {s: row_mapping(d["dec_t"], d["valid"], d["axis_t"]) for s, d in data.items()}
    gt_axis = {s: np.asarray(d["gt"], float) for s, d in data.items()}
    gt_all = {s: d["gt"] for s, d in data.items()}

    def head_dens(h, d):
        dens, _dt, valid, _f = lab_head_map(h, d["tok"], d["bt"], d["bth"], d["btis"], d["locs"], d["bands"],
                                            d["axis_info"], ps, radial_on_axis=True)
        return dens, valid

    rows = []
    for h in heads:
        dens_all = {s: head_dens(h, d) for s, d in data.items()}
        for i, s in enumerate(names):
            d = data[s]
            dens, valid = dens_all[s]
            at = d["axis_t"]
            k_or, k_ct = len(d["gt"]), int(round(float((dens * valid).sum())))
            rec = {"head": h["name"], "arm": h["arm"].name, "Sample": s, "n_gt": k_or}
            for mode, k in (("oracle", k_or), ("count", k_ct)):
                own = [at[j] for j in decode_peaks(dens, d["dec_t"], valid, "oracle", k)]
                rec[f"{mode}_n"] = len(own)
                rec[f"{mode}_head_hits"] = int(hits(own, d["gt"], d["tol"]).sum())
                rec[f"{mode}_rowprior_hits"] = int(hits(row_prior_points(maps, gt_axis, s, len(own)),
                                                        d["gt"], d["tol"]).sum())
                rec[f"{mode}_axisprior_hits"] = int(hits(population_prior_points(gt_all, s, len(own)),
                                                         d["gt"], d["tol"]).sum())
                rec[f"{mode}_kprior_hits"] = int(hits(row_prior_points_k(maps, gt_axis, s, len(own)),
                                                      d["gt"], d["tol"]).sum())
                donor_hits = []
                for step in range(1, N_DONORS + 1):
                    dd, dv = dens_all[names[(i + step) % len(names)]]
                    pts = [at[j] for j in decode_peaks(dd, d["dec_t"], dv, "oracle", len(own))]
                    donor_hits.append(hits(pts, d["gt"], d["tol"]).sum())
                rec[f"{mode}_donor_hits"] = float(np.mean(donor_hits))
            rows.append(rec)
        print(f"  {h['name']}: gotowe", flush=True)

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    out = []
    for name, sub in df.groupby("head", sort=False):
        g = lambda c: sub[c].to_numpy(float)
        r = {"head": name, "arm": sub["arm"].iloc[0]}
        for mode in ("oracle", "count"):
            den = g("n_gt") if mode == "oracle" else g(f"{mode}_n")
            for m in ("head", "rowprior", "axisprior", "kprior", "donor"):
                r[f"{mode}_{m}"] = round(g(f"{mode}_{m}_hits").sum() / max(den.sum(), 1), 3)
            for m in ("rowprior", "axisprior", "kprior", "donor"):
                r[f"{mode}_head_minus_{m}_ci95"] = boot_ci(g(f"{mode}_head_hits"), g(f"{mode}_{m}_hits"), den, rng)
        out.append(r)
    table = pd.DataFrame(out)
    vh = [x.strip() for x in args.verdict_heads.split(",") if x.strip()]
    sel = table[table["head"].isin(vh)]
    lo = lambda col: sel[col].map(lambda c: c[0])
    hi = lambda col: sel[col].map(lambda c: c[1])
    n = len(sel)
    a_noedge = int(((lo("count_head_minus_rowprior_ci95") <= 0)).sum())
    b_nodiff = int(((lo("oracle_head_minus_donor_ci95") <= 0) & (hi("oracle_head_minus_donor_ci95") >= 0)).sum())
    a_edge = int((lo("count_head_minus_rowprior_ci95") > 0).sum())
    b_edge = int((lo("oracle_head_minus_donor_ci95") > 0).sum())
    need = math.ceil(2 / 3 * n) if n else 0
    if n and a_noedge >= need and b_nodiff >= need:
        verdict = "H-pasmo potwierdzona"
    elif n and a_edge >= need and b_edge >= need:
        verdict = "H-pasmo odrzucona"
    else:
        verdict = "niejednoznaczne"
    c_edge = int((lo("count_head_minus_kprior_ci95") > 0).sum())
    c_edge_or = int((lo("oracle_head_minus_kprior_ci95") > 0).sum())
    summary = {"verdict_heads": vh, "head_vs_kprior_ci_gt0_count": f"{c_edge}/{n}",
               "head_vs_kprior_ci_gt0_oracle": f"{c_edge_or}/{n}",
               "verdict_E12": ("głowica wnosi coś ponad liczbę i położenie" if n and c_edge >= need
                               else "liczba + wzorzec położeń wyjaśnia przewagę"),
               "head_vs_rowprior_ci_gt0": f"{a_edge}/{n}",
               "own_vs_donor_ci_gt0": f"{b_edge}/{n}", "verdict": verdict}
    stem = f"zegar_band_position_{args.tolerance}" + (f"_{args.tag}" if args.tag else "") \
        + (f"_limit{args.limit}" if args.limit else "")
    table.to_csv(lab_dir / f"{stem}.csv", index=False)
    df.to_csv(lab_dir / f"{stem}_per_image.csv", index=False)
    (lab_dir / f"{stem}.json").write_text(json.dumps({"tolerance": args.tolerance, "n_donors": N_DONORS,
                                                      "summary": summary, "heads": out}, indent=2, default=str),
                                          encoding="utf-8")
    pd.set_option("display.width", 250)
    print("\n" + table.to_string(index=False))
    print("\n", json.dumps(summary, ensure_ascii=False))
    print(f"\nZapisano: {lab_dir / f'{stem}.json'}")


if __name__ == "__main__":
    main()
