"""07.10 — E8 (plan 30.09 §7): a training-free spectral decoder on the row similarity structure.

Idea from TokenCut / Normalized Cut (Shi & Malik 2000; Wang et al. 2022), moved from "object vs
background" to "cut the radius into consecutive stretches": rings are where the look of the
canvas rows changes. For one image the 44 canvas rows (tissue rows only, ordered nucleus → edge)
get an embedding, rows are compared by cosine similarity, and the row sequence is cut into k + 1
CONTIGUOUS segments maximising Shi & Malik's normalised association

    Σ_segments  assoc(A, A) / assoc(A, V),     W = max(cos, 0) after centring the rows,

solved by dynamic programming (contiguity makes the 1-D cut tractable; exact without the
spacing rule, which is checked against each state's best predecessor). The k boundaries,
each placed midway between the last row of one segment and the first row of the next, are the
ring positions. Consecutive boundaries keep the greedy decoder's minimum spacing
(DEFAULT_MIN_GAP_T, in the rows' own t).

Two embeddings, both fixed before looking at ZEGAR:
  head  the B6 head's own row embedding, normalize(proj(pool(tokens))) — the representation its
        self-similarity features are computed from (trained on age only);
  raw   tissue-weighted mean of the frozen DINOv2 tokens over each row — no training at all.
Rows are centred (the image's mean row embedding subtracted) before the cosine, because DINOv2
tokens share a large common component that makes every cosine ≈ 0.9.

k as everywhere: oracle (experts' count, sposób 1) and count (the head's own Σp, sposób 2).
Compared on the same image with the same k:
  greedy  the lab decoder on the head's map (what zegar_report.py scores);
  prior   population prior — k most frequent ring positions on the other 41 images (kontrola 2);
  rule    gaps shrinking 0.8× outward (kontrola 1).
Criterion (plan §2.3): precision(ncut) − precision(prior) with bootstrap CI95 > 0 in ≥ 4/5 seeds.
Nothing here is tuned on ZEGAR.

    python scripts/diagnostics/zegar_ncut_decoder.py --arms B6
Writes <lab-dir>/zegar_ncut_decoder_<tolerance>[_<tag>].{json,csv} + _per_image.csv.
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
import torch.nn.functional as F

import expert_annotation_eval as ev
from density_head_lab import row_layout
from expert_annotation_eval_wedge import (N_SAMPLES_AXIS, collect_lab_heads, decode_peaks, geometric_t,
                                          lab_head_map, load_lab_backbone, load_lab_head,
                                          prepare_sample, ring_tolerances)
from scripts.run_pipeline import load_merged_config
from src.peak_decoding import DEFAULT_MIN_GAP_T
from zegar_report import FRACTION, band_inputs, boot_ci, hits, population_prior_points

VALID_ROW = 0.5               # a row takes part when at least half of it is tissue


# ---------------------------------------------------------------------------
# The cut
# ---------------------------------------------------------------------------

def affinity(e: np.ndarray) -> np.ndarray:
    """W = max(cos, 0) between centred rows of e (n, d)."""
    z = e - e.mean(0, keepdims=True)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return np.clip(z @ z.T, 0.0, None)


def ncut_boundaries(W: np.ndarray, t: np.ndarray, k: int,
                    min_gap: float = DEFAULT_MIN_GAP_T) -> list[int]:
    """Start indices of segments 2 … k+1 of the best contiguous (k+1)-way normalised-association
    cut of the ordered rows (t increasing). Fewer than k when k boundaries do not fit min_gap."""
    n = len(t)
    if k <= 0 or n < 2:
        return []
    k = min(k, n - 1)
    deg = W.sum(1)
    P = np.zeros((n + 1, n + 1))
    P[1:, 1:] = W.cumsum(0).cumsum(1)
    dcum = np.concatenate([[0.0], deg.cumsum()])
    # score[i, m] = assoc(A, A) / assoc(A, V) for the segment rows i … m−1
    i_idx, m_idx = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    inner = P[m_idx, m_idx] - P[i_idx, m_idx] - P[m_idx, i_idx] + P[i_idx, i_idx]
    vol = dcum[m_idx] - dcum[i_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where((m_idx > i_idx) & (vol > 0), inner / vol, -np.inf)
    score = np.where((m_idx > i_idx) & (vol <= 0), 0.0, score)      # an all-zero segment scores 0
    # boundary position of a segment starting at row s (s ≥ 1): midway between rows s−1 and s
    bpos = np.full(n + 1, np.nan)
    bpos[1:n] = 0.5 * (t[:-1] + t[1:])
    neg = -np.inf
    # best[j][m]: rows 0 … m−1 cut into j+1 segments, the last ending at m (exclusive)
    best = np.full((k + 1, n + 1), neg)
    arg = np.zeros((k + 1, n + 1), int)
    best[0, 1:] = score[0, 1:]
    for j in range(1, k + 1):
        for m in range(j + 1, n + 1):
            cand = np.full(m, neg)
            for i in range(j, m):                       # last segment = rows i … m−1, i ≥ j
                if not np.isfinite(best[j - 1, i]):
                    continue
                if j >= 2:                              # spacing to the previous boundary
                    prev = arg[j - 1, i]
                    if bpos[i] - bpos[prev] < min_gap - 1e-12:
                        continue
                cand[i] = best[j - 1, i] + score[i, m]
            if np.isfinite(cand).any():
                arg[j, m] = int(np.argmax(cand))
                best[j, m] = cand[arg[j, m]]
    if not np.isfinite(best[k, n]):
        return ncut_boundaries(W, t, k - 1, min_gap)
    starts, m = [], n
    for j in range(k, 0, -1):
        i = arg[j, m]
        starts.append(i)
        m = i
    return sorted(starts)


# ---------------------------------------------------------------------------
# Row embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def head_rows(head, tokens: torch.Tensor, tissue: torch.Tensor) -> np.ndarray:
    """B6's own row embedding (44, proj_dim)."""
    z = head.pool(tokens, tissue)
    return F.normalize(head.proj(z), dim=-1)[0].numpy()


@torch.no_grad()
def raw_rows(tokens: torch.Tensor, tissue: torch.Tensor, shapes) -> np.ndarray:
    """Tissue-weighted mean DINOv2 token per canvas row (44, D) — no learned weights."""
    out = []
    for off, R, C in row_layout(shapes):
        x = tokens[0, off:off + R * C].reshape(R, C, -1)
        w = tissue[0, off:off + R * C].reshape(R, C, 1).clamp(min=1e-6)
        out.append((x * w).sum(1) / w.sum(1))
    return torch.cat(out, 0).numpy()


def decode_ncut(e: np.ndarray, dec_t: np.ndarray, valid: np.ndarray, k: int, to_axis_t) -> list:
    """Ring positions (axis t) from the cut of the tissue rows."""
    rows = np.flatnonzero(valid >= VALID_ROW)
    rows = rows[np.argsort(dec_t[rows], kind="stable")]
    if len(rows) < 2 or k <= 0:
        return []
    starts = ncut_boundaries(affinity(e[rows]), dec_t[rows], k)
    return [0.5 * (to_axis_t(int(rows[s - 1])) + to_axis_t(int(rows[s]))) for s in starts]


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab-dir", default="experiments/density_head_lab/raw")
    ap.add_argument("--arms", default="B6")
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
    heads = [h for h in heads if h["arm"].head == "rows_selfsim"]
    if not heads:
        sys.exit("Brak głowic B6 (rows_selfsim).")
    print(f"E8 — dekoder Normalized Cut, {len(heads)} głowic: {[h['name'] for h in heads]}", flush=True)
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
        tissue = torch.cat(btis, 1)
        for h in heads:
            tok = tokens[(h["backbone_from"], h["checkpoint_sha"])]
            dens, dec_t, valid, to_axis_t = lab_head_map(h, tok, bt, bth, btis, locs, bands,
                                                         axis_info, ps, radial_on_axis=True)
            emb = {"head": head_rows(h["head"], tok, tissue), "raw": raw_rows(tok, tissue, h["head"].shapes)}
            rec = {"head": h["name"], "arm": h["arm"].name, "Sample": sample, "n_gt": len(gt_t)}
            for mode in ("oracle", "count"):
                g_idx = decode_peaks(dens, dec_t, valid, mode, len(gt_t))
                k = len(gt_t) if mode == "oracle" else int(round(float((dens * valid).sum())))
                pts = {"greedy": [to_axis_t(i) for i in g_idx],
                       "prior": population_prior_points(gt_all, sample, k),
                       "rule": geometric_t(k),
                       **{f"ncut_{name}": decode_ncut(e, dec_t, valid, k, to_axis_t) for name, e in emb.items()}}
                rec[f"{mode}_k"] = k
                for name, p in pts.items():
                    rec[f"{mode}_{name}_n"] = len(p)
                    rec[f"{mode}_{name}_hits"] = int(hits(p, gt_t, tol).sum())
            rows.append(rec)
        print(f"  [{n_img}/{len(samples)}] {sample}", flush=True)

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    methods = ("greedy", "ncut_head", "ncut_raw", "prior", "rule")
    out = []
    for name, sub in df.groupby("head", sort=False):
        g = lambda c: sub[c].to_numpy(float)
        r = {"head": name, "arm": sub["arm"].iloc[0]}
        for mode in ("oracle", "count"):
            # oracle: coverage = hits / expert rings; count: precision = hits / points placed
            for m in methods:
                den = g("n_gt") if mode == "oracle" else g(f"{mode}_{m}_n")
                r[f"{mode}_{m}"] = round(g(f"{mode}_{m}_hits").sum() / max(den.sum(), 1), 3)
            den_ref = g("n_gt") if mode == "oracle" else g(f"{mode}_k")
            for m in ("ncut_head", "ncut_raw"):
                r[f"{mode}_{m}_minus_prior_ci95"] = boot_ci(g(f"{mode}_{m}_hits"), g(f"{mode}_prior_hits"), den_ref, rng)
                r[f"{mode}_{m}_minus_greedy_ci95"] = boot_ci(g(f"{mode}_{m}_hits"), g(f"{mode}_greedy_hits"), den_ref, rng)
        out.append(r)
    table = pd.DataFrame(out)
    verdict = {}
    for m in ("ncut_head", "ncut_raw"):
        for mode in ("oracle", "count"):
            lo = table[f"{mode}_{m}_minus_prior_ci95"].map(lambda c: c[0])
            n_pos = int((lo > 0).sum())
            verdict[f"{mode}_{m}"] = {"ci_prior_gt0": f"{n_pos}/{len(table)}",
                                      "pass": n_pos >= math.ceil(0.8 * len(table))}
    stem = f"zegar_ncut_decoder_{args.tolerance}" + (f"_{args.tag}" if args.tag else "") \
        + (f"_limit{args.limit}" if args.limit else "")
    table.to_csv(lab_dir / f"{stem}.csv", index=False)
    df.to_csv(lab_dir / f"{stem}_per_image.csv", index=False)
    (lab_dir / f"{stem}.json").write_text(json.dumps(
        {"tolerance": args.tolerance,
         "params": {"affinity": "max(cos, 0) of centred rows", "objective": "normalised association",
                    "valid_row": VALID_ROW, "min_gap": DEFAULT_MIN_GAP_T},
         "verdict": verdict, "heads": out}, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print("\n" + table.to_string(index=False))
    print("\nWerdykt (CI95 ncut − rozkład populacyjny > 0):", json.dumps(verdict, ensure_ascii=False))
    print(f"\nZapisano: {lab_dir / f'{stem}.json'}")


if __name__ == "__main__":
    main()
