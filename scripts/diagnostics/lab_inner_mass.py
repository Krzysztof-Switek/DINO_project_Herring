"""01.10 — a training-side detector of the "nucleus texture" failure of lab row heads.

In about a quarter of the seeds (A5 s3, B1 s4, B2 s1, B2 s3) the head puts 36–79 % of its ZEGAR points
in the inner third of the radius, where the experts put none. ZEGAR must not be used to reject
heads (it is the test set), so this measures the same thing on the cache's VALIDATION split:

    inner_mass = Σ p over rows with t < 1/3  /  Σ p over all rows      (tissue-weighted, per image)

reported as the mean over val images. Row t is the contour-normalised radius of the canvas row
(the evaluation's axis t differs by the axis/contour ratio, close enough for a 1/3 split).
Each val batch is read from the memmap once and fed to every head.

07.10 — second detector, the "band seam" shortcut: the 44 rows are 4 band canvases stacked
(t 0–0.6, 0.6–0.8, 0.8–0.9, 0.9–1), and some heads put their points on the last/first row of
adjacent bands (on ZEGAR: B1 39–86 % of points, B6 seeds 1 and 3 79 % / 22 %, B6 seeds 0, 2, 4
0 %; 6 of 44 rows are seam rows, 14 % by chance). Measured the same way on the val split:

    seam_mass = Σ p over seam rows  /  Σ p over all rows

    python scripts/diagnostics/lab_inner_mass.py --arms A5,B1,B2,B6
Writes (merges into) <lab-dir>/inner_mass_val.csv; with --zegar-report it joins the ZEGAR inner-third share.
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
import pandas as pd
import torch

from density_head_lab import CACHE_ROOT, TokenCache, head_forward, is_row_head
from expert_annotation_eval_wedge import collect_lab_heads, load_lab_head

INNER = 1.0 / 3.0


@torch.no_grad()
def inner_mass(logits: torch.Tensor, row_t: torch.Tensor, row_valid: torch.Tensor) -> torch.Tensor:
    """(B,) share of each image's tissue-weighted Σp lying in rows with t < 1/3."""
    p = torch.sigmoid(logits) * row_valid
    return (p * (row_t < INNER)).sum(1) / p.sum(1).clamp(min=1e-8)


def seam_rows(shapes) -> list[int]:
    """Row indices of the last row of each band and the first row of the next one."""
    from density_head_lab import row_layout
    starts, r0 = [], 0
    for _off, R, _C in row_layout(shapes):
        starts.append(r0)
        r0 += R
    return sorted({i for s in starts[1:] for i in (s - 1, s)})


@torch.no_grad()
def seam_mass(logits: torch.Tensor, row_valid: torch.Tensor, seams: list[int]) -> torch.Tensor:
    """(B,) share of each image's tissue-weighted Σp lying on band-seam rows."""
    p = torch.sigmoid(logits) * row_valid
    return p[:, seams].sum(1) / p.sum(1).clamp(min=1e-8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="raw")
    ap.add_argument("--arms", default="A5")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--lab-dir", default=None,
                    help="directory with <arm>_seed<s>/ (default experiments/density_head_lab/<cache>)")
    ap.add_argument("--zegar-report", nargs="*", default=[],
                    help="zegar_report_*.json files whose inner_third_share to join")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    lab_dir = (Path(args.lab_dir) if args.lab_dir
               else PROJECT_ROOT / "experiments" / "density_head_lab" / args.cache)
    if not lab_dir.is_absolute():
        lab_dir = PROJECT_ROOT / lab_dir
    heads = [load_lab_head(p) for p in collect_lab_heads(lab_dir, [a.strip() for a in args.arms.split(",")], False)]
    heads = [h for h in heads if is_row_head(h["arm"])]
    cache = TokenCache(CACHE_ROOT / args.cache, "quarter")
    val = cache.rows_of("val")
    sums = {h["name"]: [] for h in heads}
    seam = {h["name"]: [] for h in heads}
    for i in range(0, len(val), args.batch_size):
        b = cache.batch(val[i:i + args.batch_size], torch.device("cpu"))
        for h in heads:
            logits, row_valid = head_forward(h["arm"], h["head"], b)
            row_t, _ = h["head"].rows(b["t"], b["valid"])
            sums[h["name"]].append(inner_mass(logits, row_t, row_valid))
            seam[h["name"]].append(seam_mass(logits, row_valid, seam_rows(h["head"].shapes)))
        print(f"  {min(i + args.batch_size, len(val))}/{len(val)}", flush=True)
    rows = []
    for h in heads:
        m = torch.cat(sums[h["name"]]).numpy()
        rows.append({"head": h["name"], "arm": h["arm"].name, "val_inner_mass": round(float(m.mean()), 4),
                     "val_images_inner_gt_half": round(float((m > 0.5).mean()), 4),
                     "val_seam_mass": round(float(torch.cat(seam[h["name"]]).mean()), 4)})
    df = pd.DataFrame(rows)
    zeg = {}
    for path in args.zegar_report:
        for r in json.loads(Path(path).read_text(encoding="utf-8"))["heads"]:
            zeg[r["head"]] = r["inner_third_share"]
    if zeg:
        df["zegar_inner_third_share"] = df["head"].map(zeg)
    out = lab_dir / "inner_mass_val.csv"
    if out.exists():                      # merge: later runs update their heads, keep the others
        old = pd.read_csv(out)
        df = pd.concat([old[~old["head"].isin(df["head"])], df], ignore_index=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))
    print(f"\nZapisano: {lab_dir / 'inner_mass_val.csv'}")


if __name__ == "__main__":
    main()
