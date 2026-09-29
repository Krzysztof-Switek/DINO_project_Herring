"""24.09 — re-score an existing checkpoint on val AND test, dumping raw CORAL logits.

NO TRAINING. One forward pass per split with weights that already exist. This is the
input every later decoding experiment needs, because `src/inference.py` used to consume
`out["coral_logits"]` and discard it — so nothing on disk supports threshold search,
calibration, top-k ranking or per-fish logit averaging without a fresh pass.

It also fixes a second gap: the pipeline only ever scored `split="test"`
(`scripts/run_pipeline.py::_step_infer`), so **no validation predictions exist for any
run**. Tuning a decoder on test would be cheating; this script produces the val file that
makes honest tuning possible.

WHERE TO RUN THIS: on the server. The local machine has no image share mounted
(`Z:` absent) and a CPU-only torch build.

    python scripts/diagnostics/coral_score_splits.py \
        --run 06.08_attention_first --ckpt best.pt --splits val,test

Writes, per split:  experiments/logits/<run>__<ckpt>/<split>/predictions.{csv,json}

The checkpoint's own embedded `cfg` is authoritative for architecture (it is what the
weights were trained with); only paths and the localization branches are overridden --
see `prepare_cfg`. Plan: "plans and summaries/24.09_CORAL_plan.md" (Etap 1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"
IMAGE_DIR_LOCAL = "Z:/Photo/Otolithes/HER/Processed"

# Fields whose value changes what the CORAL head sees, so they are printed and asserted
# rather than trusted. Named after the RUN IDENTITY line added to the trainer on 22.09,
# which exists because a run once silently used pure defaults instead of its config.
IDENTITY_FIELDS = (
    "model.backbone", "model.num_age_classes", "model.head_type", "model.use_metadata",
    "data.image_size", "data.patch_size", "data.mask_background",
    "data.quarter_age_adjustment_enabled", "data.quarter_age_adjustment_campaigns",
)


def resolve_image_dir(explicit: str | None) -> str:
    """Pick the image share: explicit flag, else whichever of the two paths exists."""
    if explicit:
        return explicit
    for cand in (IMAGE_DIR_SERVER, IMAGE_DIR_LOCAL):
        if Path(cand).exists():
            return cand
    raise SystemExit(
        "No image directory found. Pass --image-dir explicitly.\n"
        f"  tried: {IMAGE_DIR_SERVER}\n         {IMAGE_DIR_LOCAL}"
    )


def dotted(cfg, path: str):
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def prepare_cfg(ckpt_cfg: dict, labels_csv: Path, image_dir: str):
    """Rebuild the run's config, then strip the localization branches for scoring.

    Why it is safe to strip them: at inference the CORAL head reads only the CLS token of
    the full 518 px square image, and the density branch is a separate forward pass under
    `torch.no_grad()` on wedge/strip canvases (`src/model.py:949, 985`). Dropping it
    changes no input to CORAL — it only avoids rebuilding the wedge/band caches, which for
    the bands geometry is four extra backbone passes and four cached PNGs per image.

    `mask_background` is deliberately NOT touched: it changes the pixels CORAL sees.
    `quarter_age_adjustment_*` is NOT touched either: it defines the target this model was
    trained against, and silently flipping it would score the model on a different task.

    The claim that stripping is inert is verified, not assumed — `verify_against_production`
    re-decodes the test split and demands a row-for-row match with the run's own
    predictions.csv.
    """
    from src.config import OtolithConfig
    cfg = OtolithConfig(**ckpt_cfg)
    cfg.data.labels_csv = str(labels_csv)
    cfg.data.image_dir = image_dir
    cfg.inference.dump_coral_logits = True
    # Localization off for speed; CORAL-neutral by construction, verified below.
    cfg.model.use_density_head = False
    cfg.data.dual_branch_wedge = False      # wedge / angular-band canvases
    cfg.data.dual_branch_density = False    # dendro-strip canvases
    cfg.data.multi_wycinek_k = 1            # one wycinek per sample (the default)
    cfg.data.num_workers = 0
    return cfg


def coral_thresholds(model) -> np.ndarray:
    """The learned boundary thresholds theta_0 < ... < theta_{K-2}.

    Reconstructed the same way `src/model.py::_coral_logits` builds them:
    ``theta = theta0`` then ``theta0 + cumsum(softplus(gaps))``. Saved alongside the
    logits so a later analysis can recover the scalar ``g`` itself (``g = logit_k +
    theta_k``) instead of working with an arbitrary affine image of it.
    """
    with torch.no_grad():
        gaps = torch.nn.functional.softplus(model.coral_gaps)
        theta = torch.cat([model.coral_theta0,
                           model.coral_theta0 + torch.cumsum(gaps, dim=0)])
    return theta.detach().cpu().numpy().astype(float)


def verify_against_production(out_csv: Path, run: str) -> dict:
    """Test-split gate: re-decoding the dump at 0.5 must reproduce the recorded run.

    If this fails, the scoring config differs from the one that produced the run's numbers
    and every downstream comparison would be against a different model. Returns a report
    dict instead of raising, so the caller decides how loud to be.
    """
    prod = PROJECT_ROOT / "outputs" / run / "emb_on_emb" / "predictions.csv"
    if not prod.exists():
        return {"checked": False, "reason": f"no production predictions at {prod}"}
    a = pd.read_csv(out_csv).set_index("image_id")
    b = pd.read_csv(prod).set_index("image_id")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"checked": False, "reason": "no overlapping image_id"}
    cols = sorted(c for c in a.columns if c.startswith("coral_logit_"))
    probs = torch.sigmoid(torch.from_numpy(a.loc[common, cols].to_numpy())).numpy()
    redecoded = (probs > 0.5).sum(axis=1)
    agree_dump = int((redecoded == a.loc[common, "predicted_age"].to_numpy()).sum())
    agree_prod = int((a.loc[common, "predicted_age"].to_numpy()
                      == b.loc[common, "predicted_age"].to_numpy()).sum())
    return {
        "checked": True, "n_common": int(len(common)),
        "redecode_matches_dump": agree_dump,
        "matches_production": agree_prod,
        "ok": agree_dump == len(common) and agree_prod == len(common),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir name under outputs/")
    ap.add_argument("--ckpt", default="best.pt", help="best.pt | best_age.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--labels", default="data/labels_embedded.csv")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.inference import load_model_from_checkpoint, run_inference
    from src.dataset import OtolithDataset

    run_dir = PROJECT_ROOT / "outputs" / args.run
    ckpt_path = run_dir / "checkpoints" / "embedded" / args.ckpt
    if not ckpt_path.exists():
        alt = run_dir / "checkpoints_embedded" / args.ckpt
        if alt.exists():
            ckpt_path = alt
        else:
            raise SystemExit(f"checkpoint not found: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = blob.get("cfg") or {}
    if not ckpt_cfg:
        raise SystemExit(f"{ckpt_path} carries no embedded cfg — cannot rebuild the model")

    labels_csv = PROJECT_ROOT / args.labels
    cfg = prepare_cfg(ckpt_cfg, labels_csv, resolve_image_dir(args.image_dir))

    print(f"SCORING IDENTITY  run={args.run}  ckpt={args.ckpt}  epoch={blob.get('epoch')}",
          flush=True)
    for f in IDENTITY_FIELDS:
        print(f"  {f} = {dotted(cfg, f)}", flush=True)

    model = load_model_from_checkpoint(cfg, ckpt_path)
    thetas = coral_thresholds(model)

    out_root = Path(args.out) if args.out else (
        PROJECT_ROOT / "experiments" / "logits" / f"{args.run}__{Path(args.ckpt).stem}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "coral_thresholds.json").write_text(
        json.dumps({"theta": list(thetas),
                    "note": "g = coral_logit_k + theta[k]; identical for every k"},
                   indent=2), encoding="utf-8")

    report: dict = {"run": args.run, "ckpt": args.ckpt, "epoch": blob.get("epoch"),
                    "splits": {}}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ds = OtolithDataset(cfg, split=split)
        loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=0)
        out_dir = out_root / split
        print(f"\n[{split}] n={len(ds)} -> {out_dir}")
        summary = run_inference(cfg, model, loader, out_dir)
        report["splits"][split] = summary
        if split == "test":
            gate = verify_against_production(out_dir / "predictions.csv", args.run)
            report["splits"][split]["verification"] = gate
            print(f"  weryfikacja wzgledem produkcyjnego predictions.csv: {gate}")
            if gate.get("checked") and not gate.get("ok"):
                print("  !! NIEZGODNOSC — config scoringu rozni sie od tego, ktory dal "
                      "liczby biegu. Nie buduj na tym dumpie zadnych porownan.")

    (out_root / "scoring_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_root}")


if __name__ == "__main__":
    main()
