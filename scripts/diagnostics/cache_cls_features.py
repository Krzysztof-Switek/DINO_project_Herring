"""24.09 — cache the DINOv2 CLS token for every labelled Embedded image, once.

WHY. The CORAL age head reads ONLY the CLS token (`src/model.py:925-929`). With the backbone
frozen the CLS token is a deterministic function of the image, so it can be computed once and
reused for an unlimited number of head-level experiments that then take seconds instead of a
20-hour fine-tuning run. That is what makes a properly seeded A/B affordable at all.

WHAT IT BUYS, AND WHAT IT DOES NOT. Any comparison run on this cache isolates the head
perfectly: identical features, identical architecture, only the thing under test differs. It
does NOT reproduce the fine-tuned regime — production unfreezes the backbone at epoch 6
(`training.freeze_backbone_epochs: 5`), and a backbone that adapts to a changed target can
gain more than the head alone. So results here are a clean lower bound plus a direction, and a
winner still has to survive a full run.

DETERMINISM. Features are extracted with the VAL transform for every split (resize, to-tensor,
ImageNet normalise) — no flips, no colour jitter — so the cache is reproducible. Background
masking follows the config, because it changes the pixels the backbone sees.

    python scripts/diagnostics/cache_cls_features.py --tag reg_masked_518

Writes  data/cls_cache/<tag>/{features.npy, index.csv, meta.json}.
A cache is refused for reuse unless its meta.json matches the requested identity, so features
from two different preprocessings can never silently mix.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

CACHE_ROOT = PROJECT_ROOT / "data" / "cls_cache"
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"
IMAGE_DIR_LOCAL = "Z:/Photo/Otolithes/HER/Processed"

# Anything here changes the features, so it is recorded and checked on reuse.
IDENTITY_FIELDS = ("model.backbone", "data.image_size", "data.patch_size",
                   "data.mask_background")


def resolve_image_dir(explicit: str | None) -> str:
    if explicit:
        return explicit
    for cand in (IMAGE_DIR_SERVER, IMAGE_DIR_LOCAL):
        if Path(cand).exists():
            return cand
    raise SystemExit("No image directory found; pass --image-dir.")


def dotted(cfg, path: str):
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def build_cfg(labels_csv: Path, image_dir: str, backbone: str, image_size: int,
              mask_background: bool):
    """A config for feature extraction only: age head, no localisation branches."""
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.data.labels_csv = str(labels_csv)
    cfg.data.image_dir = image_dir
    cfg.data.image_size = image_size
    cfg.data.mask_background = mask_background
    cfg.data.num_workers = 0
    cfg.model.backbone = backbone
    cfg.model.head_type = "coral"
    cfg.model.use_density_head = False
    cfg.training.device = "auto"
    return cfg


def deterministic_dataset(cfg, split: str):
    """The split's dataset with the VAL transform forced, whatever the split is.

    `OtolithDataset` picks its transform from the split name, so a train-split dataset would
    apply random flips and colour jitter and the cache would not be reproducible. The
    constructor takes an explicit transform, so the production class is reused as-is rather
    than reimplemented.
    """
    from src.dataset import OtolithDataset, build_transforms
    tf = build_transforms(cfg.data.image_size, "val")
    return OtolithDataset(cfg, split=split, transform=tf)


def meta_for(cfg) -> dict:
    return {f: dotted(cfg, f) for f in IDENTITY_FIELDS} | {"transform": "val (deterministic)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="reg_masked_518")
    ap.add_argument("--labels", default="data/labels_embedded.csv")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--backbone", default="dinov2_vits14_reg")
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--no-mask", action="store_true",
                    help="extract WITHOUT background masking (production masks by default)")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    from src.inference import load_model_from_checkpoint  # noqa: F401  (import check)
    from src.model import OtolithModel
    from src.utils import resolve_device

    cfg = build_cfg(PROJECT_ROOT / args.labels, resolve_image_dir(args.image_dir),
                    args.backbone, args.image_size, not args.no_mask)
    out_dir = CACHE_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_for(cfg)

    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        mismatch = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if mismatch:
            raise SystemExit(
                f"Cache '{args.tag}' was built with a different identity: {mismatch}\n"
                f"Use a different --tag rather than mixing features.")

    device = resolve_device(cfg.training.device)
    print(f"CACHE IDENTITY  tag={args.tag}  device={device}")
    for k, v in meta.items():
        print(f"  {k} = {v}")

    model = OtolithModel(cfg).to(device)
    model.eval()

    feats: list[np.ndarray] = []
    rows: list[dict] = []
    t0 = time.time()
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ds = deterministic_dataset(cfg, split)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"\n[{split}] n={len(ds)}", flush=True)
        done = 0
        with torch.no_grad():
            for batch in loader:
                out = model.backbone.forward_features(batch["image"].to(device))
                cls = out["x_norm_clstoken"].detach().cpu().numpy().astype(np.float32)
                feats.append(cls)
                for i in range(cls.shape[0]):
                    rows.append({"image_id": batch["image_id"][i], "split": split,
                                 "age": int(batch["age"][i].item())})
                done += cls.shape[0]
                if done % 320 == 0:
                    rate = done / max(time.time() - t0, 1e-9)
                    print(f"  {done}/{len(ds)}  {rate:.2f} img/s", flush=True)

    X = np.concatenate(feats, axis=0)
    index = pd.DataFrame(rows)
    np.save(out_dir / "features.npy", X)
    index.to_csv(out_dir / "index.csv", index=False)
    meta_path.write_text(json.dumps(meta | {"n": int(len(index)), "dim": int(X.shape[1])},
                                    indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir}  features {X.shape}  in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
