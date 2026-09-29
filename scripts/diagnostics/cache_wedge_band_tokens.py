"""30.09 (L5) — cache the backbone patch tokens of the four wedge bands, once per backbone.

WHY. The density branch reads the band canvases through the backbone under ``torch.no_grad()``
(`src/model.py`, ``density_image_bands`` path), so its input is a deterministic function of
the image and of the backbone weights. Four backbone passes per image dominate the ~5 h epoch
of the wedge run; with the tokens stored, a density-head experiment only runs the head. This is
what makes the density-head lab (arms A0–A5 × 5 seeds, `plans and summaries/
29.09_ewdge_b_przerwany_PLAN_TO_do.md` §5) affordable — the same pattern as
`cache_cls_features.py` for the quarter A/B of the CORAL head.

TWO VARIANTS, one script:
  (a) ``--backbone-from raw``       — pretrained ``dinov2_vits14_reg``, never fine-tuned;
  (b) ``--backbone-from <ckpt.pt>`` — the backbone of a trained checkpoint, e.g. wedge_b's
      ``best_age.pt`` (epoch 15, tuned for age).
Comparing a lab arm on (a) vs (b) separates "the head input drifts while the backbone is tuned
for age" from everything else.

DETERMINISM. Band canvases come from the production ``OtolithDataset.
_build_wedge_bands_tensors_and_polar`` with the random horizontal flip switched off, so every
split gets the same, reproducible canvases (the val transform is used for normalisation).

LABELS. Both the recorded age and the quarter-adjusted age (``_effective_age`` with the
adjustment ON) are stored; a lab arm picks which one it counts.

    python scripts/diagnostics/cache_wedge_band_tokens.py --tag raw
    python scripts/diagnostics/cache_wedge_band_tokens.py --tag wedge_b_best_age \\
        --backbone-from outputs/22.09_wedge_b/checkpoints/embedded/best_age.pt

Writes ``data/wedge_band_tokens/<tag>/``:
  tokens.npy        float16 (N, P, D)  — concatenated band patch tokens, P = sum of band patches
  polar_t.npy       float32 (N, P)     — radial position t of each token
  polar_theta.npy   float32 (N, P)     — angular position of each token
  tissue_valid.npy  float16 (N, P)     — tissue fraction per token (density_count_loss mask)
  done.npy          bool    (N,)       — resume marker, flushed every ``--flush-every`` images
  index.csv, meta.json
The run resumes where it stopped; a cache with a different identity is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

CACHE_ROOT = PROJECT_ROOT / "data" / "wedge_band_tokens"
BASE_CONFIG = PROJECT_ROOT / "configs" / "config_wedge_b.yaml"
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"
IMAGE_DIR_LOCAL = "Z:/Photo/Otolithes/HER/Processed"

# Config fields that change the canvases or the tokens; recorded and checked on reuse.
IDENTITY_FIELDS = ("model.backbone", "data.patch_size", "data.mask_background",
                   "data.wedge_delta_theta_deg", "data.wedge_band_edges_t",
                   "data.wedge_band_n_angle_patches", "data.wedge_band_n_radius_patches")


def resolve_image_dir(explicit: Optional[str]) -> str:
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


def file_sha256(path: Path, limit: int = 1 << 24) -> str:
    """SHA-256 of the first ``limit`` bytes — enough to tell two checkpoints apart cheaply."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def build_cfg(labels_csv: Path, image_dir: str):
    from scripts.run_pipeline import load_merged_config
    cfg = load_merged_config(BASE_CONFIG, None)
    cfg.data.labels_csv = str(labels_csv)
    cfg.data.image_dir = image_dir
    cfg.data.num_workers = 0
    cfg.training.device = "auto"
    return cfg


def band_layout(cfg) -> list[int]:
    """Patches per band, in concatenation order (same order as the model's torch.cat)."""
    return [a * r for a, r in zip(cfg.data.wedge_band_n_angle_patches,
                                  cfg.data.wedge_band_n_radius_patches)]


def load_backbone(cfg, backbone_from: str):
    """Production backbone: raw pretrained, or the ``backbone.*`` weights of a checkpoint.

    Returns ``(backbone, identity_dict)``. A checkpoint is loaded into a full ``OtolithModel``
    built from the checkpoint's own config with ``strict=True``, so a partial or mismatched
    state dict fails loudly instead of leaving random weights behind.
    """
    from src.config import OtolithConfig
    from src.model import OtolithModel
    if backbone_from == "raw":
        model = OtolithModel(cfg)
        return model.backbone, {"backbone_from": "raw", "checkpoint_sha": None,
                                "checkpoint_epoch": None}
    path = Path(backbone_from)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck_cfg = OtolithConfig(**ck["cfg"])
    if ck_cfg.model.backbone != cfg.model.backbone:
        raise SystemExit(f"checkpoint backbone {ck_cfg.model.backbone} != config {cfg.model.backbone}")
    model = OtolithModel(ck_cfg)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    # POSIX form so the identity is the same on the Windows workstation and the Linux server.
    shown = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    return model.backbone, {"backbone_from": shown.as_posix(),
                            "checkpoint_sha": file_sha256(path),
                            "checkpoint_epoch": int(ck.get("epoch", -1))}


def meta_for(cfg, backbone_identity: dict) -> dict:
    return ({f: dotted(cfg, f) for f in IDENTITY_FIELDS} | backbone_identity
            | {"canvases": "production _build_wedge_bands_tensors_and_polar, no flip",
               "dtype": "float16"})


def check_identity(meta_path: Path, meta: dict) -> None:
    if not meta_path.exists():
        return
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    mismatch = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
    if mismatch:
        raise SystemExit(f"Cache at {meta_path.parent} was built with a different identity: "
                         f"{mismatch}\nUse a different --tag rather than mixing tokens.")


class CacheWriter:
    """Pre-allocated .npy memmaps + a ``done`` marker, so an interrupted run resumes."""

    def __init__(self, out_dir: Path, n: int, n_patches: int, dim: int):
        from numpy.lib.format import open_memmap
        self.out_dir = out_dir
        shapes = {"tokens": ((n, n_patches, dim), np.float16),
                  "polar_t": ((n, n_patches), np.float32),
                  "polar_theta": ((n, n_patches), np.float32),
                  "tissue_valid": ((n, n_patches), np.float16),
                  "done": ((n,), np.bool_)}
        self.arr = {}
        for name, (shape, dtype) in shapes.items():
            p = out_dir / f"{name}.npy"
            if p.exists():
                a = np.load(p, mmap_mode="r+")
                if a.shape != shape or a.dtype != dtype:
                    raise SystemExit(f"{p} has shape {a.shape}/{a.dtype}, expected {shape}/{dtype}")
            else:
                a = open_memmap(p, mode="w+", dtype=dtype, shape=shape)
            self.arr[name] = a

    @property
    def done(self) -> np.ndarray:
        return self.arr["done"]

    def write(self, i: int, tokens: np.ndarray, t: np.ndarray, theta: np.ndarray,
              valid: np.ndarray) -> None:
        self.arr["tokens"][i] = tokens
        self.arr["polar_t"][i] = t
        self.arr["polar_theta"][i] = theta
        self.arr["tissue_valid"][i] = valid
        self.arr["done"][i] = True

    def flush(self) -> None:
        for a in self.arr.values():
            a.flush()


def fill_cache(writer: CacheWriter, items: Iterable[tuple[int, list]],
               backbone_fn: Callable[[torch.Tensor], torch.Tensor],
               batch_size: int = 8, flush_every: int = 64,
               log: Callable[[str], None] = print, total: Optional[int] = None) -> int:
    """Run the backbone over band canvases and store the concatenated patch tokens.

    ``items`` yields ``(row_index, bands)`` where ``bands`` is the dataset's list of
    ``(image, polar_t, polar_theta, tissue_valid)`` per band. Bands of the same index share a
    canvas size across images, so each band is batched on its own and the per-band tokens are
    concatenated in band order — exactly the model's ``torch.cat(band_tokens, dim=1)``.
    Returns the number of images written.
    """
    pending: list[tuple[int, list]] = []
    written = 0
    t0 = time.time()

    def run() -> None:
        nonlocal written
        n_bands = len(pending[0][1])
        per_band = []
        for b in range(n_bands):
            imgs = torch.stack([bands[b][0] for _i, bands in pending])
            with torch.no_grad():
                per_band.append(backbone_fn(imgs).float().cpu())
        toks = torch.cat(per_band, dim=1).numpy()
        for j, (i, bands) in enumerate(pending):
            t = np.concatenate([bd[1].reshape(-1).numpy() for bd in bands])
            th = np.concatenate([bd[2].reshape(-1).numpy() for bd in bands])
            v = np.concatenate([bd[3].reshape(-1).numpy() for bd in bands])
            writer.write(i, toks[j].astype(np.float16), t, th, v.astype(np.float16))
        written += len(pending)
        pending.clear()
        if written % flush_every < batch_size:
            writer.flush()
            rate = written / max(time.time() - t0, 1e-9)
            log(f"  {written}{'/' + str(total) if total else ''}  {rate:.2f} img/s")

    for i, bands in items:
        pending.append((i, bands))
        if len(pending) == batch_size:
            run()
    if pending:
        run()
    writer.flush()
    return written


def label_rows(ds, split: str) -> pd.DataFrame:
    """image_id, split, recorded age, quarter-adjusted age, campaign, fish key — in ds order."""
    from scripts.prepare_labels import extract_campaign_token
    campaigns = set(ds.cfg.data.quarter_age_adjustment_campaigns)
    rows = []
    for _, r in ds.df.iterrows():
        iid = str(r["image_id"])
        rec = int(r["age"])
        camp = extract_campaign_token(iid)
        rows.append({"image_id": iid, "split": split, "age_recorded": rec,
                     "age_quarter": max(rec - 1, 0) if camp in campaigns else rec,
                     "campaign": camp,
                     "fish_key": r.get("neutral_fish_key", None)})
    return pd.DataFrame(rows)


def _no_flip() -> bool:
    """Module-level (not a lambda) so the dataset still pickles into DataLoader workers."""
    return False


def deterministic_dataset(cfg, split: str):
    """Production dataset whose band canvases are reproducible for every split.

    Two random elements of the train split are switched off: the shared horizontal flip
    (``_decide_wedge_hflip``) and the ColorJitter inside the band transform, which the dataset
    builds from the split name. Missing the second one made cached tokens differ from a fresh
    pass by up to 46 % of their range (found by the cache-vs-model check on real images).
    """
    from src.dataset import OtolithDataset, build_transforms
    ds = OtolithDataset(cfg, split=split, transform=build_transforms(cfg.data.image_size, "val"))
    ds._decide_wedge_hflip = _no_flip
    ds.wedge_band_transform_no_flip = build_transforms(1, "val", include_flips=False, wedge=True)
    return ds


class _BandItems(torch.utils.data.Dataset):
    """Picklable wrapper so band extraction can run in DataLoader workers."""

    def __init__(self, ds, image_ids: list[str], rows: list[int]):
        self.ds, self.image_ids, self.rows = ds, image_ids, rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, k: int):
        return self.rows[k], self.ds._build_wedge_bands_tensors_and_polar(self.image_ids[k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="cache name, e.g. raw / wedge_b_best_age")
    ap.add_argument("--backbone-from", default="raw", help="'raw' or a checkpoint path")
    ap.add_argument("--labels", default="data/labels_embedded.csv")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers for band extraction")
    ap.add_argument("--flush-every", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke run)")
    args = ap.parse_args()

    from src.utils import resolve_device

    cfg = build_cfg(PROJECT_ROOT / args.labels, resolve_image_dir(args.image_dir))
    backbone, bb_identity = load_backbone(cfg, args.backbone_from)
    meta = meta_for(cfg, bb_identity)
    out_dir = CACHE_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    check_identity(meta_path, meta)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    datasets = {s: deterministic_dataset(cfg, s) for s in splits}
    index = pd.concat([label_rows(datasets[s], s) for s in splits], ignore_index=True)
    if args.limit:
        index = index.iloc[: args.limit].reset_index(drop=True)
    index_path = out_dir / "index.csv"
    if index_path.exists():
        old = pd.read_csv(index_path)
        if list(old["image_id"]) != list(index["image_id"]):
            raise SystemExit(f"{index_path} lists different images; use a different --tag.")
    index.to_csv(index_path, index=False)

    layout = band_layout(cfg)
    device = resolve_device(cfg.training.device)
    backbone = backbone.to(device).eval()
    dim = int(backbone.embed_dim)
    meta_path.write_text(json.dumps(meta | {"n": int(len(index)), "n_patches": int(sum(layout)),
                                            "band_patches": layout, "dim": dim,
                                            "splits": splits}, indent=2), encoding="utf-8")

    writer = CacheWriter(out_dir, len(index), sum(layout), dim)
    todo = np.flatnonzero(~writer.done)
    print(f"CACHE IDENTITY  tag={args.tag}  device={device}  n={len(index)}  "
          f"patches={sum(layout)} {layout}  dim={dim}")
    for k, v in meta.items():
        print(f"  {k} = {v}")
    print(f"  do zrobienia: {len(todo)} / {len(index)}", flush=True)

    def backbone_fn(imgs: torch.Tensor) -> torch.Tensor:
        return backbone.forward_features(imgs.to(device))["x_norm_patchtokens"]

    t0 = time.time()
    for split in splits:
        rows = [int(i) for i in todo if index.at[i, "split"] == split]
        if not rows:
            continue
        print(f"\n[{split}] {len(rows)} obrazów", flush=True)
        items = _BandItems(datasets[split], [index.at[i, "image_id"] for i in rows], rows)
        loader = torch.utils.data.DataLoader(items, batch_size=None, shuffle=False,
                                             num_workers=args.workers,
                                             persistent_workers=False)
        fill_cache(writer, loader, backbone_fn, batch_size=args.batch_size,
                   flush_every=args.flush_every, total=len(rows),
                   log=lambda m: print(m, flush=True))
    n_done = int(writer.done.sum())
    print(f"\nzapisano {out_dir}  gotowe {n_done}/{len(index)}  w {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
