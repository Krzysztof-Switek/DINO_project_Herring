"""Short, cheap fine-tune of ONLY ``density_head`` at the (optionally cropped) higher
resolution used for report/localization inference (``candidates.density_image_size`` /
``candidates.density_crop_to_otolith``, Zmiana A+B, 22.07) — backbone/CORAL/MIL stay
frozen; ``density`` is already stop-gradient in ``OtolithModel.forward()``, so this can
only ever move ``density_head``'s own weights (zero risk to age accuracy by construction).

Motivated by a measured train/inference resolution mismatch: on a 30-card local
comparison (see ``plans and summaries/22.07_TO_DO.MD``), the raw ``density`` signal got
WORSE (+9.4% mean_dist_final_to_classical_px) when fed the bigger/cropped grid than
``density_head`` was actually trained on, even though the fused ``dp`` method still
improved. This fine-tune targets exactly that gap.

Usage:
    python scripts/finetune_density_head.py \\
        --checkpoint outputs/20.07_reg/checkpoints/embedded/best.pt \\
        --output outputs/20.07_reg/checkpoints/embedded/best_density_hires.pt \\
        --image-dir "Z:/Photo/Otolithes/HER/Processed"

Only Embedded (see [[embedded_only_scope]] in project memory) — ``--labels`` defaults to
``data/labels_embedded.csv``.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune density_head only, at candidates.density_image_size/crop")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--config-embedded", default="configs/config_embedded.yaml")
    p.add_argument("--checkpoint", required=True, help="Input checkpoint to fine-tune from")
    p.add_argument("--output", required=True, help="Output checkpoint path (never overwrites input)")
    p.add_argument("--labels", default="data/labels_embedded.csv")
    p.add_argument("--image-dir", default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from scripts.run_pipeline import load_merged_config
    from src.dataset import DensityFineTuneDataset
    from src.inference import load_model_from_checkpoint
    from src.model import density_count_loss
    from src.utils import resolve_device, seed_everything

    cfg = load_merged_config(PROJECT_ROOT / args.config, PROJECT_ROOT / args.config_embedded)
    if args.image_dir:
        cfg.data.image_dir = args.image_dir
    cfg.data.labels_csv = args.labels
    if not cfg.model.use_density_head:
        raise SystemExit("cfg.model.use_density_head must be True — nothing to fine-tune")
    if not cfg.data.mask_background:
        raise SystemExit("cfg.data.mask_background must be True — DensityFineTuneDataset requires it")

    seed_everything(cfg.project.seed)
    device = resolve_device(cfg.training.device)

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = load_model_from_checkpoint(cfg, args.checkpoint)
    model.to(device)
    if not hasattr(model, "density_head"):
        raise SystemExit("Loaded model has no density_head")

    # Freeze EVERYTHING except density_head. density is already stop-gradient in
    # forward() (patches.detach()) — this freeze is a belt-and-suspenders guarantee
    # that backbone/CORAL/MIL literally cannot move, not just "shouldn't".
    for p in model.parameters():
        p.requires_grad = False
    for p in model.density_head.parameters():
        p.requires_grad = True
    model.eval()                  # backbone/CORAL/MIL: frozen semantics
    model.density_head.train()    # only density_head's own Dropout active

    size = cfg.candidates.density_image_size or cfg.data.image_size
    crop = cfg.candidates.density_crop_to_otolith
    pad = cfg.candidates.density_crop_pad_frac
    print(f"density_image_size={size} crop_to_otolith={crop} pad_frac={pad}", flush=True)

    train_ds = DensityFineTuneDataset(cfg, split="train", density_image_size=size,
                                      crop_to_otolith=crop, pad_frac=pad)
    val_ds = DensityFineTuneDataset(cfg, split="val", density_image_size=size,
                                    crop_to_otolith=crop, pad_frac=pad)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers)
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

    optimizer = torch.optim.AdamW(model.density_head.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = copy.deepcopy(model.density_head.state_dict())
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.density_head.train()
        train_loss_sum, n = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            ages = batch["age"].to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feats = model.backbone.forward_features(images)
                patches = feats["x_norm_patchtokens"]
            density = torch.sigmoid(model.density_head(patches).squeeze(-1))
            loss = density_count_loss(density, ages, cfg.model.density_conc_weight,
                                      cfg.model.density_tv_weight)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
            n += images.size(0)
        train_loss = train_loss_sum / max(n, 1)

        model.density_head.eval()
        val_loss_sum, val_n, active_sum, age_sum = 0.0, 0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                ages = batch["age"].to(device)
                feats = model.backbone.forward_features(images)
                density = torch.sigmoid(model.density_head(feats["x_norm_patchtokens"]).squeeze(-1))
                loss = density_count_loss(density, ages, cfg.model.density_conc_weight,
                                          cfg.model.density_tv_weight)
                bs = images.size(0)
                val_loss_sum += loss.item() * bs
                active_sum += (density > 0.5).sum(dim=1).float().sum().item()
                age_sum += ages.float().sum().item()
                val_n += bs
        val_loss = val_loss_sum / max(val_n, 1)
        density_active = active_sum / max(val_n, 1)
        mean_age = age_sum / max(val_n, 1)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"density_active={density_active:.2f} mean_age={mean_age:.2f}", flush=True)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.density_head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stop @ epoch {epoch} (best_val_loss={best_val:.4f})", flush=True)
                break

    model.density_head.load_state_dict(best_state)

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt["model_state_dict"] = model.state_dict()
    ckpt["density_finetune"] = {
        "density_image_size": size,
        "density_crop_to_otolith": crop,
        "base_checkpoint": str(args.checkpoint),
        "best_val_density_loss": best_val,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
