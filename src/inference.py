"""Inference: run predictions, collect results, save CSV and JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import OtolithConfig
from src.dataset import decode_age_ordinal
from src.model import OtolithModel
from src.utils import resolve_device


def run_inference(
    cfg: OtolithConfig,
    model: OtolithModel,
    loader: DataLoader,
    output_dir: str | Path,
) -> Dict:
    """Run model inference on a DataLoader.

    Saves per-sample results to:
        output_dir/predictions.csv
        output_dir/predictions.json

    Returns a summary dict with n_samples, mean_mae, median_mae.
    target_age and abs_error are None when the dataset has no labels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.training.device)
    model.to(device)
    model.eval()

    records: List[Dict] = []

    K = cfg.model.num_age_classes

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            metadata = batch.get("metadata")
            if metadata is not None:
                metadata = metadata.to(device)

            out = model(images, metadata=metadata)
            # Prefer CORAL when present (continuity with prior reports);
            # fall back to the MIL count for mil-only models. With the top-k
            # concentration loss the age is the number of ACTIVE patches
            # (prob>0.5) ≈ age, not the (slightly inflated) sum of probs.
            if "coral_logits" in out:
                pred_ages = decode_age_ordinal(out["coral_logits"])
            else:
                pred_ages = (out["patch_probs"] > 0.5).sum(dim=1).long().clamp(0, K - 1)

            has_labels = "age" in batch
            batch_size = images.size(0)

            for i in range(batch_size):
                pred_age = int(pred_ages[i].item())

                record: Dict = {
                    "image_id": batch["image_id"][i],
                    "predicted_age": pred_age,
                    "target_age": None,
                    "abs_error": None,
                    "metadata_used": bool(cfg.model.use_metadata),
                }

                if has_labels:
                    target_age = int(batch["age"][i].item())
                    record["target_age"] = target_age
                    record["abs_error"] = abs(pred_age - target_age)

                records.append(record)

    # ------------------------------------------------------------------ #
    # Save CSV
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(records)
    csv_path = output_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------ #
    # Save JSON
    # ------------------------------------------------------------------ #
    json_path = output_dir / "predictions.json"
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    errors = [r["abs_error"] for r in records if r["abs_error"] is not None]
    mean_mae   = float(np.mean(errors))   if errors else None
    median_mae = float(np.median(errors)) if errors else None

    summary = {
        "n_samples": len(records),
        "mean_mae": mean_mae,
        "median_mae": median_mae,
    }

    print(f"Inference complete: {len(records)} samples")
    if mean_mae is not None:
        print(f"  MAE  mean={mean_mae:.3f}  median={median_mae:.3f}")

    return summary


def load_model_from_checkpoint(
    cfg: OtolithConfig,
    checkpoint_path: str | Path,
    backbone: Optional[nn.Module] = None,
) -> OtolithModel:
    """Create OtolithModel and restore weights from a saved checkpoint.

    A backbone can be injected (useful for tests with a mock backbone).
    The model is returned in eval mode on the configured device.
    """
    model = OtolithModel(cfg, backbone=backbone)
    device = resolve_device(cfg.training.device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = dict(ckpt["model_state_dict"])
    model_shapes = model.state_dict()
    # Drop keys whose SHAPE no longer matches the current architecture — e.g. inserting a
    # layer inside a sub-module (density_head's LayerNorm, 22.07) shifts every subsequent
    # key's shape. Plain strict=False only tolerates missing/unexpected KEYS, not a shape
    # mismatch on a key present in both, and would otherwise hard-crash instead of falling
    # back to random init for just the changed sub-module (same spirit as the missing-key
    # case below — that sub-module needs retraining, the rest of the checkpoint is fine).
    shape_mismatched = [k for k in state_dict
                        if k in model_shapes and state_dict[k].shape != model_shapes[k].shape]
    for k in shape_mismatched:
        del state_dict[k]
    # strict=False allows loading old checkpoints that don't have patch_head
    # (the new MIL head will then be randomly initialised — retraining needed).
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys or shape_mismatched:
        import warnings
        warnings.warn(
            "Checkpoint loaded non-strictly: "
            f"{len(result.missing_keys)} missing key(s), "
            f"{len(result.unexpected_keys)} unexpected key(s), "
            f"{len(shape_mismatched)} shape-mismatched key(s) dropped. "
            "Affected layers are randomly initialised — retrain before trusting outputs. "
            f"(missing={result.missing_keys[:5]}, unexpected={result.unexpected_keys[:5]}, "
            f"shape_mismatched={shape_mismatched[:5]})",
            RuntimeWarning,
            stacklevel=2,
        )
    model.to(device)
    model.eval()
    return model
