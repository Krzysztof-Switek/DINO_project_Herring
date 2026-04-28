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

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            metadata = batch.get("metadata")
            if metadata is not None:
                metadata = metadata.to(device)
            logits = model(images, metadata=metadata)
            pred_ages = decode_age_ordinal(logits)   # (B,) int

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
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model
