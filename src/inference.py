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

from scripts.prepare_labels import extract_campaign_token
from src.config import OtolithConfig
from src.dataset import decode_age_ordinal
from src.report_common import compute_metrics
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

    With ``cfg.inference.dump_coral_logits`` (default False) each record also carries
    ``coral_logit_00 .. coral_logit_{K-2}`` — the raw boundary logits before the sigmoid.
    Off by default so existing outputs stay byte-identical.

    With ``cfg.inference.aggregate_per_fish`` (default False) a second file is written,
    ``output_dir/predictions_per_fish.csv``, holding one age per fish; the returned summary
    then also carries ``n_fish``, ``mean_mae_fish`` and ``exact_fish``. ``predictions.csv``
    is unchanged in either case.

    With ``cfg.inference.rebase_quarter_offset`` (default False) and the quarter adjustment
    enabled, ``predicted_age`` / ``target_age`` (and so every metric) are on the recorded-age
    scale, and the raw ring-scale values are kept in ``predicted_rings`` / ``target_rings``.
    See :func:`quarter_offsets`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.training.device)
    model.to(device)
    model.eval()

    records: List[Dict] = []

    K = cfg.model.num_age_classes
    dump_logits = bool(getattr(cfg.inference, "dump_coral_logits", False))
    per_fish = bool(getattr(cfg.inference, "aggregate_per_fish", False))
    logit_rows: List[np.ndarray] = []      # only filled when per_fish is on
    rebase = quarter_rebase_active(cfg)
    if cfg.data.quarter_age_adjustment_enabled and not rebase:
        print("  UWAGA: quarter_age_adjustment_enabled=True bez inference.rebase_quarter_offset "
              "— predicted_age to liczba stref, nie wiek zapisany (metryki nieporownywalne)")

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
                offset = int(quarter_offsets(cfg, [batch["image_id"][i]])[0]) if rebase else 0

                record: Dict = {
                    "image_id": batch["image_id"][i],
                    "predicted_age": pred_age + offset,
                    "target_age": None,
                    "abs_error": None,
                    "metadata_used": bool(cfg.model.use_metadata),
                }
                if rebase:
                    record["predicted_rings"] = pred_age
                    record["target_rings"] = None

                # Optional raw CORAL logits (24.09). Without them predictions.csv holds
                # only the decoded integer, so no threshold/calibration/top-k analysis is
                # possible after the fact. Column k is `g - theta_k`; since the thetas are
                # fixed per model, column 0 is an affine image of the scalar `g` itself —
                # which is why every possible decoder is just a partition of one axis.
                if dump_logits and "coral_logits" in out:
                    for k in range(out["coral_logits"].shape[-1]):
                        record[f"coral_logit_{k:02d}"] = round(
                            float(out["coral_logits"][i, k].item()), 6)
                if per_fish and "coral_logits" in out:
                    logit_rows.append(out["coral_logits"][i].detach().cpu().numpy())

                if has_labels:
                    target_age = int(batch["age"][i].item())
                    record["target_age"] = target_age + offset
                    record["abs_error"] = abs(pred_age - target_age)
                    if rebase:
                        record["target_rings"] = target_age

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

    if per_fish and logit_rows:
        fish_summary = _write_per_fish(cfg, df, np.stack(logit_rows), output_dir)
        summary.update(fish_summary)
        if fish_summary.get("mean_mae_fish") is not None:
            print(f"  per ryba: n={fish_summary['n_fish']}  "
                  f"MAE={fish_summary['mean_mae_fish']:.3f}  "
                  f"exact={100 * fish_summary['exact_fish']:.2f}%")

    return summary


def quarter_rebase_active(cfg: OtolithConfig) -> bool:
    """True when predictions must be moved from the ring scale back to recorded age."""
    return bool(getattr(cfg.inference, "rebase_quarter_offset", False)) and bool(
        cfg.data.quarter_age_adjustment_enabled)


def quarter_offsets(cfg: OtolithConfig, image_ids) -> np.ndarray:
    """Season offset per image: 1 for the quarter-adjusted campaigns, else 0.

    Inverse of ``OtolithDataset._effective_age`` (same campaign parser, same campaign list),
    so ``ring_scale + offset`` is the recorded age. The only case it cannot invert is the
    ``max(age - 1, 0)`` clamp of a recorded age-0 Q1 fish; the Embedded labels have none
    (checked on outputs/06.08_attention_first: target_rings + offset == recorded for all
    1105 test images). Every error is unchanged by the rebase (the same offset lands on
    prediction and target); what it fixes is the scale reported to the user and compared
    across runs.
    """
    campaigns = set(cfg.data.quarter_age_adjustment_campaigns)
    return np.asarray([1 if extract_campaign_token(str(i)) in campaigns else 0
                       for i in image_ids], dtype=int)


def _fish_keys_for(cfg: OtolithConfig, image_ids: pd.Series) -> Optional[pd.Series]:
    """``image_id -> fish key`` from the labels CSV, or None when it cannot be resolved.

    Read from ``cfg.data.labels_csv`` rather than from the loader, because the loader may be
    a ``Subset`` and because the label file is the authoritative source of the fish grouping
    that the train/val/test split itself was built on
    (``scripts/prepare_labels.py::assign_split_by_fish``).
    """
    path = Path(cfg.data.labels_csv)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[1]
        path = root / path
    if not path.exists():
        return None
    try:
        lab = pd.read_csv(path, usecols=["image_id", "neutral_fish_key"])
    except (ValueError, KeyError):
        return None
    mapping = lab.set_index("image_id")["neutral_fish_key"]
    mapped = image_ids.map(mapping)
    return None if mapped.isna().all() else mapped


def _write_per_fish(cfg: OtolithConfig, df: pd.DataFrame, logits: np.ndarray,
                    output_dir: Path) -> Dict:
    """One age per fish: mean CORAL logits over its photos, then decode once.

    Averaging the LOGITS is the same thing as averaging the scalar ``g`` the head computes,
    because every logit is ``g - theta_k`` and the thetas are constant — so this halves the
    variance of the decision variable. Averaging the two already-decoded integers cannot do
    that, and rounding their mean is not even well defined for a 0.5 split.
    """
    fish = _fish_keys_for(cfg, df["image_id"])
    if fish is None:
        print("  per ryba: pominiete — brak neutral_fish_key w labels_csv")
        return {"n_fish": 0, "mean_mae_fish": None, "median_mae_fish": None,
                "exact_fish": None,
                "per_fish_note": "no neutral_fish_key resolved from labels_csv"}

    n_unmapped = int(fish.isna().sum())
    if n_unmapped:
        print(f"  per ryba: {n_unmapped} z {len(fish)} zdjec bez neutral_fish_key "
              f"w labels_csv — pominiete w agregacji")

    work = pd.DataFrame({"fish_id": fish.values, "target_age": df["target_age"].values})
    for k in range(logits.shape[1]):
        work[f"_l{k}"] = logits[:, k]
    lcols = [f"_l{k}" for k in range(logits.shape[1])]
    grouped = work.groupby("fish_id", dropna=True)
    agg = grouped[lcols].mean()
    agg["n_images"] = grouped.size()
    # A fish has one age; take the first non-null label of the group.
    agg["target_age"] = grouped["target_age"].first()

    pred = decode_age_ordinal(torch.from_numpy(agg[lcols].to_numpy(dtype=np.float32)))
    pred = pred.numpy().astype(int)
    rebase = quarter_rebase_active(cfg)
    if rebase:
        # target_age in df is already on the recorded scale; both photos of a fish come
        # from the same haul, so the first image's campaign gives the fish's offset.
        first_iid = pd.Series(df["image_id"].values, index=work.index).groupby(
            work["fish_id"], dropna=True).first().reindex(agg.index)
        offset = quarter_offsets(cfg, first_iid.values)
    out = pd.DataFrame({
        "fish_id": agg.index.astype(str),
        "predicted_age": pred + offset if rebase else pred,
        "target_age": agg["target_age"].values,
        "n_images": agg["n_images"].values.astype(int),
    })
    if rebase:
        out["predicted_rings"] = pred
    out["abs_error"] = (out["predicted_age"] - out["target_age"]).abs()
    out.to_csv(output_dir / "predictions_per_fish.csv", index=False)

    labelled = out.dropna(subset=["target_age"])
    if labelled.empty:
        return {"n_fish": int(len(out)), "mean_mae_fish": None,
                "median_mae_fish": None, "exact_fish": None}
    m = compute_metrics(labelled["target_age"].values, labelled["predicted_age"].values)
    return {"n_fish": int(len(out)), "mean_mae_fish": m["MAE"],
            "median_mae_fish": m["MedAE"], "exact_fish": m["Exact"],
            "n_images_without_fish_key": n_unmapped}


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
