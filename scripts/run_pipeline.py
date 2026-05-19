"""Full Embedded vs NotEmbedded pipeline orchestrator.

Steps:
  1  scan     — scan image dir + build labels CSVs
  2  train_e  — train Embedded model
  3  train_n  — train NotEmbedded model
  4  infer_ee — infer Embedded model on Embedded test set
  5  infer_nn — infer NotEmbedded model on NotEmbedded test set
  6  infer_en — CROSS: infer Embedded model on NotEmbedded test set
  7  infer_ne — CROSS: infer NotEmbedded model on Embedded test set
  8  cards    — generate increment annotation cards
  9  report   — build comparison HTML report

Usage:
    python scripts/run_pipeline.py \\
        --image-dir "Z:/Photo/Otolithes/HER/Processed" \\
        --excel     data/analysisWithOtolithPhoto.xlsx \\
        --output-dir outputs/ \\
        [--base-config configs/config.yaml] \\
        [--config-embedded     configs/config_embedded.yaml] \\
        [--config-not-embedded configs/config_not_embedded.yaml] \\
        [--skip-scan]   \\
        [--skip-train]  \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time as _time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_merged_config(base_path: Path | None, override_path: Path | None):
    """Load base + override YAML files and return merged OtolithConfig."""
    from src.config import OtolithConfig

    base_raw: dict = {}
    if base_path and Path(base_path).exists():
        raw = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))
        if raw:
            base_raw = raw

    override_raw: dict = {}
    if override_path and Path(override_path).exists():
        raw = yaml.safe_load(Path(override_path).read_text(encoding="utf-8"))
        if raw:
            override_raw = raw

    merged = _deep_update(base_raw, override_raw)
    return OtolithConfig(**merged)


# ---------------------------------------------------------------------------
# Pipeline state tracking
# ---------------------------------------------------------------------------

STEPS = [
    "scan", "train_e", "train_n",
    "infer_ee", "infer_nn", "infer_en", "infer_ne",
    "cards", "report",
]


def _save_state(state_path: Path, completed: list[str]) -> None:
    state_path.write_text(
        json.dumps({"completed_steps": completed}, indent=2), encoding="utf-8"
    )


def _load_state(state_path: Path) -> list[str]:
    if not state_path.exists():
        return []
    try:
        return json.loads(state_path.read_text(encoding="utf-8")).get("completed_steps", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Individual step runners
# ---------------------------------------------------------------------------

def _step_scan(args, data_dir: Path) -> tuple[Path, Path]:
    """Scan image directory and produce labels_embedded.csv + labels_not_embedded.csv."""
    from src.scan_labels import build_combined_labels

    combined = build_combined_labels(
        image_dir=Path(args.image_dir),
        excel_path=Path(args.excel),
        train=args.train,
        val=args.val,
        seed=args.seed,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(data_dir / "labels_combined.csv", index=False)
    emb_path = data_dir / "labels_embedded.csv"
    notemb_path = data_dir / "labels_not_embedded.csv"
    combined[combined.preprocessing_type == "Embedded"].to_csv(emb_path, index=False)
    combined[combined.preprocessing_type == "NotEmbedded"].to_csv(notemb_path, index=False)
    print(f"  Saved: {emb_path}, {notemb_path}")
    return emb_path, notemb_path


def _parse_train_log(log_path: Path) -> list[dict]:
    """Parsuje train.log trainera → lista słowników per epoka.

    Format linii: [timestamp] epoch=  N  train_loss=X.XXXX  val_loss=X.XXXX  val_mae=X.XXX
    """
    if not log_path.exists():
        return []
    pattern = re.compile(
        r"epoch=\s*(\d+)\s+"
        r"train_loss=([\d.]+)\s+"
        r"val_loss=([\d.]+)\s+"
        r"val_mae=([\d.]+)"
    )
    rows: list[dict] = []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        m = pattern.search(line)
        if m:
            try:
                rows.append({
                    "epoch":      int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_loss":   float(m.group(3)),
                    "val_mae":    float(m.group(4)),
                })
            except ValueError:
                continue
    return rows


def _step_train(cfg, labels_csv: Path) -> tuple[Path, list[dict]]:
    """Train model and return (path to best checkpoint, per-epoch training logs)."""
    from src.dataset import OtolithDataset
    from src.model import OtolithModel
    from src.trainer import Trainer
    from torch.utils.data import DataLoader

    cfg_copy = cfg.model_copy(deep=True)
    cfg_copy.data.labels_csv = str(labels_csv)

    train_ds = OtolithDataset(cfg_copy, split="train")
    val_ds = OtolithDataset(cfg_copy, split="val")

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=cfg.data.num_workers)

    model = OtolithModel(cfg_copy)
    trainer = Trainer(cfg_copy, model, train_loader, val_loader)
    trainer.fit()

    # Parse training log (Trainer resolves log_dir relative to PROJECT_ROOT)
    log_dir = Path(cfg.training.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    training_log_data = _parse_train_log(log_dir / "train.log")

    best_ckpt = trainer.checkpoint_dir / "best.pt"
    if not best_ckpt.exists():
        # Fallback: checkpointy bez best.pt (wznowienie po starej wersji kodu)
        ckpt_files = sorted(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint saved in {trainer.checkpoint_dir}")
        best_ckpt_src = min(ckpt_files, key=lambda p: float(p.stem.split("_loss")[-1]))
        import shutil as _shutil
        _shutil.copy2(best_ckpt_src, best_ckpt)
    print(f"  Best checkpoint: {best_ckpt}")
    return best_ckpt, training_log_data


def _step_infer(cfg, ckpt_path: Path, labels_csv: Path, output_dir: Path) -> Path:
    """Run inference for all test samples; then interpretation+candidates for top-k only."""
    import pandas as pd
    from src.dataset import OtolithDataset
    from src.inference import run_inference, load_model_from_checkpoint
    from src.interpretation import run_interpretation
    from src.candidates import run_candidates
    from torch.utils.data import DataLoader, Subset

    cfg_copy = cfg.model_copy(deep=True)
    cfg_copy.data.labels_csv = str(labels_csv)

    # Resolve original image directory for full-resolution outputs
    image_dir = Path(cfg_copy.data.image_dir)
    if not image_dir.is_absolute():
        image_dir = PROJECT_ROOT / image_dir

    test_ds = OtolithDataset(cfg_copy, split="test")
    loader = DataLoader(test_ds, batch_size=cfg.training.batch_size,
                        shuffle=False, num_workers=cfg.data.num_workers)

    model = load_model_from_checkpoint(cfg_copy, ckpt_path)

    # --- Step A: inference on all test samples ---
    print("  Inferencja (wszystkie próbki)...")
    run_inference(cfg_copy, model, loader, output_dir)

    # --- Step B: select top-k best + top-k worst for interpretation ---
    pred_csv = output_dir / "predictions.csv"
    top_k = cfg.inference.increment_samples.top_k_best + cfg.inference.increment_samples.top_k_worst
    top_ids: set[str] = set()
    if pred_csv.exists():
        try:
            pred_df = pd.read_csv(pred_csv)
            if "age" in pred_df.columns and "predicted_age" in pred_df.columns:
                pred_df["_err"] = (pred_df["predicted_age"] - pred_df["age"]).abs()
                pred_df = pred_df.sort_values("_err")
                k_best  = cfg.inference.increment_samples.top_k_best
                k_worst = cfg.inference.increment_samples.top_k_worst
                top_ids = (
                    set(pred_df.head(k_best)["image_id"].astype(str))
                    | set(pred_df.tail(k_worst)["image_id"].astype(str))
                )
                print(f"  Interpretacja dla {len(top_ids)} próbek "
                      f"({k_best} najlepszych + {k_worst} najgorszych)")
        except Exception as e:
            print(f"  Uwaga: nie można wybrać top-k ({e}) — interpretacja dla wszystkich")

    if top_ids:
        # Build a filtered dataset with only the top-k image_ids
        top_indices = [i for i, row in test_ds.df.iterrows()
                       if str(row["image_id"]) in top_ids]
        subset_ds = Subset(test_ds, top_indices)
        interp_loader = DataLoader(subset_ds, batch_size=cfg.training.batch_size,
                                   shuffle=False, num_workers=cfg.data.num_workers)
    else:
        interp_loader = loader

    print("  Heatmapy i nakładki (oryginalna rozdzielczość)...")
    run_interpretation(cfg_copy, model, interp_loader, output_dir, image_dir=image_dir)

    print("  Kandydaci przyrostów (oryginalna rozdzielczość)...")
    run_candidates(cfg_copy, model, interp_loader, output_dir, image_dir=image_dir)

    print(f"  Predictions: {pred_csv}")
    return pred_csv


def _compute_axis_data_for_samples(
    samples: list[dict],
    image_dir: Path,
    cfg,
    ckpt_path: Path,
    cond_dir: Path,
) -> tuple[dict, dict]:
    """Compute importance grids and reasoning-card axis data for the given samples.

    For each sample we run the model once (importance grid + last-ordinal sigmoid),
    then resolve the otolith mask (cached under ``cond_dir/masks/`` if present,
    otherwise segmented on-the-fly and saved there), compute the measurement
    axis, sample the importance profile along it and detect peaks.

    Args:
        samples:    list of prediction rows (each at least ``image_id``)
        image_dir:  directory holding the original-resolution photos
        cfg:        config to drive ``compute_patch_importance``
        ckpt_path:  checkpoint to load for this condition
        cond_dir:   ``output_dir / cond_key`` — mask cache lives here

    Returns:
        grids:     ``{image_id → (H_p, W_p) ndarray}``
        axis_data: ``{image_id → {mask, axis_info, peak_indices, line_xy, profile_1d}}``
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage

    from src.candidates import find_candidate_peaks
    from src.dataset import build_transforms
    from src.inference import load_model_from_checkpoint
    from src.interpretation import compute_patch_importance
    from src.otolith_axis import (
        detect_axis,
        find_centroid,
        find_farthest_edge,
        load_mask,
        sample_profile_along_axis,
        save_mask,
    )

    grids: dict = {}
    axis_data: dict = {}

    if not samples:
        return grids, axis_data

    mask_dir = cond_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint(cfg, ckpt_path)
    model.eval()
    device = next(model.parameters()).device
    transform = build_transforms(cfg.data.image_size, "test")

    min_dist = cfg.candidates.min_peak_distance
    prominence = cfg.candidates.prominence_threshold
    n_samples_axis = 50

    missing_images = 0
    failed_axes = 0
    for row in samples:
        iid = str(row["image_id"])
        img_path = image_dir / iid
        if not img_path.exists():
            print(f"    [cards] obraz nie znaleziony: {img_path}")
            missing_images += 1
            continue
        try:
            img_pil = PILImage.open(img_path).convert("RGB")
            tensor = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                grid = compute_patch_importance(model, tensor).cpu().numpy()
            grids[iid] = grid
        except Exception as e:
            print(f"    [cards] błąd dla {iid}: {e}")
            continue

        # --- Resolve axis: cached mask → re-derive centroid/far_edge; else segment ---
        orig_rgb = np.array(img_pil, dtype=np.uint8)
        H_img, W_img = orig_rgb.shape[:2]
        stem = Path(iid).stem
        mask_path = mask_dir / f"{stem}_mask.png"

        import cv2
        axis_info = None
        mask_arr = load_mask(mask_path)
        if mask_arr is not None:
            cent = find_centroid(mask_arr)
            far = find_farthest_edge(mask_arr, cent) if cent else None
            if cent and far:
                contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    axis_info = {
                        "mask":      mask_arr,
                        "centroid":  cent,
                        "far_edge":  far,
                        "contour":   contour,
                        "length_px": float(np.hypot(far[0] - cent[0],
                                                     far[1] - cent[1])),
                    }
        if axis_info is None:
            axis_info = detect_axis(orig_rgb)
            if axis_info is not None:
                mask_arr = axis_info["mask"]
                save_mask(mask_arr, mask_path)

        if axis_info is None:
            failed_axes += 1
            axis_data[iid] = {
                "mask": None, "axis_info": None,
                "peak_indices": None, "line_xy": None, "profile_1d": None,
            }
            continue

        profile_1d, line_xy = sample_profile_along_axis(
            grid, axis_info["centroid"], axis_info["far_edge"],
            H_img, W_img, n_samples=n_samples_axis,
        )
        peak_indices = find_candidate_peaks(profile_1d, min_dist, prominence)

        axis_data[iid] = {
            "mask":         mask_arr,
            "axis_info":    axis_info,
            "peak_indices": peak_indices,
            "line_xy":      line_xy,
            "profile_1d":   profile_1d,
        }

    total = len(samples)
    print(f"    [cards] obliczono gridy dla {len(grids)}/{total} próbek "
          f"(brak obrazu: {missing_images}, segmentacja nieudana: {failed_axes})")
    if len(grids) == 0 and total > 0:
        print(f"    [cards] OSTRZEŻENIE: 0/{total} próbek przetworzonych — "
              f"sprawdź --image-dir={image_dir}")
    return grids, axis_data


def _reload_cards_from_disk(output_dir: Path) -> dict[str, list[Path]]:
    """Reconstruct ``{best, worst}`` card lists from ``output_dir/cards/**/*.png``.

    Called when the cards step is skipped (already completed in an earlier run)
    so the comparison report still gets card paths instead of an empty dict.
    """
    cards: dict[str, list[Path]] = {"best": [], "worst": []}
    cards_root = Path(output_dir) / "cards"
    if not cards_root.exists():
        return cards
    for label in ("best", "worst"):
        for png in sorted(cards_root.glob(f"**/{label}/*.png")):
            cards[label].append(png)
    return cards


def _step_cards(
    pred_csvs: dict[str, Path],
    cfg_emb,
    image_dir: Path,
    output_dir: Path,
    cond_models: dict[str, tuple],
) -> dict[str, list[Path]]:
    """Generate 6-panel reasoning cards for best/worst predictions of each condition.

    cond_models: cond_key → (cfg, ckpt_path)
    """
    from src.visualization import select_top_k_samples, save_reasoning_cards

    if not Path(image_dir).exists():
        print(f"    [cards] OSTRZEŻENIE: --image-dir nie istnieje: {image_dir} "
              f"— karty zostaną pominięte")
        return {"best": [], "worst": []}

    top_k_best = cfg_emb.inference.increment_samples.top_k_best
    top_k_worst = cfg_emb.inference.increment_samples.top_k_worst
    cards: dict[str, list[Path]] = {"best": [], "worst": []}

    for cond_key, pred_csv in pred_csvs.items():
        if not Path(pred_csv).exists():
            continue
        try:
            best, worst = select_top_k_samples(pred_csv, top_k_best, top_k_worst)
        except Exception as e:
            print(f"    [cards] {cond_key}: nie można wczytać predictions.csv ({e})")
            continue

        cfg_cond, ckpt_cond = cond_models.get(cond_key, (None, None))
        if cfg_cond is None or ckpt_cond is None or not ckpt_cond.exists():
            print(f"    [cards] {cond_key}: brak checkpointu — pomijam")
            continue

        cond_dir = output_dir / cond_key
        all_samples = list(best) + list(worst)
        importance_grids, axis_data = _compute_axis_data_for_samples(
            all_samples, image_dir, cfg_cond, ckpt_cond, cond_dir,
        )

        cards_dir = output_dir / "cards" / cond_key
        best_saved = save_reasoning_cards(
            best, image_dir, importance_grids, axis_data,
            cards_dir / "best", "best",
        )
        worst_saved = save_reasoning_cards(
            worst, image_dir, importance_grids, axis_data,
            cards_dir / "worst", "worst",
        )
        cards["best"].extend(best_saved)
        cards["worst"].extend(worst_saved)

    return cards


def _compute_dataset_stats(labels_combined_csv: Path) -> dict:
    """Compute dataset statistics from labels_combined.csv for Section A of the report."""
    import pandas as pd

    # Try pipeline output dir first, then project data/ directory
    candidates = [
        labels_combined_csv,
        PROJECT_ROOT / "data" / "labels_combined.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)

    if csv_path is None:
        return {"counts": {}, "orphan_count": "N/A", "age_distributions": {}}

    df = pd.read_csv(csv_path)

    counts: dict = {}
    for ptype in ["Embedded", "NotEmbedded"]:
        sub = df[df["preprocessing_type"] == ptype]
        counts[ptype] = {
            s: int((sub["split"] == s).sum()) for s in ["train", "val", "test"]
        }

    age_distributions: dict = {}
    for ptype in ["Embedded", "NotEmbedded"]:
        sub = df[
            (df["preprocessing_type"] == ptype)
            & (df["split"].notna())
            & (df["age"] >= 0)
        ]
        if not sub.empty:
            age_distributions[ptype] = sub["age"].dropna().astype(int).tolist()

    orphan_count = int(df["orphan"].sum()) if "orphan" in df.columns else "N/A"

    return {
        "counts": counts,
        "orphan_count": orphan_count,
        "age_distributions": age_distributions,
    }


def _write_pipeline_summary(
    output_dir: Path,
    training_logs: dict[str, list[dict]],
    results_dfs: dict,
    completed_steps: list[str],
) -> None:
    """Zapisuje pipeline_summary.json — szybka weryfikacja wyników bez otwierania HTML."""
    from src.comparison_report import compute_metrics

    training_summary: dict[str, dict] = {}
    for model_key, logs in training_logs.items():
        if not logs:
            training_summary[model_key] = {
                "epochs_completed": 0,
                "best_val_mae": None,
                "final_train_loss": None,
            }
        else:
            training_summary[model_key] = {
                "epochs_completed": len(logs),
                "best_val_mae":     round(min(r["val_mae"] for r in logs), 4),
                "final_train_loss": round(logs[-1]["train_loss"], 4),
            }

    inference_summary: dict[str, dict] = {}
    for cond_key, df in results_dfs.items():
        if df is None or df.empty:
            inference_summary[cond_key] = None
            continue
        if "age" not in df.columns or "predicted_age" not in df.columns:
            inference_summary[cond_key] = {"n_samples": len(df), "error": "missing columns"}
            continue
        m = compute_metrics(df["age"].values, df["predicted_age"].values)
        inference_summary[cond_key] = {
            "n_samples": int(len(df)),
            "MAE":    round(m["MAE"],    4),
            "RMSE":   round(m["RMSE"],   4),
            "Acc1yr": round(m["Acc1yr"], 4),
            "Bias":   round(m["Bias"],   4),
        }

    summary = {
        "generated_at":    _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps_completed": completed_steps,
        "training":        training_summary,
        "inference":       inference_summary,
    }
    out_path = output_dir / "pipeline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Pipeline summary: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OtolithDino — Embedded vs NotEmbedded pipeline")
    p.add_argument("--image-dir", default="Z:/Photo/Otolithes/HER/Processed")
    p.add_argument("--excel", default="data/analysisWithOtolithPhoto.xlsx")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--base-config", default="configs/config.yaml")
    p.add_argument("--config-embedded", default="configs/config_embedded.yaml",
                   dest="config_embedded")
    p.add_argument("--config-not-embedded", default="configs/config_not_embedded.yaml",
                   dest="config_not_embedded")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-scan", action="store_true",
                   help="Skip step 1; use existing labels CSVs in output-dir/data/")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip steps 2-3; use existing checkpoints")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned steps without executing them")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "pipeline_state.json"
    completed = _load_state(state_path)

    base_cfg_path = Path(args.base_config)
    cfg_emb = load_merged_config(base_cfg_path, Path(args.config_embedded))
    cfg_notemb = load_merged_config(base_cfg_path, Path(args.config_not_embedded))

    if args.dry_run:
        skip_set: set[str] = set()
        if args.skip_scan:
            skip_set.add("scan")
        if args.skip_train:
            skip_set.update({"train_e", "train_n"})
        print("=== DRY RUN — pipeline steps ===")
        for i, step in enumerate(STEPS, 1):
            status = "SKIP" if step in skip_set else "RUN "
            prev = "(completed)" if step in completed else ""
            print(f"  {i}. [{status}] {step} {prev}")
        return

    print("=" * 60)
    print("OtolithDino — pipeline Embedded vs NotEmbedded")
    print("=" * 60)

    data_dir = output_dir / "data"

    # --- Step 1: scan ---
    if not args.skip_scan and "scan" not in completed:
        print("\n[1/9] SCAN — budowanie labels CSVs")
        emb_labels, notemb_labels = _step_scan(args, data_dir)
        completed.append("scan")
        _save_state(state_path, completed)
    else:
        print("\n[1/9] SCAN — pominięty")
        def _resolve_labels(pipeline_path: Path, cfg_path: str) -> Path:
            if pipeline_path.exists():
                return pipeline_path
            p = Path(cfg_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            return p

        emb_labels = _resolve_labels(
            data_dir / "labels_embedded.csv", cfg_emb.data.labels_csv)
        notemb_labels = _resolve_labels(
            data_dir / "labels_not_embedded.csv", cfg_notemb.data.labels_csv)
        print(f"  Embedded labels:    {emb_labels}")
        print(f"  NotEmbedded labels: {notemb_labels}")

    # Determine checkpoint paths from config
    ckpt_emb_dir = Path(cfg_emb.training.checkpoint_dir)
    ckpt_notemb_dir = Path(cfg_notemb.training.checkpoint_dir)
    if not ckpt_emb_dir.is_absolute():
        ckpt_emb_dir = PROJECT_ROOT / ckpt_emb_dir
    if not ckpt_notemb_dir.is_absolute():
        ckpt_notemb_dir = PROJECT_ROOT / ckpt_notemb_dir
    ckpt_emb = ckpt_emb_dir / "best.pt"
    ckpt_notemb = ckpt_notemb_dir / "best.pt"

    # Training logs — populated by _step_train; empty when --skip-train or step already done
    logs_emb: list[dict] = []
    logs_notemb: list[dict] = []

    # --- Steps 2-3: training ---
    if not args.skip_train:
        if "train_e" not in completed:
            print("\n[2/9] TRAIN — Embedded")
            ckpt_emb, logs_emb = _step_train(cfg_emb, emb_labels)
            completed.append("train_e")
            _save_state(state_path, completed)
        else:
            print("\n[2/9] TRAIN Embedded — pominięty (już wykonany)")

        if "train_n" not in completed:
            print("\n[3/9] TRAIN — NotEmbedded")
            ckpt_notemb, logs_notemb = _step_train(cfg_notemb, notemb_labels)
            completed.append("train_n")
            _save_state(state_path, completed)
        else:
            print("\n[3/9] TRAIN NotEmbedded — pominięty (już wykonany)")
    else:
        print("\n[2-3/9] TRAIN — pominięty (--skip-train)")

    # --- Steps 4-7: inference + interpretation + candidates (4 conditions) ---
    conditions = [
        ("infer_ee", "emb_on_emb",          cfg_emb,    ckpt_emb,    emb_labels),
        ("infer_nn", "notemb_on_notemb",     cfg_notemb, ckpt_notemb, notemb_labels),
        ("infer_en", "cross_emb_on_notemb",  cfg_emb,    ckpt_emb,    notemb_labels),
        ("infer_ne", "cross_notemb_on_emb",  cfg_notemb, ckpt_notemb, emb_labels),
    ]
    pred_csvs: dict[str, Path] = {}
    step_nums = {"infer_ee": 4, "infer_nn": 5, "infer_en": 6, "infer_ne": 7}

    for step_name, cond_key, cfg, ckpt, labels_csv in conditions:
        n = step_nums[step_name]
        infer_dir = output_dir / cond_key
        if step_name not in completed:
            print(f"\n[{n}/9] INFER — {cond_key}")
            _step_infer(cfg, ckpt, labels_csv, infer_dir)
            completed.append(step_name)
            _save_state(state_path, completed)
        else:
            print(f"\n[{n}/9] INFER {cond_key} — pominięty (już wykonany)")
        pred_csvs[cond_key] = infer_dir / "predictions.csv"

    # --- Step 8: increment cards ---
    # Mapping: condition key → (cfg, ckpt_path) used by that model
    cond_models = {
        "emb_on_emb":          (cfg_emb,    ckpt_emb),
        "notemb_on_notemb":    (cfg_notemb, ckpt_notemb),
        "cross_emb_on_notemb": (cfg_emb,    ckpt_emb),
        "cross_notemb_on_emb": (cfg_notemb, ckpt_notemb),
    }

    if "cards" not in completed:
        print("\n[8/9] CARDS — karty rozumowania")
        image_dir = Path(args.image_dir)
        increment_cards = _step_cards(
            pred_csvs, cfg_emb, image_dir, output_dir, cond_models
        )
        completed.append("cards")
        _save_state(state_path, completed)
    else:
        print("\n[8/9] CARDS — pominięty (już wykonany); wczytuję karty z dysku")
        increment_cards = _reload_cards_from_disk(output_dir)
        n_best = len(increment_cards.get("best", []))
        n_worst = len(increment_cards.get("worst", []))
        print(f"  Wczytano: best={n_best}, worst={n_worst}")

    # --- Prepare results_dfs (needed for both report and summary) ---
    import pandas as pd
    results_dfs: dict[str, pd.DataFrame | None] = {}
    for cond_key, csv_path in pred_csvs.items():
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            if "target_age" in df.columns and "age" not in df.columns:
                df = df.rename(columns={"target_age": "age"})
            results_dfs[cond_key] = df
        else:
            results_dfs[cond_key] = None

    # --- Step 9: comparison report ---
    if "report" not in completed:
        print("\n[9/9] REPORT — raport porównawczy")
        from src.comparison_report import build_comparison_report

        model_info = {
            "backbone": cfg_emb.model.backbone,
            "num_age_classes": cfg_emb.model.num_age_classes,
            "use_metadata": cfg_emb.model.use_metadata,
            "ckpt_embedded": str(ckpt_emb),
            "ckpt_not_embedded": str(ckpt_notemb),
        }

        dataset_stats = _compute_dataset_stats(data_dir / "labels_combined.csv")

        report_path = output_dir / "comparison_report.html"
        training_logs = {"embedded": logs_emb, "not_embedded": logs_notemb}
        build_comparison_report(
            results=results_dfs,
            training_logs=training_logs,
            increment_cards=increment_cards,
            dataset_stats=dataset_stats,
            output_path=report_path,
            model_info=model_info,
        )
        completed.append("report")
        _save_state(state_path, completed)
        print(f"  Report: {report_path}")
    else:
        print("\n[9/9] REPORT — pominięty (już wykonany)")

    # --- Pipeline summary (always written) ---
    _write_pipeline_summary(
        output_dir=output_dir,
        training_logs={"embedded": logs_emb, "not_embedded": logs_notemb},
        results_dfs=results_dfs,
        completed_steps=completed,
    )

    print("\n=== Pipeline zakończony ===")
    print(f"Raport:          {output_dir / 'comparison_report.html'}")
    print(f"Pipeline summary: {output_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()