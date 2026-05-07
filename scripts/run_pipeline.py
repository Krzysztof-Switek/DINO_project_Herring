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

    ckpt_files = sorted(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint saved in {trainer.checkpoint_dir}")
    # Pick checkpoint with lowest encoded val_loss from filename
    best_ckpt_src = min(ckpt_files, key=lambda p: float(p.stem.split("_loss")[-1]))
    import shutil as _shutil
    best_ckpt = trainer.checkpoint_dir / "best.pt"
    _shutil.copy2(best_ckpt_src, best_ckpt)
    print(f"  Best checkpoint: {best_ckpt} (← {best_ckpt_src.name})")
    return best_ckpt, training_log_data


def _step_infer(cfg, ckpt_path: Path, labels_csv: Path, output_dir: Path) -> Path:
    """Run inference for one condition, return predictions.csv path."""
    from src.dataset import OtolithDataset
    from src.inference import run_inference, load_model_from_checkpoint
    from torch.utils.data import DataLoader

    cfg_copy = cfg.model_copy(deep=True)
    cfg_copy.data.labels_csv = str(labels_csv)

    test_ds = OtolithDataset(cfg_copy, split="test")
    loader = DataLoader(test_ds, batch_size=cfg.training.batch_size,
                        shuffle=False, num_workers=cfg.data.num_workers)

    model = load_model_from_checkpoint(cfg_copy, ckpt_path)
    run_inference(cfg_copy, model, loader, output_dir)

    pred_csv = output_dir / "predictions.csv"
    print(f"  Predictions: {pred_csv}")
    return pred_csv


def _step_cards(
    pred_csvs: dict[str, Path],
    cfg_emb,
    image_dir: Path,
    output_dir: Path,
) -> dict[str, list[Path]]:
    """Generate increment annotation cards for best/worst predictions."""
    from src.visualization import select_top_k_samples, save_increment_cards

    top_k_best = cfg_emb.inference.increment_samples.top_k_best
    top_k_worst = cfg_emb.inference.increment_samples.top_k_worst
    cards: dict[str, list[Path]] = {"best": [], "worst": []}

    for cond_key, pred_csv in pred_csvs.items():
        if not Path(pred_csv).exists():
            continue
        try:
            best, worst = select_top_k_samples(pred_csv, top_k_best, top_k_worst)
        except Exception:
            continue
        cards_dir = output_dir / "cards" / cond_key
        # Importance grids not available here; cards with missing grids are skipped silently
        best_saved = save_increment_cards(best, image_dir, {}, {}, cards_dir / "best", "best")
        worst_saved = save_increment_cards(worst, image_dir, {}, {}, cards_dir / "worst", "worst")
        cards["best"].extend(best_saved)
        cards["worst"].extend(worst_saved)

    return cards


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
        # Resolve labels paths: prefer pipeline data_dir, fall back to config paths
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

    # --- Steps 4-7: inference (4 conditions) ---
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
    if "cards" not in completed:
        print("\n[8/9] CARDS — karty przyrostów")
        image_dir = Path(args.image_dir)
        increment_cards = _step_cards(pred_csvs, cfg_emb, image_dir, output_dir)
        completed.append("cards")
        _save_state(state_path, completed)
    else:
        print("\n[8/9] CARDS — pominięty (już wykonany)")
        increment_cards = {"best": [], "worst": []}

    # --- Prepare results_dfs (needed for both report and summary) ---
    import pandas as pd
    results_dfs: dict[str, pd.DataFrame | None] = {}
    for cond_key, csv_path in pred_csvs.items():
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            # predictions.csv uses "target_age"; normalise to "age"
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

        report_path = output_dir / "comparison_report.html"
        training_logs = {"embedded": logs_emb, "not_embedded": logs_notemb}
        build_comparison_report(
            results=results_dfs,
            training_logs=training_logs,
            increment_cards=increment_cards,
            dataset_stats={"counts": {}, "orphan_count": "N/A", "age_distributions": {}},
            output_path=report_path,
            model_info=model_info,
        )
        completed.append("report")
        _save_state(state_path, completed)
        print(f"  Report: {report_path}")
    else:
        print("\n[9/9] REPORT — pominięty (już wykonany)")

    # --- Pipeline summary (always written, niezależnie od trybu) ---
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
