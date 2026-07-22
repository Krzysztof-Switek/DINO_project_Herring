"""Regenerate cards + report for outputs/20.07_reg using the CURRENT codebase (all of
today's/yesterday's post-hoc fixes: E1 clustering, margin-filter fix, arc-aware scoring,
DP spread_weight, Krok2 classical signal). Reuses predictions.csv (skip slow inference —
predictions are unaffected by post-hoc report/localization code)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

RUN_DIR = PROJECT_ROOT / "outputs" / "20.07_reg"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "20.07_reg_local"
CKPT = RUN_DIR / "checkpoints" / "embedded" / "best.pt"
IMAGE_DIR = Path("Z:/Photo/Otolithes/HER/Processed")

from scripts.run_pipeline import (
    load_merged_config, _parse_train_log, _step_cards, _compute_dataset_stats,
    _write_pipeline_summary,
)

combined_labels = RUN_DIR / "data" / "labels_combined.csv"
train_log = RUN_DIR / "logs" / "embedded" / "train.log"

cfg = load_merged_config(PROJECT_ROOT / "configs" / "config.yaml",
                         PROJECT_ROOT / "configs" / "config_embedded.yaml")
cfg.data.image_dir = str(IMAGE_DIR)
print(f"backbone={cfg.model.backbone} mask_background={cfg.data.mask_background} "
      f"top_k_best={cfg.inference.increment_samples.top_k_best} "
      f"top_k_worst={cfg.inference.increment_samples.top_k_worst}", flush=True)

from src.utils import configure_attention
configure_attention(cfg.interpretation.disable_fused_attention)

cond_key = "emb_on_emb"
src_predictions = RUN_DIR / cond_key / "predictions.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / cond_key).mkdir(parents=True, exist_ok=True)
dst_predictions = OUTPUT_DIR / cond_key / "predictions.csv"
if not dst_predictions.exists():
    import shutil
    shutil.copy(src_predictions, dst_predictions)
pred_csvs = {cond_key: dst_predictions}

print("[1/3] CARDS (reusing predictions.csv, current code)", flush=True)
cond_models = {cond_key: (cfg, CKPT)}
increment_cards, opencv_reference, localization_methods, localization_walkthrough = _step_cards(
    pred_csvs, cfg, Path(cfg.data.image_dir), OUTPUT_DIR, cond_models)

import pandas as pd
results_dfs = {}
for ck, csv_path in pred_csvs.items():
    df = pd.read_csv(csv_path)
    if "target_age" in df.columns and "age" not in df.columns:
        df = df.rename(columns={"target_age": "age"})
    results_dfs[ck] = df
training_logs = {"embedded": _parse_train_log(train_log), "not_embedded": []}

print("[2/3] REPORT", flush=True)
from src.comparison_report import build_comparison_report
model_info = {"backbone": cfg.model.backbone, "num_age_classes": cfg.model.num_age_classes,
              "use_metadata": cfg.model.use_metadata, "ckpt_embedded": str(CKPT), "ckpt_not_embedded": ""}
dataset_stats = _compute_dataset_stats(combined_labels, active_ptypes=["Embedded"])
report_path = OUTPUT_DIR / "comparison_report.html"
build_comparison_report(
    results=results_dfs, training_logs=training_logs, increment_cards=increment_cards,
    dataset_stats=dataset_stats, output_path=report_path, model_info=model_info,
    opencv_reference=opencv_reference, localization_methods=localization_methods,
    localization_walkthrough=localization_walkthrough,
)

print("[3/3] SUMMARY", flush=True)
_write_pipeline_summary(output_dir=OUTPUT_DIR, training_logs=training_logs,
                        results_dfs=results_dfs, completed_steps=["infer_ee", "cards", "report"])

print("=== DONE ===", flush=True)
print(f"Raport: {report_path}", flush=True)
print(f"Rozmiar: {report_path.stat().st_size / 1e6:.1f} MB", flush=True)
