"""23.07: regenerate ONLY cards + report for the 966px full training run
(``outputs/data/DD.MM_hires966``, launched via ``main_hires966.py``) using the CURRENT
code — needed because the training process is a single, long-lived Python interpreter
that keeps using whatever ``src/ring_extraction.py`` was on disk when it started, so a
``git pull`` mid-run (or even right after it finishes, before this script runs) does NOT
retroactively apply the new width-prior/arc-continuity selection logic (§4.22,
``outputs/DINO_proces.md``) to that run's own ``cards``/``report`` steps.

Does NOT retrain — reuses the already-trained checkpoint and the already-computed
``predictions.csv`` (age/CORAL untouched by this change). Writes to a SEPARATE output
directory so the original run's own (stale-code) cards/report stay intact for comparison
if wanted.

Usage (on the server, AFTER the 966px run has finished and you've ``git pull``ed):
    python scripts/diagnostics/regen_hires966_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import configure_attention
from scripts.run_pipeline import (
    load_merged_config, _parse_train_log, _step_cards, _compute_dataset_stats,
    _write_pipeline_summary,
)

# Auto-discover the run dir (outputs/data/DD.MM_hires966) instead of hardcoding the date —
# main_hires966.py fixes RUN_TAG once at process start, and this script may run on a
# different day than the training started.
_candidates = sorted((PROJECT_ROOT / "outputs" / "data").glob("*_hires966"))
if not _candidates:
    sys.exit("Nie znaleziono outputs/data/*_hires966 — czy trening main_hires966.py "
             "(MODE='full') już się zakończył?")
if len(_candidates) > 1:
    print(f"[uwaga] Więcej niż jeden katalog *_hires966, biorę najnowszy: {_candidates[-1]}")
RUN_DIR = _candidates[-1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data" / f"{RUN_DIR.name}_widthprior"
CKPT = RUN_DIR / "checkpoints" / "embedded" / "best.pt"
IMAGE_DIR = Path("/home/kswitek/Documents/Photo/Otolithes/HER/Processed")  # server path (main.py convention)

if not CKPT.exists():
    sys.exit(f"Brak checkpointu: {CKPT} — trening jeszcze trwa albo padł przed zapisem best.pt?")

cfg = load_merged_config(PROJECT_ROOT / "configs" / "config_hires966.yaml",
                         PROJECT_ROOT / "configs" / "config_embedded.yaml")
cfg.data.image_dir = str(IMAGE_DIR)
print(f"RUN_DIR={RUN_DIR}", flush=True)
print(f"width_decay_weight={cfg.candidates.width_decay_weight} "
      f"width_ceiling_weight={cfg.candidates.width_ceiling_weight} "
      f"(nowa logika — jeśli tu widzisz stare wartości, git pull się nie powiódł)", flush=True)
configure_attention(cfg.interpretation.disable_fused_attention)

combined_labels = RUN_DIR / "data" / "labels_combined.csv"
train_log = RUN_DIR / "logs" / "embedded" / "train.log"

cond_key = "emb_on_emb"
src_predictions = RUN_DIR / cond_key / "predictions.csv"
if not src_predictions.exists():
    sys.exit(f"Brak predictions.csv: {src_predictions} — czy krok infer_ee się zakończył?")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / cond_key).mkdir(parents=True, exist_ok=True)
dst_predictions = OUTPUT_DIR / cond_key / "predictions.csv"
if not dst_predictions.exists():
    import shutil
    shutil.copy(src_predictions, dst_predictions)
pred_csvs = {cond_key: dst_predictions}

print("[1/3] CARDS (nowa logika: arc-aware + width prior)", flush=True)
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