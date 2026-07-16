"""Zbuduj komplet raportów z ISTNIEJĄCEGO checkpointu — bez treningu, bez kasowania.

Użycie jak w main.py: ustaw RUN_DIR w bloku KONFIGURACJA poniżej i uruchom (▶ w PyCharm
albo `python scripts/report_from_checkpoint.py`). Nie podaje się nic w wierszu poleceń.

Skrypt sam ustala (z RUN_DIR):
  * checkpoint  = <RUN_DIR>/checkpoints/embedded/best.pt
  * train.log   = <RUN_DIR>/logs/embedded/train.log
  * labels      = <RUN_DIR>/data/labels_*.csv
  * zdjęcia     = main.IMAGE_DIR (ta sama ścieżka co trening, wg LOCATION w main.py)
  * wynik       = <RUN_DIR>_preview  (OBOK biegu, np. outputs/data/13.07_preview)

Robi dokładnie to, co kroki 4–9 „normalnego" pipeline'u (`scripts/run_pipeline.py`) —
woła TE SAME funkcje (`_step_infer`, `_step_cards`, `build_comparison_report`,
`_write_pipeline_summary`), więc predykcje, karty, `comparison_report.html`,
`localization_quality.json` i `pipeline_summary.json` są takie same jak na końcu
zwykłego biegu. Pomija tylko skan + trening (bierze gotowy best.pt).

Bezpieczne do uruchomienia W TRAKCIE trwającego treningu: pisze WYŁĄCZNIE do
<RUN_DIR>_preview, NIE kasuje niczego (żadnego rmtree), z biegu tylko CZYTA.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# KONFIGURACJA — zmień tylko tutaj (jak w main.py)
# ============================================================

RUN_DIR = "outputs/data/13.07"   # katalog biegu (źródło best.pt / train.log / labels)

DEVICE = None                    # None = z configu (auto → GPU) | "cpu" = nie dziel GPU z treningiem

# Opcjonalne nadpisania — None = wylicz automatycznie z RUN_DIR:
OUTPUT_DIR = None                # None → <RUN_DIR>_preview (obok biegu)
CHECKPOINT = None                # None → <RUN_DIR>/checkpoints/embedded/best.pt
IMAGE_DIR  = None                # None → main.IMAGE_DIR (ta sama ścieżka co trening)

# ============================================================

# Te same bloki, których używa run_pipeline w krokach 4–9 — zero własnej logiki raportu.
from scripts.run_pipeline import (          # noqa: E402
    load_merged_config,
    _parse_train_log,
    _step_infer,
    _step_cards,
    _compute_dataset_stats,
    _write_pipeline_summary,
)


def _default_image_dir() -> str | None:
    """Ta sama ścieżka do zdjęć co trening — z main.py (LOCATION → serwer/lokalnie)."""
    try:
        from main import IMAGE_DIR as MAIN_IMAGE_DIR
        return MAIN_IMAGE_DIR
    except Exception:
        return None


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    run_dir = _resolve(Path(RUN_DIR))
    if not run_dir.is_dir():
        print(f"[błąd] Katalog biegu nie istnieje: {run_dir}  (popraw RUN_DIR)")
        return 1

    # --- Wszystko wyliczane z RUN_DIR (chyba że nadpisane w KONFIGURACJI) ---
    output_dir = _resolve(Path(OUTPUT_DIR)) if OUTPUT_DIR \
        else run_dir.parent / f"{run_dir.name}_preview"
    if output_dir.resolve() == run_dir.resolve():
        print("[błąd] OUTPUT_DIR nie może być tym samym katalogiem co bieg.")
        return 1

    ckpt = _resolve(Path(CHECKPOINT)) if CHECKPOINT \
        else run_dir / "checkpoints" / "embedded" / "best.pt"
    if not ckpt.exists():
        print(f"[błąd] Nie znaleziono checkpointu: {ckpt}")
        return 1

    emb_labels = run_dir / "data" / "labels_embedded.csv"
    combined_labels = run_dir / "data" / "labels_combined.csv"
    if not emb_labels.exists():                       # fallback: kanoniczny <project>/data
        alt = PROJECT_ROOT / "data" / "labels_embedded.csv"
        if alt.exists():
            emb_labels = alt
            combined_labels = PROJECT_ROOT / "data" / "labels_combined.csv"
        else:
            print(f"[błąd] Brak labels_embedded.csv w {run_dir / 'data'} ani w <project>/data.")
            return 1

    train_log = run_dir / "logs" / "embedded" / "train.log"

    image_dir = IMAGE_DIR or _default_image_dir()
    if not image_dir:
        print("[błąd] Nie udało się ustalić IMAGE_DIR (brak main.IMAGE_DIR). Ustaw IMAGE_DIR w KONFIGURACJI.")
        return 1
    if not Path(image_dir).is_dir():
        print(f"[błąd] Katalog zdjęć nie istnieje: {image_dir} "
              f"(sprawdź LOCATION w main.py albo ustaw IMAGE_DIR).")
        return 1

    # --- Config: dokładnie jak run_pipeline (merge base + embedded override) ---
    cfg = load_merged_config(
        PROJECT_ROOT / "configs" / "config.yaml",
        PROJECT_ROOT / "configs" / "config_embedded.yaml",
    )
    cfg.data.image_dir = str(image_dir)
    if DEVICE:
        cfg.training.device = DEVICE
    cfg.training.checkpoint_dir = str((output_dir / "checkpoints" / "embedded").resolve())
    cfg.training.log_dir = str((output_dir / "logs" / "embedded").resolve())

    # Prawdziwa uwaga CLS na kartach — PRZED importem backbone (jak w pipeline).
    from src.utils import configure_attention
    configure_attention(cfg.interpretation.disable_fused_attention)

    output_dir.mkdir(parents=True, exist_ok=True)     # NIE kasujemy niczego
    print("=" * 60)
    print("RAPORT Z CHECKPOINTU (bez treningu, bez kasowania)")
    print(f"  bieg       : {run_dir}")
    print(f"  checkpoint : {ckpt}")
    print(f"  train.log  : {'jest' if train_log.exists() else 'BRAK → Sekcja B pusta'}")
    print(f"  zdjęcia    : {cfg.data.image_dir}")
    print(f"  wynik      : {output_dir}")
    print(f"  device     : {cfg.training.device}")
    print("=" * 60)

    # --- Krok 4: inferencja + interpretacja + kandydaci (ten sam _step_infer) ---
    cond_key = "emb_on_emb"
    infer_dir = output_dir / cond_key
    print(f"\n[1/4] INFER — {cond_key}")
    _step_infer(cfg, ckpt, emb_labels, infer_dir)
    pred_csvs = {cond_key: infer_dir / "predictions.csv"}

    # --- Krok 8: karty rozumowania (ten sam _step_cards) ---
    print("\n[2/4] CARDS — karty rozumowania")
    cond_models = {cond_key: (cfg, ckpt)}
    increment_cards, opencv_reference, localization_methods, localization_walkthrough = _step_cards(
        pred_csvs, cfg, Path(cfg.data.image_dir), output_dir, cond_models)

    # --- Wyniki + logi treningu (parsujemy ISTNIEJĄCY train.log) ---
    import pandas as pd
    results_dfs = {}
    for ck, csv_path in pred_csvs.items():
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            if "target_age" in df.columns and "age" not in df.columns:
                df = df.rename(columns={"target_age": "age"})
            results_dfs[ck] = df
        else:
            results_dfs[ck] = None
    training_logs = {"embedded": _parse_train_log(train_log), "not_embedded": []}

    # --- Krok 9: raport porównawczy (ten sam build_comparison_report) ---
    print("\n[3/4] REPORT — raport porównawczy")
    from src.comparison_report import build_comparison_report
    model_info = {
        "backbone": cfg.model.backbone,
        "num_age_classes": cfg.model.num_age_classes,
        "use_metadata": cfg.model.use_metadata,
        "ckpt_embedded": str(ckpt),
        "ckpt_not_embedded": "",
    }
    dataset_stats = _compute_dataset_stats(combined_labels, active_ptypes=["Embedded"])
    report_path = output_dir / "comparison_report.html"
    build_comparison_report(
        results=results_dfs,
        training_logs=training_logs,
        increment_cards=increment_cards,
        dataset_stats=dataset_stats,
        output_path=report_path,
        model_info=model_info,
        opencv_reference=opencv_reference,
        localization_methods=localization_methods,
        localization_walkthrough=localization_walkthrough,
    )

    # --- Summary (ten sam _write_pipeline_summary) ---
    print("\n[4/4] SUMMARY")
    _write_pipeline_summary(
        output_dir=output_dir,
        training_logs=training_logs,
        results_dfs=results_dfs,
        completed_steps=["infer_ee", "cards", "report"],   # bez scan/train — raport z checkpointu
    )

    print("\n=== Gotowe ===")
    print(f"Raport:           {report_path}")
    print(f"Pipeline summary: {output_dir / 'pipeline_summary.json'}")
    print(f"Localization:     {output_dir / 'localization_quality.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
