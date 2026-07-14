"""Zbuduj komplet raportów z ISTNIEJĄCEGO checkpointu — bez treningu, bez kasowania.

Robi dokładnie to, co kroki 4–9 „normalnego" pipeline'u (`scripts/run_pipeline.py`),
ale START-uje od gotowego `best.pt` i pomija skan + trening. Woła **te same funkcje**
co pipeline (`_step_infer`, `_step_cards`, `build_comparison_report`,
`_write_pipeline_summary`), więc predykcje, karty, `comparison_report.html`,
`localization_quality.json` i `pipeline_summary.json` są takie same, jak na końcu
zwykłego biegu. NIE reimplementuje nic osobno.

Bezpieczeństwo (można uruchomić W TRAKCIE trwającego treningu):
  * pisze WYŁĄCZNIE do --output-dir (osobny katalog),
  * z katalogu treningu tylko CZYTA best.pt / train.log / labels_*.csv,
  * NIE robi żadnego rmtree / czyszczenia (w przeciwieństwie do run_pipeline).

Uwaga o wyścigu: `best.pt` jest nadpisywany, gdy trening trafi lepszą epokę. Żeby na
pewno nie wczytać na wpół zapisanego pliku podczas treningu, skopiuj go najpierw i wskaż
kopię przez --checkpoint:
    cp outputs/data/13.07/checkpoints/embedded/best.pt ~/best_13.07.pt

Uwaga o GPU: uruchomiony w trakcie treningu dzieli GPU z treningiem (może zwolnić oba
albo zabraknąć pamięci). Do podglądu można wymusić CPU przez --device cpu.

Przykłady:
    # podgląd w trakcie treningu (osobny katalog, kopia checkpointu, CPU):
    python scripts/report_from_checkpoint.py \\
        --run-dir  outputs/data/13.07 \\
        --checkpoint ~/best_13.07.pt \\
        --output-dir outputs/data/13.07_preview \\
        --image-dir /home/kswitek/Documents/Photo/Otolithes/HER/Processed \\
        --device cpu

    # po zakończeniu/przerwaniu treningu (użyj best.pt z katalogu biegu):
    python scripts/report_from_checkpoint.py \\
        --run-dir  outputs/data/13.07 \\
        --output-dir outputs/data/13.07_report \\
        --image-dir /home/kswitek/Documents/Photo/Otolithes/HER/Processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Te same bloki, których używa run_pipeline w krokach 4–9 — zero własnej logiki raportu.
from scripts.run_pipeline import (          # noqa: E402
    load_merged_config,
    _parse_train_log,
    _step_infer,
    _step_cards,
    _compute_dataset_stats,
    _collect_candidate_overlays,
    _build_split_lookup,
    _write_pipeline_summary,
)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Raporty z istniejącego checkpointu (bez treningu, bez kasowania)")
    p.add_argument("--run-dir", required=True,
                   help="Katalog trwającego/zakończonego biegu (źródło best.pt, train.log, labels)")
    p.add_argument("--output-dir", default=None,
                   help="Katalog na raporty (domyślnie: <run-dir>_report). MUSI być inny niż run-dir.")
    p.add_argument("--checkpoint", default=None,
                   help="Ścieżka do .pt (domyślnie <run-dir>/checkpoints/embedded/best.pt). "
                        "W trakcie treningu wskaż KOPIĘ, by nie trafić na wpół zapisany plik.")
    p.add_argument("--image-dir", required=True,
                   help="Katalog oryginalnych zdjęć (do inferencji/kart) — jak w treningu")
    p.add_argument("--base-config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    p.add_argument("--config-embedded", default=str(PROJECT_ROOT / "configs" / "config_embedded.yaml"),
                   dest="config_embedded")
    p.add_argument("--labels-dir", default=None,
                   help="Katalog z labels_*.csv (domyślnie <run-dir>/data, fallback: <project>/data)")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"],
                   help="Wymuś urządzenie (np. cpu, gdy trening zajmuje GPU). Domyślnie z configu.")
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv=None) -> int:
    args = _parse_args(argv)

    run_dir = _resolve(Path(args.run_dir))
    output_dir = _resolve(Path(args.output_dir)) if args.output_dir else run_dir.parent / f"{run_dir.name}_report"
    if output_dir.resolve() == run_dir.resolve():
        print("[błąd] --output-dir nie może być tym samym katalogiem co --run-dir "
              "(chronimy trening przed nadpisaniem).")
        return 1

    ckpt = _resolve(Path(args.checkpoint)) if args.checkpoint else \
        run_dir / "checkpoints" / "embedded" / "best.pt"
    if not ckpt.exists():
        print(f"[błąd] Nie znaleziono checkpointu: {ckpt}")
        return 1

    labels_dir = _resolve(Path(args.labels_dir)) if args.labels_dir else run_dir / "data"
    emb_labels = labels_dir / "labels_embedded.csv"
    combined_labels = labels_dir / "labels_combined.csv"
    if not emb_labels.exists():
        # fallback: kanoniczny katalog projektu
        alt = PROJECT_ROOT / "data" / "labels_embedded.csv"
        if alt.exists():
            emb_labels = alt
            combined_labels = PROJECT_ROOT / "data" / "labels_combined.csv"
        else:
            print(f"[błąd] Brak labels_embedded.csv w {labels_dir} ani w <project>/data.")
            return 1

    train_log = run_dir / "logs" / "embedded" / "train.log"

    # --- Config: dokładnie jak run_pipeline (merge base + embedded override) ---
    cfg = load_merged_config(Path(args.base_config), Path(args.config_embedded))
    cfg.data.image_dir = args.image_dir              # autorytatywne dla inferencji/kart
    if args.device:
        cfg.training.device = args.device
    # Katalogi checkpointów/logów wskazujemy na OUTPUT (nie używane do zapisu tu, ale spójnie).
    cfg.training.checkpoint_dir = str((output_dir / "checkpoints" / "embedded").resolve())
    cfg.training.log_dir = str((output_dir / "logs" / "embedded").resolve())

    # Prawdziwa uwaga CLS na kartach — musi paść PRZED importem backbone (jak w pipeline).
    from src.utils import configure_attention
    configure_attention(cfg.interpretation.disable_fused_attention)

    output_dir.mkdir(parents=True, exist_ok=True)    # NIE kasujemy niczego
    print("=" * 60)
    print("RAPORT Z CHECKPOINTU (bez treningu, bez kasowania)")
    print(f"  checkpoint : {ckpt}")
    print(f"  labels     : {emb_labels}")
    print(f"  train.log  : {train_log}  ({'jest' if train_log.exists() else 'BRAK → Sekcja B pusta'})")
    print(f"  image-dir  : {cfg.data.image_dir}")
    print(f"  output     : {output_dir}")
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
    increment_cards, opencv_reference = _step_cards(
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
    logs_emb = _parse_train_log(train_log)
    training_logs = {"embedded": logs_emb, "not_embedded": []}

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
    candidate_overlays = _collect_candidate_overlays(output_dir, pred_csvs.keys())
    split_lookup = _build_split_lookup(combined_labels)
    report_path = output_dir / "comparison_report.html"
    build_comparison_report(
        results=results_dfs,
        training_logs=training_logs,
        increment_cards=increment_cards,
        dataset_stats=dataset_stats,
        output_path=report_path,
        model_info=model_info,
        candidate_overlays=candidate_overlays,
        split_lookup=split_lookup,
        opencv_reference=opencv_reference,
    )

    # --- Summary (ten sam _write_pipeline_summary) ---
    print("\n[4/4] SUMMARY")
    _write_pipeline_summary(
        output_dir=output_dir,
        training_logs=training_logs,
        results_dfs=results_dfs,
        completed_steps=["infer_ee", "cards", "report"],   # bez scan/train — to raport z checkpointu
    )

    print("\n=== Gotowe ===")
    print(f"Raport:           {report_path}")
    print(f"Pipeline summary: {output_dir / 'pipeline_summary.json'}")
    print(f"Localization:     {output_dir / 'localization_quality.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
