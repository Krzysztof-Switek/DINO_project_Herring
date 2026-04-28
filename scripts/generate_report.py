"""Generate a comprehensive HTML training report.

Usage (minimal — only data section):
    python scripts/generate_report.py --labels data/labels.csv --out outputs/report.html

Usage (full — after training + inference + interpretation + candidates):
    python scripts/generate_report.py ^
        --labels    data/labels.csv ^
        --log       logs/train.log ^
        --preds     outputs/predictions.csv ^
        --heatmaps  outputs/heatmaps ^
        --overlays  outputs/overlays ^
        --cand_json outputs/candidates ^
        --cand_ovl  outputs/candidates_overlays ^
        --out       outputs/report.html

All arguments are optional — missing sections are skipped with a notice in the report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _opt(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    return p if not path_str.startswith("data/") and not path_str.startswith("outputs/") \
        else PROJECT_ROOT / path_str if not p.is_absolute() else p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generuj raport HTML z treningu OtolithDino")
    p.add_argument("--labels",    default=None, help="data/labels.csv")
    p.add_argument("--log",       default=None, help="logs/train.log")
    p.add_argument("--preds",     default=None, help="outputs/predictions.csv")
    p.add_argument("--heatmaps",  default=None, help="outputs/heatmaps/")
    p.add_argument("--overlays",  default=None, help="outputs/overlays/")
    p.add_argument("--cand_json", default=None, help="outputs/candidates/")
    p.add_argument("--cand_ovl",  default=None, help="outputs/candidates_overlays/")
    p.add_argument("--out",       default="outputs/report.html", help="Sciezka wyjsciowa")
    p.add_argument("--n_samples", type=int, default=12,
                   help="Liczba przykladowych zdjec w sekcjach interpretacji i kandydatow")
    return p.parse_args()


def _resolve(val: str | None) -> Path | None:
    if val is None:
        return None
    p = Path(val)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / val


def main() -> None:
    args = parse_args()

    labels_csv       = _resolve(args.labels)
    log_path         = _resolve(args.log)
    predictions_csv  = _resolve(args.preds)
    heatmaps_dir     = _resolve(args.heatmaps)
    overlays_dir     = _resolve(args.overlays)
    cand_json_dir    = _resolve(args.cand_json)
    cand_overlays_dir= _resolve(args.cand_ovl)
    out_path         = _resolve(args.out)

    # Status summary
    print("=" * 55)
    print("OtolithDino -- generowanie raportu HTML")
    print("=" * 55)

    def _status(name: str, path: Path | None) -> None:
        if path is None:
            print(f"  [SKIP] {name:20s} -- nie podano")
        elif path.exists():
            print(f"  [OK]   {name:20s}  {path}")
        else:
            print(f"  [MISS] {name:20s}  {path}  (brak pliku)")

    _status("labels.csv",      labels_csv)
    _status("train.log",       log_path)
    _status("predictions.csv", predictions_csv)
    _status("heatmaps/",       heatmaps_dir)
    _status("overlays/",       overlays_dir)
    _status("candidates/",     cand_json_dir)
    _status("cand_overlays/",  cand_overlays_dir)
    print(f"  Wyjscie:          {out_path}")
    print(f"  Przykladow:       {args.n_samples}")

    print("\nGenerowanie raportu ...")

    from src.report import build_html_report, save_report

    html = build_html_report(
        labels_csv        = labels_csv,
        log_path          = log_path,
        predictions_csv   = predictions_csv,
        heatmaps_dir      = heatmaps_dir,
        overlays_dir      = overlays_dir,
        cand_json_dir     = cand_json_dir,
        cand_overlays_dir = cand_overlays_dir,
        n_image_samples   = args.n_samples,
    )

    save_report(html, out_path)
    print("OK gotowe.")


if __name__ == "__main__":
    main()
