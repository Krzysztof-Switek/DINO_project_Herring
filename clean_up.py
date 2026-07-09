#!/usr/bin/env python
"""clean_up.py — reset the pipeline for a fresh training run.

Removes the GENERATED training / inference / report artifacts so the next run
starts clean, WITHOUT touching source, configs, raw data (Excel), the
segmentation-review folders, or the state-tracking files that must survive
(state.json, progress.md, next_step.txt, controller.py — see CLAUDE.md).

What it removes
---------------
Default (safe, fast re-train on the SAME scan):
  * checkpoints/*.pt              (all model checkpoints incl. best.pt)
  * logs/train.log*              (training logs)
  * <output>/pipeline_state.json (resume state → otherwise steps are "skipped")
  * <output>/pipeline_summary.json, report*.html, comparison_report.html
  * <output>/predictions.*       and the per-condition run dirs
  * <output>/cards, heatmaps, overlays, masks, candidates, candidates_overlays

With --with-scan (also forces a fresh scan of Z:):
  * <output>/data/labels_*.csv, <output>/data/scan_stats.json
  * data/labels_*.csv, data/scan_stats.json   (project-level scan output)

--all = default + --with-scan.

It NEVER deletes: state-tracking files, configs/, src/, tests/, scripts/,
data/*.xlsx, plans and summaries/, project_context/, and any output subdir it
doesn't recognise (e.g. outputs/09.07, outputs/_segcheck_new) — those are left
alone so a targeted clean can't nuke a past run or the segmentation review.

Two ways to run
---------------
1. Green "Run" triangle (IDE) — no CLI args are passed, so it uses the SETTINGS
   block below. Edit those constants and hit Run. It starts in DRY_RUN (preview
   only) so an accidental click deletes nothing; set DRY_RUN = False to clean.
2. Terminal — CLI flags override the SETTINGS block:
     python clean_up.py --dry-run        # preview only, delete nothing
     python clean_up.py -y               # clean without prompting
     python clean_up.py --with-scan -y   # also drop the scanned labels
     python clean_up.py --output-dir outputs/myrun -y
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Force UTF-8 output so the Polish messages don't crash on a server whose locale
# is C/POSIX (e.g. under cron / a bare SSH session). No-op on Python < 3.7.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent

# ===========================================================================
#  USTAWIENIA — edytuj i uruchom zielonym trójkątem ▶ (Run).
#  Uruchomienie z terminala z flagami (np. --with-scan -y) nadpisuje te wartości.
# ===========================================================================
DRY_RUN    = True    # True = tylko podgląd (nic nie usuwa). Ustaw False, żeby wyczyścić.
WITH_SCAN  = True   # True = usuń też etykiety skanu (labels_*.csv) → wymusza ponowny skan.
ASSUME_YES = False   # True = nie pytaj o potwierdzenie (potrzebne bez terminala / w cronie).
OUTPUT_DIR     = "outputs"       # katalog wyników pipeline'u do wyczyszczenia
CHECKPOINT_DIR = "checkpoints"   # katalog checkpointów (*.pt)
LOG_DIR        = "logs"          # katalog logów (train.log*)
# ===========================================================================

# Files/dirs that must NEVER be removed, whatever else happens.
PROTECTED = {
    "state.json", "progress.md", "next_step.txt", "controller.py",
    "CLAUDE.md", "clean_up.py",
}

# Per-run condition sub-directories written under the output dir.
CONDITION_DIRS = [
    "emb_on_emb", "notemb_on_notemb", "emb_on_notemb", "notemb_on_emb",
    "cross_emb_on_notemb", "cross_notemb_on_emb",
]
# Standard generated sub-directories under the output dir.
OUTPUT_SUBDIRS = [
    "cards", "heatmaps", "overlays", "masks", "candidates", "candidates_overlays",
]
# Standard generated files under the output dir.
OUTPUT_FILES = [
    "pipeline_state.json", "pipeline_summary.json",
    "comparison_report.html", "report.html", "report_preview.html",
    "predictions.csv", "predictions.json",
]
# Scan artefacts (removed only with --with-scan / --all).
SCAN_GLOBS = ["labels_*.csv", "scan_stats.json"]


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GB"


def _is_safe(path: Path) -> bool:
    """True only if `path` is inside PROJECT_ROOT and not a protected name."""
    try:
        path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    return path.name not in PROTECTED


def _collect(output_dir: Path, checkpoint_dir: Path, log_dir: Path,
             with_scan: bool) -> list[Path]:
    targets: list[Path] = []

    # checkpoints/*.pt
    if checkpoint_dir.exists():
        targets += sorted(checkpoint_dir.glob("*.pt"))

    # logs/train.log*
    if log_dir.exists():
        targets += sorted(log_dir.glob("train.log*"))

    # output run artefacts
    for name in OUTPUT_FILES:
        p = output_dir / name
        if p.exists():
            targets.append(p)
    for name in OUTPUT_SUBDIRS + CONDITION_DIRS:
        p = output_dir / name
        if p.exists():
            targets.append(p)

    # scan artefacts (opt-in)
    if with_scan:
        for base in (output_dir / "data", PROJECT_ROOT / "data"):
            if base.exists():
                for pat in SCAN_GLOBS:
                    targets += sorted(base.glob(pat))

    # de-dupe, keep only safe paths
    seen: set[Path] = set()
    out: list[Path] = []
    for t in targets:
        r = t.resolve()
        if r in seen:
            continue
        seen.add(r)
        if _is_safe(t):
            out.append(t)
        else:
            print(f"  [skip — protected/outside root] {t}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reset the pipeline for a fresh training run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Defaults come from the SETTINGS block (so the green-triangle / no-arg run
    # uses them); passing a flag on the CLI overrides. BooleanOptionalAction adds
    # a --no-<flag> variant so the CLI can turn a SETTINGS default off too.
    ap.add_argument("--output-dir", default=OUTPUT_DIR,
                    help=f"pipeline output dir to clean (default: {OUTPUT_DIR})")
    ap.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR,
                    help=f"checkpoint dir (default: {CHECKPOINT_DIR})")
    ap.add_argument("--log-dir", default=LOG_DIR,
                    help=f"log dir (default: {LOG_DIR})")
    ap.add_argument("--with-scan", action=argparse.BooleanOptionalAction, default=WITH_SCAN,
                    help="also delete scanned labels + scan_stats (forces re-scan)")
    ap.add_argument("--all", action="store_true",
                    help="everything: default + --with-scan")
    ap.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=DRY_RUN,
                    help="show what would be removed, delete nothing")
    ap.add_argument("-y", "--yes", action=argparse.BooleanOptionalAction, default=ASSUME_YES,
                    help="don't ask for confirmation")
    args = ap.parse_args()

    def _resolve(d: str) -> Path:
        p = Path(d)
        return p if p.is_absolute() else PROJECT_ROOT / p

    output_dir = _resolve(args.output_dir)
    checkpoint_dir = _resolve(args.checkpoint_dir)
    log_dir = _resolve(args.log_dir)
    with_scan = args.with_scan or args.all

    targets = _collect(output_dir, checkpoint_dir, log_dir, with_scan)

    if not targets:
        print("Nic do wyczyszczenia — pipeline już czysty.")
        return 0

    total = 0
    print("\nDo usunięcia:\n")
    for t in targets:
        if t.is_dir():
            size = _dir_size(t)
            kind = "dir "
        else:
            try:
                size = t.stat().st_size
            except OSError:
                size = 0
            kind = "file"
        total += size
        rel = t.relative_to(PROJECT_ROOT)
        print(f"  {kind}  {_human(size):>9}  {rel}")
    print(f"\n  Razem: {len(targets)} pozycji, {_human(total)}")
    if not with_scan:
        print("  (etykiety skanu ZACHOWANE — użyj --with-scan, aby wymusić ponowny skan Z:)")

    if args.dry_run:
        print("\n--dry-run: nic nie usunięto.")
        return 0

    if not args.yes:
        try:
            ans = input("\nUsunąć powyższe? [t/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in {"t", "tak", "y", "yes"}:
            print("Przerwano — nic nie usunięto.")
            return 1

    removed = 0
    for t in targets:
        try:
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink()
            removed += 1
        except OSError as e:
            print(f"  [błąd] {t}: {e}")
    print(f"\nUsunięto {removed}/{len(targets)} pozycji ({_human(total)}). "
          f"Pipeline gotowy do świeżego treningu.")
    if not with_scan:
        print("Uruchom trening (skan pominięty, etykiety zachowane) lub dodaj --with-scan by przeskanować od nowa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
