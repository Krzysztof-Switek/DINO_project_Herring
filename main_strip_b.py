"""03.09 — Eksperyment "wielokrotne wycinki" (Track B / "Wariant 2"): jak Track A
(main_strip_a.py, wciąż osobny, żywy bieg — ten plik go nie dotyka), ale głowica density
dostaje wiele kandydackich wycinków na próbkę zamiast jednego, z wyborem najlepszego wg
zgodności liczby przyrostów z prawdziwym wiekiem.

Głowica wieku (CORAL+MIL) nietknięta — identyczna receptura co w Track A i Run N.
Pełne uzasadnienie: plans and summaries/03.09_multi_wycinek_plan.md, nagłówek
configs/config_strip_b.yaml — przeczytaj PRZED odpaleniem.

Osobny plik od main_strip_a.py i innych — nie nadpisuje żadnego dotychczasowego
biegu/configu. Kliknij ▶ na serwerze.
"""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# KONFIGURACJA — zmień tylko tutaj
# ============================================================

LOCATION = "server"   # "server" → serwer (Linux)  |  "local" → Twój komp (Windows, Z:)
                      # ↑ przełącznik ścieżki do zdjęć — zmień gdy zmieniasz maszynę

EMBEDDED_ONLY = True  # True = trenuj/raportuj TYLKO Embedded (pomija NotEmbedded i cross)

RESCAN = False   # False: Zmiana C jest WYŁĄCZONA w tym biegu, więc kolumna "campaign" nie jest
                 # potrzebna — reuse istniejących data/labels_*.csv (splity deterministyczne,
                 # seed=42, nic się nie zmieniło w danych na dysku od ostatniego skanu). Ustaw
                 # True jeśli dane na Z: faktycznie się zmieniły od ostatniego RESCAN=True biegu.

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Ścieżka do zdjęć — dwie stałe, LOCATION wybiera jedną (identycznie jak main_strip_a.py).
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"  # serwer (Linux)
IMAGE_DIR_LOCAL  = "Z:/Photo/Otolithes/HER/Processed"                       # Twój komp (Windows)
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

RUN_TAG = datetime.now().strftime("%d.%m") + "_strip_b"
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config_strip_b.yaml")

ARGV = [
    "--base-config",          BASE_CONFIG,
    "--image-dir",            IMAGE_DIR,
    "--excel",                EXCEL_PATH,
    "--output-dir",           OUTPUT_DIR,
    "--config-embedded",      str(PROJECT_ROOT / "configs" / "config_embedded.yaml"),
    "--config-not-embedded",  str(PROJECT_ROOT / "configs" / "config_not_embedded.yaml"),
]
if RESCAN:
    ARGV.append("--rescan")
if EMBEDDED_ONLY:
    ARGV.append("--embedded-only")

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline import main  # noqa: E402

if __name__ == "__main__":
    # Sanity-check the image dir UP FRONT — inaczej błąd wyskakuje dopiero w
    # środku treningu (jak w plans and summaries/błąd.md).
    if not Path(IMAGE_DIR).is_dir():
        sys.exit(
            f"[main_strip_b] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna,\n"
            f"       albo popraw IMAGE_DIR_SERVER / IMAGE_DIR_LOCAL powyżej."
        )
    print(f"[main_strip_b] LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}  RESCAN={RESCAN}")
    print(f"[main_strip_b] BASE_CONFIG={BASE_CONFIG}")
    print(f"[main_strip_b] OUTPUT_DIR={OUTPUT_DIR}")
    print("[main_strip_b] UWAGA: karty report.html (density/kandydaci) NIE są świadome gałęzi "
          "paska ani wielu kandydatów — ufać tylko predictions.csv/pipeline_summary.json (wiek); "
          "lokalizacja liczona osobno przez dedykowany skrypt ewaluacji ZEGAR.")
    main(ARGV)
