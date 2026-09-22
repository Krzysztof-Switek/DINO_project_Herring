"""09.09 — Eksperyment "wycinek kątowy" (polar wedge): wiek na całym obrazie, lokalizacja na
rozwinięciu Daugmanowym ograniczonym do sektora Δθ wokół osi jądro→brzeg.

Jedna izolowana zmienna względem Run N (config_radial_attention.yaml): geometria wejścia głowicy
density. Głowica wieku (CORAL+MIL) czyta wyłącznie zwykły kwadratowy obraz, dokładnie jak w Run N.
Pełne uzasadnienie: C:\\Users\\kswitek\\.claude\\plans\\wise-moseying-rivest.md,
outputs/09.09_masked_wizualizacje/report4.html, nagłówek configs/config_wedge_a.yaml —
przeczytaj PRZED odpaleniem.

Osobny plik od main_strip_a.py i innych — nie nadpisuje żadnego dotychczasowego biegu/configu.
Kliknij ▶ na serwerze.
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

RUN_TAG = datetime.now().strftime("%d.%m") + "_wedge_a"
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config_wedge_a.yaml")

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
            f"[main_wedge_a] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna,\n"
            f"       albo popraw IMAGE_DIR_SERVER / IMAGE_DIR_LOCAL powyżej."
        )
    # (22.09) Sanity-check the BASE_CONFIG the same way — a config that exists locally but was
    # never committed does NOT reach the server, and scripts/run_pipeline.py used to fall back to
    # pure OtolithConfig() defaults without a word (19h41m wasted, see
    # plans and summaries/22.09_wedge_b_analiza.md). Fail here, in 2 seconds, not after a night.
    if not Path(BASE_CONFIG).is_file():
        sys.exit(
            f"[main_wedge_a] BASE_CONFIG nie istnieje: {BASE_CONFIG!r}\n"
            f"       Ten plik DEFINIUJE eksperyment — bez niego bieg wytrenowałby zupełnie co\n"
            f"       innego. Najczęstsza przyczyna: config nie został zacommitowany i nie\n"
            f"       dojechał na serwer (sprawdź `git status` na maszynie lokalnej)."
        )
    print(f"[main_wedge_a] LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}  RESCAN={RESCAN}")
    print(f"[main_wedge_a] BASE_CONFIG={BASE_CONFIG}")
    print(f"[main_wedge_a] OUTPUT_DIR={OUTPUT_DIR}")
    print("[main_wedge_a] UWAGA: karty report.html (density/kandydaci) NIE są świadome gałęzi "
          "wycinka — ufać tylko predictions.csv/pipeline_summary.json (wiek); lokalizacja wymaga "
          "osobnego eval skryptu, wzorem scripts/diagnostics/expert_annotation_eval_strip.py.")
    main(ARGV)
