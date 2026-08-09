"""09.08 — Izolacja Zmiany A (attention density head) od Zmiany B (windowed E9).

Dopełnia macierz 2×2: baza (A off/B off) = produkcja 0,7993; A off/B on = `08.08_isolate_b`
(density budzi się @e16, ale nadal głównie na krawędzi); A on/B on = Run K (`06.08_attention_
first`, density martwa 50/50 epok). Ten bieg = brakująca komórka (A on/B off) — sprawdza, czy
sama architektura uwagi budzi się bez okienkowej straty, czy to dopiero interakcja A+B zabiła
density w Run K. PRIORYTET tej całej serii to poprawna lokalizacja przyrostów, nie sama
mechanika "czy density > 0" — po analizie wyników OBEJRZYJ realne karty
(outputs/data/DD.MM_isolate_a/cards/), nie tylko liczby z localization_quality.json. Pełne
uzasadnienie w nagłówku configs/config_isolate_a.yaml — przeczytaj PRZED odpaleniem.

Osobny plik od main.py/main_e9.py/main_attention_first.py/main_isolate_b.py — nie nadpisuje
żadnego dotychczasowego biegu/configu. Kliknij ▶ na serwerze.
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

# Ścieżka do zdjęć — dwie stałe, LOCATION wybiera jedną (identycznie jak main.py/main_e9.py).
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"  # serwer (Linux)
IMAGE_DIR_LOCAL  = "Z:/Photo/Otolithes/HER/Processed"                       # Twój komp (Windows)
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

RUN_TAG = datetime.now().strftime("%d.%m") + "_isolate_a"
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config_isolate_a.yaml")

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
            f"[main_isolate_a] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna,\n"
            f"       albo popraw IMAGE_DIR_SERVER / IMAGE_DIR_LOCAL powyżej."
        )
    print(f"[main_isolate_a] LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}  RESCAN={RESCAN}")
    print(f"[main_isolate_a] BASE_CONFIG={BASE_CONFIG}")
    print(f"[main_isolate_a] OUTPUT_DIR={OUTPUT_DIR}")
    main(ARGV)
