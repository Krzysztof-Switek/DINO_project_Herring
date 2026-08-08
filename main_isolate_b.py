"""08.08 — Izolacja Zmiany B (windowed E9) od Zmiany A (attention density head).

Follow-up po `outputs/06.08_attention_first` (Run K, A+B+C naraz): density_active=0.0000
przez wszystkie 50 epok, więc Run K nie mówi nic o tym, czy Zmiana A czy B (czy obie) są
winne. Ten bieg zmienia DOKŁADNIE JEDNĄ rzecz względem produkcyjnej receptury 22.07_reg —
Zmiana B włączona, Zmiana A i C wyłączone. Pełne uzasadnienie w nagłówku
configs/config_isolate_b.yaml — przeczytaj PRZED odpaleniem.

Osobny plik od main.py/main_e9.py/main_attention_first.py — nie nadpisuje żadnego
dotychczasowego biegu/configu. Kliknij ▶ na serwerze.
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

RUN_TAG = datetime.now().strftime("%d.%m") + "_isolate_b"
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config_isolate_b.yaml")

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
            f"[main_isolate_b] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna,\n"
            f"       albo popraw IMAGE_DIR_SERVER / IMAGE_DIR_LOCAL powyżej."
        )
    print(f"[main_isolate_b] LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}  RESCAN={RESCAN}")
    print(f"[main_isolate_b] BASE_CONFIG={BASE_CONFIG}")
    print(f"[main_isolate_b] OUTPUT_DIR={OUTPUT_DIR}")
    main(ARGV)
