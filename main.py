"""Root entry point — kliknij ▶ w PyCharm aby uruchomić pipeline.

Zmień MODE poniżej przed uruchomieniem:
  "demo"  — 1 epoka, outputs/data/demo/     (szybki test całego pipeline'u)
  "full"  — pełny trening, outputs/data/<DD.MM>/

Każdy run jest PEŁNY i OD ZERA: run_pipeline czyści katalog runu przed startem,
nie pomija żadnych kroków i pisze świeży train.log (11.07 TO-DO Punkty 2–3).
"""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# KONFIGURACJA — zmień tylko tutaj
# ============================================================

MODE = "full"   # "demo" → 1 epoka, szybki test pipeline'u | "full" → pełny trening

LOCATION = "server"   # "server" → serwer (Linux)  |  "local" → Twój komp (Windows, Z:)
                      # ↑ przełącznik ścieżki do zdjęć — zmień gdy zmieniasz maszynę

EMBEDDED_ONLY = True  # True = trenuj/raportuj TYLKO Embedded (pomija NotEmbedded i cross)

RESCAN = False  # True  = przebuduj data/labels_*.csv od nowa (skan ~18k zdjęć, kilka minut)
                # False = użyj istniejących data/labels_*.csv jeśli są
                #         (splity deterministyczne przy seed=42 → wynik ten sam)

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Ścieżka do zdjęć — dwie stałe, LOCATION wybiera jedną.
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"  # serwer (Linux)
IMAGE_DIR_LOCAL  = "Z:/Photo/Otolithes/HER/Processed"                       # Twój komp (Windows)
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

# Jeden folder z datą na cały run: checkpoints/, logs/, predictions, raport, summary.
RUN_TAG = "demo" if MODE == "demo" else datetime.now().strftime("%d.%m")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" /
                  ("config_demo.yaml" if MODE == "demo" else "config.yaml"))

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
            f"[main] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna,\n"
            f"       albo popraw IMAGE_DIR_SERVER / IMAGE_DIR_LOCAL w main.py."
        )
    print(f"[main] LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}")
    print(f"[main] OUTPUT_DIR={OUTPUT_DIR}")
    main(ARGV)
