"""Uruchomienie treningu na podniesionej rozdzielczości 966px — kliknij ▶ na serwerze.

Osobny plik od main.py (który zostaje domyślnym launcherem na 518px) — nie nadpisuje
żadnego dotychczasowego biegu/configu. CORAL+MIL+density trenują RAZEM na 966px od
epoki 1 (zamiast post-hoc hires tylko dla density, jak w 22.07_reg). Kontekst i
uzasadnienie: outputs/DINO_proces.md §4.21, configs/config_hires966.yaml.

Zmień MODE poniżej przed uruchomieniem:
  "demo"  — 1 epoka, kilkanaście obrazów, configs/config_demo_hires966.yaml
            (sanity-check: łapie OOM/błędy kształtu na 966px w kilka minut, PRZED
            wielogodzinnym pełnym biegiem — zalecane uruchomić raz jako pierwsze)
  "full"  — pełny trening, configs/config_hires966.yaml, outputs/data/<DD.MM>_hires966/
"""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# KONFIGURACJA — zmień tylko tutaj
# ============================================================

MODE = "demo"   # "demo" → 1 epoka, szybki sanity-check na 966px | "full" → pełny trening

LOCATION = "server"   # "server" → serwer (Linux)  |  "local" → Twój komp (Windows, Z:)
                      # ↑ przełącznik ścieżki do zdjęć — zmień gdy zmieniasz maszynę

EMBEDDED_ONLY = True  # True = trenuj/raportuj TYLKO Embedded (pomija NotEmbedded i cross)

RESCAN = False  # True  = przebuduj data/labels_*.csv od nowa (skan ~18k zdjęć, kilka minut)
                # False = użyj istniejących data/labels_*.csv jeśli są
                #         (splity deterministyczne przy seed=42 → wynik ten sam)

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Ścieżka do zdjęć — dwie stałe, LOCATION wybiera jedną (identycznie jak main.py).
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"  # serwer (Linux)
IMAGE_DIR_LOCAL  = "Z:/Photo/Otolithes/HER/Processed"                       # Twój komp (Windows)
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

# Osobny tag runu (_hires966), żeby NIE kolidować z main.py (który tego samego dnia
# pisze do outputs/data/<DD.MM>/ bez sufiksu).
RUN_TAG = "demo_hires966" if MODE == "demo" else datetime.now().strftime("%d.%m") + "_hires966"
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" /
                  ("config_demo_hires966.yaml" if MODE == "demo" else "config_hires966.yaml"))

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
            f"[main_hires966] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna,\n"
            f"       albo popraw IMAGE_DIR_SERVER / IMAGE_DIR_LOCAL powyżej."
        )
    print(f"[main_hires966] MODE={MODE}  LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}")
    print(f"[main_hires966] BASE_CONFIG={BASE_CONFIG}")
    print(f"[main_hires966] OUTPUT_DIR={OUTPUT_DIR}")
    main(ARGV)
