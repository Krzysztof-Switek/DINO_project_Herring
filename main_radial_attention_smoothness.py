"""17.08 — Strojenie straty density: osłabić wymuszaną punktowość (density_conc_weight 1,0→0,3,
density_tv_weight 0,03→0,3), jedyna zmiana względem Run N (config_radial_attention.yaml).

Po dwóch z rzędu negatywnych wynikach strojenia SAMEJ ARCHITEKTURY radial_attention (Zmiana B
13-14.08, density_attn_window_deg 15-17.08 — oba nierozróżnialne od Run N na ZEGAR), nowa hipoteza
ze strony STRATY: density_count_loss (src/model.py:116) ma człon `l_conc`, który dosłownie każe
wypchnąć dokładnie ⌈age⌉ komórek do 1,0 i resztę do 0,0 — matematyczna definicja punktowości,
nigdy niestrojona (density_conc_weight=1,0 w KAŻDYM biegu w historii projektu). Pełne uzasadnienie
w nagłówku configs/config_radial_attention_smoothness.yaml i
plans and summaries/6.08_PODSUMOWANIE_PROCESU.md Etap 26/27 — przeczytaj PRZED odpaleniem.

Osobny plik od main_radial_attention.py/main_radial_attention_wide_window.py — nie nadpisuje
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

# Ścieżka do zdjęć — dwie stałe, LOCATION wybiera jedną (identycznie jak main_radial_attention.py).
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"  # serwer (Linux)
IMAGE_DIR_LOCAL  = "Z:/Photo/Otolithes/HER/Processed"                       # Twój komp (Windows)
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

RUN_TAG = datetime.now().strftime("%d.%m") + "_radial_attention_smoothness"
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config_radial_attention_smoothness.yaml")

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
            f"[main_radial_attention_smoothness] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna,\n"
            f"       albo popraw IMAGE_DIR_SERVER / IMAGE_DIR_LOCAL powyżej."
        )
    print(f"[main_radial_attention_smoothness] LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}  RESCAN={RESCAN}")
    print(f"[main_radial_attention_smoothness] BASE_CONFIG={BASE_CONFIG}")
    print(f"[main_radial_attention_smoothness] OUTPUT_DIR={OUTPUT_DIR}")
    main(ARGV)
