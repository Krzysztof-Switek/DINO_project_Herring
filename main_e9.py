"""Faza 1 planu E9 (plans and summaries/28.07_966_comparison_TO_DO.md) — tanie
dostrojenie prioru koncentryczności na 518px, PRZED ewentualnym przeniesieniem
najlepszej wagi na configs/config_hires966.yaml. Kliknij ▶ na serwerze.

Osobny plik od main.py/main_hires966.py — nie nadpisuje żadnego dotychczasowego
biegu/configu. Sam config (configs/config_e9_w*.yaml) to pełna, samodzielna kopia
config.yaml (== receptura 22.07_reg) z DWOMA nowymi polami
(density_concentricity_weight/_bins) — jedyna zmienna w tych biegach.

Zmień WEIGHT poniżej PRZED KAŻDYM z trzech biegów (uruchamiane sekwencyjnie, jeden
po drugim — to jest CPU, nie ma równoległości; ~13h/wariant):
  "0.1"  → configs/config_e9_w0.1.yaml   (WARIANT A, rząd wielkości niżej)
  "1.0"  → configs/config_e9_w1.0.yaml   (WARIANT B, środkowy punkt bracketingu)
  "10.0" → configs/config_e9_w10.0.yaml  (WARIANT C, rząd wielkości wyżej)

Bramka PRZED oceną lokalizacji: wiek (CORAL) musi wyjść bit-identyczny z
22.07_reg/config.yaml na każdym z trzech wariantów (density_concentricity_loss jest
stop-gradient, patrz tests/test_stage3_model.py::
test_density_concentricity_loss_stop_gradient_safe) — regresja wieku = błąd
implementacji, nie oczekiwany efekt uboczny.
"""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# KONFIGURACJA — zmień tylko tutaj
# ============================================================

WEIGHT = "1.0"   # "0.1" | "1.0" | "10.0" — który wariant configs/config_e9_w*.yaml odpalić

LOCATION = "server"   # "server" → serwer (Linux)  |  "local" → Twój komp (Windows, Z:)
                      # ↑ przełącznik ścieżki do zdjęć — zmień gdy zmieniasz maszynę

EMBEDDED_ONLY = True  # True = trenuj/raportuj TYLKO Embedded (pomija NotEmbedded i cross)

RESCAN = False  # True  = przebuduj data/labels_*.csv od nowa (skan ~18k zdjęć, kilka minut)
                # False = użyj istniejących data/labels_*.csv jeśli są
                #         (splity deterministyczne przy seed=42 → wynik ten sam)

# ============================================================

if WEIGHT not in ("0.1", "1.0", "10.0"):
    sys.exit(f"[main_e9] WEIGHT={WEIGHT!r} nieznany — użyj \"0.1\", \"1.0\" albo \"10.0\"")

PROJECT_ROOT = Path(__file__).resolve().parent

# Ścieżka do zdjęć — dwie stałe, LOCATION wybiera jedną (identycznie jak main.py).
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"  # serwer (Linux)
IMAGE_DIR_LOCAL  = "Z:/Photo/Otolithes/HER/Processed"                       # Twój komp (Windows)
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

# Osobny tag runu per wariant, żeby trzy sekwencyjne biegi nie nadpisywały się
# nawzajem (i nie kolidowały z main.py, które tego samego dnia pisze do
# outputs/data/<DD.MM>/ bez sufiksu).
RUN_TAG = datetime.now().strftime("%d.%m") + f"_e9_w{WEIGHT}"
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" / f"config_e9_w{WEIGHT}.yaml")

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
            f"[main_e9] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna,\n"
            f"       albo popraw IMAGE_DIR_SERVER / IMAGE_DIR_LOCAL powyżej."
        )
    print(f"[main_e9] WEIGHT={WEIGHT}  LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}")
    print(f"[main_e9] BASE_CONFIG={BASE_CONFIG}")
    print(f"[main_e9] OUTPUT_DIR={OUTPUT_DIR}")
    main(ARGV)
