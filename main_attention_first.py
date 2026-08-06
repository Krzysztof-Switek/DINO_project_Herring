"""05.08 — Uwaga modelu jako podstawa lokalizacji: Zmiana A (attention density
head) + Zmiana B (windowed E9) + Zmiana C ("-1" dla pierwszego półrocza), wszystkie
NARAZ (plans and summaries/5.08_plan_TO_DO.md, plan sesji: zaktualizuj-pami-projektu-
peppy-crane.md). Kliknij ▶ na serwerze.

Osobny plik od main.py/main_e9.py/main_hires966.py — nie nadpisuje żadnego
dotychczasowego biegu/configu. Sam config (configs/config_attention_first.yaml) to
pełna, samodzielna kopia config.yaml (== receptura 22.07_reg) z TRZEMA nowymi polami
naraz — czytaj nagłówek tamtego pliku PRZED odpaleniem, zwłaszcza akapit o
bezpieczeństwie (Zmiana A/B są bezpieczne konstrukcyjnie jak E9; Zmiana C NIE JEST —
zmienia sam cel treningu CORAL dla ~40% datasetu).

Jeśli wynik tego biegu będzie niejednoznaczny (np. wiek się pogorszy, nie wiadomo
czy to Zmiana C czy interakcja A/B), kolejny krok to izolacja pojedynczej flagi w
osobnym biegu — dokładnie ten sam schemat co 15.07_rx -> 16.07_rx.
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

RESCAN = True   # ZALECANE True na PIERWSZYM biegu tego configu: materializuje kolumnę
                # "campaign" w data/labels_embedded.csv (Zmiana C czyta ją bezpośrednio,
                # zamiast dociągać z nazwy pliku w locie przy każdym __getitem__ — działa
                # tak czy tak, ale rescan raz na start jest czystszy). Splity deterministyczne
                # (seed=42) więc kolejne biegi mogą wrócić na False, jeśli nic się nie zmieniło
                # w danych na dysku (skan ~18k zdjęć trwa kilka minut).

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Ścieżka do zdjęć — dwie stałe, LOCATION wybiera jedną (identycznie jak main.py/main_e9.py).
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"  # serwer (Linux)
IMAGE_DIR_LOCAL  = "Z:/Photo/Otolithes/HER/Processed"                       # Twój komp (Windows)
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

RUN_TAG = datetime.now().strftime("%d.%m") + "_attention_first"
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config_attention_first.yaml")

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
            f"[main_attention_first] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna,\n"
            f"       albo popraw IMAGE_DIR_SERVER / IMAGE_DIR_LOCAL powyżej."
        )
    print(f"[main_attention_first] LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}  RESCAN={RESCAN}")
    print(f"[main_attention_first] BASE_CONFIG={BASE_CONFIG}")
    print(f"[main_attention_first] OUTPUT_DIR={OUTPUT_DIR}")
    main(ARGV)
