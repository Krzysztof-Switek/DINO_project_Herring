"""28.08 — Opcja A, powtórka z INNYM SEEDEM (project.seed 42→7), żeby sprawdzić czy Z05/Z38
(katastrofalne regresje względem Run N w pierwszym biegu, seed=42) to powtarzalna właściwość
mechanizmu, czy stochastyka jednego treningu. Pełne uzasadnienie w nagłówku
configs/config_zegar_semi_weak_seed7.yaml i plans and summaries/6.08_PODSUMOWANIE_PROCESU.md
Etap 33 — przeczytaj PRZED odpaleniem.

Te same dane co pierwszy bieg (data/ZEGAR/Processed/, data/zegar_semi_weak_*.csv — NIE
regenerowane, ten sam podział 30 trening / 12 held-out) — jeśli te pliki nie są jeszcze
zsynchronizowane na serwerze, zrób to tak samo jak przy pierwszym biegu Opcji A.

Osobny plik od main_zegar_semi_weak.py — nie nadpisuje żadnego dotychczasowego biegu/configu
(RUN_TAG ma dopisek "_seed7", więc output ląduje w INNYM katalogu niż pierwszy bieg nawet
gdybyś odpalił oba tego samego dnia).
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

RESCAN = False   # False: reuse istniejących data/labels_*.csv, tak samo jak w pierwszym biegu
                 # Opcji A. UWAGA: rescan dotyczy WYŁĄCZNIE głównego labels_csv/image_dir — pliki
                 # ZEGAR (data/zegar_semi_weak_*.csv) są od niego całkowicie niezależne.

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Ścieżka do zdjęć — dwie stałe, LOCATION wybiera jedną (identycznie jak main_zegar_semi_weak.py).
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"  # serwer (Linux)
IMAGE_DIR_LOCAL  = "Z:/Photo/Otolithes/HER/Processed"                       # Twój komp (Windows)
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

# "_seed7" na stałe w tagu (nie tylko w nazwie configu) — gwarantuje osobny katalog wyjściowy
# nawet gdybyś odpalił ten i pierwszy bieg (main_zegar_semi_weak.py) tego samego dnia.
RUN_TAG = datetime.now().strftime("%d.%m") + "_zegar_semi_weak_seed7"
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config_zegar_semi_weak_seed7.yaml")

# Pliki ZEGAR — te same co w pierwszym biegu, sanity-checkowane niżej.
ZEGAR_IMAGE_DIR = PROJECT_ROOT / "data" / "ZEGAR" / "Processed"
ZEGAR_LABELS_CSV = PROJECT_ROOT / "data" / "zegar_semi_weak_labels.csv"
ZEGAR_TARGETS_CSV = PROJECT_ROOT / "data" / "zegar_semi_weak_targets.csv"

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
    # Sanity-check katalogu produkcyjnego UP FRONT — inaczej błąd wyskakuje dopiero w
    # środku treningu (jak w plans and summaries/błąd.md).
    if not Path(IMAGE_DIR).is_dir():
        sys.exit(
            f"[main_zegar_semi_weak_seed7] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna,\n"
            f"       albo popraw IMAGE_DIR_SERVER / IMAGE_DIR_LOCAL powyżej."
        )
    # Sanity-check plików ZEGAR — bez nich zegar_position_weight>0 uczyłby się z pustego
    # zbioru (has_zegar_target zawsze False), po cichu no-op zamiast błędu.
    for p, label in [(ZEGAR_IMAGE_DIR, "katalog crop-ów ZEGAR"),
                      (ZEGAR_LABELS_CSV, "zegar_semi_weak_labels.csv"),
                      (ZEGAR_TARGETS_CSV, "zegar_semi_weak_targets.csv")]:
        if not p.exists():
            sys.exit(
                f"[main_zegar_semi_weak_seed7] Brak {label}: {p}\n"
                f"       Uruchom najpierw: python scripts/prepare_zegar_semi_weak_data.py"
            )
    print(f"[main_zegar_semi_weak_seed7] LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}  RESCAN={RESCAN}")
    print(f"[main_zegar_semi_weak_seed7] BASE_CONFIG={BASE_CONFIG}")
    print(f"[main_zegar_semi_weak_seed7] ZEGAR: {ZEGAR_IMAGE_DIR}")
    print(f"[main_zegar_semi_weak_seed7] OUTPUT_DIR={OUTPUT_DIR}")
    main(ARGV)
