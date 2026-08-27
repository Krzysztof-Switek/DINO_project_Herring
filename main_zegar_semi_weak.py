"""26.08 — Opcja A: semi-slaby nadzor. Dodatkowy czlon straty ciagnie glowice density w
strone prawdziwych pozycji pierscieni z ~30 obrazow ZEGAR, obok glownej petli slabego
nadzoru na ~18 700 obrazach. Pelne uzasadnienie w naglowku configs/config_zegar_semi_weak.yaml
i plans and summaries/6.08_PODSUMOWANIE_PROCESU.md Etap 32 — przeczytaj PRZED odpaleniem.

WYMAGA wczesniejszego uruchomienia scripts/prepare_zegar_semi_weak_data.py (lokalnie, dane
ZEGAR sa w repo) i skopiowania/synchronizacji wynikowych plikow
(data/ZEGAR/Processed/, data/zegar_semi_weak_*.csv) na serwer razem z reszta repo.

Osobny plik od main_radial_attention*.py — nie nadpisuje zadnego dotychczasowego biegu/configu.
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
                 # UWAGA: rescan dotyczy WYŁĄCZNIE głównego labels_csv/image_dir — pliki ZEGAR
                 # (data/zegar_semi_weak_*.csv) są od niego całkowicie niezależne, patrz
                 # config_zegar_semi_weak.yaml.

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Ścieżka do zdjęć — dwie stałe, LOCATION wybiera jedną (identycznie jak main_radial_attention.py).
IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"  # serwer (Linux)
IMAGE_DIR_LOCAL  = "Z:/Photo/Otolithes/HER/Processed"                       # Twój komp (Windows)
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

RUN_TAG = datetime.now().strftime("%d.%m") + "_zegar_semi_weak"
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "data" / RUN_TAG)
BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config_zegar_semi_weak.yaml")

# Pliki ZEGAR — sanity-checkowane niżej, żeby błąd wyskoczył od razu, nie w środku treningu.
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
            f"[main_zegar_semi_weak] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
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
                f"[main_zegar_semi_weak] Brak {label}: {p}\n"
                f"       Uruchom najpierw: python scripts/prepare_zegar_semi_weak_data.py"
            )
    print(f"[main_zegar_semi_weak] LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}  RESCAN={RESCAN}")
    print(f"[main_zegar_semi_weak] BASE_CONFIG={BASE_CONFIG}")
    print(f"[main_zegar_semi_weak] ZEGAR: {ZEGAR_IMAGE_DIR}")
    print(f"[main_zegar_semi_weak] OUTPUT_DIR={OUTPUT_DIR}")
    main(ARGV)
