"""Root entry point — kliknij ▶ w PyCharm aby uruchomić pipeline.

Zmień MODE poniżej przed uruchomieniem:
  "demo"  — 1 epoka, outputs/demo/    (szybki test całego pipeline'u)
  "full"  — 50 epok, outputs/         (pełny trening Embedded + NotEmbedded)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# ============================================================
# KONFIGURACJA — zmień tylko tutaj
# ============================================================

MODE = "full"   # "demo"  → 1 epoka, szybki test pipeline'u
                # "full"  → pełny trening (50 epok na model)

SKIP_SCAN  = False   # True  = używaj istniejących data/labels_*.csv
                    # False = skanuj Z: od nowa (kilka minut)

SKIP_TRAIN = False  # True  = pomija trening; używa istniejących checkpoints

FRESH      = False  # True  = kasuje pipeline_state.json i robi pełny re-run od zera
                    #         (użyj, gdy „wszystkie kroki pominięte bo już wykonane”)

EMBEDDED_ONLY = True  # True = trenuj/raportuj TYLKO Embedded (szybciej;
                       #        pomija NotEmbedded i cross-domain). Kod NotEmbedded zostaje.

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Zdjęcia leżą na dysku sieciowym Windows (Z:) lokalnie, a pod home na serwerze.
# Bierzemy PIERWSZĄ istniejącą ścieżkę (env var ma pierwszeństwo), więc TEN SAM
# main.py działa na obu maszynach bez edycji. UWAGA: "Z:/..." na Linuxie to
# ścieżka WZGLĘDNA — doklejała się do PROJECT_ROOT (stąd błąd
# ".../DINO_project_Herring/Z:/Photo/..."). Nadpisz w dowolnej chwili:
#   export OTOLITH_IMAGE_DIR=/pełna/ścieżka/do/Processed
_IMAGE_DIR_CANDIDATES = [
    os.environ.get("OTOLITH_IMAGE_DIR"),
    "/home/kswitek/Documents/Photo/Otolithes/HER/Processed",
    "/home/kswitek/Documents/Photo/Otolithes/HER",
    "Z:/Photo/Otolithes/HER/Processed",
]
IMAGE_DIR = next((p for p in _IMAGE_DIR_CANDIDATES if p and Path(p).is_dir()),
                 "Z:/Photo/Otolithes/HER/Processed")
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

if MODE == "demo":
    BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config_demo.yaml")
    OUTPUT_DIR  = str(PROJECT_ROOT / "outputs" / "demo")
else:
    BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config.yaml")
    OUTPUT_DIR  = str(PROJECT_ROOT / "outputs")

ARGV = [
    "--base-config",          BASE_CONFIG,
    "--image-dir",            IMAGE_DIR,
    "--excel",                EXCEL_PATH,
    "--output-dir",           OUTPUT_DIR,
    "--config-embedded",      str(PROJECT_ROOT / "configs" / "config_embedded.yaml"),
    "--config-not-embedded",  str(PROJECT_ROOT / "configs" / "config_not_embedded.yaml"),
]
if SKIP_SCAN:
    ARGV.append("--skip-scan")
if SKIP_TRAIN:
    ARGV.append("--skip-train")
if FRESH:
    ARGV.append("--fresh")
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
            f"       Ustaw poprawną ścieżkę serwerową:\n"
            f"         export OTOLITH_IMAGE_DIR=/pełna/ścieżka/do/Processed\n"
            f"       (albo popraw listę _IMAGE_DIR_CANDIDATES w main.py).\n"
            f"       Znajdź katalog ze zdjęciami np.:  find ~ -type d -name Processed"
        )
    print(f"[main] IMAGE_DIR = {IMAGE_DIR}")
    main(ARGV)


test