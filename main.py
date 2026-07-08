"""Root entry point — kliknij ▶ w PyCharm aby uruchomić pipeline.

Zmień MODE poniżej przed uruchomieniem:
  "demo"  — 1 epoka, outputs/demo/    (szybki test całego pipeline'u)
  "full"  — 50 epok, outputs/         (pełny trening Embedded + NotEmbedded)
"""
from __future__ import annotations
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
# /home/kswitek/Documents/Photo/Otolithes/HER
IMAGE_DIR  = "Z:/Photo/Otolithes/HER/Processed"
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
    main(ARGV)
