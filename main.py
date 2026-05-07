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

MODE = "demo"   # "demo"  → 1 epoka, szybki test pipeline'u
                # "full"  → pełny trening (50 epok na model)

SKIP_SCAN  = True   # True  = używaj istniejących data/labels_*.csv
                    # False = skanuj Z: od nowa (kilka minut)

SKIP_TRAIN = False  # True  = pomija trening; używa istniejących checkpoints

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR  = "Z:/Photo/Otolithes/HER/Processed"
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

if MODE == "demo":
    BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config_demo.yaml")
    OUTPUT_DIR  = str(PROJECT_ROOT / "outputs" / "demo")
else:
    BASE_CONFIG = str(PROJECT_ROOT / "configs" / "config.yaml")
    OUTPUT_DIR  = str(PROJECT_ROOT / "outputs")

sys.argv = [
    "run_pipeline.py",
    "--base-config",          BASE_CONFIG,
    "--image-dir",            IMAGE_DIR,
    "--excel",                EXCEL_PATH,
    "--output-dir",           OUTPUT_DIR,
    "--config-embedded",      str(PROJECT_ROOT / "configs" / "config_embedded.yaml"),
    "--config-not-embedded",  str(PROJECT_ROOT / "configs" / "config_not_embedded.yaml"),
]
if SKIP_SCAN:
    sys.argv.append("--skip-scan")
if SKIP_TRAIN:
    sys.argv.append("--skip-train")

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
