"""24.09 — CZYSTY A/B korekty kwartalnej, 6 biegów (2 ramiona × 3 ziarna).

CO ROZSTRZYGA. `06.08_attention_first` (korekta ON) daje 53,85 % exact, `09.08_isolate_a` (OFF)
43,71 % — ale te dwa biegi różniły się TAKŻE `density_concentricity_weight` (E9, 1,0 vs 0,0)
i całą trajektorią optymalizacji, więc +10,1 pkt proc. nie jest przypisane fladze. Tutaj E9
jest wyłączone w OBU ramionach, a `configs/config_quarter_ab_{off,on}.yaml` różnią się
**dokładnie jednym polem** — przypięte testem `tests/test_quarter_ab_configs.py`.

DLACZEGO 3 ZIARNA. Rozrzut exact między biegami o rozsprzęgniętych gałęziach sięga ±6 pkt
proc., a błąd standardowy przy n=1105 to ~1,5 pkt proc. Jeden bieg na ramię nie rozstrzygnąłby
efektu mniejszego niż ~4 pkt proc. `project.seed` idzie z `--train-seed`, więc **podział danych
nie jest ruszany** — porównanie zostaje na tych samych obrazach.

ZANIM UWIERZYSZ W WYNIK. Sprawdź linię `RUN IDENTITY` w każdym `train.log` —
`data.quarter_age_adjustment_enabled` musi być różne między ramionami i identyczne w obrębie
ramienia. To ta sama pułapka, która zamieniła bieg 09.09 w pusty baseline (config nigdy nie
trafił na serwer, `load_merged_config` po cichu wziął domyślne).

WYNIK CZĄSTKOWY, KTÓRY JUŻ JEST. Ten sam A/B na zamrożonym backbone (cache tokenów CLS,
`scripts/diagnostics/coral_head_ab.py`) daje czystą dolną granicę efektu z 10 ziarnami —
patrz `experiments/quarter_ab/`. Ten bieg dokłada brakujący kawałek: co zyskuje backbone,
który ma swobodę dostosowania się do zmienionego celu.

Analiza po zakończeniu: `python scripts/diagnostics/coral_quarter_ab_report.py`.

Uruchomienie na serwerze (sekwencyjnie, ~19-20 h na bieg):
    python main_quarter_ab.py
albo jedno ramię/ziarno osobno:
    python main_quarter_ab.py --arm on --seed 7
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# KONFIGURACJA — zmień tylko tutaj
# ============================================================

LOCATION = "server"   # "server" → serwer (Linux)  |  "local" → Twój komp (Windows, Z:)

EMBEDDED_ONLY = True  # True = trenuj/raportuj TYLKO Embedded

RESCAN = False        # Korekta kwartalna czyta kampanię z nazwy pliku, gdy kolumna "campaign"
                      # nie istnieje (src/dataset.py::_effective_age), więc istniejące
                      # data/labels_*.csv wystarczają. Splity muszą zostać TE SAME — rescan
                      # przebudowałby je i unieważnił porównanie z historią.

SEEDS = [42, 7, 13]   # project.seed per bieg; podział danych NIE jest tym ruszany
ARMS = ["off", "on"]

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"
IMAGE_DIR_LOCAL = "Z:/Photo/Otolithes/HER/Processed"
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
EXCEL_PATH = str(PROJECT_ROOT / "data" / "analysisWithOtolithPhoto.xlsx")

DATE_TAG = datetime.now().strftime("%d.%m")


def argv_for(arm: str, seed: int) -> list[str]:
    run_tag = f"{DATE_TAG}_quarter_ab_{arm}_seed{seed}"
    argv = [
        "--base-config", str(PROJECT_ROOT / "configs" / f"config_quarter_ab_{arm}.yaml"),
        "--image-dir", IMAGE_DIR,
        "--excel", EXCEL_PATH,
        "--output-dir", str(PROJECT_ROOT / "outputs" / "data" / run_tag),
        "--config-embedded", str(PROJECT_ROOT / "configs" / "config_embedded.yaml"),
        "--config-not-embedded", str(PROJECT_ROOT / "configs" / "config_not_embedded.yaml"),
        "--train-seed", str(seed),
    ]
    if RESCAN:
        argv.append("--rescan")
    if EMBEDDED_ONLY:
        argv.append("--embedded-only")
    return argv


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B korekty kwartalnej")
    ap.add_argument("--arm", choices=ARMS, default=None, help="tylko to ramię")
    ap.add_argument("--seed", type=int, default=None, help="tylko to ziarno")
    ap.add_argument("--dry-run", action="store_true", help="wypisz plan i wyjdź")
    args = ap.parse_args()

    if not Path(IMAGE_DIR).is_dir():
        sys.exit(
            f"[main_quarter_ab] Katalog zdjęć nie istnieje: {IMAGE_DIR!r}\n"
            f"       LOCATION = {LOCATION!r} — sprawdź czy to właściwa maszyna."
        )
    for arm in ARMS:
        cfg = PROJECT_ROOT / "configs" / f"config_quarter_ab_{arm}.yaml"
        if not cfg.exists():
            sys.exit(f"[main_quarter_ab] Brak configu: {cfg}")

    arms = [args.arm] if args.arm else ARMS
    seeds = [args.seed] if args.seed is not None else SEEDS
    plan = [(a, s) for a in arms for s in seeds]

    print(f"[main_quarter_ab] LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}  RESCAN={RESCAN}")
    print(f"[main_quarter_ab] {len(plan)} biegow: " +
          ", ".join(f"{a}/seed{s}" for a, s in plan))
    print("[main_quarter_ab] Po kazdym biegu sprawdz linie RUN IDENTITY w train.log: "
          "data.quarter_age_adjustment_enabled musi byc rozne miedzy ramionami.")
    if args.dry_run:
        for a, s in plan:
            print(f"  {a}/seed{s}: " + " ".join(argv_for(a, s)))
        return

    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.run_pipeline import main as run_pipeline

    for a, s in plan:
        print(f"\n{'=' * 60}\n[main_quarter_ab] RAMIE={a}  SEED={s}\n{'=' * 60}", flush=True)
        run_pipeline(argv_for(a, s))


if __name__ == "__main__":
    main()
