"""30.09 — S2: laboratorium głowicy gęstości na serwerze, od testów do tabeli wyników.

SERWER NIE MA GPU — wszystko liczy się na CPU. Zakres wybrany pod to (30.09, "priorytet
wedge_c"): serwer liczy A3 × 3 ziarna (~47 min/epokę na serwerze), potem A4 × 3 (maks. 20 epok, stop 3 epoki
po dojrzeniu, zapis stanu co epokę), komputer lokalny równolegle A5. A0/A1/A2 i kontrola na
cache (b) są odłożone — można je dopisać do ARMS_RAW / ARMS_CONTROL.

CO ROZSTRZYGA. Który przepis głowicy gęstości dojrzewa niezawodnie na geometrii wycinka
(5852 patchy), zanim puścimy kolejny pełny bieg (~5 h/epokę). Bieg wedge_b zapadł się do
stanu absorbującego (logity ≈ −13). Plan i uzasadnienie ramion:
`plans and summaries/29.09_ewdge_b_przerwany_PLAN_TO_do.md` §5 i §8.

CO ROBI, PO KOLEI (każdy krok pomija to, co już jest gotowe, więc przerwany bieg wznawia
się zwykłym ponownym uruchomieniem):
  1. kontrole: katalog zdjęć, etykiety, checkpoint wedge_b (gdy potrzebny), CPU;
  2. testy laboratorium i cache (pytest) — bez zielonych testów nic dalej nie rusza;
  3. cache tokenów pasm: (a) surowy DINOv2; (b) backbone z best_age.pt wedge_b tylko, gdy
     ARMS_CONTROL nie jest puste;
  4. ramiona ARMS_RAW × SEEDS na cache (a) (+ ARMS_CONTROL na cache (b));
  5. raport: experiments/density_head_lab/WYNIKI.md + wyniki.json.

A5 i A4 liczą się lokalnie; ich wyniki dołączy ten sam raport, jeśli katalogi
experiments/density_head_lab/raw/A5_seed*, A4_seed* zostaną tu skopiowane.

    python main_density_lab.py              # całość
    python main_density_lab.py --report     # tylko raport z tego, co już policzone
    python main_density_lab.py --dry-run    # plan i kontrole, bez liczenia
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# KONFIGURACJA — zmień tylko tutaj
# ============================================================

LOCATION = "server"          # "server" → serwer (Linux)  |  "local" → Twój komp (Windows, Z:)

ARMS_RAW = ["A3", "A4"]      # A3 główny kandydat, A4 okno kanwy (przeniesione z lokalnego: serwer ~3,4× szybszy); A0/A1/A2 odłożone
ARMS_CONTROL = []            # ["A0"] = kontrola na cache (b), wymaga zbudowania drugiego cache (~28 GB)
SEEDS = [0, 1, 2]
EPOCHS = 20                  # bramka patrzy tylko na epoki <= 20
STOP_AFTER_MATURE = 3        # ziarno, które dojrzało, kończy 3 epoki później
BATCH_SIZE = 16
LABEL = "quarter"            # cel liczenia: "quarter" (wiek po korekcie kwartalnej) | "recorded"
CACHE_WORKERS = 8            # procesy ekstrakcji kanw przy budowie cache

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PY = sys.executable

IMAGE_DIR_SERVER = "/home/kswitek/Documents/Photo/Otolithes/HER/Processed"
IMAGE_DIR_LOCAL = "Z:/Photo/Otolithes/HER/Processed"
IMAGE_DIR = IMAGE_DIR_SERVER if LOCATION == "server" else IMAGE_DIR_LOCAL
CKPT_B = PROJECT_ROOT / ("outputs/data/22.09_wedge_b" if LOCATION == "server"
                         else "outputs/22.09_wedge_b") / "checkpoints" / "embedded" / "best_age.pt"
LABELS = PROJECT_ROOT / "data" / "labels_embedded.csv"
CACHE_ROOT = PROJECT_ROOT / "data" / "wedge_band_tokens"
LAB_ROOT = PROJECT_ROOT / "experiments" / "density_head_lab"
LOG = PROJECT_ROOT / "logs" / "main_density_lab.log"

CACHES = {"raw": []} | ({"wedge_b_best_age": ["--backbone-from", str(CKPT_B)]} if ARMS_CONTROL else {})
TESTS = ["tests/test_density_head_lab.py", "tests/test_cache_wedge_band_tokens.py",
         "tests/test_stage4_trainer.py"]

# Maturity gate of the plan (§5): active ≥ 1 and zero_ratio < 0.5 before epoch 20, in ≥ 4/5 seeds;
# with fewer than 5 seeds every seed has to mature.
GATE_SEEDS = 4 if len(SEEDS) >= 5 else len(SEEDS)
# wedge_b's own state from e16 on: 97–99 % of the zero-map loss.
COLLAPSED_RATIO = 0.9


def log(msg: str) -> None:
    line = f"[main_density_lab {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str]) -> None:
    """Run a step, streaming its output to the console and the log; stop on failure."""
    log("$ " + " ".join(cmd))
    with LOG.open("a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", bufsize=1)
        for line in p.stdout:
            if "xFormers is not available" in line or line.strip().startswith("warnings.warn"):
                continue
            print(line, end="", flush=True)
            f.write(line)
        rc = p.wait()
    if rc != 0:
        log(f"KROK NIEUDANY (kod {rc}) — przerwano. Szczegóły: {LOG}")
        sys.exit(rc)


def cache_complete(tag: str) -> bool:
    done = CACHE_ROOT / tag / "done.npy"
    if not done.exists():
        return False
    import numpy as np
    return bool(np.load(done).all())


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def preflight() -> None:
    problems = []
    if not Path(IMAGE_DIR).is_dir():
        problems.append(f"brak katalogu zdjęć {IMAGE_DIR!r} (LOCATION={LOCATION!r})")
    if ARMS_CONTROL and not CKPT_B.exists():
        problems.append(f"brak checkpointu wedge_b {CKPT_B}")
    if not LABELS.exists():
        problems.append(f"brak etykiet {LABELS}")
    for t in TESTS:
        if not (PROJECT_ROOT / t).exists():
            problems.append(f"brak pliku {t} — zrób git pull")
    if problems:
        for p in problems:
            log("BŁĄD: " + p)
        sys.exit(1)
    import os
    import torch
    log(f"CPU: {os.cpu_count()} rdzeni, torch {torch.__version__}, wątki {torch.get_num_threads()} "
        f"(serwer nie ma GPU — wszystko na CPU)")


def step_tests() -> None:
    run([PY, "-m", "pytest", *TESTS, "-q"])


def step_caches() -> None:
    for tag, extra in CACHES.items():
        if cache_complete(tag):
            log(f"cache {tag}: kompletny — pomijam")
            continue
        run([PY, "scripts/diagnostics/cache_wedge_band_tokens.py", "--tag", tag,
             "--workers", str(CACHE_WORKERS), *extra])


def step_lab() -> None:
    common = ["--seeds", ",".join(map(str, SEEDS)), "--epochs", str(EPOCHS),
              "--stop-after-mature", str(STOP_AFTER_MATURE), "--device", "cpu",
              "--batch-size", str(BATCH_SIZE), "--label", LABEL, "--skip-done"]
    for cache, arms in (("raw", ARMS_RAW), ("wedge_b_best_age", ARMS_CONTROL)):
        if not arms:
            continue
        for arm in arms:           # one arm per call, so a failure costs at most one arm
            run([PY, "scripts/diagnostics/density_head_lab.py", "--cache", cache,
                 "--arms", arm, *common])
        write_report()             # partial report after each cache


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def seed_rows(cache: str) -> list[dict]:
    import pandas as pd
    rows = []
    for d in sorted((LAB_ROOT / cache).glob("*_seed*")):
        s_path, m_path = d / "summary.json", d / "metrics.csv"
        if not (s_path.exists() and m_path.exists()):
            continue
        s = json.loads(s_path.read_text(encoding="utf-8"))
        m = pd.read_csv(m_path)
        last = m.iloc[-1]
        rows.append({"cache": cache, "arm": s["arm"], "seed": s["seed"],
                     "matured": bool(s["matured"]), "matured_epoch": s["matured_epoch"],
                     "epochs": int(last["epoch"]),
                     "zero_ratio_e0": float(m.iloc[0]["zero_ratio"]),
                     "min_zero_ratio": float(m["zero_ratio"].min()),
                     "final_zero_ratio": float(last["zero_ratio"]),
                     "final_active": float(last["active"]),
                     "final_max_logit": float(last["max_logit"]),
                     "final_median_logit": float(last["median_logit"]),
                     "final_sum_p_over_age": float(last["sum_p_over_age"]),
                     "collapsed": bool(last["zero_ratio"] > COLLAPSED_RATIO and last["active"] < 1.0),
                     "head_pt": str((d / "head.pt").relative_to(PROJECT_ROOT).as_posix())})
    return rows


def arm_table(rows: list[dict]) -> list[dict]:
    import numpy as np
    out = []
    for cache in dict.fromkeys(r["cache"] for r in rows):
        for arm in sorted({r["arm"] for r in rows if r["cache"] == cache}):
            rs = [r for r in rows if r["cache"] == cache and r["arm"] == arm]
            ep = [r["matured_epoch"] for r in rs if r["matured_epoch"] is not None]
            n_mat = sum(r["matured"] for r in rs)
            out.append({
                "cache": cache, "arm": arm, "seeds": len(rs), "matured": n_mat,
                "gate": ("TAK" if n_mat >= GATE_SEEDS else
                         "NIE" if len(rs) - n_mat > len(SEEDS) - GATE_SEEDS else "w toku"),
                "median_matured_epoch": float(np.median(ep)) if ep else None,
                "collapsed_seeds": sum(r["collapsed"] for r in rs),
                "median_final_zero_ratio": float(np.median([r["final_zero_ratio"] for r in rs])),
                "median_final_active": float(np.median([r["final_active"] for r in rs])),
                "median_final_max_logit": float(np.median([r["final_max_logit"] for r in rs])),
            })
    return out


ARM_DESC = {"A0": "przepis wedge_b bez zmian", "A1": "+ bias z priorem",
            "A2": "+ BCE na logitach", "A3": "prior + BCE",
            "A4": "A3 + okno kanwy 21 patchy", "A5": "A3 + głowica wierszowa (44)"}


def verdicts(table: list[dict]) -> list[str]:
    get = {(t["cache"], t["arm"]): t for t in table}
    lines = []
    ctrl = get.get(("wedge_b_best_age", "A0"))
    if ctrl:
        ok = ctrl["collapsed_seeds"] == ctrl["seeds"] and ctrl["matured"] == 0
        lines.append(f"Kontrola laboratorium (A0 na tokenach backbone'u wedge_b): zapaść w "
                     f"{ctrl['collapsed_seeds']}/{ctrl['seeds']} ziaren, dojrzało {ctrl['matured']}. "
                     + ("Laboratorium odtwarza zjawisko z serwera." if ok else
                        "Laboratorium NIE odtwarza w pełni zapaści — wyniki ramion czytać ostrożnie."))
    a0 = get.get(("raw", "A0"))
    if a0:
        lines.append(f"Mechanizm 3 (dryf wejścia): A0 na zamrożonym DINOv2 dojrzało w "
                     f"{a0['matured']}/{a0['seeds']} ziaren — "
                     + ("dryf był warunkiem koniecznym zapaści." if a0["gate"] == "TAK" else
                        "sam brak dryfu nie wystarcza; przyczyna leży w stracie, inicjalizacji lub oknie."))
    passing = [t for t in table if t["cache"] == "raw" and t["gate"] == "TAK"]
    if passing:
        lines.append(f"Ramiona przechodzące bramkę dojrzałości (≥ {GATE_SEEDS}/{len(SEEDS)} ziaren): "
                     + ", ".join(f"{t['arm']} ({t['matured']}/{t['seeds']}, mediana e"
                                 f"{t['median_matured_epoch']:.0f})" for t in passing)
                     + ". Następny krok: ocena ZEGAR ich wag (L8).")
    elif table:
        lines.append("Żadne ramię nie przeszło jeszcze bramki dojrzałości.")
    return lines


def write_report() -> None:
    rows = seed_rows("raw") + seed_rows("wedge_b_best_age")
    table = arm_table(rows)
    LAB_ROOT.mkdir(parents=True, exist_ok=True)
    (LAB_ROOT / "wyniki.json").write_text(json.dumps({"arms": table, "seeds": rows,
                                                      "verdicts": verdicts(table)},
                                                     indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Laboratorium głowicy gęstości — wyniki", "",
          f"Bramka: `active ≥ 1` i `zero_ratio < 0,5` w epokach 1–20, w ≥ {GATE_SEEDS}/{len(SEEDS)} ziaren. "
          f"Zapaść: końcowe `zero_ratio > {COLLAPSED_RATIO}` i `active < 1`. Cel liczenia: `{LABEL}`.", "",
          "## Werdykty", ""] + [f"- {v}" for v in verdicts(table)] + [
          "", "## Ramiona", "",
          "| cache | ramię | opis | dojrzało | bramka | mediana epoki | zapaść | zero_ratio końc. | active końc. | max logit końc. |",
          "|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for t in table:
        ep = f"{t['median_matured_epoch']:.0f}" if t["median_matured_epoch"] is not None else "—"
        md.append(f"| {t['cache']} | {t['arm']} | {ARM_DESC.get(t['arm'], '')} | {t['matured']}/{t['seeds']} "
                  f"| {t['gate']} | {ep} | {t['collapsed_seeds']}/{t['seeds']} "
                  f"| {t['median_final_zero_ratio']:.3f} | {t['median_final_active']:.2f} "
                  f"| {t['median_final_max_logit']:.2f} |")
    md += ["", "## Ziarna", "",
           "| cache | ramię | ziarno | dojrzało (epoka) | zero_ratio e0 → min → końc. | active końc. | max / mediana logit | Σp/wiek | wagi |",
           "|---|---|---:|---|---|---:|---|---:|---|"]
    for r in rows:
        mat = f"tak (e{r['matured_epoch']})" if r["matured"] else ("zapaść" if r["collapsed"] else "nie")
        md.append(f"| {r['cache']} | {r['arm']} | {r['seed']} | {mat} "
                  f"| {r['zero_ratio_e0']:.3f} → {r['min_zero_ratio']:.3f} → {r['final_zero_ratio']:.3f} "
                  f"| {r['final_active']:.2f} | {r['final_max_logit']:.2f} / {r['final_median_logit']:.2f} "
                  f"| {r['final_sum_p_over_age']:.2f} | `{r['head_pt']}` |")
    (LAB_ROOT / "WYNIKI.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"raport: {LAB_ROOT / 'WYNIKI.md'}  ({len(rows)} ziaren)")


def main() -> None:
    ap = argparse.ArgumentParser(description="S2: laboratorium głowicy gęstości")
    ap.add_argument("--report", action="store_true", help="tylko raport z gotowych wyników")
    ap.add_argument("--dry-run", action="store_true", help="kontrole i plan, bez liczenia")
    args = ap.parse_args()

    if args.report:
        write_report()
        return
    log(f"LOCATION={LOCATION}  IMAGE_DIR={IMAGE_DIR}  CKPT_B={CKPT_B}")
    log(f"plan: cache {list(CACHES)}; ramiona raw {ARMS_RAW} × ziarna {SEEDS}; "
        f"kontrola {ARMS_CONTROL} na wedge_b_best_age; {EPOCHS} epok, batch {BATCH_SIZE}, cel {LABEL}")
    preflight()
    if args.dry_run:
        for tag in CACHES:
            log(f"cache {tag}: {'kompletny' if cache_complete(tag) else 'do zbudowania / dokończenia'}")
        return
    step_tests()
    step_caches()
    step_lab()
    write_report()
    log("KONIEC — wyniki w experiments/density_head_lab/WYNIKI.md")


if __name__ == "__main__":
    main()
