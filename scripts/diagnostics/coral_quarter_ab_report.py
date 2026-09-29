"""24.09 — analiza pełnego A/B korekty kwartalnej (2 ramiona × 3 ziarna, fine-tuning).

Czyta katalogi biegów wyprodukowane przez `main_quarter_ab.py` i składa porównanie parowane.
Nic nie liczy na GPU — wszystko z `predictions.csv`.

CO ROBI, ŻEBY PORÓWNANIE BYŁO UCZCIWE

1. **Sprowadza oba ramiona do jednej skali.** Ramię ON trenowało na „liczbie widocznych
   pierścieni", ramię OFF na wieku zapisanym. Predykcje ON są rebasowane (+1 dla połowów Q1,
   `src.report_common.rebase_to_recorded`) i OBA ramiona są punktowane przeciw wiekowi
   zapisanemu z `data/labels_embedded.csv`. Bez tego porównywałoby się dwa różne zadania.
2. **Weryfikuje, że bieg naprawdę miał tę flagę, jaką deklaruje jego katalog.** Skala celu jest
   odczytywana z `target_age` w `predictions.csv` (`src.report_common.target_definition`), nie
   z nazwy katalogu — bo raz już się zdarzyło, że config nigdy nie trafił na serwer i bieg
   policzył czyste domyślne (`plans and summaries/22.09_wedge_b_analiza.md`).
3. **Paruje po ziarnie.** To samo ziarno = ta sama inicjalizacja w obu ramionach, więc różnica
   per ziarno usuwa wariancję inicjalizacji.
4. **Sprawdza, że zbiór testowy jest ten sam** we wszystkich biegach, przez hash zbioru
   `image_id`.

    python scripts/diagnostics/coral_quarter_ab_report.py

Zapisuje experiments/quarter_ab_full/{metrics.json, per_run.csv, notes.md} i narrację do
"plans and summaries/".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.report_common import (
    compute_metrics,
    compute_per_age_metrics,
    macro_exact,
    rebase_to_recorded,
    target_definition,
)

LABELS = PROJECT_ROOT / "data" / "labels_embedded.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "quarter_ab_full"
NARRATIVE = PROJECT_ROOT / "plans and summaries" / "24.09_quarter_ab_wyniki.md"
RUN_RE = re.compile(r"quarter_ab_(off|on)_seed(\d+)$")


def discover_runs(roots: list[Path]) -> list[dict]:
    """Every `*_quarter_ab_{arm}_seed{n}` directory that has scored predictions."""
    found = []
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            m = RUN_RE.search(d.name)
            if not m:
                continue
            csv = d / "emb_on_emb" / "predictions.csv"
            if csv.exists():
                found.append({"dir": d, "arm": m.group(1), "seed": int(m.group(2)),
                              "csv": csv})
    return found


def testset_signature(image_ids) -> str:
    return hashlib.sha1("\n".join(sorted(map(str, image_ids))).encode()).hexdigest()[:12]


def score_run(run: dict, recorded: pd.Series) -> dict:
    """Metrics for one run, on the recorded-age scale, with the arm's flag verified."""
    df = pd.read_csv(run["csv"]).dropna(subset=["target_age"])
    joined = df.merge(recorded.rename("rec"), left_on="image_id", right_index=True)
    scale = target_definition(joined["image_id"].values, joined["target_age"].values,
                             joined["rec"].values)

    expected = "rings(-1 for Q1)" if run["arm"] == "on" else "recorded"
    flag_ok = scale == expected

    pred = (rebase_to_recorded(joined["image_id"].values, joined["predicted_age"].values)
            if scale == "rings(-1 for Q1)" else joined["predicted_age"].to_numpy(dtype=float))
    truth = joined["rec"].to_numpy(dtype=float)
    m = compute_metrics(truth, pred)
    pa = compute_per_age_metrics(truth, pred)

    fish_csv = run["dir"] / "emb_on_emb" / "predictions_per_fish.csv"
    fish = None
    if fish_csv.exists():
        fdf = pd.read_csv(fish_csv).dropna(subset=["target_age"])
        if not fdf.empty:
            # The per-fish file is on the arm's own target scale, and the fish key carries the
            # campaign, so the same rebase applies.
            fpred = (rebase_to_recorded(fdf["fish_id"].values, fdf["predicted_age"].values)
                     if scale == "rings(-1 for Q1)" else
                     fdf["predicted_age"].to_numpy(dtype=float))
            ftruth = (fdf["target_age"].to_numpy(dtype=float)
                      + (np.isin([str(x).split("_")[1] for x in fdf["fish_id"]],
                                 ("BITS1q", "BITS2q")).astype(float)
                         if scale == "rings(-1 for Q1)" else 0.0))
            fm = compute_metrics(ftruth, fpred)
            fish = {"n_fish": int(len(fdf)), "Exact": fm["Exact"], "MAE": fm["MAE"],
                    "Acc1yr": fm["Acc1yr"]}

    return {"arm": run["arm"], "seed": run["seed"], "run": run["dir"].name,
            "target_scale": scale, "flag_as_declared": flag_ok,
            "n": m["N"], "signature": testset_signature(joined["image_id"]),
            "Exact": m["Exact"], "Acc1yr": m["Acc1yr"], "MAE": m["MAE"],
            "MedAE": m["MedAE"], "Bias": m["Bias"], "MacroExact": macro_exact(pa),
            "fish": fish}


def paired_stats(rows: pd.DataFrame, field: str = "Exact") -> dict:
    """Per-seed differences ON minus OFF, over seeds present in BOTH arms."""
    off = rows[rows["arm"] == "off"].set_index("seed")[field]
    on = rows[rows["arm"] == "on"].set_index("seed")[field]
    common = sorted(set(off.index) & set(on.index))
    if not common:
        return {"n_seeds": 0, "note": "no seed present in both arms"}
    d = np.array([on[s] - off[s] for s in common], dtype=float)
    out = {"n_seeds": len(common), "seeds": common,
           "per_seed": [float(x) for x in d],
           "mean": float(d.mean()),
           "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
           "wins_for_ON": int((d > 0).sum())}
    if len(d) > 1:
        t = stats.ttest_rel([on[s] for s in common], [off[s] for s in common])
        out["p_two_sided"] = float(t.pvalue)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="*", default=None,
                    help="katalogi z biegami (domyslnie outputs/ i outputs/data/)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    roots = ([Path(r) for r in args.roots] if args.roots
             else [PROJECT_ROOT / "outputs", PROJECT_ROOT / "outputs" / "data"])
    runs = discover_runs(roots)
    if not runs:
        raise SystemExit(
            "Nie znaleziono zadnego biegu *_quarter_ab_{off,on}_seed*.\n"
            "Odpal najpierw: python main_quarter_ab.py (na serwerze).")

    recorded = pd.read_csv(LABELS).set_index("image_id")["age"]
    rows = pd.DataFrame([score_run(r, recorded) for r in runs])
    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.drop(columns=["fish"]).to_csv(out_dir / "per_run.csv", index=False)

    problems = []
    bad_flag = rows[~rows["flag_as_declared"]]
    for _, r in bad_flag.iterrows():
        problems.append(
            f"`{r['run']}`: katalog deklaruje ramie **{r['arm'].upper()}**, ale skala celu w "
            f"predictions.csv to `{r['target_scale']}`. Config prawdopodobnie nie trafil do "
            f"biegu — ten bieg NIE wchodzi do porownania.")
    if rows["signature"].nunique() > 1:
        problems.append("Biegi nie maja wspolnego zbioru testowego — porownanie parowane "
                        "jest niewazne.")

    valid = rows[rows["flag_as_declared"]]
    payload = {
        "question": "czy data.quarter_age_adjustment_enabled podnosi exact accuracy",
        "regime": "pelny fine-tuning, 2 ramiona x N ziaren",
        "scoring": "oba ramiona punktowane przeciw wiekowi ZAPISANEMU; ramie ON rebasowane",
        "runs": json.loads(rows.to_json(orient="records")),
        "paired_exact": paired_stats(valid, "Exact"),
        "paired_mae": paired_stats(valid, "MAE"),
        "arms": {arm: {f: {"mean": float(g[f].mean()),
                           "sd": float(g[f].std(ddof=1)) if len(g) > 1 else 0.0}
                       for f in ("Exact", "Acc1yr", "MAE", "MacroExact", "Bias")}
                 for arm, g in valid.groupby("arm")},
        "problems": problems,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    write_notes(payload, valid)
    pe = payload["paired_exact"]
    if pe.get("n_seeds"):
        print(f"delta exact = {100 * pe['mean']:+.2f} pp +- {100 * pe['sd']:.2f} (sd), "
              f"ON wins {pe['wins_for_ON']}/{pe['n_seeds']}, p={pe.get('p_two_sided')}")
    for p in problems:
        print("PROBLEM: " + p)
    print(f"wrote {out_dir} and {NARRATIVE}")


def write_notes(payload: dict, valid: pd.DataFrame) -> None:
    def pct(x): return f"{100 * x:.2f} %"
    L = ["# 24.09 — Wyniki A/B korekty kwartalnej (pelny fine-tuning)", "",
         f"Rezim: {payload['regime']}.  Punktacja: {payload['scoring']}.",
         "", "Skrypt: `scripts/diagnostics/coral_quarter_ab_report.py`. "
         "Biegi: `main_quarter_ab.py`.", ""]
    if payload["problems"]:
        L += ["## Problemy — przeczytaj PRZED liczbami", ""]
        L += [f"- {p}" for p in payload["problems"]] + [""]
    L += ["## Per bieg", "",
          "| bieg | ramie | ziarno | skala celu | exact | +-1 rok | MAE | bias |",
          "|---|---|---:|---|---:|---:|---:|---:|"]
    for _, r in valid.sort_values(["arm", "seed"]).iterrows():
        L.append(f"| `{r['run']}` | {r['arm'].upper()} | {r['seed']} | `{r['target_scale']}` "
                 f"| **{pct(r['Exact'])}** | {pct(r['Acc1yr'])} | {r['MAE']:.4f} "
                 f"| {r['Bias']:+.3f} |")
    L += ["", "## Ramiona", "",
          "| ramie | exact | +-1 rok | MAE | macro exact | bias |",
          "|---|---:|---:|---:|---:|---:|"]
    for arm in ("off", "on"):
        a = payload["arms"].get(arm)
        if not a:
            continue
        L.append(f"| {arm.upper()} | **{pct(a['Exact']['mean'])}** ± {100 * a['Exact']['sd']:.2f} "
                 f"| {pct(a['Acc1yr']['mean'])} | {a['MAE']['mean']:.4f} ± {a['MAE']['sd']:.4f} "
                 f"| {pct(a['MacroExact']['mean'])} | {a['Bias']['mean']:+.3f} |")
    pe, pm = payload["paired_exact"], payload["paired_mae"]
    if pe.get("n_seeds"):
        L += ["", "## Test parowany po ziarnach", "",
              f"- delta exact = **{100 * pe['mean']:+.2f} pkt proc.** (sd {100 * pe['sd']:.2f})",
              f"- delta MAE = **{pm['mean']:+.4f}**",
              f"- ON wygrywa w **{pe['wins_for_ON']}/{pe['n_seeds']}** ziarnach",
              f"- p (dwustronne, parowane) = {pe.get('p_two_sided')}", "",
              "Per ziarno: " + ", ".join(f"{100 * d:+.2f}" for d in pe["per_seed"]), ""]
    NARRATIVE.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
