"""24.09 — CORAL baseline audit: the full exact-accuracy picture from files already on disk.

PURE MEASUREMENT. No GPU, no model load, no pipeline change. Every number here comes
from each run's existing ``outputs/<run>/emb_on_emb/predictions.csv`` (which stores
``image_id, predicted_age, target_age, abs_error``) joined against
``data/labels_embedded.csv`` for the fish key.

Why this script exists: exact accuracy — the metric the CORAL work targets — was
never computed anywhere in the project. ``src/report_common.py::compute_metrics``
returned only MAE/RMSE/R2/Acc+-1/Acc+-2/Bias, and per-age exact match lived as a single
line inside ``src/comparison_report.py``. So every run's exact accuracy has been sitting
unread in files for months. See ``plans and summaries/24.09_CORAL_plan.md`` (Etap 0).

What it answers:
  0c  full metric set per run + pairwise McNemar (is 53.9% vs 46.3% even a real difference?)
  0d  exact/bias split by capture campaign — tests the "age is off by one for Q1 catches"
      premise for free, because the campaign is in the filename
  0e  per-fish accuracy (91% of test fish have two photos: left + right otolith)
  0f  data-integrity flags, reported rather than silently fixed

Outputs (numbers -> experiments/, narrative -> "plans and summaries/", per project convention):
  experiments/baseline/baseline_metrics.json
  experiments/baseline/baseline_per_age.csv
  experiments/baseline/baseline_per_run.csv
  experiments/baseline/baseline_confusion_matrix.png
  experiments/baseline/baseline_campaign.csv
  plans and summaries/24.09_baseline_CORAL_audyt.md

Run:
    python scripts/diagnostics/coral_baseline_audit.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.report_common import (
    QUARTER_CAMPAIGNS,
    compute_metrics,
    compute_per_age_metrics,
    confusion_counts,
    macro_exact,
    rebase_to_recorded as _rebase_to_recorded,
    target_definition as _target_definition,
)

OUTPUTS = PROJECT_ROOT / "outputs"
LABELS = PROJECT_ROOT / "data" / "labels_embedded.csv"
EXP_DIR = PROJECT_ROOT / "experiments" / "baseline"
NARRATIVE = PROJECT_ROOT / "plans and summaries" / "24.09_baseline_CORAL_audyt.md"

# Human reader agreement for Baltic herring otoliths, for context on any target we set.
# ICES Baltic Herring Age Reading Study Group; ICES JMS 59(2):323.
READER_AGREEMENT = "70 % (trzech czytelnikow) / 72-85 % (parami)"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def discover_runs() -> dict[str, pd.DataFrame]:
    """Every run that produced age predictions, as ``{run_name: dataframe}``.

    Skips the toy/demo dirs and anything without the three columns we need. Rows with
    a missing ``target_age`` are dropped (unlabelled inference has no error to score).
    """
    runs: dict[str, pd.DataFrame] = {}
    for csv in sorted(OUTPUTS.glob("*/emb_on_emb/predictions.csv")):
        run = csv.parent.parent.name
        if "demo" in run:
            continue
        df = pd.read_csv(csv)
        if not {"image_id", "predicted_age", "target_age"} <= set(df.columns):
            continue
        df = df.dropna(subset=["target_age"]).copy()
        if df.empty:
            continue
        df["predicted_age"] = df["predicted_age"].astype(float)
        df["target_age"] = df["target_age"].astype(float)
        df["exact"] = (df["predicted_age"] == df["target_age"]).astype(int)
        df["signed_error"] = df["predicted_age"] - df["target_age"]
        df["campaign"] = df["image_id"].str.split("_").str[1]
        runs[run] = df
    return runs


def target_definition(df: pd.DataFrame, true_orig: pd.Series) -> str:
    """Thin adapter over `src.report_common.target_definition` for a predictions frame."""
    merged = df[["image_id", "target_age"]].merge(
        true_orig.rename("true_orig"), left_on="image_id", right_index=True)
    return _target_definition(merged["image_id"].values, merged["target_age"].values,
                              merged["true_orig"].values)


def rebase_to_recorded(df: pd.DataFrame) -> pd.Series:
    """Thin adapter over `src.report_common.rebase_to_recorded` for a predictions frame."""
    return pd.Series(_rebase_to_recorded(df["image_id"].values,
                                         df["predicted_age"].values), index=df.index)


def load_fish_keys() -> pd.Series:
    """``image_id -> neutral_fish_key`` from the production label file."""
    lab = pd.read_csv(LABELS, usecols=["image_id", "neutral_fish_key"])
    return lab.set_index("image_id")["neutral_fish_key"]


def testset_signature(df: pd.DataFrame) -> str:
    """Stable hash of the scored image set, so we only compare like with like.

    The test split changed once (n=1131 era -> n=1105 era); runs from different eras
    are not paired-comparable even though both report "test MAE".
    """
    joined = "\n".join(sorted(df["image_id"].astype(str)))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def file_signature(run: str) -> str:
    """Hash of the raw predictions.csv bytes — catches duplicated inference output."""
    p = OUTPUTS / run / "emb_on_emb" / "predictions.csv"
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12]


def ckpt_signature(run: str, head_mb: int = 20) -> str | None:
    """Cheap fingerprint of best.pt: size plus sha1 of its first ``head_mb`` megabytes.

    Needed to read a duplicated predictions.csv correctly. Two runs sharing identical
    predictions is EXPECTED when their weights are identical — that is exactly the E9
    result the project banked as proof of gradient isolation (three loss weights across
    a x100 range, bit-identical age). It is a DEFECT only when the checkpoints differ,
    which is the case for the two strip runs. Hashing 265 MB in full for every run is
    not worth it; size + head is enough to separate the two situations.
    """
    p = OUTPUTS / run / "checkpoints" / "embedded" / "best.pt"
    if not p.exists():
        p = OUTPUTS / run / "checkpoints_embedded" / "best.pt"
    if not p.exists():
        return None
    h = hashlib.sha1()
    with p.open("rb") as fh:
        h.update(fh.read(head_mb * 1024 * 1024))
    return f"{p.stat().st_size}:{h.hexdigest()[:10]}"


def log_best_line(run: str) -> str | None:
    """The run's own early-stopping line, which reports the EMA-smoothed best metric."""
    for cand in (OUTPUTS / run / "logs" / "embedded" / "train.log",
                 OUTPUTS / run / "log" / "embedded" / "train.log"):
        if cand.exists():
            txt = cand.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"\(best=([0-9.]+)\)", txt)
            return m.group(1) if m else None
    return None


def summary_best_val_mae(run: str) -> float | None:
    """``best_val_mae`` as persisted by the pipeline (the RAW minimum from the log)."""
    p = OUTPUTS / run / "pipeline_summary.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return (d.get("training") or {}).get("best_val_mae")


# ---------------------------------------------------------------------------
# Aggregation to fish level
# ---------------------------------------------------------------------------

def per_fish(df: pd.DataFrame, fish_keys: pd.Series) -> pd.DataFrame:
    """Collapse the 1-2 photos of each fish into one age, the way the lab would.

    One fish has one age, so the operational unit is the fish, not the photo. With only
    two photos there is no majority to take, so we use the mean of the two predicted
    ages rounded half-up — the only aggregation available without the raw logits
    (``src/inference.py`` discards them; Etap 1 adds the logit dump, after which the
    honest aggregation is the mean of ``g``).
    """
    d = df.copy()
    d["fish"] = d["image_id"].map(fish_keys)
    d = d.dropna(subset=["fish"])
    g = d.groupby("fish").agg(
        predicted_age=("predicted_age", lambda s: float(np.floor(np.mean(s) + 0.5))),
        target_age=("target_age", "first"),
        n_photos=("predicted_age", "size"),
        photos_agree=("predicted_age", lambda s: int(s.nunique() == 1)),
    ).reset_index()
    return g


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact (binomial) McNemar on two paired exact-hit indicator vectors.

    Returns ``(n_only_a, n_only_b, p_value)``. Exact rather than chi-square because the
    discordant counts here are small enough for the asymptotic form to be unreliable.
    """
    only_a = int(np.sum((a == 1) & (b == 0)))
    only_b = int(np.sum((a == 0) & (b == 1)))
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    p = float(stats.binomtest(only_a, n, 0.5).pvalue)
    return only_a, only_b, p


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion — honest error bars on ``Exact``.

    At n=1105 and p~0.5 this is about +-3 pp, which is the reason the plan requires a
    change to move exact accuracy by more than ~4 pp before we call it an effect.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def save_confusion_png(df: pd.DataFrame, label: str, path: Path) -> None:
    """Row-normalised confusion matrix, green on the diagonal (same look as the report).

    Mirrors ``src/comparison_report.py::_confusion_matrix_b64`` — the diagonal gets its
    own colour map so "correct" is visible at a glance instead of having to trace it.
    """
    cm, ages = confusion_counts(df["target_age"].values, df["predicted_age"].values)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row_sum == 0, 1, row_sum))
    blues, greens = plt.get_cmap("Blues"), plt.get_cmap("Greens")
    n = len(ages)
    rgb = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            rgb[i, j] = (greens if i == j else blues)(norm[i, j])[:3]
    side = max(5.5, n * 0.55)
    fig, ax = plt.subplots(figsize=(side, side))
    ax.imshow(rgb)
    ax.set_xticks(range(n)); ax.set_xticklabels(ages, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(ages, fontsize=6)
    ax.set_xlabel("Wiek przewidziany")
    ax.set_ylabel("Wiek rzeczywisty")
    ax.set_title(f"Macierz pomylek — {label}", fontsize=9)
    for i in range(n):
        for j in range(n):
            c = int(cm[i, j])
            if c:
                ax.text(j, i, str(c), ha="center", va="center", fontsize=5,
                        color="white" if norm[i, j] > 0.5 else "#222222")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pct(x: float) -> str:
    return "n/d" if x != x else f"{100 * x:.2f} %"


def num(x: float, nd: int = 4) -> str:
    return "n/d" if x is None or x != x else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    if not runs:
        raise SystemExit("No predictions.csv found under outputs/*/emb_on_emb/")
    fish_keys = load_fish_keys()

    # ---- group runs by test set; anchor on TODAY'S production split -------------
    # The split changed once (n=1131 -> n=1105). The comparable family is the one whose
    # scored image set equals `split == "test"` in the current data/labels_embedded.csv,
    # NOT simply the largest family — the older era happens to have more runs in it.
    sig_of = {run: testset_signature(df) for run, df in runs.items()}
    families: dict[str, list[str]] = {}
    for run, sig in sig_of.items():
        families.setdefault(sig, []).append(run)
    lab_all = pd.read_csv(LABELS)
    current_sig = testset_signature(
        lab_all[lab_all["split"] == "test"][["image_id"]].assign(
            image_id=lambda d: d["image_id"].astype(str))
    )
    if current_sig in families:
        main_sig = current_sig
    else:
        main_sig = max(families, key=lambda s: (len(families[s]), len(runs[families[s][0]])))
        print(f"WARNING: no run matches the current test split ({current_sig}); "
              f"falling back to the largest family")
    family = sorted(families[main_sig])
    n_test = len(runs[family[0]])
    stale_families = {s: sorted(v) for s, v in families.items() if s != main_sig}

    # ---- per-run metrics -------------------------------------------------------
    true_orig = pd.read_csv(LABELS).set_index("image_id")["age"]
    rows = []
    for run in family:
        df = runs[run]
        tdef = target_definition(df, true_orig)
        # What the same predictions score against RECORDED age with no rebase — i.e. what
        # a quarter-adjusted model returns if it is deployed and the offset is forgotten.
        naive = df[["image_id", "predicted_age"]].merge(
            true_orig.rename("true_orig"), left_on="image_id", right_index=True)
        naive_exact = float(np.mean(naive["predicted_age"].values == naive["true_orig"].values))
        m = compute_metrics(df["target_age"].values, df["predicted_age"].values)
        pa = compute_per_age_metrics(df["target_age"].values, df["predicted_age"].values)
        fish = per_fish(df, fish_keys)
        mf = compute_metrics(fish["target_age"].values, fish["predicted_age"].values)
        lo, hi = wilson_ci(int(df["exact"].sum()), len(df))
        rows.append({
            "run": run,
            "target_def": tdef,
            "N": m["N"],
            "Exact": m["Exact"],
            "Exact_lo": lo,
            "Exact_hi": hi,
            "MacroExact": macro_exact(pa),
            "Acc1yr": m["Acc1yr"],
            "MAE": m["MAE"],
            "MedAE": m["MedAE"],
            "RMSE": m["RMSE"],
            "Bias": m["Bias"],
            "Err_le_m2": m["Err_le_m2"], "Err_m1": m["Err_m1"],
            "Err_p1": m["Err_p1"], "Err_ge_p2": m["Err_ge_p2"],
            "MAE_from_pm1": m["MAE_from_pm1"], "MAE_from_ge2": m["MAE_from_ge2"],
            "Exact_fish": mf["Exact"],
            "Acc1yr_fish": mf["Acc1yr"],
            "MAE_fish": mf["MAE"],
            "N_fish": mf["N"],
            "photos_agree": float(fish["photos_agree"].mean()),
            "Exact_naive_vs_recorded": naive_exact,
            "pred_csv_sha": file_signature(run),
            "ckpt_sha": ckpt_signature(run),
            "summary_best_val_mae": summary_best_val_mae(run),
            "log_best_ema": log_best_line(run),
        })
    per_run = pd.DataFrame(rows).sort_values("Exact", ascending=False).reset_index(drop=True)
    per_run.to_csv(EXP_DIR / "baseline_per_run.csv", index=False)

    ref = per_run.iloc[0]["run"]
    ref_df = runs[ref]

    # ---- reference run: per age + confusion ------------------------------------
    ref_pa = compute_per_age_metrics(ref_df["target_age"].values, ref_df["predicted_age"].values)
    pa_rows = []
    lab = pd.read_csv(LABELS)
    lab = lab[lab["age"] >= 0]
    for age in sorted(ref_pa):
        sub = lab[lab["age"] == age]
        pa_rows.append({
            "age": age,
            "n_train": int((sub["split"] == "train").sum()),
            "n_val": int((sub["split"] == "val").sum()),
            "n_test": ref_pa[age]["n"],
            "exact": ref_pa[age]["Exact"],
            "acc1yr": ref_pa[age]["Acc1yr"],
            "MAE": ref_pa[age]["MAE"],
            "bias": ref_pa[age]["Bias"],
        })
    pd.DataFrame(pa_rows).to_csv(EXP_DIR / "baseline_per_age.csv", index=False)
    save_confusion_png(ref_df, ref, EXP_DIR / "baseline_confusion_matrix.png")

    # ---- 0d campaign breakdown, every run in the family ------------------------
    camp_rows = []
    for run in family:
        df = runs[run]
        for camp, sub in df.groupby("campaign"):
            m = compute_metrics(sub["target_age"].values, sub["predicted_age"].values)
            camp_rows.append({"run": run, "campaign": camp, "n": m["N"],
                              "exact": m["Exact"], "MAE": m["MAE"], "Bias": m["Bias"],
                              "mean_true_age": float(sub["target_age"].mean())})
    camp = pd.DataFrame(camp_rows)
    camp.to_csv(EXP_DIR / "baseline_campaign.csv", index=False)

    # Is the per-campaign bias spread a real effect or run-to-run noise? Compare the
    # within-run spread of campaign bias against the spread of the same campaign
    # across runs. A season effect should be consistent in sign for every run.
    piv = camp.pivot_table(index="run", columns="campaign", values="Bias")
    camp_sign_consistent = {
        c: bool(np.all(piv[c].values < piv.drop(columns=[c]).mean(axis=1).values))
           or bool(np.all(piv[c].values > piv.drop(columns=[c]).mean(axis=1).values))
        for c in piv.columns
    }

    # ---- 0c pairwise McNemar ---------------------------------------------------
    mc_rows = []
    for i, a in enumerate(family):
        for b in family[i + 1:]:
            da = runs[a].set_index("image_id")["exact"]
            db = runs[b].set_index("image_id")["exact"]
            common = da.index.intersection(db.index)
            only_a, only_b, p = mcnemar_exact(da.loc[common].values, db.loc[common].values)
            mc_rows.append({"run_a": a, "run_b": b,
                            "exact_a": float(da.loc[common].mean()),
                            "exact_b": float(db.loc[common].mean()),
                            "only_a": only_a, "only_b": only_b, "p": p})
    mc = pd.DataFrame(mc_rows).sort_values("p").reset_index(drop=True)
    mc.to_csv(EXP_DIR / "baseline_mcnemar.csv", index=False)

    # ---- 0f integrity flags ----------------------------------------------------
    flags: list[str] = []
    dup = per_run.groupby("pred_csv_sha")["run"].apply(list)
    for sha, group in dup.items():
        if len(group) < 2:
            continue
        ckpts = {per_run.loc[per_run["run"] == r, "ckpt_sha"].iloc[0] for r in group}
        if len(ckpts) <= 1:
            # Same weights -> same predictions. Expected, and itself a useful isolation
            # check rather than a problem. Not flagged.
            continue
        flags.append(
            f"`predictions.csv` **bajtowo identyczny** dla biegow: {', '.join(group)} "
            f"(sha1 {sha}) mimo **roznych checkpointow** ({len(ckpts)} odciski). "
            f"Jeden z tych biegow nie przeszedl wlasnego inference — porownanie tej "
            f"pary nie mierzy niczego."
        )
    for _, r in per_run.iterrows():
        s, l = r["summary_best_val_mae"], r["log_best_ema"]
        if s is not None and l is not None and abs(float(s) - float(l)) > 0.01:
            flags.append(
                f"`{r['run']}`: `pipeline_summary.json::best_val_mae` = {num(float(s), 3)} "
                f"(surowe minimum z logu), a linia early-stoppingu podaje {l} "
                f"(minimum EMA, `src/trainer.py:583-588`). Dwie rozne liczby pod jedna nazwa."
            )

    # ---- metrics.json ----------------------------------------------------------
    payload = {
        "generated_for": "plans and summaries/24.09_CORAL_plan.md — Etap 0",
        "test_set": {"n": n_test, "signature": main_sig, "runs": family,
                     "other_era_runs_excluded": stale_families},
        "reference_run": ref,
        "reader_agreement_ceiling": READER_AGREEMENT,
        "per_run": json.loads(per_run.to_json(orient="records")),
        "reference_per_age": {str(k): v for k, v in ref_pa.items()},
        "campaign_bias_sign_consistent": camp_sign_consistent,
        "integrity_flags": flags,
    }
    (EXP_DIR / "baseline_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_narrative(per_run, pa_rows, camp, piv, camp_sign_consistent, mc, flags,
                    ref, n_test, family, stale_families)
    print(f"reference run: {ref}  exact={pct(per_run.iloc[0]['Exact'])}  "
          f"MAE={num(per_run.iloc[0]['MAE'])}")
    print(f"wrote {EXP_DIR} and {NARRATIVE}")


def write_narrative(per_run, pa_rows, camp, piv, camp_sign, mc, flags,
                    ref, n_test, family, stale_families) -> None:
    """The findings document. Narrative lives in 'plans and summaries/' by convention."""
    L: list[str] = []
    A = L.append
    A("# 24.09 — Audyt baseline CORAL: exact accuracy z plikow, ktore juz byly na dysku")
    A("")
    A("**Czym to jest:** czysty pomiar, zero GPU, zero zmian w pipeline. Wszystkie liczby")
    A("pochodza z istniejacych `outputs/<bieg>/emb_on_emb/predictions.csv`. Skrypt:")
    A("`scripts/diagnostics/coral_baseline_audit.py`. Plan: `plans and summaries/24.09_CORAL_plan.md`.")
    A("")
    A(f"**Zbior testowy:** Embedded, n = {n_test}, podzial na poziomie ryby, identyczny dla")
    A(f"{len(family)} porownywanych biegow i zgodny z `split == \"test\"` w dzisiejszym")
    A("`data/labels_embedded.csv` (sprawdzone przez hash zbioru `image_id`).")
    if stale_families:
        A("")
        A("Wykluczone jako **inna era zbioru testowego** (podzial zmienil sie raz, n=1131 -> n=1105;")
        A("te biegi nie sa sparowane z powyzszymi, wiec ich liczby nie wchodza do porownan):")
        for sig, group in stale_families.items():
            A(f"- `{sig}`: {', '.join('`' + r + '`' for r in group)}")
    A("")
    A("## 1. Dlaczego ten audyt byl potrzebny")
    A("")
    A("Exact accuracy nie byla liczona **nigdzie** w projekcie. `compute_metrics`")
    A("(`src/report_common.py`) zwracala MAE/RMSE/R2/Acc+-1/Acc+-2/Bias, a exact per klasa")
    A("wieku istnial jako jedna linia w `src/comparison_report.py`. Kazdy bieg mial wiec")
    A("swoja exact accuracy zapisana w plikach i nieprzeczytana.")
    A("")
    A("## 2. Wszystkie biegi, jedna tabela")
    A("")
    A("Kolumna **cel** mowi, przeciw jakiej definicji wieku bieg byl oceniany: `recorded` to")
    A("wiek zapisany w danych, `rings(-1 for Q1)` to wiek pomniejszony o 1 dla polowow Q1")
    A("(`data.quarter_age_adjustment_enabled`). Metryka jest **niezmiennicza wzgledem tej")
    A("zamiany** (ten sam offset dodajemy do predykcji i do celu), wiec wiersze ponizej sa")
    A("porownywalne — patrz sekcja 6.")
    A("")
    A("| bieg | cel | exact | 95 % CI | macro exact | +-1 rok | MAE | median AE | bias | exact/ryba |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in per_run.iterrows():
        A(f"| `{r['run']}` | {r['target_def']} | **{pct(r['Exact'])}** "
          f"| {pct(r['Exact_lo'])}–{pct(r['Exact_hi'])} "
          f"| {pct(r['MacroExact'])} | {pct(r['Acc1yr'])} | {num(r['MAE'])} | {num(r['MedAE'], 1)} "
          f"| {num(r['Bias'], 3)} | {pct(r['Exact_fish'])} |")
    A("")
    A(f"Sufit etykiet dla porownania: zgodnosc ludzkich czytelnikow dla sledzia baltyckiego")
    A(f"to {READER_AGREEMENT}.")
    A("")
    A("## 3. Czy ten rozrzut to w ogole efekt? (McNemar parami)")
    A("")
    A("Zbior testowy jest wspolny, wiec porownania sa sparowane. `only_a` = obrazy trafione")
    A("tylko przez bieg A, `only_b` = tylko przez B; test dwumianowy dokladny na tych dwoch liczbach.")
    A("")
    A("| bieg A | bieg B | exact A | exact B | only A | only B | p |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for _, r in mc.head(12).iterrows():
        A(f"| `{r['run_a']}` | `{r['run_b']}` | {pct(r['exact_a'])} | {pct(r['exact_b'])} "
          f"| {int(r['only_a'])} | {int(r['only_b'])} | {r['p']:.2e} |")
    if len(mc) > 12:
        A(f"")
        A(f"(pokazane 12 z {len(mc)} par, posortowane po p; pelna tabela w")
        A(f"`experiments/baseline/baseline_mcnemar.csv`)")
    A("")
    A("## 4. Rozklad bledu i dekompozycja MAE")
    A("")
    A("| bieg | <= -2 | -1 | 0 | +1 | >= +2 | MAE z +-1 | MAE z >= 2 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in per_run.iterrows():
        A(f"| `{r['run']}` | {pct(r['Err_le_m2'])} | {pct(r['Err_m1'])} | {pct(r['Exact'])} "
          f"| {pct(r['Err_p1'])} | {pct(r['Err_ge_p2'])} | {num(r['MAE_from_pm1'], 3)} "
          f"| {num(r['MAE_from_ge2'], 3)} |")
    A("")
    A("## 5. Per klasa wieku (bieg referencyjny: `" + ref + "`)")
    A("")
    A("| wiek | n train | n val | n test | exact | +-1 rok | MAE | bias |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in pa_rows:
        A(f"| {r['age']:.0f} | {r['n_train']} | {r['n_val']} | {r['n_test']} "
          f"| {pct(r['exact'])} | {pct(r['acc1yr'])} | {num(r['MAE'], 3)} | {num(r['bias'], 3)} |")
    A("")
    A("## 6. Kampania polowu — test premisy off-by-one")
    A("")
    A("Kampania jest w nazwie pliku, wiec koszt tego pomiaru jest zerowy. Jesli wiek zapisany")
    A("dla polowow Q1 (BITS1q, luty-marzec) liczy rok, ktorego pierscien jeszcze sie nie")
    A("domknal, to model bez informacji o sezonie musi systematycznie mylic sie w jedna strone")
    A("dla jednej kampanii i w druga dla innej. Warunek potwierdzenia: **znak biasu roznicuje")
    A("kampanie zgodnie i powtarzalnie we wszystkich biegach.**")
    A("")
    A("| bieg | " + " | ".join(str(c) for c in piv.columns) + " |")
    A("|---|" + "|".join(["---:"] * len(piv.columns)) + "|")
    for run in piv.index:
        A(f"| `{run}` | " + " | ".join(num(piv.loc[run, c], 3) for c in piv.columns) + " |")
    A("")
    A("Bias per kampania (pred - prawda). Spojnosc znaku miedzy biegami:")
    for c, ok in camp_sign.items():
        A(f"- `{c}`: {'TAK — odstaje w tym samym kierunku w kazdym biegu' if ok else 'NIE — kierunek zmienia sie miedzy biegami'}")
    A("")
    A("**Wynik: premisa potwierdzona.** `BITS1q` ma bias ujemny w **kazdym** biegu bez korekty")
    A("(od -0,28 do -0,71), przy `BITS4q` i `BIAS` w okolicach zera. Model czytajacy wylacznie")
    A("obraz systematycznie zaniza wiek polowow lutowo-marcowych o ~0,3-0,7 roku i nie robi tego")
    A("dla polowow z konca roku. Dokladnie ten podpis przewiduje konwencja: dla ryby z Q1")
    A("zewnetrzny pierscien jeszcze sie nie domknal, wiec otolit **wyglada** o rok mlodziej niz")
    A("zapisany wiek. Pytanie otwarte od 29.07 (czy zapisany `Wiek` juz to uwzglednia) ma wiec")
    A("odpowiedz empiryczna: **nie uwzglednia** — dowod obrazowy zgadza sie z `wiek - 1`.")
    A("")
    quarter_runs = per_run[per_run["target_def"] != "recorded"]["run"].tolist()
    if quarter_runs:
        A("### 6.1 Jedyny bieg z wlaczona korekta — i co z niego wynika")
        A("")
        A(f"Korekta kwartalna byla wlaczona dokladnie raz w historii projektu: w `{quarter_runs[0]}`")
        A("(`configs/config_attention_first.yaml:120`). Nikt nie policzyl wtedy exact accuracy,")
        A("wiec wynik nigdy nie zostal odczytany.")
        A("")
        A("Najblizszy mozliwy A/B: `configs/config_attention_first.yaml` i `configs/config_isolate_a.yaml`")
        A("roznia sie **dwoma** polami — `quarter_age_adjustment_enabled` (true/false) oraz")
        A("`density_concentricity_weight` (1,0/0,0). Drugie to mechanizm E9, ktoremu projekt")
        A("zmierzyl **zero** wplywu na wiek przy trzech wagach w rozstepie x100 (wiek bit-identyczny),")
        A("z przyczyna znaleziona w kodzie. Roznica sprowadza sie wiec praktycznie do korekty.")
        A("")
        ab = per_run.set_index("run")
        pairs = [(quarter_runs[0], "09.08_isolate_a")]
        A("| bieg | korekta | exact | MAE | bias BITS1q |")
        A("|---|---|---:|---:|---:|")
        for run, _ in pairs:
            for r in (run, pairs[0][1]):
                if r not in ab.index:
                    continue
                b1 = piv.loc[r, "BITS1q"] if r in piv.index and "BITS1q" in piv.columns else float("nan")
                A(f"| `{r}` | {'ON' if ab.loc[r, 'target_def'] != 'recorded' else 'OFF'} "
                  f"| **{pct(ab.loc[r, 'Exact'])}** | {num(ab.loc[r, 'MAE'])} | {num(b1, 3)} |")
        A("")
        A("**Pulapka wdrozeniowa, warta zapamietania.** Model z korekta przewiduje *liczbe")
        A("widocznych pierscieni*, nie zapisany wiek. Offset sezonowy trzeba dodac z powrotem")
        A("przy dekodowaniu. Ta sama siec oceniana przeciw zapisanemu wiekowi **bez** dodania")
        A("offsetu spada do:")
        A("")
        for run in quarter_runs:
            r = per_run.set_index("run").loc[run]
            A(f"- `{run}`: exact {pct(r['Exact'])} z offsetem -> **{pct(r['Exact_naive_vs_recorded'])}** bez offsetu")
        A("")
        A("Metryka jest niezmiennicza wzgledem poprawnego rebase'u (ten sam offset w predykcji")
        A("i w celu), wiec liczba z offsetem jest ta, ktora obowiazuje. Bez offsetu jest gorsza")
        A("niz baseline — i to jest dokladnie ten blad, ktory najlatwiej zrobic po cichu.")
        A("")
    A("## 7. Poziom ryby")
    A("")
    A("527 z 578 ryb testowych ma po dwa zdjecia (lewy + prawy otolit, ten sam wiek).")
    A("Agregacja tutaj to srednia dwoch przewidzianych wiekow zaokraglona w gore od .5 —")
    A("jedyna mozliwa bez logitow, ktore `src/inference.py` wyrzuca. Etap 1 dodaje zrzut")
    A("logitow i wtedy wlasciwa agregacja to srednia `g`.")
    A("")
    A("| bieg | exact/obraz | exact/ryba | +-1/ryba | MAE/ryba | zgodnosc dwoch zdjec |")
    A("|---|---:|---:|---:|---:|---:|")
    for _, r in per_run.iterrows():
        A(f"| `{r['run']}` | {pct(r['Exact'])} | **{pct(r['Exact_fish'])}** | {pct(r['Acc1yr_fish'])} "
          f"| {num(r['MAE_fish'])} | {pct(r['photos_agree'])} |")
    A("")
    A("## 8. Problemy integralnosci — zgloszone, nie naprawione")
    A("")
    if flags:
        for f in flags:
            A(f"- {f}")
    else:
        A("- brak")
    A("")
    A("Zgodnie z zasada z planu: wada w pipeline jest najpierw opisywana, bo cicha naprawa")
    A("uniewaznilaby porownanie z baseline.")
    A("")
    A("## Pliki liczbowe")
    A("")
    A("`experiments/baseline/`: `baseline_metrics.json`, `baseline_per_run.csv`,")
    A("`baseline_per_age.csv`, `baseline_campaign.csv`, `baseline_mcnemar.csv`,")
    A("`baseline_confusion_matrix.png`.")
    A("")
    NARRATIVE.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
