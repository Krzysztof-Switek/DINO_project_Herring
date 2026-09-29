"""24.09 — CORAL decoding lab: how much exact accuracy is reachable without retraining.

Works purely on the logit dump produced by `scripts/diagnostics/coral_score_splits.py`.
No GPU, no model, no images.

THE POINT. The head is CORAL in the Cao et al. form: one scalar ``g = w^T cls`` compared
against K-1 increasing thresholds (`src/model.py:1029-1042`). Every boundary logit is
``g - theta_k``, so the whole age decision lives on ONE axis. Therefore **any** decoder --
the current `(sigmoid > 0.5).sum()`, a tuned global threshold, per-boundary thresholds,
`argmax_k P(y=k)` -- is nothing but a choice of K-1 cutpoints on that axis. That makes the
best possible decoder computable exactly, by dynamic programming, which turns "should we
tune the threshold?" from a search into a single measurement with a hard ceiling.

If the ceiling sits near today's 0.5 result, decoding on ``g`` alone is exhausted and the
bottleneck is the representation (Etap 3 of the plan). If it sits well above, the gain is
free. A separate family is measured alongside it: a decoder that ALSO sees the capture
campaign, which can absorb the season-dependent bias measured in Etap 0 and is therefore
not bounded by the single-axis ceiling.

Selection happens on VAL only; test is scored once, at the end, with the val-selected rule.

    python scripts/diagnostics/coral_decode_lab.py \
        --dump experiments/logits/06.08_attention_first__best

Outputs:
  experiments/threshold/<name>/metrics.json
  experiments/threshold/<name>/decoders.csv
  experiments/threshold/<name>/per_age.csv
  experiments/threshold/<name>/predictions.csv        (test, best val decoder)
  experiments/threshold/<name>/threshold_vs_exact.png
  experiments/threshold/<name>/notes.md

Plan: "plans and summaries/24.09_CORAL_plan.md" (Etap 1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.report_common import (
    campaign_of,
    compute_metrics,
    compute_per_age_metrics,
    macro_exact,
    rebase_to_recorded,
    target_definition,
)

LABELS = PROJECT_ROOT / "data" / "labels_embedded.csv"


# ---------------------------------------------------------------------------
# Loading the dump
# ---------------------------------------------------------------------------

def load_split(dump_dir: Path, split: str) -> pd.DataFrame:
    csv = dump_dir / split / "predictions.csv"
    if not csv.exists():
        raise SystemExit(f"missing {csv} — run coral_score_splits.py first")
    df = pd.read_csv(csv)
    cols = sorted(c for c in df.columns if c.startswith("coral_logit_"))
    if not cols:
        raise SystemExit(f"{csv} has no coral_logit_* columns "
                         f"(was inference.dump_coral_logits enabled?)")
    return df.dropna(subset=["target_age"]).reset_index(drop=True)


def logit_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("coral_logit_"))


def recover_g(df: pd.DataFrame, thetas: np.ndarray | None) -> np.ndarray:
    """The scalar the head actually computes.

    ``logit_k = g - theta_k``, so ``g = logit_k + theta_k`` for every k — averaging over k
    just denoises the float rounding in the dump. Without the thresholds file, column 0 is
    used as-is: it is ``g - theta_0``, an affine image of g, which is enough for anything
    that only needs the ORDER of samples along the axis (all decoders here).
    """
    L = df[logit_columns(df)].to_numpy(dtype=float)
    if thetas is None:
        return L[:, 0]
    return (L + np.asarray(thetas, dtype=float)[None, :]).mean(axis=1)


def class_probabilities(df: pd.DataFrame) -> np.ndarray:
    """``P(y = k)`` for k = 0..K-1, from the cumulative boundary probabilities.

    CORAL gives ``P(y > k)``; the class probability is the difference of neighbours, with
    ``P(y > -1) = 1`` and ``P(y > K-1) = 0``. Rank monotonicity is structural here, so
    these differences are non-negative and sum to 1 without clipping — but we clip at 0
    anyway to stay robust to float rounding in the dump.
    """
    p_gt = 1.0 / (1.0 + np.exp(-np.clip(df[logit_columns(df)].to_numpy(dtype=float), -60, 60)))
    ones = np.ones((len(p_gt), 1))
    zeros = np.zeros((len(p_gt), 1))
    cum = np.hstack([ones, p_gt, zeros])          # P(y > -1) ... P(y > K-1)
    probs = np.clip(cum[:, :-1] - cum[:, 1:], 0.0, None)
    return probs / probs.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Decoders — each maps the dump to an integer age vector
# ---------------------------------------------------------------------------

def decode_global_threshold(df: pd.DataFrame, thr: float) -> np.ndarray:
    """``(sigmoid(logits) > thr).sum()`` — the production rule with a tunable threshold."""
    p = 1.0 / (1.0 + np.exp(-np.clip(df[logit_columns(df)].to_numpy(dtype=float), -60, 60)))
    return (p > thr).sum(axis=1).astype(int)


def decode_per_boundary(df: pd.DataFrame, thrs: np.ndarray) -> np.ndarray:
    """One threshold per boundary.

    Counting (rather than stopping at the first failure) can break monotonicity once the
    thresholds differ per boundary, so the count is taken over the cumulative AND: age =
    the longest run of satisfied boundaries from the start. That keeps the output a valid
    ordinal decode instead of a sum of possibly non-monotone indicators.
    """
    p = 1.0 / (1.0 + np.exp(-np.clip(df[logit_columns(df)].to_numpy(dtype=float), -60, 60)))
    passed = p > np.asarray(thrs)[None, :]
    return np.cumprod(passed, axis=1).sum(axis=1).astype(int)


def decode_argmax(df: pd.DataFrame) -> np.ndarray:
    """``argmax_k P(y = k)`` — the Bayes rule for 0-1 loss, i.e. for exact accuracy.

    The production decoder counts boundaries above 0.5, which is the median of the implied
    distribution and therefore an MAE-flavoured rule. So argmax is the natural candidate
    when the target metric is exact accuracy.

    BUT it is Bayes-optimal only if that implied distribution is actually CALIBRATED, and
    here it need not be: ``P(y=k)`` is a difference of consecutive sigmoids, so a class
    sitting in an unusually wide learned threshold gap collects probability mass for
    geometric reasons rather than evidential ones and can win the argmax regardless of g.
    On a synthetic head with irregular thresholds this decoder scored well BELOW plain
    count@0.5. That is why the ceiling is computed by the DP over cutpoints, which
    maximises exact accuracy directly and assumes nothing about calibration — argmax is
    just one more candidate to be measured, not the answer.
    """
    return class_probabilities(df).argmax(axis=1).astype(int)


def decode_cutpoints(g: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Age = how many cutpoints ``g`` exceeds. The general form of every decoder here."""
    return np.searchsorted(np.asarray(cuts, dtype=float), g, side="right").astype(int)


# ---------------------------------------------------------------------------
# The ceiling: best possible monotone partition of the g axis
# ---------------------------------------------------------------------------

def optimal_cutpoints(g: np.ndarray, y: np.ndarray, n_classes: int,
                      cost: str = "exact") -> tuple[np.ndarray, float]:
    """Exact DP for the best K-1 cutpoints on the g axis. Returns ``(cuts, score)``.

    A decoder on a single axis assigns non-decreasing labels 0..K-1 to samples sorted by g.
    So the task is: cut the sorted sequence into at most K consecutive blocks and label them
    in increasing order, maximising hits (``cost="exact"``) or minimising total absolute
    error (``cost="mae"``). Labels may be skipped (an empty block), which is what lets the
    optimum ignore ages the model never separates.

    ``f[a][j]`` = best score for the first j samples using labels 0..a. Because the block
    score is a difference of prefix counts, the inner maximisation collapses to a running
    prefix maximum, giving O(n*K) rather than the naive O(n^2*K).

    This is the hard ceiling for any decoder that sees ONLY ``g``: no threshold scheme,
    calibration or per-boundary tuning can beat it, because they are all special cases of a
    monotone partition of this same axis.

    It does NOT bound a decoder given extra information. ``optimal_cutpoints_per_group``
    also sees the capture campaign and therefore lives in a strictly larger family — it can
    and does score above this number, which is not a contradiction but the point: the gain
    comes from information the single axis never had.
    """
    order = np.argsort(g, kind="mergesort")
    ys = np.asarray(y, dtype=int)[order]
    n = len(ys)
    # prefix[a][j] = score contributed by labelling samples 0..j-1 as class a
    prefix = np.zeros((n_classes, n + 1))
    for a in range(n_classes):
        step = (ys == a).astype(float) if cost == "exact" else -np.abs(ys - a).astype(float)
        prefix[a, 1:] = np.cumsum(step)

    NEG = -np.inf
    f = np.full((n_classes, n + 1), NEG)
    back = np.zeros((n_classes, n + 1), dtype=int)
    # label 0 must start at sample 0
    f[0] = prefix[0]
    for a in range(1, n_classes):
        best_val, best_i = NEG, 0
        for j in range(n + 1):
            cand = f[a - 1, j] - prefix[a, j]
            if cand > best_val:
                best_val, best_i = cand, j
            f[a, j] = best_val + prefix[a, j]
            back[a, j] = best_i

    # Backtrack the block starts. Label 0 owns [0, b_1), label a owns [b_a, b_{a+1}),
    # label K-1 owns [b_{K-1}, n) — so there are exactly K-1 internal boundaries, and
    # every one of them becomes a cutpoint.
    starts = []
    j = n
    for a in range(n_classes - 1, 0, -1):
        j = back[a, j]
        starts.append(j)
    starts.reverse()                           # b_1 .. b_{K-1}, len == n_classes - 1

    gs = np.asarray(g, dtype=float)[order]
    cuts = []
    for b in starts:
        if b <= 0:
            cuts.append(gs[0] - 1.0)           # label skipped: cut below every sample
        elif b >= n:
            cuts.append(gs[-1] + 1.0)          # label skipped: cut above every sample
        else:
            cuts.append((gs[b - 1] + gs[b]) / 2.0)
    cuts = np.asarray(cuts, dtype=float)

    # Report the score the cutpoints ACTUALLY realise, not the DP's internal optimum.
    # They coincide unless two samples share the same g, in which case no cut can separate
    # them and the realisable ceiling is genuinely lower. With a float scalar out of a
    # network that is vanishingly rare, but the honest number is the achievable one.
    realised = (float(np.sum(decode_cutpoints(g, cuts) == np.asarray(y, dtype=int)))
                if cost == "exact"
                else -float(np.sum(np.abs(decode_cutpoints(g, cuts)
                                          - np.asarray(y, dtype=int)))))
    return cuts, realised


# ---------------------------------------------------------------------------
# Every decoder as one callable of (g, campaign)
# ---------------------------------------------------------------------------

def rule_from_cuts(cuts: np.ndarray):
    """Decoder from a fixed cutpoint vector."""
    return lambda g, camp: decode_cutpoints(g, cuts)


def rule_from_group_cuts(cuts_by_group: dict):
    return lambda g, camp: decode_per_group(g, camp, cuts_by_group)


def rule_from_group_offsets(cuts: np.ndarray, offsets: dict):
    return lambda g, camp: decode_with_group_offsets(g, camp, cuts, offsets)


def rule_argmax(thetas: np.ndarray):
    """``argmax_k P(y=k)`` expressed on g, so it works on a fish-averaged g too.

    Every other rule here is a function of g alone; keeping argmax tied to a dataframe of
    logits would have made it the one rule that cannot be applied to an averaged axis, and
    that special case is what previously left the selected rule without a fish-level row.
    """
    def predict(g, camp):
        logits = np.asarray(g, dtype=float)[:, None] - np.asarray(thetas, dtype=float)[None, :]
        p_gt = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        cum = np.hstack([np.ones((len(p_gt), 1)), p_gt, np.zeros((len(p_gt), 1))])
        probs = np.clip(cum[:, :-1] - cum[:, 1:], 0.0, None)
        return probs.argmax(axis=1).astype(int)
    return predict


def thetas_from_dump(df: pd.DataFrame, g: np.ndarray) -> np.ndarray:
    """Recover the boundary thresholds: ``theta_k = g - logit_k``, identical for every row."""
    return g[0] - df[logit_columns(df)].to_numpy(dtype=float)[0]


# ---------------------------------------------------------------------------
# Fitting procedures (factored out so cross-validation can refit per fold)
# ---------------------------------------------------------------------------

def fit_global_threshold(df: pd.DataFrame, y: np.ndarray, grid: np.ndarray) -> float:
    """Best single threshold on P(y>k), by exact accuracy with the plan's tie-break."""
    rows = []
    for thr in grid:
        pred = decode_global_threshold(df, thr)
        err = pred - np.asarray(y, dtype=int)
        rows.append((float(np.mean(err == 0)), float(np.mean(np.abs(err) <= 1)),
                     -float(np.mean(np.abs(err))), -abs(float(np.mean(err))), float(thr)))
    return max(rows)[4]


def fit_per_boundary_thresholds(df: pd.DataFrame, y: np.ndarray, init: float,
                                n_boundaries: int, rounds: int = 3) -> np.ndarray:
    """Coordinate ascent on one threshold per boundary, starting from a single value."""
    thrs = np.full(n_boundaries, float(init))
    y = np.asarray(y, dtype=int)
    for _ in range(rounds):
        for b in range(n_boundaries):
            best = (-1.0, thrs[b])
            for cand in np.arange(0.10, 0.91, 0.02):
                trial = thrs.copy()
                trial[b] = cand
                score = float(np.mean(decode_per_boundary(df, trial) == y))
                if score > best[0]:
                    best = (score, cand)
            thrs[b] = best[1]
    return thrs


# ---------------------------------------------------------------------------
# Honest comparison between fitted rules, without touching test
# ---------------------------------------------------------------------------

def fish_folds(image_ids, fish_keys: pd.Series, n_folds: int = 2,
               seed: int = 42) -> np.ndarray:
    """Fold index per row, split at FISH level so a fish's two photos never separate.

    Splitting by row would put the left and right otolith of the same fish on both sides of
    the split; the fold would then leak, and every fitted rule would look like it
    generalises better than it does. Same reason the train/val/test split itself is
    fish-level (`scripts/prepare_labels.py::assign_split_by_fish`).
    """
    fish = pd.Series(image_ids).map(fish_keys).fillna(pd.Series(image_ids)).values
    uniq = np.unique(fish)
    rng = np.random.default_rng(seed)
    assign = {f: i % n_folds for i, f in enumerate(rng.permutation(uniq))}
    return np.array([assign[f] for f in fish], dtype=int)


def cv_compare_rules(df: pd.DataFrame, g: np.ndarray, y: np.ndarray,
                     groups: np.ndarray, folds: np.ndarray, n_classes: int,
                     grid: np.ndarray) -> dict:
    """Out-of-fold exact accuracy for every decoder, fitted and refitted inside val only.

    This is what selection must use. Ranking decoders by their score on the very val data
    they were fitted to ranks them by CAPACITY: the shared partition has K-1 free
    parameters, per-group partitions have (K-1) per group, per-boundary thresholds K-1, the
    offset form K-1 plus one per group, and `production`/`argmax` have none at all. Measured
    here, in-sample val put a 48-parameter rule on top while out-of-fold put it level with
    or below a 16-parameter one.

    Comparing val-to-test shrinkage would also answer it, but would spend the test set on a
    selection decision. Folds are split at fish level (`fish_folds`).
    """
    y = np.asarray(y, dtype=int)
    keys = ["production", "tuned_global", "per_boundary", "argmax",
            "optimal_partition", "optimal_partition_mae", "per_campaign",
            "campaign_offset"]
    out: dict[str, list] = {k: [] for k in keys}
    for f in np.unique(folds):
        fit, held = folds != f, folds == f
        df_fit, df_held = df[fit].reset_index(drop=True), df[held].reset_index(drop=True)
        y_fit, y_held = y[fit], y[held]
        g_fit, g_held = g[fit], g[held]
        gr_fit, gr_held = groups[fit], groups[held]

        def score(pred):
            return float(np.mean(pred == y_held))

        out["production"].append(score(decode_global_threshold(df_held, 0.5)))
        out["argmax"].append(score(decode_argmax(df_held)))

        thr = fit_global_threshold(df_fit, y_fit, grid)
        out["tuned_global"].append(score(decode_global_threshold(df_held, thr)))

        thrs = fit_per_boundary_thresholds(df_fit, y_fit, thr, n_classes - 1)
        out["per_boundary"].append(score(decode_per_boundary(df_held, thrs)))

        cuts, _ = optimal_cutpoints(g_fit, y_fit, n_classes)
        out["optimal_partition"].append(score(decode_cutpoints(g_held, cuts)))

        # Cross-validated too, even though it optimises MAE rather than exact accuracy:
        # otherwise its val figure would sit in the same table as out-of-fold ones while
        # being in-sample. It simply loses the exact-accuracy selection, as it should.
        cuts_m, _ = optimal_cutpoints(g_fit, y_fit, n_classes, cost="mae")
        out["optimal_partition_mae"].append(score(decode_cutpoints(g_held, cuts_m)))

        per_grp = optimal_cutpoints_per_group(g_fit, y_fit, gr_fit, n_classes)
        out["per_campaign"].append(score(decode_per_group(g_held, gr_held, per_grp)))

        oc, off = fit_offsets_and_cuts(g_fit, y_fit, gr_fit, n_classes)
        out["campaign_offset"].append(score(
            decode_with_group_offsets(g_held, gr_held, oc, off)))
    return {k: {"folds": v, "mean": float(np.mean(v))} for k, v in out.items()}


# ---------------------------------------------------------------------------
# Candidate ranking / uncertainty
# ---------------------------------------------------------------------------

def topk_table(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Top-k age candidates with scores, margin and entropy, per image."""
    probs = class_probabilities(df)
    order = np.argsort(-probs, axis=1)
    out = {"image_id": df["image_id"].values, "target_age": df["target_age"].values}
    for i in range(k):
        out[f"age_candidate_{i + 1}"] = order[:, i]
        out[f"score_{i + 1}"] = probs[np.arange(len(probs)), order[:, i]]
    out["margin"] = out["score_1"] - out["score_2"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["entropy"] = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    return pd.DataFrame(out)


def adjacency_rate(tk: pd.DataFrame) -> float:
    """Share of images whose runner-up age is a NEIGHBOUR of the top candidate.

    Measured, not assumed. It caps what any candidate reranker could recover: a reranker
    can only swap the prediction for one of the listed candidates, so if the runner-up is
    almost always the neighbour, reranking can only ever convert +-1 errors into hits — it
    can never reach an error of 2 or more. Rows with no real second candidate (a saturated
    distribution, score_2 == 0) are excluded, since their ranking is a tie among zeros.
    """
    real = tk[tk["score_2"] > 1e-9]
    if real.empty:
        return float("nan")
    return float(np.mean(np.abs(real["age_candidate_1"] - real["age_candidate_2"]) == 1))


def topk_accuracy(tk: pd.DataFrame, k: int) -> float:
    hit = np.zeros(len(tk), dtype=bool)
    for i in range(1, k + 1):
        hit |= tk[f"age_candidate_{i}"].values == tk["target_age"].values
    return float(hit.mean())


def selective_accuracy(tk: pd.DataFrame, fractions=(0.25, 0.5, 0.75, 1.0)) -> dict:
    """Exact accuracy on the most confident X% of samples, ranked by margin.

    If confidence tracks correctness, this is the basis for a HIGH CONFIDENCE /
    REVIEW REQUIRED split — a practical result even when overall accuracy stalls.
    """
    order = np.argsort(-tk["margin"].values)
    correct = (tk["age_candidate_1"].values == tk["target_age"].values)[order]
    return {f"top_{int(100 * f)}pct": float(correct[:max(1, int(f * len(correct)))].mean())
            for f in fractions}


# ---------------------------------------------------------------------------
# Fish-level aggregation
# ---------------------------------------------------------------------------

def fish_level_rule(df: pd.DataFrame, g: np.ndarray, groups: np.ndarray, rule,
                    fish_keys: pd.Series) -> dict:
    """Average the scalar g over each fish's photos, then apply ``rule`` once.

    A fish has one age, and 91 % of test fish have two photos (left + right otolith). The
    right aggregation is over the model's continuous evidence, not over two already-rounded
    integers — averaging g halves the variance of the decision variable, which the integer
    average cannot do. Both photos come from the same haul, so the campaign is constant
    within a fish and simply carries through the grouping.
    """
    d = pd.DataFrame({"image_id": df["image_id"].values, "g": g, "grp": groups,
                      "target_age": df["target_age"].values})
    d["fish"] = d["image_id"].map(fish_keys)
    d = d.dropna(subset=["fish"])
    if d.empty:
        return {"n_fish": 0, "note": "no image_id matched data/labels_embedded.csv"}
    agg = d.groupby("fish").agg(g=("g", "mean"), grp=("grp", "first"),
                               target_age=("target_age", "first")).reset_index()
    pred = rule(agg["g"].values, agg["grp"].values)
    m = compute_metrics(agg["target_age"].values, pred)
    m["n_fish"] = int(len(agg))
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(key: str, name: str, y: np.ndarray, pred: np.ndarray) -> dict:
    """One decoder's metrics. ``key`` is the stable identity across splits.

    The display ``name`` differs between splits (a rule fitted on val is labelled as such on
    test), so selection must join on ``key`` — matching on the label silently fails to find
    the val-selected rule's test row.
    """
    m = compute_metrics(y, pred)
    m["rule_key"] = key
    m["decoder"] = name
    m["MacroExact"] = macro_exact(compute_per_age_metrics(y, pred))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="experiments/logits/<run>__<ckpt>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", default="0.20:0.80:0.005",
                    help="global threshold grid lo:hi:step")
    args = ap.parse_args()

    dump = Path(args.dump)
    if not dump.is_absolute():
        dump = PROJECT_ROOT / dump
    out_dir = Path(args.out) if args.out else (
        PROJECT_ROOT / "experiments" / "threshold" / dump.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    thetas = None
    tfile = dump / "coral_thresholds.json"
    if tfile.exists():
        thetas = np.asarray(json.loads(tfile.read_text(encoding="utf-8"))["theta"], dtype=float)

    val, test = load_split(dump, "val"), load_split(dump, "test")
    K = len(logit_columns(val)) + 1
    g_val, g_test = recover_g(val, thetas), recover_g(test, thetas)
    y_val = val["target_age"].to_numpy(dtype=int)
    y_test = test["target_age"].to_numpy(dtype=int)
    fish_keys = pd.read_csv(LABELS, usecols=["image_id", "neutral_fish_key"]).set_index(
        "image_id")["neutral_fish_key"]

    # Which age scale is this dump on? Exact/MAE/Acc+-1 are invariant between the two
    # (the same season offset lands on prediction and target), but the PER-AGE breakdown
    # is not: on the rings scale a Q1 fish sits one class lower. Detect and report it so
    # the per-age table is never read against the wrong axis.
    recorded = pd.read_csv(LABELS).set_index("image_id")["age"]
    scale = {}
    for name, df in (("val", val), ("test", test)):
        joined = df[["image_id", "target_age"]].merge(
            recorded.rename("rec"), left_on="image_id", right_index=True)
        scale[name] = (target_definition(joined["image_id"].values,
                                        joined["target_age"].values, joined["rec"].values)
                       if len(joined) else "unknown (no image_id matched the label file)")

    # --- global threshold sweep, selected on VAL ------------------------------
    lo, hi, step = (float(x) for x in args.grid.split(":"))
    grid = np.arange(lo, hi + step / 2, step)
    sweep = [evaluate("global_sweep", f"global@{t:.3f}", y_val, decode_global_threshold(val, t)) | {"thr": t}
             for t in grid]
    sweep_df = pd.DataFrame(sweep)
    sweep_df["abs_bias"] = sweep_df["Bias"].abs()
    best_thr = fit_global_threshold(val, y_val, grid)

    # --- per-boundary thresholds, coordinate ascent on VAL --------------------
    thrs = fit_per_boundary_thresholds(val, y_val, best_thr, K - 1)

    # --- the ceiling ---------------------------------------------------------
    cuts_val, hits_val = optimal_cutpoints(g_val, y_val, K, cost="exact")
    cuts_mae, _ = optimal_cutpoints(g_val, y_val, K, cost="mae")

    # Campaign-conditional decoding. The campaign is in the filename, hence known at
    # inference time; a global rule cannot use it, a per-group rule can.
    camp_val = campaign_of(val["image_id"].values)
    camp_test = campaign_of(test["image_id"].values)
    cuts_by_camp = optimal_cutpoints_per_group(g_val, y_val, camp_val, K)
    offset_cuts, camp_offsets = fit_offsets_and_cuts(g_val, y_val, camp_val, K)
    # Which campaign rule actually generalises? Decided inside val, never on test.
    folds = fish_folds(val["image_id"].values, fish_keys)

    rows_val = [
        evaluate("production", "production  count@0.5", y_val, decode_global_threshold(val, 0.5)),
        evaluate("tuned_global", f"tuned global count@{best_thr:.3f}", y_val, decode_global_threshold(val, best_thr)),
        evaluate("per_boundary", "per-boundary thresholds", y_val, decode_per_boundary(val, thrs)),
        evaluate("argmax", "argmax P(y=k)", y_val, decode_argmax(val)),
        evaluate("optimal_partition", "CEILING optimal partition of g", y_val, decode_cutpoints(g_val, cuts_val)),
        evaluate("optimal_partition_mae", "optimal partition (MAE objective)", y_val, decode_cutpoints(g_val, cuts_mae)),
        evaluate("per_campaign", "per-campaign optimal partition", y_val,
                 decode_per_group(g_val, camp_val, cuts_by_camp)),
        evaluate("campaign_offset", "shared partition + per-campaign shift of g", y_val,
                 decode_with_group_offsets(g_val, camp_val, offset_cuts, camp_offsets)),
    ]
    # Everything above is FITTED on val, so val numbers are optimistic by construction.
    # The honest figure is the same rule applied to the untouched test split.
    rows_test = [
        evaluate("production", "production  count@0.5", y_test, decode_global_threshold(test, 0.5)),
        evaluate("tuned_global", f"tuned global count@{best_thr:.3f}", y_test, decode_global_threshold(test, best_thr)),
        evaluate("per_boundary", "per-boundary thresholds", y_test, decode_per_boundary(test, thrs)),
        evaluate("argmax", "argmax P(y=k)", y_test, decode_argmax(test)),
        evaluate("optimal_partition", "val-optimal partition applied to test", y_test, decode_cutpoints(g_test, cuts_val)),
        evaluate("optimal_partition_mae", "val-optimal partition (MAE objective)", y_test, decode_cutpoints(g_test, cuts_mae)),
        evaluate("per_campaign", "per-campaign partition (fitted on val)", y_test,
                 decode_per_group(g_test, camp_test, cuts_by_camp)),
        evaluate("campaign_offset",
                 "shared partition + per-campaign shift (fitted on val)", y_test,
                 decode_with_group_offsets(g_test, camp_test, offset_cuts, camp_offsets)),
    ]
    dec = pd.concat([pd.DataFrame(rows_val).assign(split="val"),
                     pd.DataFrame(rows_test).assign(split="test")], ignore_index=True)
    # Pick the rule the way the plan says: best Exact on VAL, ties by Acc1yr then MAE.
    # Selection: OUT-OF-FOLD inside val. Using the in-sample val score would rank the
    # decoders by how many parameters they fit, which is exactly what happened when this
    # lab first ran (a 48-parameter rule on top in-sample, level or worse out-of-fold).
    cv = cv_compare_rules(val, g_val, y_val, camp_val, folds, K, grid)
    selected_key = max(cv, key=lambda k: cv[k]["mean"])
    sel_val = dec[(dec["split"] == "val") & (dec["rule_key"] == selected_key)]
    selected_rule = str(sel_val.iloc[0]["decoder"]) if len(sel_val) else selected_key
    sel_test = dec[(dec["split"] == "test") & (dec["rule_key"] == selected_key)]
    dec.to_csv(out_dir / "decoders.csv", index=False)
    sweep_df.to_csv(out_dir / "threshold_sweep_val.csv", index=False)

    # --- candidates, confidence, fish level ----------------------------------
    tk_val, tk_test = topk_table(val), topk_table(test)
    topk = {"val": {f"top{k}": topk_accuracy(tk_val, k) for k in (1, 2, 3)}
                   | {"adjacency_rate": adjacency_rate(tk_val)},
            "test": {f"top{k}": topk_accuracy(tk_test, k) for k in (1, 2, 3)}
                    | {"adjacency_rate": adjacency_rate(tk_test)}}
    selective = {"val": selective_accuracy(tk_val), "test": selective_accuracy(tk_test)}
    thetas_d = thetas_from_dump(val, g_val)
    rule_table = {
        "production": rule_from_cuts(cutpoints_for_threshold(val, g_val, 0.5)),
        "tuned_global": rule_from_cuts(cutpoints_for_threshold(val, g_val, best_thr)),
        "per_boundary": rule_from_cuts(cutpoints_for_per_boundary(val, g_val, thrs)),
        "argmax": rule_argmax(thetas_d),
        "optimal_partition": rule_from_cuts(cuts_val),
        "optimal_partition_mae": rule_from_cuts(cuts_mae),
        "per_campaign": rule_from_group_cuts(cuts_by_camp),
        "campaign_offset": rule_from_group_offsets(offset_cuts, camp_offsets),
    }
    fish, windows = {}, {}
    for split, df_, gg, cc, yy in (("val", val, g_val, camp_val, y_val),
                                   ("test", test, g_test, camp_test, y_test)):
        for name, rule in rule_table.items():
            fish[f"{split}_{name}"] = fish_level_rule(df_, gg, cc, rule, fish_keys)
            pred = rule(gg, cc)
            windows[f"{split}_{name}"] = {
                f"window_pm{r}": float(np.mean(np.abs(pred - yy) <= r)) for r in range(3)}
    tk_test.to_csv(out_dir / "topk_test.csv", index=False)

    pd.DataFrame({
        "image_id": test["image_id"], "target_age": y_test,
        "predicted_age_production": decode_global_threshold(test, 0.5),
        "predicted_age_best": decode_cutpoints(g_test, cuts_val),
        "g": g_test,
    }).to_csv(out_dir / "predictions.csv", index=False)

    per_age = compute_per_age_metrics(y_test, decode_cutpoints(g_test, cuts_val))
    pd.DataFrame([{"age": a, **v} for a, v in sorted(per_age.items())]).to_csv(
        out_dir / "per_age.csv", index=False)

    # --- threshold vs exact accuracy ----------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(sweep_df["thr"], 100 * sweep_df["Exact"], lw=1.8, label="val exact")
    test_sweep = [float(np.mean(decode_global_threshold(test, t) == y_test)) for t in grid]
    ax.plot(grid, 100 * np.asarray(test_sweep), lw=1.2, ls="--", label="test exact")
    ax.axvline(0.5, color="#888", lw=1, ls=":", label="produkcja 0,5")
    ax.axvline(best_thr, color="#B3157B", lw=1.2, label=f"wybrany {best_thr:.3f}")
    ax.axhline(100 * hits_val / len(y_val), color="#16704F", lw=1, ls="-.",
               label="sufit (optymalny podzial g, val)")
    ax.set_xlabel("prog na P(wiek > k)"); ax.set_ylabel("exact accuracy [%]")
    ax.set_title("Prog vs trafienie dokladne")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "threshold_vs_exact.png", dpi=130); plt.close(fig)

    payload = {
        "dump": str(dump), "K": K,
        "n": {"val": int(len(val)), "test": int(len(test))},
        "target_scale": scale,
        "campaigns_with_own_cutpoints": [k for k in cuts_by_camp if k != "__global__"],
        "campaign_offsets_on_g": camp_offsets,
        "cv_within_val": cv,
        "selection_basis": "out-of-fold exact accuracy, 2-fold by fish, inside val",
        "campaign_offset_shared_cutpoints": [float(x) for x in offset_cuts],
        "selected_on_val": {"global_threshold": best_thr,
                            "per_boundary_thresholds": [float(x) for x in thrs],
                            "optimal_cutpoints": [float(x) for x in cuts_val]},
        "ceiling_val_exact": float(hits_val / len(y_val)),
        "decoders": json.loads(dec.to_json(orient="records")),
        "topk": topk, "selective_accuracy_by_margin": selective, "fish_level": fish,
        "reranker_window_bound": windows,
        "val_selected_rule": selected_rule,
        "val_selected_rule_key": selected_key,
        "val_selected_rule_test": (json.loads(sel_test.iloc[[0]].to_json(orient="records"))[0]
                                   if len(sel_test) else None),
        "val_selected_rule_test_fish": fish.get(f"test_{selected_key}"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    write_notes(out_dir, dec, payload, best_thr)
    print(f"ceiling (val, fitted) = {100 * hits_val / len(y_val):.2f} %  "
          f"| production val = {100 * rows_val[0]['Exact']:.2f} %")
    print(f"wrote {out_dir}")


def cutpoints_for_threshold(df: pd.DataFrame, g: np.ndarray, thr: float) -> np.ndarray:
    """The cutpoints on g that reproduce ``count@thr`` exactly.

    Needed so fish-level aggregation can decode an AVERAGED g with the production rule:
    the rule is normally expressed on per-image probabilities, but averaging happens on g,
    so the rule has to be re-expressed on the same axis. ``P(y>k) > thr`` is equivalent to
    ``g > theta_k + logit(thr)``; the theta values are recovered from any single row, since
    ``theta_k = g - logit_k``.
    """
    L = df[logit_columns(df)].to_numpy(dtype=float)
    theta = g[0] - L[0]                       # identical for every row by construction
    return theta + float(np.log(thr / (1.0 - thr)))


def cutpoints_for_per_boundary(df: pd.DataFrame, g: np.ndarray,
                               thrs: np.ndarray) -> np.ndarray:
    """The per-boundary rule, re-expressed as cutpoints on g.

    Boundary k passes when ``P(y>k) > thr_k``, i.e. ``g > theta_k + logit(thr_k)``. The
    decode is the longest satisfied prefix, so "boundaries 0..k all pass" means g exceeds
    the RUNNING MAXIMUM of those per-boundary cuts. Taking the cummax therefore gives an
    equivalent, monotone cutpoint vector — which is what lets the same rule be applied to a
    fish-averaged g, and also makes explicit that per-boundary tuning never leaves the
    single-axis family the ceiling bounds.
    """
    L = df[logit_columns(df)].to_numpy(dtype=float)
    theta = g[0] - L[0]
    thrs = np.clip(np.asarray(thrs, dtype=float), 1e-6, 1 - 1e-6)
    return np.maximum.accumulate(theta + np.log(thrs / (1.0 - thrs)))


def optimal_cutpoints_per_group(g: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                n_classes: int, min_n: int = 150) -> dict:
    """One cutpoint vector per group (here: per survey campaign).

    Motivation, measured in Etap 0: a model reading only the image under-predicts Q1-caught
    fish by ~0.3-0.7 years in EVERY run without the quarter adjustment, and is unbiased on
    Q3/Q4 catches. That is a systematic, group-constant shift, and the capture campaign is
    known at inference time — so a decoder allowed to use it can absorb the shift without
    any retraining, which a single global rule provably cannot.

    Groups with fewer than ``min_n`` validation samples fall back to the global rule: 16
    cutpoints fitted on a handful of samples would be pure overfitting.
    """
    global_cuts, _ = optimal_cutpoints(g, y, n_classes, cost="exact")
    out = {"__global__": global_cuts}
    for grp in np.unique(groups):
        mask = groups == grp
        if int(mask.sum()) < min_n:
            continue
        out[str(grp)], _ = optimal_cutpoints(g[mask], y[mask], n_classes, cost="exact")
    return out


def optimal_group_offsets(g: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          cuts: np.ndarray, lo: float = -2.0, hi: float = 2.0,
                          step: float = 0.02) -> dict:
    """ONE scalar shift of ``g`` per group, on top of a single shared partition.

    This is the model class the measurement actually calls for. Etap 0 found a
    group-CONSTANT bias (Q1 catches under-predicted by ~0.3-0.7 years in every run, Q3/Q4
    unbiased) -- one number per group, not a free re-cutting of the whole axis. Fitting a
    full per-group partition instead spends K-1 parameters per group and duly overfits: on
    the two real dumps it shrinks 6.4-8.8 pp from val to test, against 5.0-5.5 pp for the
    global partition. Three scalars cannot do that.

    The grid spans +-2.0 on the g scale; the learned thresholds sit ~0.9 apart, so that is
    about +-2 age classes of shift -- far more than the measured effect needs.
    """
    offsets = {}
    grid = np.arange(lo, hi + step / 2, step)
    for grp in np.unique(groups):
        mask = groups == grp
        gg, yy = g[mask], np.asarray(y, dtype=int)[mask]
        scores = [float(np.mean(decode_cutpoints(gg + d, cuts) == yy)) for d in grid]
        offsets[str(grp)] = float(grid[int(np.argmax(scores))])
    return offsets


def fit_offsets_and_cuts(g: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        n_classes: int, rounds: int = 4) -> tuple:
    """Alternate between per-group offsets and the shared partition. Returns ``(cuts, offsets)``.

    Fitting the offsets on top of a partition that was itself fitted to the UNCORRECTED axis
    is circular: that partition already absorbed some of the group bias by placing its
    cutpoints between the groups, so there is less left for the offsets to remove -- and if
    the uncorrected optimum is degenerate (several partitions tie), it can land on one that
    no single per-group shift can repair. Measured on a synthetic group-constant shift, the
    single-pass version recovered 73 % where the shift is worth ~100 %.

    So: fit offsets given the current cuts, refit the cuts on the shifted axis, repeat. Each
    step is the exact optimum of its own subproblem and the objective is bounded, so the
    score is non-decreasing; a handful of rounds is enough in practice. Still only
    ``n_groups`` free parameters beyond the shared partition.
    """
    offsets = {str(grp): 0.0 for grp in np.unique(groups)}
    cuts, _ = optimal_cutpoints(g, y, n_classes, cost="exact")
    best = (cuts, dict(offsets),
            float(np.mean(decode_with_group_offsets(g, groups, cuts, offsets) == y)))
    for _ in range(rounds):
        offsets = optimal_group_offsets(g, y, groups, cuts)
        shift = np.array([offsets.get(str(grp), 0.0) for grp in groups], dtype=float)
        cuts, _ = optimal_cutpoints(g + shift, y, n_classes, cost="exact")
        score = float(np.mean(decode_with_group_offsets(g, groups, cuts, offsets) == y))
        if score > best[2] + 1e-12:
            best = (cuts, dict(offsets), score)
    return best[0], best[1]


def decode_with_group_offsets(g: np.ndarray, groups: np.ndarray, cuts: np.ndarray,
                              offsets: dict) -> np.ndarray:
    """Shared cutpoints, per-group shift of g. An unseen group gets no shift."""
    shift = np.array([offsets.get(str(grp), 0.0) for grp in groups], dtype=float)
    return decode_cutpoints(np.asarray(g, dtype=float) + shift, cuts)


def decode_per_group(g: np.ndarray, groups: np.ndarray, cuts_by_group: dict) -> np.ndarray:
    """Apply each group's own cutpoints, falling back to the global vector."""
    pred = np.empty(len(g), dtype=int)
    for grp in np.unique(groups):
        mask = groups == grp
        cuts = cuts_by_group.get(str(grp), cuts_by_group["__global__"])
        pred[mask] = decode_cutpoints(g[mask], cuts)
    return pred


def candidate_window(g: np.ndarray, cuts: np.ndarray, y: np.ndarray,
                     radius: int = 1) -> dict:
    """Coverage of a +-radius window around a DECODED age, not around argmax.

    The top-k table ranks candidates by ``P(y=k)``, whose top-1 is argmax — and argmax is
    measurably a worse point estimate here than the threshold decoders. Ranking candidates
    that way therefore understates what a reranker could reach, because the window is
    centred on the wrong age. This measures the honest bound instead: if a reranker could
    always pick the right age out of {decode-r .. decode+r}, this is what it would score.
    """
    pred = decode_cutpoints(g, cuts)
    y = np.asarray(y, dtype=int)
    return {f"window_pm{r}": float(np.mean(np.abs(pred - y) <= r))
            for r in range(0, radius + 1)}


def write_notes(out_dir: Path, dec: pd.DataFrame, payload: dict, best_thr: float) -> None:
    def pct(x): return f"{100 * x:.2f} %"
    L = ["# Notatki — laboratorium dekodowania CORAL", "",
         f"Dump: `{payload['dump']}`  ·  K = {payload['K']}  ·  "
         f"n val = {payload['n']['val']}, n test = {payload['n']['test']}", "",
         "Wszystkie reguly wybrane **wylacznie na val**; test policzony raz, na koncu.", "",
         f"**Skala celu:** val = `{payload['target_scale']['val']}`, "
         f"test = `{payload['target_scale']['test']}`.",
         "Przy skali `rings(-1 for Q1)` model przewiduje liczbe widocznych pierscieni, a nie",
         "zapisany wiek. Exact / MAE / +-1 sa **niezmiennicze** wzgledem tej zamiany, ale",
         "tabela per wiek nie jest: ryba z Q1 lezy o klase nizej. Do porownan z zapisanym",
         "wiekiem trzeba dodac offset sezonowy (`src.report_common.rebase_to_recorded`).", "",
         "## Dekodery", "",
         "| dekoder | split | exact | macro exact | +-1 rok | MAE | bias |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in dec.iterrows():
        L.append(f"| {r['decoder']} | {r['split']} | **{pct(r['Exact'])}** "
                 f"| {pct(r['MacroExact'])} | {pct(r['Acc1yr'])} | {r['MAE']:.4f} "
                 f"| {r['Bias']:+.3f} |")
    L += ["", "## Sufit — i czego on NIE ogranicza", "",
          f"Optymalny podzial osi `g` na val daje **{pct(payload['ceiling_val_exact'])}**.",
          "Zadna zmiana progu, kalibracji ani progow per granica nie moze tego przebic —",
          "wszystkie sa szczegolnym przypadkiem monotonicznego podzialu tej samej osi.",
          "Liczba dopasowana na val jest optymistyczna; wiersz *val-optimal partition",
          "applied to test* pokazuje, ile z tego zostaje.", "",
          "**Dekoder warunkowany kampania lezy POZA ta rodzina** i moze byc wyzszy — widzi",
          "informacje, ktorej jedna os nigdy nie miala (kampania polowu, obecna w nazwie",
          "pliku). To nie sprzecznosc z sufitem, tylko jego zakres.", "",
          "## Kandydaci i niepewnosc", "",
          "| split | top-1 | top-2 | top-3 |", "|---|---:|---:|---:|"]
    for s in ("val", "test"):
        t = payload["topk"][s]
        L.append(f"| {s} | {pct(t['top1'])} | {pct(t['top2'])} | {pct(t['top3'])} |")
    adj = payload["topk"]["test"].get("adjacency_rate", float("nan"))
    L += ["", f"**Zmierzony** udzial przypadkow, w ktorych drugi kandydat jest sasiadem",
          f"pierwszego: **{pct(adj)}** (test). To NIE jest wlasnosc strukturalna — przy",
          "nierownych odstepach progow `P(y=k)` nie musi byc unimodalne. Im blizej 100 %,",
          "tym mocniej top-2 zbliza sie do Acc+-1, i tym scislej ograniczone jest to, co",
          "jakikolwiek reranking kandydatow moze odzyskac: wylacznie bledy o 1 rok.", "",
          "| split | 25 % najpewniejszych | 50 % | 75 % | 100 % |", "|---|---:|---:|---:|---:|"]
    for s in ("val", "test"):
        sa = payload["selective_accuracy_by_margin"][s]
        L.append(f"| {s} | {pct(sa['top_25pct'])} | {pct(sa['top_50pct'])} "
                 f"| {pct(sa['top_75pct'])} | {pct(sa['top_100pct'])} |")
    L += ["", "## Poziom ryby (srednia `g` po zdjeciach jednej ryby)", "",
          "| regula | n ryb | exact | +-1 rok | MAE |", "|---|---:|---:|---:|---:|"]
    for key, m in payload["fish_level"].items():
        if not m.get("n_fish"):
            L.append(f"| {key} | 0 | — | — | — |")
            continue
        L.append(f"| {key} | {m['n_fish']} | {pct(m['Exact'])} | {pct(m['Acc1yr'])} "
                 f"| {m['MAE']:.4f} |")
    cv = payload.get("cv_within_val") or {}
    if cv:
        L += ["", "## Wybor reguly — 2-krotna CV wewnatrz val (podzial po rybie)", "",
              "Reguly roznia sie pojemnoscia o rzad wielkosci, wiec porownanie na danych, na",
              "ktorych byly dopasowane, rankinguje je po liczbie parametrow. Skurcz val->test",
              "odpowiedzialby na to pytanie, ale wydalby zbior testowy na decyzje selekcyjna.",
              "Podzial po **rybie**, zeby dwa zdjecia tej samej ryby nie trafily po obu stronach.",
              "",
              "| regula | parametry | exact out-of-fold |", "|---|---|---:|"]
        params = {"production": "0", "argmax": "0", "tuned_global": "1",
                  "per_boundary": f"K-1 = {payload['K'] - 1}",
                  "optimal_partition": f"K-1 = {payload['K'] - 1}",
                  "optimal_partition_mae": f"K-1 = {payload['K'] - 1} (cel MAE)",
                  "per_campaign": f"(K-1) x kampanie = {(payload['K'] - 1) * 3}",
                  "campaign_offset": f"{payload['K'] - 1} + 1 na kampanie"}
        for k, v in cv.items():
            L.append(f"| {k} | {params.get(k, '?')} | {pct(v['mean'])} |")
        L.append("")
    L += ["", "## Gorna granica dla rerankera kandydatow", "",
          "Ile dalby reranker, ktory ZAWSZE wybiera poprawny wiek z okna +-r wokol",
          "zdekodowanego. Liczone wokol realnego dekodera, nie wokol `argmax` — okno",
          "wycentrowane na slabszym estymatorze zanizalaby te granice.", "",
          "| regula | +-0 (exact) | +-1 | +-2 |", "|---|---:|---:|---:|"]
    for key, w in payload["reranker_window_bound"].items():
        L.append(f"| {key} | {pct(w['window_pm0'])} | {pct(w['window_pm1'])} "
                 f"| {pct(w['window_pm2'])} |")
    sel_t = payload.get("val_selected_rule_test") or {}
    sel_f = payload.get("val_selected_rule_test_fish") or {}
    L += ["", "## Dyscyplina selekcji — ktora liczba obowiazuje", "",
          f"Regula wybrana: {payload['val_selected_rule']} "
          f"(`{payload['val_selected_rule_key']}`).",
          f"Podstawa wyboru: {payload.get('selection_basis', 'n/d')}.", "",
          "**Wynik tego etapu, na tescie, jeden raz:**",
          f"- per obraz: exact **{pct(sel_t.get('Exact', float('nan')))}**, "
          f"MAE {sel_t.get('MAE', float('nan')):.4f}, bias {sel_t.get('Bias', float('nan')):+.3f}"
          if sel_t else "- per obraz: n/d",
          (f"- per ryba: exact **{pct(sel_f.get('Exact', float('nan')))}**, "
           f"MAE {sel_f.get('MAE', float('nan')):.4f}, n = {sel_f.get('n_fish')}")
          if sel_f.get("n_fish") else "- per ryba: n/d", "",
          "To jej wynik na tescie jest wynikiem tego etapu. Jesli inna regula wypadla na",
          "tescie lepiej, ta informacja pochodzi z testu i **nie wolno** jej uzyc do wyboru —",
          "inaczej stroimy na zbiorze testowym. Pozostale wiersze testowe sa raportowane",
          "wylacznie dla pelnego obrazu.", "",
          f"Wybrany prog globalny: **{best_thr:.3f}** (produkcja: 0,5).", ""]
    (out_dir / "notes.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
