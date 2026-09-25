"""Shared primitives for the HTML report generators.

Both report generators build self-contained HTML with base64-embedded PNGs:

  * ``src/report.py``            — single training-run report (report.html)
  * ``src/comparison_report.py`` — Embedded vs NotEmbedded report (comparison_report.html)

This module holds the pieces they have in common (figure/image → base64,
``<img>`` tag, regression metrics) so there is a single source of truth.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# base64 encoders
# ---------------------------------------------------------------------------

def fig_to_b64(fig: "plt.Figure", dpi: int = 110) -> str:
    """Render a matplotlib figure to a base64 PNG string and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def png_to_b64(path: str | Path) -> str | None:
    """Return the base64 of an existing PNG file, or None if it is missing."""
    path = Path(path)
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def pil_to_b64(path: str | Path, max_px: int = 300) -> str:
    """Load an image, downscale it to fit ``max_px``, return it as base64 PNG."""
    img = PILImage.open(path).convert("RGB")
    img.thumbnail((max_px, max_px), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def img_tag(b64: str, alt: str = "", style: str = "") -> str:
    """A responsive ``<img>`` tag for a base64 PNG (``max-width:100%`` always applied)."""
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;{style}">'


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Regression metrics for age prediction.

    Returns dict with keys: MAE, RMSE, R2, Acc1yr, Acc2yr, Bias  (the original six,
    every existing caller depends on them) plus, added 24.09 for the CORAL
    exact-accuracy work (`plans and summaries/24.09_CORAL_plan.md`):

      * ``Exact``      — share of |pred - true| == 0.  THE target metric; it was
                         computed nowhere in the project before 24.09, only per age
                         class inside ``src/comparison_report.py``.
      * ``MedAE``      — median absolute error.
      * ``Err_*``      — the signed-error decomposition behind MAE:
                         ``Err_le_m2`` (<= -2), ``Err_m1`` (-1), ``Err_0`` (0),
                         ``Err_p1`` (+1), ``Err_ge_p2`` (>= +2).  These five sum to 1.0
                         and answer "is the model confusing N with N+/-1, or is it
                         further out" without a second pass over the data.
      * ``MAE_from_pm1`` / ``MAE_from_ge2`` — how much of MAE each group contributes.
      * ``N`` — sample count, so a metrics dict is self-describing.

    Keys are ADDITIVE on purpose: nothing is renamed or removed, so
    ``src/comparison_report.py``, ``scripts/run_pipeline.py`` and
    ``src/entrypoint.py`` keep working untouched.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    abs_err = np.abs(errors)
    n = int(errors.size)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    try:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except Exception:
        r2 = float("nan")
    acc1 = float(np.mean(abs_err <= 1.0))
    acc2 = float(np.mean(abs_err <= 2.0))
    bias = float(np.mean(errors))

    exact = float(np.mean(abs_err == 0.0))
    pm1 = abs_err == 1.0
    ge2 = abs_err >= 2.0
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "Acc1yr": acc1, "Acc2yr": acc2, "Bias": bias,
        # --- added 24.09 ---
        "Exact": exact,
        "MedAE": float(np.median(abs_err)),
        "Err_le_m2": float(np.mean(errors <= -2.0)),
        "Err_m1": float(np.mean(errors == -1.0)),
        "Err_0": exact,
        "Err_p1": float(np.mean(errors == 1.0)),
        "Err_ge_p2": float(np.mean(errors >= 2.0)),
        "MAE_from_pm1": float(np.sum(abs_err[pm1]) / n) if n else float("nan"),
        "MAE_from_ge2": float(np.sum(abs_err[ge2]) / n) if n else float("nan"),
        "N": n,
    }


def compute_per_age_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[int, dict[str, float]]:
    """Per-true-age-class breakdown: ``{age: {n, MAE, Exact, Acc1yr, Bias}}``.

    Same definitions as ``src/comparison_report.py:451-454`` (per-age MAE /
    exact-match accuracy / signed bias / count), lifted here so the report and the
    CORAL experiment scripts read from one source instead of two copies.
    Ages are returned as ints, sorted ascending by the caller if needed.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    out: dict[int, dict[str, float]] = {}
    for a in np.unique(y_true):
        mask = y_true == a
        err = errors[mask]
        out[int(a)] = {
            "n": int(mask.sum()),
            "MAE": float(np.mean(np.abs(err))),
            "Exact": float(np.mean(err == 0.0)),
            "Acc1yr": float(np.mean(np.abs(err) <= 1.0)),
            "Bias": float(np.mean(err)),
        }
    return out


def macro_exact(per_age: dict[int, dict[str, float]]) -> float:
    """Unweighted mean of per-age Exact — the rare classes count as much as age 4.

    Reported alongside global ``Exact`` because class balancing trades one for the
    other; a single number cannot show that trade.
    """
    vals = [m["Exact"] for m in per_age.values()]
    return float(np.mean(vals)) if vals else float("nan")


def confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Raw confusion counts and the shared age axis, as ``(matrix, ages)``.

    ``matrix[i, j]`` = number of samples with true age ``ages[i]`` predicted as
    ``ages[j]``.  Numbers only — plotting stays in the report modules
    (``src/comparison_report.py:381``, ``src/report.py:478``) so their look is untouched.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ages = sorted({int(v) for v in np.concatenate([y_true, y_pred])})
    idx = {a: i for i, a in enumerate(ages)}
    cm = np.zeros((len(ages), len(ages)), dtype=int)
    for t_, p_ in zip(y_true, y_pred):
        cm[idx[int(t_)], idx[int(p_)]] += 1
    return cm, ages

# ---------------------------------------------------------------------------
# Age target scale (24.09)
# ---------------------------------------------------------------------------

# Campaigns whose catch predates that year's ring closing, so the target becomes
# "complete rings visible" = recorded age - 1. Mirrors the default of
# `data.quarter_age_adjustment_campaigns` in src/config.py.
QUARTER_CAMPAIGNS = ("BITS1q", "BITS2q")


def campaign_of(image_ids) -> "np.ndarray":
    """Survey campaign parsed out of the filename: ``<year>_<CAMPAIGN>_<species>_...``."""
    return np.asarray([str(i).split("_")[1] if "_" in str(i) else "" for i in image_ids])


def target_definition(image_ids, target_age, recorded_age) -> str:
    """Which age scale a set of predictions was scored against.

    ``data.quarter_age_adjustment_enabled`` (`src/dataset.py::_effective_age`) subtracts 1
    from the age of Q1-caught fish, turning the target from "recorded calendar age" into
    "complete rings visible". The labels CSV on disk is NOT rewritten, so comparing a
    predictions file's ``target_age`` against the production labels is the only way to tell
    which scale a run used — and mixing the two scales silently compares different tasks.

    Returns ``"recorded"``, ``"rings(-1 for Q1)"`` or ``"unknown-shift"``.
    """
    delta = np.asarray(recorded_age, dtype=float) - np.asarray(target_age, dtype=float)
    if np.all(delta == 0):
        return "recorded"
    expected = np.isin(campaign_of(image_ids), QUARTER_CAMPAIGNS).astype(float)
    if np.all(delta == expected):
        return "rings(-1 for Q1)"
    return "unknown-shift"


def rebase_to_recorded(image_ids, predicted_age) -> "np.ndarray":
    """Express a "rings" model's predictions on the recorded-age scale.

    A Q1 fish showing N complete rings is recorded as age N+1, so the season offset is added
    back. Note every error is UNCHANGED by this (the same offset lands on prediction and
    target), which is what makes the two scales fairly comparable — and why forgetting the
    offset, rather than the rebase itself, is the thing that costs accuracy.
    """
    offset = np.isin(campaign_of(image_ids), QUARTER_CAMPAIGNS).astype(float)
    return np.asarray(predicted_age, dtype=float) + offset
