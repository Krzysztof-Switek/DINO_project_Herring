"""Comparative HTML report for Embedded vs NotEmbedded otolith pipeline.

Generates a self-contained HTML file with sections:
  A — Dataset statistics
  B — Training curves (loss, LR, val_MAE)
  C — Evaluation metrics (MAE, RMSE, R², Acc±1yr, Acc±2yr, Bias) for 4 conditions
  D — Cross-evaluation summary table with automatic comment
  E — Increment annotation cards (best / worst predictions)
  F — Model and configuration info
  G — Increment-dot gallery (model-drawn dots for every annotated test image)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Shared report primitives (single source of truth). compute_metrics is
# re-exported so existing `from src.comparison_report import compute_metrics`
# imports (run_pipeline, tests) keep working.
from src.report_common import compute_metrics, fig_to_b64, img_tag, pil_to_b64, png_to_b64

__all__ = ["compute_metrics", "cross_comment", "normalize_result_keys",
           "build_comparison_report"]


# Colour follows the CONDITION (the entity), consistently across EVERY chart —
# not matplotlib's per-figure default cycle. 4-hue categorical palette validated
# with the dataviz validate_palette.js (light mode: all ≥3:1 contrast).
_COND_COLORS = {
    "emb_on_emb":       "#2a78d6",   # blue
    "notemb_on_notemb": "#eb6834",   # orange
    "emb_on_notemb":    "#008300",   # green
    "notemb_on_emb":    "#4a3aa7",   # violet
}
_LABEL_COLORS = {
    "Emb → Emb":            "#2a78d6",
    "NotEmb → NotEmb":      "#eb6834",
    "Emb → NotEmb ★ CROSS": "#008300",
    "NotEmb → Emb ★ CROSS": "#4a3aa7",
}
_PTYPE_COLORS = {"Embedded": "#2a78d6", "NotEmbedded": "#eb6834"}
# Split identity — reuses the same validated categorical hues (train/val/test).
_SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#008300"}
_INK, _MUTED, _GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _style_ax(ax) -> None:
    """Recessive hairline grid + muted axes (dataviz chrome). Call on every axis."""
    ax.set_axisbelow(True)
    ax.grid(True, color=_GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelcolor=_INK)
    ax.title.set_color(_INK)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

def cross_comment(own_mae: float, cross_mae: float) -> str:
    """Return automatic generalization comment based on cross/own MAE ratio."""
    if np.isnan(own_mae) or np.isnan(cross_mae):
        return "Brak danych do oceny generalizacji (brak wyników cross)."
    if cross_mae < 1.5 * own_mae:
        return "Model generalizuje dobrze (cross MAE < 1.5 × own MAE)."
    return "Słaba generalizacja cross-domain (cross MAE ≥ 1.5 × own MAE)."


# Canonical condition keys used by the report. The pipeline may deliver the
# cross-domain results under a ``cross_`` prefix (cross_emb_on_notemb, …), so we
# normalise both spellings to the bare canonical key below.
CANONICAL_CONDITIONS = ("emb_on_emb", "notemb_on_notemb", "emb_on_notemb", "notemb_on_emb")


def normalize_result_keys(results: dict) -> dict:
    """Accept both bare and ``cross_``-prefixed condition keys.

    ``scripts/run_pipeline.py`` stores cross-domain results as
    ``cross_emb_on_notemb`` / ``cross_notemb_on_emb`` while the report sections
    look them up as ``emb_on_notemb`` / ``notemb_on_emb``. This maps the prefixed
    keys onto the canonical ones without clobbering an already-canonical entry.
    """
    normalized = dict(results)
    for key, df in results.items():
        if key.startswith("cross_"):
            normalized.setdefault(key[len("cross_"):], df)
    return normalized


# ---------------------------------------------------------------------------
# Plot helpers — thin adapters over src/report_common
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    return fig_to_b64(fig, dpi=100)


_load_png_b64 = png_to_b64


def _img_tag(b64: str, width: str = "100%") -> str:
    return img_tag(b64, style=f"width:{width};")


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_a(dataset_stats: dict, active_ptypes: list[str] | None = None) -> str:
    counts       = dataset_stats.get("counts", {})
    fish_counts  = dataset_stats.get("fish_counts", {})
    age_by_split = dataset_stats.get("age_by_split", {})
    age_dists    = dataset_stats.get("age_distributions", {})
    funnel       = dataset_stats.get("funnel")
    orphans      = dataset_stats.get("orphan_count", "N/A")

    ptypes = active_ptypes or dataset_stats.get("active_ptypes") or ["Embedded", "NotEmbedded"]
    # keep only preparation types that actually carry data
    present_ptypes = [p for p in ptypes
                      if counts.get(p) or age_by_split.get(p) or age_dists.get(p)]
    ptypes = present_ptypes or ptypes

    html = '<section id="A"><h2>A. Statystyki zbioru danych</h2>'

    # 1) Data funnel: disk → parsed → labeled → orphans
    if funnel:
        html += ('<p class="cap">Lejek danych — ile zdjęć przeszło z dysku do zbioru '
                 'uczącego (gdzie i dlaczego ubywa).</p>')
        steps = [
            ("Na dysku (zeskanowane)",        funnel.get("on_disk")),
            ("Sparsowane (poprawna nazwa)",   funnel.get("parsed")),
            ("Z metadanymi = labeled",        funnel.get("labeled")),
            ("Sieroty (bez metadanych)",      funnel.get("orphans")),
        ]
        html += '<table border="1" cellpadding="4" cellspacing="0"><tr>'
        html += "".join(f"<th>{name}</th>" for name, _ in steps) + "</tr><tr>"
        html += "".join(f"<td>{'—' if v is None else v}</td>" for _, v in steps)
        html += "</tr></table>"
        html += (f'<p class="cap">Sparsowane wg typu: Embedded '
                 f'<b>{funnel.get("embedded", "?")}</b>, NotEmbedded '
                 f'<b>{funnel.get("notembedded", "?")}</b>.</p>')
    else:
        html += f'<p>Sieroty (bez metadanych w Excel): <b>{orphans}</b></p>'

    # 2) Per-split counts — images AND unique fish
    html += ('<p class="cap">Liczności per split — obrazy i unikalne ryby '
             '(podział per-ryba, bez wycieku między splitami).</p>')
    html += ('<table border="1" cellpadding="4" cellspacing="0">'
             '<tr><th>Typ</th><th>Split</th><th>Obrazy</th><th>Ryby</th></tr>')
    for ptype in ptypes:
        for split in ["train", "val", "test"]:
            n_img = counts.get(ptype, {}).get(split, 0)
            n_fish = fish_counts.get(ptype, {}).get(split)
            html += (f"<tr><td>{ptype}</td><td>{split}</td><td>{n_img}</td>"
                     f"<td>{'—' if n_fish is None else n_fish}</td></tr>")
    html += "</table>"

    # 3) Age distribution per split — small multiples (makes a split skew obvious)
    html += _section_a_age_charts(ptypes, age_by_split, age_dists)

    html += "</section>"
    return html


def _section_a_age_charts(ptypes, age_by_split: dict, age_dists: dict) -> str:
    """Per-split age histograms (small multiples), or a single-histogram fallback."""
    html = ""
    for ptype in ptypes:
        by_split = age_by_split.get(ptype)
        if by_split and any(by_split.get(s) for s in ("train", "val", "test")):
            fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
            for ax, split in zip(axes, ("train", "val", "test")):
                ages = by_split.get(split, [])
                ax.hist(ages, bins=range(0, 21), color=_SPLIT_COLORS[split])
                mean = float(np.mean(ages)) if ages else float("nan")
                ax.set_title(f"{split} (n={len(ages)}, śr={mean:.1f})")
                ax.set_xlabel("Wiek (lata)")
                ax.set_xticks(range(0, 21, 4))
                _style_ax(ax)
            axes[0].set_ylabel("Liczba obrazów")
            fig.suptitle(f"Rozkład wieku per split — {ptype}", color=_INK)
            fig.tight_layout()
            html += _img_tag(_fig_to_b64(fig))
            plt.close(fig)
            html += ('<p class="cap">Rozkłady train/val/test powinny się pokrywać; '
                     'rozjazd (np. test = tylko stare ryby) sygnalizuje zły podział.</p>')
        elif age_dists.get(ptype):
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.hist(age_dists[ptype], bins=range(0, 21),
                    color=_PTYPE_COLORS.get(ptype, "#888888"))
            ax.set_xlabel("Wiek (lata)")
            ax.set_ylabel("Liczba obrazów")
            ax.set_title(f"Rozkład wieku — {ptype}")
            ax.set_xticks(range(0, 21, 2))
            _style_ax(ax)
            fig.tight_layout()
            html += _img_tag(_fig_to_b64(fig))
            plt.close(fig)
    return html


def _section_b(training_logs: dict) -> str:
    html = '<section id="B"><h2>B. Dane treningowe</h2>'
    for key, logs in training_logs.items():
        if not logs:
            continue
        epochs     = [r.get("epoch", i) for i, r in enumerate(logs)]
        train_loss = [r.get("train_loss", float("nan")) for r in logs]
        val_loss   = [r.get("val_loss", float("nan")) for r in logs]
        val_mae    = [r.get("val_mae", float("nan")) for r in logs]
        lr         = [r.get("lr", float("nan")) for r in logs]

        def _col(name):
            return [r.get(name, float("nan")) for r in logs]
        coral_loss = _col("coral_loss")
        mil_loss   = _col("mil_loss")
        mil_active = _col("mil_active")
        mean_age   = _col("mean_age")
        has_components = any(not np.isnan(v) for v in coral_loss) or \
                         any(not np.isnan(v) for v in mil_loss)
        has_localise = any(not np.isnan(v) for v in mil_active)

        best_idx = int(np.nanargmin(val_mae)) if any(not np.isnan(v) for v in val_mae) else None
        best_mae = val_mae[best_idx] if best_idx is not None else float("nan")
        # best_idx is the POSITION in this run's log; the real epoch NUMBER is
        # epochs[best_idx]. Show the epoch number (not the list index) in the table —
        # a concatenated train.log used to make them differ (11.07 TO-DO Punkt 5).
        best_epoch_num = epochs[best_idx] if best_idx is not None else None

        html += f"<h3>{key}</h3>"

        # Row 1: train/val loss, val_MAE, LR (always available)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        axes[0].plot(epochs, train_loss, label="train_loss", color="#2a78d6", linewidth=2)
        axes[0].plot(epochs, val_loss, label="val_loss", color="#eb6834", linewidth=2)
        if best_idx is not None:
            axes[0].axvline(epochs[best_idx], color=_MUTED, linestyle="--", alpha=0.7)
        axes[0].set_title("Loss: train vs val")
        axes[0].legend()
        axes[1].plot(epochs, val_mae, color="#008300", linewidth=2)
        if best_idx is not None:
            axes[1].axvline(epochs[best_idx], color=_MUTED, linestyle="--", alpha=0.7,
                            label=f"best={best_mae:.3f}")
        axes[1].set_title("val_MAE")
        axes[1].legend()
        axes[2].plot(epochs, lr, color="#4a3aa7", linewidth=2)
        axes[2].set_title("Learning rate")
        for ax in axes:
            ax.set_xlabel("Epoka")
            _style_ax(ax)
        fig.tight_layout()
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)
        html += ('<p class="cap">Loss: rozjazd train↓ / val↑ = przeuczenie. '
                 'val_MAE: linia = najlepsza epoka. LR: harmonogram uczenia.</p>')

        # Row 2: CORAL vs MIL loss, #active-vs-age (only if trainer logged them)
        if has_components or has_localise:
            n = int(has_components) + int(has_localise)
            fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3), squeeze=False)
            axes = axes[0]
            i = 0
            if has_components:
                axes[i].plot(epochs, coral_loss, label="CORAL (liczy wiek)",
                             color="#2a78d6", linewidth=2)
                axes[i].plot(epochs, mil_loss, label="MIL (lokalizuje)",
                             color="#eb6834", linewidth=2)
                axes[i].set_title("Strata CORAL vs MIL")
                axes[i].legend()
                axes[i].set_xlabel("Epoka")
                _style_ax(axes[i])
                i += 1
            if has_localise:
                axes[i].plot(epochs, mil_active, label="#aktywnych (p>0.5)",
                             color="#008300", linewidth=2)
                axes[i].plot(epochs, mean_age, label="średni wiek", color=_MUTED,
                             linewidth=2, linestyle="--")
                axes[i].set_title("Lokalizacja MIL: #aktywnych vs wiek")
                axes[i].legend()
                axes[i].set_xlabel("Epoka")
                _style_ax(axes[i])
            fig.tight_layout()
            html += _img_tag(_fig_to_b64(fig))
            plt.close(fig)
            html += ('<p class="cap">Która głowica się uczy (CORAL vs MIL) oraz czy MIL '
                     '<b>lokalizuje</b>: #aktywnych patchy powinno zbiegać do średniego wieku.</p>')

        html += f"""
<table border="1" cellpadding="4"><tr>
<th>Model</th><th>Best epoch</th><th>Best val_MAE</th>
</tr><tr>
<td>{key}</td><td>{best_epoch_num}</td><td>{best_mae:.4f}</td>
</tr></table><br>"""

    html += "</section>"
    return html


def _section_c(results: dict, single: bool = False) -> str:
    condition_labels = {
        "emb_on_emb":       "Emb → Emb",
        "notemb_on_notemb": "NotEmb → NotEmb",
        "emb_on_notemb":    "Emb → NotEmb ★ CROSS",
        "notemb_on_emb":    "NotEmb → Emb ★ CROSS",
    }
    if single:
        # Only keep conditions that carry data (embedded-only → Emb → Emb).
        condition_labels = {
            k: v.replace(" ★ CROSS", "") for k, v in condition_labels.items()
            if results.get(k) is not None and not results[k].empty
        }
    heading = "C. Metryki ewaluacyjne" + ("" if single else " (4 warunki)")
    html = f'<section id="C"><h2>{heading}</h2>'

    metric_rows = []
    scatter_figs = []

    for cond_key, label in condition_labels.items():
        df = results.get(cond_key)
        if df is None or df.empty:
            continue
        y_true = df["age"].values
        y_pred = df["predicted_age"].values
        m = compute_metrics(y_true, y_pred)
        metric_rows.append({
            "Warunek": label,
            "MAE": f"{m['MAE']:.3f}",
            "RMSE": f"{m['RMSE']:.3f}",
            "R²": f"{m['R2']:.3f}",
            "Acc±1yr": f"{m['Acc1yr']:.1%}",
            "Acc±2yr": f"{m['Acc2yr']:.1%}",
            "Bias": f"{m['Bias']:+.3f}",
        })

        # Scatter — colour = condition (consistent with all other charts)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(y_true, y_pred, alpha=0.5, s=16,
                   color=_COND_COLORS.get(cond_key, "#2a78d6"), edgecolor="none")
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], color=_MUTED, linestyle="--", linewidth=1)
        ax.set_xlabel("Wiek rzeczywisty")
        ax.set_ylabel("Wiek przewidziany")
        ax.set_title(f"{label}\nMAE={m['MAE']:.2f}")
        _style_ax(ax)
        fig.tight_layout()
        scatter_figs.append(_fig_to_b64(fig))
        plt.close(fig)

    if metric_rows:
        html += "<table border='1' cellpadding='4' cellspacing='0'>"
        html += "<tr>" + "".join(f"<th>{k}</th>" for k in metric_rows[0].keys()) + "</tr>"
        for row in metric_rows:
            html += "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"
        html += "</table>"

    # Scatter plots in a row
    if scatter_figs:
        html += "<div style='display:flex;gap:8px;flex-wrap:wrap;'>"
        for b64 in scatter_figs:
            html += f'<div style="width:22%">{_img_tag(b64, "100%")}</div>'
        html += "</div>"

    # Additional plots: MAE per age class, error distribution, box plot
    html += _plots_per_condition(results, condition_labels)

    html += "</section>"
    return html


def _confusion_matrix_b64(y_true, y_pred, label: str) -> str:
    """Row-normalised confusion matrix (true × predicted) as a base64 PNG.

    Rows = true age, cols = predicted age; cell colour = row share, cell number =
    image count. A perfect model puts everything on the diagonal.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    lo = int(min(y_true.min(), y_pred.min()))
    hi = int(max(y_true.max(), y_pred.max()))
    ages = list(range(lo, hi + 1))
    n = len(ages)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t - lo, p - lo] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    side = max(3.6, n * 0.42)
    fig, ax = plt.subplots(figsize=(side, side))
    ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(ages, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(ages, fontsize=6)
    ax.set_xlabel("Wiek przewidziany")
    ax.set_ylabel("Wiek rzeczywisty")
    ax.set_title(f"Macierz pomyłek — {label}", fontsize=9)
    for i in range(n):
        for j in range(n):
            c = int(cm[i, j])
            if c:
                ax.text(j, i, str(c), ha="center", va="center", fontsize=5,
                        color="white" if cm_norm[i, j] > 0.5 else _INK)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _plots_per_condition(results: dict, condition_labels: dict) -> str:
    html = ""
    all_errors = {}
    all_mae_per_age = {}
    all_acc_per_age = {}
    all_bias_per_age = {}
    all_n_per_age = {}
    confusion_figs = []

    for cond_key, label in condition_labels.items():
        df = results.get(cond_key)
        if df is None or df.empty:
            continue
        errors = df["predicted_age"].values - df["age"].values
        all_errors[label] = errors

        ages = sorted(df["age"].unique())
        mae_per_age = {}
        acc_per_age = {}
        bias_per_age = {}
        n_per_age = {}
        for a in ages:
            mask = (df["age"] == a).values
            mae_per_age[a] = float(np.mean(np.abs(errors[mask])))
            acc_per_age[a] = float(np.mean(errors[mask] == 0))   # exact-match accuracy
            bias_per_age[a] = float(np.mean(errors[mask]))       # signed (pred − true)
            n_per_age[a] = int(mask.sum())
        all_mae_per_age[label] = mae_per_age
        all_acc_per_age[label] = acc_per_age
        all_bias_per_age[label] = bias_per_age
        all_n_per_age[label] = n_per_age
        confusion_figs.append(
            _confusion_matrix_b64(df["age"].values, df["predicted_age"].values, label))

    if all_errors:
        # Signed-error distribution per condition. Grouped (side-by-side) bars on
        # integer-year bins — no overlap, so systematic bias (over/under-estimation)
        # and spread are legible. 0 = trafienie; ujemne = zaniżanie wieku.
        all_vals = np.concatenate([np.asarray(v, dtype=float) for v in all_errors.values()])
        lo = int(np.floor(all_vals.min()))
        hi = int(np.ceil(all_vals.max()))
        edges = np.arange(lo - 0.5, hi + 1.5, 1.0)   # bins centred on integers
        data = [np.asarray(all_errors[l], dtype=float) for l in all_errors]

        colors = [_LABEL_COLORS.get(l, "#888888") for l in all_errors]
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.hist(data, bins=edges, label=list(all_errors.keys()), color=colors)
        ax.axvline(0, color=_INK, linewidth=1.2)
        ax.set_xlabel("Błąd = przewidziany − rzeczywisty wiek (lata)")
        ax.set_ylabel("Liczba obrazów")
        ax.set_title("Rozkład błędów per warunek (0 = trafienie)")
        ax.set_xticks(range(lo, hi + 1))
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        _style_ax(ax)
        fig.tight_layout()
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)

        # Box plot of |error| — box fill = condition colour
        fig, ax = plt.subplots(figsize=(7, 4))
        labels_list = list(all_errors.keys())
        box_data = [np.abs(all_errors[l]) for l in labels_list]
        box_ticks = [l.replace(" ★ CROSS", "\n★CROSS") for l in labels_list]
        try:
            bp = ax.boxplot(box_data, tick_labels=box_ticks, patch_artist=True)
        except TypeError:
            bp = ax.boxplot(box_data, labels=box_ticks, patch_artist=True)
        for patch, l in zip(bp["boxes"], labels_list):
            patch.set_facecolor(_LABEL_COLORS.get(l, "#888888"))
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color(_INK)
        ax.set_ylabel("|błąd| (lata)")
        ax.set_title("Rozkład błędów bezwzględnych")
        _style_ax(ax)
        fig.tight_layout()
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)

    if all_mae_per_age:
        age_union = sorted({a for d in all_mae_per_age.values() for a in d})
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, mae_dict in all_mae_per_age.items():
            ages = sorted(mae_dict.keys())
            ax.plot(ages, [mae_dict[a] for a in ages], marker="o", markersize=5,
                    linewidth=2, color=_LABEL_COLORS.get(label, "#888888"), label=label)
        ax.set_xlabel("Klasa wiekowa (lata)")
        ax.set_ylabel("MAE (lata)")
        ax.set_title("MAE per klasa wiekowa")
        if age_union:
            ax.set_xticks(age_union)
        ax.legend(fontsize=7)
        _style_ax(ax)
        fig.tight_layout()
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)
        html += ('<p class="cap">MAE per klasa wiekowa — gdzie model myli się najbardziej '
                 '(zwykle rzadkie, starsze roczniki).</p>')

    if all_acc_per_age:
        age_union = sorted({a for d in all_acc_per_age.values() for a in d})
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, acc_dict in all_acc_per_age.items():
            ages = sorted(acc_dict.keys())
            ax.plot(ages, [acc_dict[a] for a in ages], marker="o", markersize=5,
                    linewidth=2, color=_LABEL_COLORS.get(label, "#888888"), label=label)
        ax.set_xlabel("Klasa wiekowa (lata)")
        ax.set_ylabel("Dokładność (pred = prawda)")
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy per klasa wiekowa")
        if age_union:
            ax.set_xticks(age_union)
        ax.legend(fontsize=7)
        _style_ax(ax)
        fig.tight_layout()
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)
        html += ('<p class="cap">Dokładność dokładnego trafienia (pred = prawda) '
                 'per klasa wiekowa.</p>')

    if all_bias_per_age:
        age_union = sorted({a for d in all_bias_per_age.values() for a in d})
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, bias_dict in all_bias_per_age.items():
            xs = sorted(bias_dict.keys())
            ax.plot(xs, [bias_dict[a] for a in xs], marker="o", markersize=5,
                    linewidth=2, color=_LABEL_COLORS.get(label, "#888888"), label=label)
        ax.axhline(0, color=_INK, linewidth=1)
        ax.set_xlabel("Klasa wiekowa (lata)")
        ax.set_ylabel("Bias = śr(pred − prawda)")
        ax.set_title("Bias per klasa wiekowa (0 = bez obciążenia; <0 = zaniża)")
        if age_union:
            ax.set_xticks(age_union)
        ax.legend(fontsize=7)
        _style_ax(ax)
        fig.tight_layout()
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)
        html += ('<p class="cap">Znakowany bias per wiek — czy model systematycznie '
                 'zaniża/zawyża zależnie od rocznika (różny od |MAE|).</p>')

    if all_n_per_age:
        age_union = sorted({a for d in all_n_per_age.values() for a in d})
        labels_list = list(all_n_per_age.keys())
        width = 0.8 / max(1, len(labels_list))
        fig, ax = plt.subplots(figsize=(8, 3.2))
        for k, label in enumerate(labels_list):
            counts = [all_n_per_age[label].get(a, 0) for a in age_union]
            xs = [a + (k - (len(labels_list) - 1) / 2.0) * width for a in age_union]
            ax.bar(xs, counts, width=width,
                   color=_LABEL_COLORS.get(label, "#888888"), label=label)
        ax.set_xlabel("Klasa wiekowa (lata)")
        ax.set_ylabel("Liczba obrazów (n)")
        ax.set_title("Liczność per klasa wiekowa (zbiór testowy)")
        if age_union:
            ax.set_xticks(age_union)
        ax.legend(fontsize=7)
        _style_ax(ax)
        fig.tight_layout()
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)
        html += ('<p class="cap">Ile obrazów testowych w każdym roczniku — słabe MAE/bias '
                 'zwykle tam, gdzie n małe (rzadkie, starsze ryby).</p>')

    if confusion_figs:
        html += "<div style='display:flex;gap:8px;flex-wrap:wrap;'>"
        for b64 in confusion_figs:
            html += f'<div style="max-width:32%">{_img_tag(b64, "100%")}</div>'
        html += "</div>"
        html += ('<p class="cap">Macierz pomyłek: wiersz = wiek rzeczywisty, kolumna = '
                 'przewidziany; kolor = udział wiersza, liczba = #obrazów. Idealnie wszystko '
                 'na przekątnej.</p>')

    return html


def _section_d(results: dict) -> str:
    def _mae(cond_key: str) -> float:
        df = results.get(cond_key)
        if df is None or df.empty:
            return float("nan")
        return float(np.mean(np.abs(df["predicted_age"].values - df["age"].values)))

    mae_ee = _mae("emb_on_emb")
    mae_nn = _mae("notemb_on_notemb")
    mae_en = _mae("emb_on_notemb")   # cross
    mae_ne = _mae("notemb_on_emb")   # cross

    comment_e = cross_comment(mae_ee, mae_en)
    comment_n = cross_comment(mae_nn, mae_ne)

    def _fmt(v: float) -> str:
        return f"{v:.3f} yr" if not np.isnan(v) else "N/A"

    return f"""
<section id="D">
<h2>D. Tabela cross-ewaluacji</h2>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th></th><th>Test: Embedded</th><th>Test: NotEmbedded</th></tr>
<tr>
  <th>Model: Embedded</th>
  <td>MAE = {_fmt(mae_ee)}</td>
  <td>MAE = {_fmt(mae_en)} ← CROSS</td>
</tr>
<tr>
  <th>Model: NotEmbedded</th>
  <td>MAE = {_fmt(mae_ne)} ← CROSS</td>
  <td>MAE = {_fmt(mae_nn)}</td>
</tr>
</table>
<p><b>Emb model cross:</b> {comment_e}</p>
<p><b>NotEmb model cross:</b> {comment_n}</p>
<p style="font-size:90%;color:#444;margin-top:1em;">
  <b>Mapy uwagi modelu</b> (w katalogach <code>heatmaps/</code> i <code>overlays/</code>
  każdej kondycji): kolormap <i>inferno</i> — ciemne = niski sygnał, jasno-żółte =
  wysoki sygnał. <code>heatmaps/</code> = czysta mapa ważności w rozdzielczości
  oryginału, <code>overlays/</code> = ta sama mapa zblendowana z oryginalnym
  zdjęciem (α=0.55) <i>wewnątrz sylwetki otolitu</i>; poza otolitem pokazujemy
  surowe zdjęcie, żeby tło nie generowało fałszywych „gorących punktów”.
</p>
</section>
"""


def _section_e(increment_cards: dict) -> str:
    html = '<section id="E"><h2>E. Karty rozumowania modelu</h2>'
    html += """
<p>Każda karta to <b>6 paneli w dwóch rzędach = dwie głowice modelu</b>. Rząd górny
(pasek granatowy) to <b>GŁOWICA WIEKU (CORAL)</b> — „na co model patrzy, licząc wiek".
Rząd dolny (pasek pomarańczowy) to <b>GŁOWICA LOKALIZACJI (density)</b> — „gdzie model
widzi przyrosty". Wszystkie panele renderowane są na zdjęciu w oryginalnej rozdzielczości;
cyjanowy obrys = kontur otolitu, żółta linia = oś biologiczna od jądra do najdalszej
krawędzi konturu.</p>
<p><b>Rząd 1 — GŁOWICA WIEKU (CORAL):</b></p>
<ol>
  <li><b>Grad-CAM werdyktu</b> — mapa (inferno), które rejony najbardziej wpływają na
      przewidziany wiek. Uwaga: na ścieżce tokenu CLS gradient bywa rozmyty i mapa jest
      wtedy niemal płaska — tytuł panelu sygnalizuje to jako „(plaski/nieinform.)". To
      znane ograniczenie tej atrybucji, nie brak renderu.</li>
  <li><b>Uwaga CLS</b> — z których patchy token CLS złożył „streszczenie" obrazu, z
      którego CORAL liczy wiek. Gdy backbone używa fused-attention (nie wystawia macierzy
      uwagi), panel pokazuje <b>oznaczone proxy L2</b> (norma tokenów) zamiast prawdziwej
      uwagi — tytuł mówi wtedy „proxy L2 patchy (CLS niedost.)".</li>
  <li><b>Werdykt</b> — zdjęcie + etykieta „Wiek: X (true: Y)" + ramka: <b>zielona</b> =
      trafny, <b>czerwona</b> = błąd.</li>
</ol>
<p><b>Rząd 2 — GŁOWICA LOKALIZACJI (density):</b></p>
<ol start="4">
  <li><b>Mapa density</b> — kolormap inferno wewnątrz otolitu: ciemne = niski sygnał,
      jasno-żółte = wysoki. Wartość = <b>prawdopodobieństwo przyrostu na patch</b>
      (odsprzęgnięta głowica density, uczona słabo — samą liczbą wieku). Gdy znaleziono
      sensowny zestaw pierścieni-konsensusu (≥2), nakładane są ich krzywe.</li>
  <li><b>Kandydaci</b> — <b>żółte kropki</b> = wszystkie lokalne maksima mapy density
      wzdłuż <b>48 promieni</b> z jądra we wszystkich kierunkach (nie jednej osi). Cyjanowy
      kontur, żółta oś pomiaru, niebieski krzyżyk = jądro. To „surowy" sygnał lokalizacji.</li>
  <li><b>Finalne (N = wiek)</b> — <b>czerwone kropki</b> = top-<code>wiek</code> „pierścieni-
      konsensusu" (piki na tym samym promieniu w wielu kierunkach), rzutowane na oś pomiaru.
      <b>Zielone puste okręgi</b> = piki klasyczne (OpenCV) — kontrola „model vs technik".</li>
</ol>
<p style="font-size:90%;color:#444;">
  <b>Wiek (werdykt) vs liczba pierścieni.</b> Werdykt wiekowy pochodzi z głowicy
  <b>liczącej</b> (CORAL, rząd 1), a kropki/finalne z głowicy <b>lokalizującej</b>
  (density, rząd 2, odsprzęgniętej stop-gradientem) — to dwa niezależne sygnały. Mogą się
  różnić (np. wiek = 3, a zlokalizowane 2 przyrosty); trafność lokalizacji jest wciąż
  wąskim gardłem — patrz sekcja H i dziennik Kierunku B.
</p>
"""
    for label, paths in increment_cards.items():
        html += f"<h3>{label.capitalize()}</h3>"
        for p in paths:
            b64 = _load_png_b64(p)
            if b64:
                html += _img_tag(b64, "95%") + "<br>"
    html += "</section>"
    return html


def _section_f(model_info: dict) -> str:
    rows = ""
    for k, v in model_info.items():
        rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
    return f"""
<section id="F">
<h2>F. Informacje o modelu i konfiguracji</h2>
<table border="1" cellpadding="4" cellspacing="0">
{rows}
</table>
</section>
"""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

_OPENCV_JS = r"""
(function(){
  function smooth(a, sigma){
    if(sigma<=0) return a.slice();
    var r=Math.max(1,Math.ceil(sigma*2)), k=[], s=0, i, j;
    for(i=-r;i<=r;i++){var w=Math.exp(-(i*i)/(2*sigma*sigma));k.push(w);s+=w;}
    return a.map(function(_,idx){var acc=0;for(j=-r;j<=r;j++){var q=Math.min(a.length-1,Math.max(0,idx+j));acc+=a[q]*k[j+r];}return acc/s;});
  }
  function findPeaks(a, prom, minD){
    var cand=[], i, t;
    for(i=1;i<a.length-1;i++){
      if(a[i]>=a[i-1] && a[i]>a[i+1]){
        var l=i; while(l>0 && a[l-1]<=a[i]) l--;
        var rr=i; while(rr<a.length-1 && a[rr+1]<=a[i]) rr++;
        var lmin=a[i]; for(t=l;t<=i;t++) lmin=Math.min(lmin,a[t]);
        var rmin=a[i]; for(t=i;t<=rr;t++) rmin=Math.min(rmin,a[t]);
        if(a[i]-Math.max(lmin,rmin)>=prom) cand.push(i);
      }
    }
    cand.sort(function(x,y){return a[y]-a[x];});
    var kept=[];
    cand.forEach(function(p){ if(kept.every(function(q){return Math.abs(q-p)>=minD;})) kept.push(p); });
    kept.sort(function(x,y){return x-y;});
    return kept;
  }
  function widget(d){
    var box=document.createElement('div');
    box.style.cssText='display:inline-block;vertical-align:top;margin:8px;border:1px solid #ddd;padding:6px;border-radius:6px;';
    var cv=document.createElement('canvas'); cv.width=d.w; cv.height=d.h; cv.style.maxWidth='100%';
    var img=new Image();
    var ctrls=document.createElement('div'); ctrls.style.cssText='font-size:12px;margin-top:4px;';
    function mk(label,min,max,step,val){
      var wrap=document.createElement('label'); wrap.style.cssText='display:block;margin:2px 0;';
      var sp=document.createElement('span'); sp.textContent=label+': ';
      var inp=document.createElement('input'); inp.type='range'; inp.min=min; inp.max=max; inp.step=step; inp.value=val; inp.style.verticalAlign='middle';
      var out=document.createElement('span'); out.textContent=val; out.style.marginLeft='4px';
      inp.addEventListener('input',function(){out.textContent=inp.value;draw();});
      wrap.appendChild(sp);wrap.appendChild(inp);wrap.appendChild(out);ctrls.appendChild(wrap);
      return inp;
    }
    var sSigma=mk('wygladzanie sigma',0,4,0.5,1);
    var sProm=mk('prominencja',0.02,0.5,0.02,0.1);
    var sDist=mk('min odstep',1,10,1,3);
    var readout=document.createElement('div'); readout.style.cssText='font-weight:bold;margin-top:4px;'; ctrls.appendChild(readout);
    function draw(){
      var ctx=cv.getContext('2d');
      ctx.clearRect(0,0,cv.width,cv.height);
      if(img.complete) ctx.drawImage(img,0,0,cv.width,cv.height);
      ctx.strokeStyle='rgba(255,220,0,0.7)';ctx.lineWidth=1.5;ctx.beginPath();
      d.line.forEach(function(p,i){ if(i===0)ctx.moveTo(p[0],p[1]); else ctx.lineTo(p[0],p[1]); }); ctx.stroke();
      var prof=smooth(d.profile, parseFloat(sSigma.value));
      var peaks=findPeaks(prof, parseFloat(sProm.value), parseInt(sDist.value));
      ctx.fillStyle='rgba(255,60,60,0.95)';
      peaks.forEach(function(idx){ var p=d.line[Math.min(idx,d.line.length-1)]; ctx.beginPath();ctx.arc(p[0],p[1],4,0,2*Math.PI);ctx.fill(); });
      readout.textContent='OpenCV wykryl: '+peaks.length+'  (model: '+d.pred_age+', prawda: '+d.true_age+')';
    }
    img.onload=draw; img.src=d.img;
    box.appendChild(cv); box.appendChild(ctrls);
    return box;
  }
  var root=document.getElementById('cv-widgets');
  if(root && typeof OPENCV_DATA!=='undefined'){ OPENCV_DATA.forEach(function(d){ root.appendChild(widget(d)); }); }
})();
"""


def _section_opencv(opencv_reference: dict | None) -> str:
    """Section H — interactive classical (OpenCV-style) increment detection for
    technicians (11.07 Punkt 7 / Kierunek A). Profile is precomputed in Python;
    smoothing + peak detection run live in the browser via sliders. Reference only —
    it does not touch the model or its verdict.
    """
    if not opencv_reference:
        return ""
    import json as _json
    items = []
    for iid, d in opencv_reference.items():
        if not d or not d.get("line") or not d.get("profile"):
            continue
        items.append({
            "id": str(iid),
            "img": d["img"], "w": d["w"], "h": d["h"],
            "line": d["line"], "profile": d["profile"],
            "true_age": d.get("true_age", 0), "pred_age": d.get("pred_age", 0),
        })
    if not items:
        return ""
    data_json = _json.dumps(items)
    html = '<section id="H"><h2>H. OpenCV — reference dla technikow (klasyczna detekcja przyrostow)</h2>'
    html += ('<p class="cap">Interaktywnie: suwaki steruja wygladzaniem i detekcja pikow profilu '
             'intensywnosci wzdluz osi odczytu. Czerwone kropki = przyrosty wykryte METODA KLASYCZNA '
             '(nie model) — porownaj z werdyktem modelu. To punkt odniesienia; klasyka bywa zawodna.</p>')
    html += '<div id="cv-widgets"></div>'
    html += '<script>const OPENCV_DATA=' + data_json + ';</script>'
    html += '<script>' + _OPENCV_JS + '</script>'
    html += '</section>'
    return html


def _section_localization_walkthrough(payload: dict | None) -> str:
    """Sekcja edukacyjna „krok po kroku": na JEDNYM otolicie pokazuje, jak z kandydatów
    z 48 promieni (density + klasyka) metoda DP wybiera finalne przyrosty (czerwone).

    ``payload`` (z run_pipeline): ``image_id, true_age, pred_age, panel_rays_b64,
    panel_final_b64, data`` gdzie ``data`` = wynik ``ring_extraction.dp_walkthrough_data``.
    """
    if not payload or not payload.get("data"):
        return ""
    d = payload["data"]
    age = int(payload.get("pred_age", 0))
    gap = float(d.get("dp_min_gap", 0.04))

    # --- Panel 2: profile 3 promieni (surowy vs znormalizowany per-promień) + piki ---
    profiles = d.get("sample_profiles") or []
    prof_b64 = ""
    if profiles:
        n = len(profiles)
        fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 2.8), squeeze=False)
        for j, pr in enumerate(profiles):
            ax = axes[0][j]
            t = pr["t"]
            ax.plot(t, pr["raw"], color="#999", lw=1.0, label="surowy")
            ax.plot(t, pr["norm"], color="#2a78d6", lw=1.6, label="znorm. [0,1]")
            for pt in pr.get("peak_t", []):
                ax.axvline(pt, color="#e01e1e", ls="--", lw=1.0)
            ax.set_title(f"promień {j + 1}", fontsize=9)
            ax.set_xlabel("t (jądro→brzeg)", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            if j == 0:
                ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        prof_b64 = _fig_to_b64(fig)
        plt.close(fig)

    # --- Panel 3: głosowanie po promieniu (piki density vs klasyka, x=t) ---
    dpk = d.get("density_peaks") or []
    cpk = d.get("classical_peaks") or []
    vote_b64 = ""
    if dpk or cpk:
        fig, ax = plt.subplots(figsize=(9, 2.6))
        dt = [p[0] for p in dpk]
        ct = [p[0] for p in cpk]
        if dt:
            ax.hist(dt, bins=40, range=(0, 1), color="#f4b400", alpha=0.75, label="density (48 promieni)")
        if ct:
            ax.hist(ct, bins=40, range=(0, 1), color="#0a9d6e", alpha=0.55, label="klasyka (48 promieni)")
        for (mt, _s, _st) in (d.get("density_clusters") or []):
            ax.axvline(mt, color="#c58a00", ls=":", lw=0.8)
        ax.set_xlabel("t (znormalizowany promień, jądro→brzeg)", fontsize=9)
        ax.set_ylabel("liczba pików (kierunków)", fontsize=9)
        ax.set_title("Głosowanie po promieniu — ile z 48 kierunków ma pik na danym promieniu", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        vote_b64 = _fig_to_b64(fig)
        plt.close(fig)

    # --- Panel 4: score pierścieni + wybór DP (wyróżnione `wiek` wybranych) ---
    merged = d.get("merged") or []
    chosen = set(round(t, 4) for t in (d.get("chosen_t") or []))
    dp_b64 = ""
    if merged:
        fig, ax = plt.subplots(figsize=(9, 2.8))
        src_color = {"consensus": "#7b1fa2", "density": "#f4b400", "classical": "#0a9d6e"}
        for (t, score, source) in merged:
            picked = round(t, 4) in chosen
            ax.bar(t, score, width=0.012,
                   color=src_color.get(source, "#888"),
                   edgecolor="#e01e1e" if picked else "none",
                   linewidth=2.2 if picked else 0.0)
        for t in (d.get("chosen_t") or []):
            ax.axvline(t, color="#e01e1e", ls="--", lw=1.0, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("t (promień)", fontsize=9)
        ax.set_ylabel("score pierścienia\n(support × siła; konsensus = suma)", fontsize=8)
        n_sel = len(d.get("chosen_t") or [])
        _cap = (f"wybrano {n_sel} z {age} (tyle odrębnych pierścieni-kandydatów)"
                if n_sel < age else f"dokładnie {age}")
        ax.set_title(f"Wybór DP: {_cap} (czerwona ramka), min. rozstaw {gap:g}", fontsize=10)
        # legenda źródeł
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=l) for l, c in
                           (("konsensus", "#7b1fa2"), ("density", "#f4b400"), ("klasyka", "#0a9d6e"))],
                  fontsize=8, loc="upper right")
        fig.tight_layout()
        dp_b64 = _fig_to_b64(fig)
        plt.close(fig)

    def _fig_block(title, desc, b64):
        if not b64:
            return ""
        return (f'<h3 style="margin:0.8em 0 0.2em;">{title}</h3>'
                f'<p style="margin:0 0 0.4em;color:#555;">{desc}</p>{_img_tag(b64)}')

    n_final = len(d.get("chosen_t") or [])
    note = ("" if n_final >= age else
            f' <b>Uwaga:</b> tu DP znalazło tylko <b>{n_final}</b> odrębnych pierścieni (wiek {age}) — '
            'gęste piki klasyki sklejają się w klastrowaniu po promieniu; to znane ograniczenie do poprawy.')
    html = (
        f'<section id="loc-walkthrough"><h2>Lokalizacja — jak wybieramy przyrosty (krok po kroku)</h2>'
        f'<p>Na jednym otolicie (<code>{payload.get("image_id","")[:60]}</code>, wiek modelu '
        f'<b>{age}</b>, prawdziwy {int(payload.get("true_age",0))}) pokazujemy, jak z kandydatów na '
        f'<b>48 promieniach</b> (density modelu + klasyka obrazu) metoda <b>DP</b> (sekcja L) wybiera '
        f'finalne przyrosty (czerwone).{note} Ta sama logika działa na wszystkich 20 otolitach niżej.</p>'
    )
    html += _fig_block(
        "Krok 1 — kandydaci ze wszystkich 48 kierunków",
        "Z jądra rzucamy 48 promieni do konturu. Wzdłuż każdego szukamy pików: "
        "<b>żółte</b> = mapa density modelu, <b>zielone</b> = klasyka (jasność obrazu). "
        "Cyjan = kontur, żółta linia = oś pomiaru.",
        payload.get("panel_rays_b64", ""))
    html += _fig_block(
        "Krok 2 — profil promienia i normalizacja per-promień",
        "Każdy promień normalizujemy osobno do [0,1] (żeby jasne i ciemne kierunki były "
        "porównywalne), potem szukamy pików (czerwone linie). Trzy przykładowe promienie:",
        prof_b64)
    html += _fig_block(
        "Krok 3 — głosowanie po promieniu",
        "Każdy pik ma promień <code>t</code> (0=jądro, 1=brzeg). Prawdziwy pierścień jest "
        "koncentryczny → pojawia się na tym samym <code>t</code> w wielu kierunkach. Skupiska = "
        "pierścienie-kandydaci.",
        vote_b64)
    html += _fig_block(
        "Krok 4 — score pierścieni i wybór DP",
        "Score pierścienia = ile kierunków go widziało × siła; gdy density i klasyka zgadzają się "
        "co do promienia, score się <b>sumuje</b> (konsensus). DP wybiera dokładnie <code>wiek</code> "
        "pierścieni o najwyższym łącznym score, z <b>wymuszonym rozstawem</b> (nie skupia pików).",
        dp_b64)
    html += _fig_block(
        "Krok 5 — rzut na oś: finalne przyrosty",
        "Wybrane promienie rzutujemy na oś pomiaru (jądro→brzeg) — <b>czerwone</b> punkty. "
        "To jest wynik sekcji L dla tego otolitu.",
        payload.get("panel_final_b64", ""))
    html += '</section>'
    return html


_LOC_METHOD_META = {
    "density":   ("I", "density (model)",
                  "Top-<code>wiek</code> pierścieni z mapy density modelu (48 promieni z jądra)."),
    "classical": ("J", "klasyka (obraz)",
                  "Top-<code>wiek</code> z klasycznej intensywności szarości (te same 48 promieni)."),
    "consensus": ("K", "fuzja (konsensus)",
                  "Pierścienie, gdzie density I klasyka zgadzają się co do promienia; "
                  "liczba = wiek (CORAL). Fallback do density, gdy za mało zgodnych."),
    "dp":        ("L", "fuzja (DP + rozstaw)",
                  "Programowanie dynamiczne: dokładnie <code>wiek</code> pierścieni z połączonej "
                  "puli density+klasyka (konsensus punktowany wyżej), z wymuszonym minimalnym "
                  "rozstawem — rozkłada przyrosty wzdłuż osi zamiast je skupiać."),
}


def _section_localization_methods(localization_methods: dict | None) -> str:
    """Bake-off metod lokalizacji: sekcje I / J / K / L (density | klasyka | konsensus | DP).

    ``localization_methods``: ``{method -> [{image_id, true_age, pred_age, b64, n_final}]}``.
    Każda metoda = osobna sekcja; ta sama pula otolitów, różny sposób wyboru finalnych.
    """
    if not localization_methods:
        return ""
    html = ""
    for method in ("density", "classical", "consensus", "dp"):
        items = localization_methods.get(method)
        if not items:
            continue
        sec_id, title, desc = _LOC_METHOD_META[method]
        html += (
            f'<section id="{sec_id}"><h2>{sec_id}. Lokalizacja — {title}</h2>'
            f'<p>{desc} <b>Czerwone</b> = finalne (N=wiek), żółte = kandydaci, '
            'cyjan = kontur otolitu, żółta linia = oś pomiaru. Ta sama pula otolitów we '
            'wszystkich metodach — porównaj, która najlepiej trafia w realne przyrosty.</p>'
            '<div style="display:flex;flex-wrap:wrap;gap:8px;">'
        )
        for it in items:
            cap = f"wiek {it['pred_age']} (true {it['true_age']}) &middot; N={it['n_final']}"
            html += (
                '<figure style="width:230px;margin:0;">'
                f'<img src="{it["b64"]}" style="width:100%;border:1px solid #ccc;border-radius:3px;">'
                f'<figcaption style="font-size:10px;color:#666;word-break:break-all;">'
                f'{cap}</figcaption></figure>'
            )
        html += '</div></section>'
    return html


def build_comparison_report(
    results: dict,
    training_logs: dict,
    increment_cards: dict,
    dataset_stats: dict,
    output_path: Path,
    model_info: dict | None = None,
    opencv_reference: dict | None = None,
    localization_methods: dict | None = None,
    localization_walkthrough: dict | None = None,
) -> None:
    """Build and write a self-contained HTML comparison report.

    Parameters
    ----------
    results       : keys emb_on_emb, notemb_on_notemb, emb_on_notemb, notemb_on_emb
                    Values are DataFrames with columns: image_id, age, predicted_age
    training_logs : keys embedded, not_embedded → list of per-epoch dicts
                    (keys: epoch, train_loss, val_loss, val_mae, lr)
    increment_cards : keys best, worst → list of PNG Paths
    dataset_stats : keys counts, age_distributions, orphan_count
    output_path   : where to write the HTML file
    model_info    : optional dict of key→value pairs for section F
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Accept cross_-prefixed keys from the pipeline (see normalize_result_keys).
    results = normalize_result_keys(results)

    if model_info is None:
        model_info = {}

    # Embedded-only runs deliver a single condition — the report becomes a
    # single-model "Raport treningu" (no cross-domain Section D, no Emb-vs-NotEmb).
    n_present = sum(1 for k in CANONICAL_CONDITIONS
                    if results.get(k) is not None and not results[k].empty)
    active_ptypes = dataset_stats.get("active_ptypes")
    single = n_present <= 1

    if single:
        ptype = (active_ptypes or ["Embedded"])[0]
        title = f"OtolithDino — Raport treningu ({ptype})"
        single_note = (
            '<p style="background:#eef2ff;padding:8px;border-left:3px solid #2a78d6;">'
            f'Tryb <b>{ptype}-only</b> — trenowany i oceniany jeden model; '
            'brak porównania cross-domain.</p>'
        )
    else:
        title = "OtolithDino — Raport porównawczy Embedded vs NotEmbedded"
        single_note = ""

    body = single_note + _section_a(dataset_stats, active_ptypes)
    body += _section_b(training_logs)
    body += _section_c(results, single=single)
    if not single:
        body += _section_d(results)     # cross-evaluation is meaningless for 1 condition
    body += _section_e(increment_cards)
    body += _section_f(model_info)
    body += _section_opencv(opencv_reference)
    body += _section_localization_walkthrough(localization_walkthrough)
    body += _section_localization_methods(localization_methods)

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{font-family:sans-serif;max-width:1400px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
table {{border-collapse:collapse;margin-bottom:1em;}}
td,th {{padding:4px 8px;}}
h2 {{color:#1a237e;}}
h3 {{color:#283593;}}
p.cap {{font-size:88%;color:#555;margin:2px 0 12px;}}
</style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
