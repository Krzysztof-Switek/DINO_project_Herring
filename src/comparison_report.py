"""Comparative HTML report for Embedded vs NotEmbedded otolith pipeline.

Generates a self-contained HTML file with sections:
  A — Dataset statistics
  B — Training curves (loss, LR, val_MAE)
  C — Evaluation metrics (MAE, RMSE, R², Acc±1yr, Acc±2yr, Bias) for 4 conditions
  D — Cross-evaluation summary table with automatic comment
  E — Increment annotation cards (best / worst predictions)
  F — Model and configuration info
"""
from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for age prediction.

    Returns dict with keys: MAE, RMSE, R2, Acc1yr, Acc2yr, Bias.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    try:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except Exception:
        r2 = float("nan")
    acc1 = float(np.mean(np.abs(errors) <= 1.0))
    acc2 = float(np.mean(np.abs(errors) <= 2.0))
    bias = float(np.mean(errors))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Acc1yr": acc1, "Acc2yr": acc2, "Bias": bias}


def cross_comment(own_mae: float, cross_mae: float) -> str:
    """Return automatic generalization comment based on cross/own MAE ratio."""
    if cross_mae < 1.5 * own_mae:
        return "Model generalizuje dobrze (cross MAE < 1.5 × own MAE)."
    return "Słaba generalizacja cross-domain (cross MAE ≥ 1.5 × own MAE)."


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure, dpi: int = 100) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _load_png_b64(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def _img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:100%;">'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_a(dataset_stats: dict) -> str:
    counts = dataset_stats.get("counts", {})
    rows = ""
    for ptype in ["Embedded", "NotEmbedded"]:
        for split in ["train", "val", "test"]:
            n = counts.get(ptype, {}).get(split, 0)
            rows += f"<tr><td>{ptype}</td><td>{split}</td><td>{n}</td></tr>"
    orphans = dataset_stats.get("orphan_count", "N/A")

    # Age histogram
    age_data = dataset_stats.get("age_distributions", {})
    plots_html = ""
    if age_data:
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = {"Embedded": "#2196F3", "NotEmbedded": "#FF9800"}
        for ptype, ages in age_data.items():
            if ages:
                ax.hist(ages, bins=range(0, 21), alpha=0.6, label=ptype, color=colors.get(ptype))
        ax.set_xlabel("Wiek (lata)")
        ax.set_ylabel("Liczba zdjęć")
        ax.set_title("Rozkład wiekowy: Embedded vs NotEmbedded")
        ax.legend()
        plots_html = _img_tag(_fig_to_b64(fig))
        plt.close(fig)

    return f"""
<section id="A">
<h2>A. Statystyki zbioru danych</h2>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Typ</th><th>Split</th><th>N zdjęć</th></tr>
{rows}
</table>
<p>Sieroty (not-embedded bez metadanych): <b>{orphans}</b></p>
{plots_html}
</section>
"""


def _section_b(training_logs: dict) -> str:
    html = '<section id="B"><h2>B. Dane treningowe</h2>'
    for key, logs in training_logs.items():
        if not logs:
            continue
        epochs = [r.get("epoch", i) for i, r in enumerate(logs)]
        train_loss = [r.get("train_loss", float("nan")) for r in logs]
        val_loss = [r.get("val_loss", float("nan")) for r in logs]
        val_mae = [r.get("val_mae", float("nan")) for r in logs]
        lr = [r.get("lr", float("nan")) for r in logs]

        best_epoch = int(np.nanargmin(val_mae)) if any(not np.isnan(v) for v in val_mae) else None
        best_mae = val_mae[best_epoch] if best_epoch is not None else float("nan")

        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        axes[0].plot(epochs, train_loss, label="train_loss")
        axes[0].plot(epochs, val_loss, label="val_loss")
        if best_epoch is not None:
            axes[0].axvline(epochs[best_epoch], color="red", linestyle="--", alpha=0.5)
        axes[0].set_title(f"Loss — {key}")
        axes[0].legend()
        axes[1].plot(epochs, val_mae)
        if best_epoch is not None:
            axes[1].axvline(epochs[best_epoch], color="red", linestyle="--", alpha=0.5,
                            label=f"best={best_mae:.3f}")
        axes[1].set_title("val_MAE")
        axes[1].legend()
        axes[2].plot(epochs, lr)
        axes[2].set_title("Learning rate")
        for ax in axes:
            ax.set_xlabel("Epoka")
        fig.tight_layout()
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)

        html += f"""
<table border="1" cellpadding="4"><tr>
<th>Model</th><th>Best epoch</th><th>Best val_MAE</th>
</tr><tr>
<td>{key}</td><td>{best_epoch}</td><td>{best_mae:.4f}</td>
</tr></table><br>"""

    html += "</section>"
    return html


def _section_c(results: dict) -> str:
    condition_labels = {
        "emb_on_emb":       "Emb → Emb",
        "notemb_on_notemb": "NotEmb → NotEmb",
        "emb_on_notemb":    "Emb → NotEmb ★ CROSS",
        "notemb_on_emb":    "NotEmb → Emb ★ CROSS",
    }
    html = '<section id="C"><h2>C. Metryki ewaluacyjne (4 warunki)</h2>'

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

        # Scatter
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(y_true, y_pred, alpha=0.4, s=8)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("True age")
        ax.set_ylabel("Predicted age")
        ax.set_title(f"{label}\nMAE={m['MAE']:.2f}")
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


def _plots_per_condition(results: dict, condition_labels: dict) -> str:
    html = ""
    all_errors = {}
    all_mae_per_age = {}

    for cond_key, label in condition_labels.items():
        df = results.get(cond_key)
        if df is None or df.empty:
            continue
        errors = df["predicted_age"].values - df["age"].values
        all_errors[label] = errors

        ages = sorted(df["age"].unique())
        mae_per_age = {}
        for a in ages:
            mask = df["age"] == a
            mae_per_age[a] = float(np.mean(np.abs(errors[mask])))
        all_mae_per_age[label] = mae_per_age

    if all_errors:
        # Error histogram
        fig, ax = plt.subplots(figsize=(7, 3))
        for label, errs in all_errors.items():
            ax.hist(errs, bins=20, alpha=0.5, label=label)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Predicted − True (lata)")
        ax.set_title("Rozkład błędów")
        ax.legend(fontsize=7)
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)

        # Box plot
        fig, ax = plt.subplots(figsize=(7, 4))
        labels_list = list(all_errors.keys())
        ax.boxplot([np.abs(all_errors[l]) for l in labels_list],
                   labels=[l.replace(" ★ CROSS", "\n★CROSS") for l in labels_list])
        ax.set_ylabel("|error| (lata)")
        ax.set_title("Rozkład błędów bezwzględnych")
        fig.tight_layout()
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)

    if all_mae_per_age:
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, mae_dict in all_mae_per_age.items():
            ages = sorted(mae_dict.keys())
            ax.plot(ages, [mae_dict[a] for a in ages], marker="o", markersize=4, label=label)
        ax.set_xlabel("Klasa wiekowa")
        ax.set_ylabel("MAE (lata)")
        ax.set_title("MAE per klasa wiekowa")
        ax.legend(fontsize=7)
        html += _img_tag(_fig_to_b64(fig))
        plt.close(fig)

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
<p>Każda karta pokazuje 6 etapów dochodzenia modelu do werdyktu wiekowego
(od surowego zdjęcia do liczby pierścieni). Wszystkie panele renderowane są
na zdjęciu w oryginalnej rozdzielczości — żółta linia to <i>oś biologiczna</i>
od jądra (centroid) do najdalszej krawędzi konturu (zwykle post-rostralnej):</p>
<ol>
  <li><b>Surowe zdjęcie</b> — wejście do modelu, bez żadnych anotacji.</li>
  <li><b>Segmentacja otolitu</b> — cyjanowy kontur, niebieski krzyżyk = jądro
      (klasyczne CV, niezależnie od modelu). Tu sprawdzasz, czy segmentacja w
      ogóle złapała otolit; jeśli nie, panele 4/5/6 pokażą placeholder
      „Segmentacja nieudana”.</li>
  <li><b>Mapa uwagi modelu</b> — kolormap <i>inferno</i> wewnątrz otolitu:
      ciemne = niski sygnał, jasno-żółte = wysoki sygnał. Wartość = norma L2
      tokenów DINOv2 (heurystyka — model w obecnej wersji nie jest jeszcze
      trenowany do bezpośredniej lokalizacji przyrostów).</li>
  <li><b>Oś pomiaru + profil 1D</b> — żółta oś biologiczna na zdjęciu, a w
      <b>prawym górnym rogu mała wstawka 1D</b> z profilem ważności wzdłuż tej
      osi. Wstawkę czyta się od góry: oś pionowa = pozycja na osi (góra = jądro,
      dół = brzeg), oś pozioma = wartość ważności. <i>Czerwone kropki + przerywane
      poziome linie</i> w tej wstawce = wykryte peaki profilu (kandydaci na
      przyrosty roczne).</li>
  <li><b>Strefy roczne</b> — obszary <i>między kolejnymi peakami</i>, każdy w
      innym kolorze (czerwony / niebieski / zielony / fioletowy …). Tu widać
      <i>jak duży obszar otolitu</i> model przypisuje każdemu rocznemu
      przyrostowi. <b>Jeśli model nie wykrył ani jednego peaku, cały otolit
      jest zalany jednym kolorem</b> — to nie jest brakujący render, to
      sygnał, że profil 1D był płaski.</li>
  <li><b>Werdykt</b> — końcowa anotacja: żółta oś + <i>ponumerowane żółte
      kropki</i> w pozycjach peaków + przewidziany wiek (zielona ramka = trafny,
      czerwona = błąd). <b>Brak ponumerowanych kropek na osi = peak-detector
      zwrócił pustą listę</b> (zob. uwaga niżej).</li>
</ol>
<p style="font-size:90%;color:#666;background:#f8f8f0;padding:8px;border-left:3px solid #c8b400;">
  <b>Dlaczego niektóre karty mają puste panele 4/5/6?</b><br>
  Peaki w profilu 1D wykrywa <code>scipy.signal.find_peaks</code> z progiem
  prominencji <code>0.1</code>. Gdy model jest jeszcze niewytrenowany
  (np. po 1 epoce w trybie <code>demo</code>, na kilkunastu obrazach), mapa
  ważności DINOv2 jest praktycznie jednorodnym szumem, profil wzdłuż osi jest
  płaski i detector zwraca pustą tablicę. <b>To zachowanie oczekiwane,
  nie błąd</b> — peaki, strefy i ponumerowane kropki pojawią się dopiero
  po pełnym treningu (<code>config.yaml</code>, 50 epok na model).
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

def build_comparison_report(
    results: dict,
    training_logs: dict,
    increment_cards: dict,
    dataset_stats: dict,
    output_path: Path,
    model_info: dict | None = None,
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

    if model_info is None:
        model_info = {}
    model_info.setdefault("Wygenerowano", datetime.now().strftime("%Y-%m-%d %H:%M"))

    body = (
        _section_a(dataset_stats)
        + _section_b(training_logs)
        + _section_c(results)
        + _section_d(results)
        + _section_e(increment_cards)
        + _section_f(model_info)
    )

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>OtolithDino — Raport porównawczy Embedded vs NotEmbedded</title>
<style>
body {{font-family:sans-serif;max-width:1400px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
table {{border-collapse:collapse;margin-bottom:1em;}}
td,th {{padding:4px 8px;}}
h2 {{color:#1a237e;}}
h3 {{color:#283593;}}
</style>
</head>
<body>
<h1>OtolithDino — Raport porównawczy Embedded vs NotEmbedded</h1>
{body}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
