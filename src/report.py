"""HTML report generator for OtolithDino training runs.

Generates a single self-contained HTML file with:
  - Section 1: Input data summary (distributions, split breakdown)
  - Section 2: Training curves (loss, MAE, backbone freeze point)
  - Section 3: Model evaluation (confusion matrix, error distribution, per-age MAE)
  - Section 4: Interpretation samples (heatmaps + overlays grid)
  - Section 5: Candidate increment markers (overlay grid + peak stats)

All plots are embedded as base64 PNG — no external dependencies, no internet required.
Missing inputs are gracefully skipped with a notice in the report.

Usage:
    from src.report import build_html_report
    html = build_html_report(labels_csv=Path("data/labels.csv"), ...)
    Path("outputs/report.html").write_text(html, encoding="utf-8")
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# Shared report primitives (single source of truth — see src/report_common.py).
from src.report_common import fig_to_b64 as _fig_to_b64
from src.report_common import img_tag as _img_tag
from src.report_common import pil_to_b64 as _pil_to_b64

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}
_PALETTE      = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336",
                 "#00BCD4", "#8BC34A", "#FF5722", "#607D8B", "#795548"]


# ---------------------------------------------------------------------------
# Utility: HTML building blocks
# ---------------------------------------------------------------------------

_SECTION_COUNTER = [0]


def _section(title: str, content: str) -> str:
    _SECTION_COUNTER[0] += 1
    idx = _SECTION_COUNTER[0]
    anchor = title.lower().replace(" ", "_").replace("/", "_")
    return (
        f'<section id="{anchor}">'
        f'<h2>{idx}. {title}</h2>'
        f'{content}'
        f'</section>\n'
    )


def _notice(msg: str) -> str:
    return f'<p class="notice">&#9888; {msg}</p>'


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    if fmt:
        df = df.copy()
        for col, f in fmt.items():
            if col in df.columns:
                df[col] = df[col].map(lambda v: f.format(v) if pd.notna(v) else "")
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


def _two_col(left: str, right: str) -> str:
    return f'<div class="two-col"><div>{left}</div><div>{right}</div></div>'


def _grid(items: List[str], cols: int = 3) -> str:
    cells = "".join(f'<div class="grid-cell">{item}</div>' for item in items)
    return f'<div class="img-grid" style="grid-template-columns:repeat({cols},1fr)">{cells}</div>'


# ---------------------------------------------------------------------------
# Section 1: Data summary
# ---------------------------------------------------------------------------

def build_data_section(labels_csv: Path) -> str:
    if not labels_csv.exists():
        return _section("Dane wejsciowe", _notice(f"Brak pliku: {labels_csv}"))

    df = pd.read_csv(labels_csv)
    parts: List[str] = []

    # ---- overview table ----
    n_total = len(df)
    n_fish  = df["fish_id"].nunique() if "fish_id" in df.columns else "n/d"
    avg_img = f"{n_total / n_fish:.1f}" if isinstance(n_fish, int) else "n/d"
    labeled = df[df["age"] >= 0] if "age" in df.columns else df
    age_min  = int(labeled["age"].min())
    age_max  = int(labeled["age"].max())
    age_mean = labeled["age"].mean()
    age_med  = labeled["age"].median()
    age_std  = labeled["age"].std()

    overview = pd.DataFrame({
        "Parametr": [
            "Wszystkich rekordow", "Unikalnych ryb (fish_id)", "Srednie zdjecia/ryba",
            "Wiek min", "Wiek max", "Wiek sredni", "Wiek mediana", "Wiek std",
        ],
        "Wartosc": [
            n_total, n_fish, avg_img,
            age_min, age_max, f"{age_mean:.2f}", f"{age_med:.1f}", f"{age_std:.2f}",
        ],
    })
    parts.append(_table(overview))

    # ---- split breakdown table ----
    if "split" in df.columns:
        split_rows = []
        for sp in ["train", "val", "test", "unlabeled"]:
            sub = df[df["split"] == sp]
            if len(sub) == 0:
                continue
            n_f = sub["fish_id"].nunique() if "fish_id" in df.columns else "n/d"
            ma  = f"{sub['age'].mean():.2f}" if sub["age"].ge(0).any() else "n/d"
            split_rows.append([sp, len(sub), n_f, f"{len(sub)/n_total*100:.1f}%", ma])
        split_df = pd.DataFrame(split_rows,
                                columns=["Split", "Zdjecia", "Ryby", "%", "Mean age"])
        parts.append("<h3>Podzial na zbiory</h3>" + _table(split_df))

    # ---- age distribution (overall) ----
    vc = labeled["age"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(vc.index, vc.values, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Wiek (lata)")
    ax.set_ylabel("Liczba zdjecic")
    ax.set_title("Rozklad wiekowy — caly zbior")
    ax.set_xticks(vc.index)
    for x, y in zip(vc.index, vc.values):
        ax.text(x, y + max(vc.values) * 0.01, str(y), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    parts.append("<h3>Rozklad wiekowy — caly zbior</h3>" + _img_tag(_fig_to_b64(fig)))

    # ---- age distribution per split ----
    if "split" in df.columns:
        splits_present = [s for s in ["train", "val", "test"] if s in df["split"].values]
        all_ages = sorted(labeled["age"].unique())
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(all_ages))
        width = 0.8 / max(len(splits_present), 1)
        for j, sp in enumerate(splits_present):
            sub = df[(df["split"] == sp) & (df["age"] >= 0)]
            counts = sub["age"].value_counts().reindex(all_ages, fill_value=0)
            ax.bar(x + j * width - (len(splits_present) - 1) * width / 2,
                   counts.values, width=width * 0.9,
                   label=sp, color=_SPLIT_COLORS.get(sp, _PALETTE[j]),
                   edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(all_ages)
        ax.set_xlabel("Wiek (lata)")
        ax.set_ylabel("Liczba zdjecic")
        ax.set_title("Rozklad wiekowy per split")
        ax.legend()
        fig.tight_layout()
        parts.append("<h3>Rozklad wiekowy per split</h3>" + _img_tag(_fig_to_b64(fig)))

    # ---- age distribution table ----
    age_table_rows = []
    for age in sorted(labeled["age"].unique()):
        row = [int(age)]
        tot = int(labeled[labeled["age"] == age].shape[0])
        row.append(tot)
        row.append(f"{tot/len(labeled)*100:.1f}%")
        if "split" in df.columns:
            for sp in ["train", "val", "test"]:
                n = int(df[(df["split"] == sp) & (df["age"] == age)].shape[0])
                row.append(n)
        age_table_rows.append(row)
    cols = ["Wiek", "Razem", "%"] + (["train", "val", "test"] if "split" in df.columns else [])
    age_df = pd.DataFrame(age_table_rows, columns=cols)
    parts.append("<h3>Tabela rozkladu wiekowego</h3>" + _table(age_df))

    # ---- sex distribution ----
    if "sex" in df.columns:
        sex_vc = df["sex"].fillna("Unknown").value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(sex_vc.index, sex_vc.values, color=_PALETTE[:len(sex_vc)], edgecolor="white")
        ax.set_xlabel("Plec")
        ax.set_ylabel("Liczba zdjecic")
        ax.set_title("Rozklad plci")
        for x, y in enumerate(sex_vc.values):
            ax.text(x, y + max(sex_vc.values) * 0.01, str(y), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        parts.append("<h3>Plec / rok polowu / subdywizja</h3>" + _img_tag(_fig_to_b64(fig), style="width:45%;vertical-align:top"))

    # ---- year distribution ----
    if "year" in df.columns:
        year_vc = df["year"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(year_vc.index.astype(str), year_vc.values, color="#607D8B", edgecolor="white")
        ax.set_xlabel("Rok")
        ax.set_ylabel("Liczba zdjecic")
        ax.set_title("Rozklad rekordow per rok polowu")
        for x, y in enumerate(year_vc.values):
            ax.text(x, y + max(year_vc.values) * 0.01, str(y), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        parts.append(_img_tag(_fig_to_b64(fig), style="width:45%;vertical-align:top;margin-left:2%"))

    # ---- subdivision distribution ----
    if "subdivision" in df.columns:
        sub_vc = df["subdivision"].fillna("Unknown").value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(sub_vc.index, sub_vc.values, color="#00BCD4", edgecolor="white")
        ax.set_xlabel("Liczba zdjecic")
        ax.set_title("Subdywizja (ICES)")
        fig.tight_layout()
        parts.append("<br>" + _img_tag(_fig_to_b64(fig), style="width:38%;vertical-align:top"))

    # ---- otolith type distribution ----
    if "otolith_type" in df.columns:
        ot_vc = df["otolith_type"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(ot_vc.index, ot_vc.values, color=["#2196F3", "#FF9800"], edgecolor="white")
        ax.set_title("Typ otolitu (Left / Right)")
        for x, y in enumerate(ot_vc.values):
            ax.text(x, y + max(ot_vc.values) * 0.01, str(y), ha="center", va="bottom")
        fig.tight_layout()
        parts.append(_img_tag(_fig_to_b64(fig), style="width:30%;vertical-align:top;margin-left:2%"))

    # ---- images per fish histogram ----
    if "fish_id" in df.columns:
        imgs_per_fish = df.groupby("fish_id").size()
        fig, ax = plt.subplots(figsize=(6, 3))
        max_count = int(imgs_per_fish.max())
        bins = np.arange(0.5, max_count + 1.5)
        ax.hist(imgs_per_fish.values, bins=bins, color="#9C27B0", edgecolor="white")
        ax.set_xlabel("Liczba zdjec per ryba")
        ax.set_ylabel("Liczba ryb")
        ax.set_title("Rozklad zdjec per ryba")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        fig.tight_layout()
        parts.append("<h3>Liczba zdjec per ryba</h3>" + _img_tag(_fig_to_b64(fig)))

    # ---- length and weight distributions ----
    if "length_mm" in df.columns and "weight_g" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        splits_present = [s for s in ["train", "val", "test"] if "split" in df.columns and s in df["split"].values]
        for j, (col, unit, ax) in enumerate([("length_mm", "mm", axes[0]), ("weight_g", "g", axes[1])]):
            data = df[df[col].notna()]
            if splits_present:
                for sp in splits_present:
                    sub = data[data["split"] == sp][col]
                    ax.hist(sub, bins=30, alpha=0.6, label=sp,
                            color=_SPLIT_COLORS.get(sp, _PALETTE[j]), edgecolor="none")
                ax.legend()
            else:
                ax.hist(data[col], bins=30, color=_PALETTE[j], edgecolor="white")
            ax.set_xlabel(f"{col} [{unit}]")
            ax.set_ylabel("Liczba zdjecic")
            ax.set_title(f"Rozklad: {col}")
        fig.tight_layout()
        parts.append("<h3>Dlugoss i masa ryb</h3>" + _img_tag(_fig_to_b64(fig)))

        # ---- scatter: length vs weight coloured by age ----
        valid = df[df["length_mm"].notna() & df["weight_g"].notna() & (df["age"] >= 0)]
        if len(valid) > 0:
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(valid["length_mm"], valid["weight_g"],
                            c=valid["age"], cmap="viridis", alpha=0.5, s=10)
            plt.colorbar(sc, ax=ax, label="Wiek")
            ax.set_xlabel("Dlugosc [mm]")
            ax.set_ylabel("Masa [g]")
            ax.set_title("Dlugosc vs masa (kolor = wiek)")
            fig.tight_layout()
            parts.append("<h3>Dlugosc vs masa</h3>" + _img_tag(_fig_to_b64(fig)))

        # ---- scatter: age vs length with trend ----
        valid2 = df[df["length_mm"].notna() & (df["age"] >= 0)]
        if len(valid2) > 0:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(valid2["age"], valid2["length_mm"], alpha=0.3, s=8, color="#2196F3")
            grp = valid2.groupby("age")["length_mm"]
            ax.errorbar(grp.mean().index, grp.mean().values, yerr=grp.std().values,
                        fmt="o-", color="#F44336", linewidth=1.5, markersize=5,
                        capsize=3, label="Srednia +/- std")
            ax.set_xlabel("Wiek (lata)")
            ax.set_ylabel("Dlugosc [mm]")
            ax.set_title("Wiek vs dlugosc (srednia +/- std)")
            ax.legend()
            fig.tight_layout()
            parts.append("<h3>Wiek vs dlugosc</h3>" + _img_tag(_fig_to_b64(fig)))

    # ---- correlation heatmap ----
    corr_cols = [c for c in ["age", "length_mm", "weight_g"] if c in df.columns]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].dropna().astype(float).corr()
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols)
        ax.set_yticklabels(corr_cols)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)
        ax.set_title("Korelacja cech")
        fig.tight_layout()
        parts.append("<h3>Korelacja age / length / weight</h3>" + _img_tag(_fig_to_b64(fig)))

    return _section("Dane wejsciowe", "\n".join(parts))


# ---------------------------------------------------------------------------
# Section 2: Training curves
# ---------------------------------------------------------------------------

def build_training_section(log_path: Path) -> str:
    if not log_path.exists():
        return _section("Krzywe treningowe", _notice(f"Brak logu: {log_path}"))

    lines = log_path.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(
        r"epoch=\s*(\d+)\s+train_loss=([\d.nan]+)\s+val_loss=([\d.nan]+)\s+val_mae=([\d.nan]+)"
    )
    freeze_pattern = re.compile(r"Backbone frozen for first (\d+) epochs")
    unfreeze_epoch = None
    rows = []
    for line in lines:
        m = pattern.search(line)
        if m:
            rows.append({
                "epoch":      int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss":   float(m.group(3)),
                "val_mae":    float(m.group(4)),
            })
        fm = freeze_pattern.search(line)
        if fm:
            unfreeze_epoch = int(fm.group(1)) + 1

    if not rows:
        return _section("Krzywe treningowe", _notice("Brak danych treningowych w logu."))

    log_df = pd.DataFrame(rows)
    best_epoch = int(log_df.loc[log_df["val_loss"].idxmin(), "epoch"])
    best_val_loss = float(log_df["val_loss"].min())
    best_val_mae  = float(log_df.loc[log_df["val_loss"].idxmin(), "val_mae"])

    parts: List[str] = []

    # Summary stats
    summary = pd.DataFrame({
        "Parametr": ["Epoki razem", "Najlepsza epoka (min val_loss)",
                     "Najlepsza val_loss", "Val MAE przy najlepszym checkpoincie",
                     "Ostatnia val_MAE"],
        "Wartosc": [len(log_df), best_epoch, f"{best_val_loss:.4f}",
                    f"{best_val_mae:.3f}", f"{float(log_df['val_mae'].iloc[-1]):.3f}"],
    })
    parts.append(_table(summary))

    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train_loss",
            color="#2196F3", linewidth=1.5)
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val_loss",
            color="#FF9800", linewidth=1.5)
    if unfreeze_epoch:
        ax.axvline(unfreeze_epoch, color="#9E9E9E", linestyle="--", linewidth=1,
                   label=f"unfreeze backbone (ep {unfreeze_epoch})")
    ax.scatter([best_epoch], [best_val_loss], color="#F44336", zorder=5, s=80,
               marker="*", label=f"best ckpt (ep {best_epoch})")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("Loss")
    ax.set_title("Train / Val Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(log_df["epoch"], log_df["val_mae"], color="#4CAF50", linewidth=1.5)
    if unfreeze_epoch:
        ax.axvline(unfreeze_epoch, color="#9E9E9E", linestyle="--", linewidth=1,
                   label=f"unfreeze (ep {unfreeze_epoch})")
    ax.scatter([best_epoch], [best_val_mae], color="#F44336", zorder=5, s=80,
               marker="*", label=f"best ckpt MAE={best_val_mae:.3f}")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("MAE (lata)")
    ax.set_title("Val MAE per epoka")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    parts.append(_img_tag(_fig_to_b64(fig)))

    # Full epoch table (last 30 epochs visible, all in table)
    parts.append("<h3>Historia epok</h3>")
    display_df = log_df.copy()
    display_df["train_loss"] = display_df["train_loss"].map("{:.4f}".format)
    display_df["val_loss"]   = display_df["val_loss"].map("{:.4f}".format)
    display_df["val_mae"]    = display_df["val_mae"].map("{:.3f}".format)
    parts.append(
        '<div style="max-height:300px;overflow-y:auto">'
        + _table(display_df)
        + "</div>"
    )

    return _section("Krzywe treningowe", "\n".join(parts))


# ---------------------------------------------------------------------------
# Section 3: Model evaluation
# ---------------------------------------------------------------------------

def build_eval_section(predictions_csv: Path, labels_csv: Optional[Path] = None) -> str:
    if not predictions_csv.exists():
        return _section("Ewaluacja modelu", _notice(f"Brak pliku: {predictions_csv}"))

    preds = pd.read_csv(predictions_csv)
    preds = preds[preds["target_age"].notna()].copy()
    preds["target_age"]    = preds["target_age"].astype(int)
    preds["predicted_age"] = preds["predicted_age"].astype(int)
    preds["abs_error"]     = preds["abs_error"].astype(float)

    if len(preds) == 0:
        return _section("Ewaluacja modelu", _notice("Brak labellowanych próbek w predictions.csv"))

    # Optionally join split info
    if labels_csv and labels_csv.exists():
        lab = pd.read_csv(labels_csv)[["image_id", "split"]]
        preds = preds.merge(lab, on="image_id", how="left")

    mean_mae   = preds["abs_error"].mean()
    median_mae = preds["abs_error"].median()
    std_mae    = preds["abs_error"].std()
    within_1   = (preds["abs_error"] <= 1).mean() * 100
    within_2   = (preds["abs_error"] <= 2).mean() * 100
    within_3   = (preds["abs_error"] <= 3).mean() * 100

    parts: List[str] = []

    # Summary table
    summary = pd.DataFrame({
        "Metryka": ["N probek", "Mean MAE", "Median MAE", "Std MAE",
                    "blad <= 1 rok (%)", "blad <= 2 lata (%)", "blad <= 3 lata (%)"],
        "Wartosc": [len(preds), f"{mean_mae:.3f}", f"{median_mae:.3f}", f"{std_mae:.3f}",
                    f"{within_1:.1f}%", f"{within_2:.1f}%", f"{within_3:.1f}%"],
    })
    parts.append(_table(summary))

    # Per-split MAE if available
    if "split" in preds.columns:
        split_rows = []
        for sp in preds["split"].dropna().unique():
            sub = preds[preds["split"] == sp]
            split_rows.append([sp, len(sub),
                                f"{sub['abs_error'].mean():.3f}",
                                f"{sub['abs_error'].median():.3f}"])
        if split_rows:
            split_df = pd.DataFrame(split_rows,
                                    columns=["Split", "N", "Mean MAE", "Median MAE"])
            parts.append("<h3>MAE per split</h3>" + _table(split_df))

    all_ages = sorted(set(preds["target_age"].unique()) | set(preds["predicted_age"].unique()))
    fig_rows = []

    # ---- confusion matrix ----
    cm = np.zeros((len(all_ages), len(all_ages)), dtype=int)
    age_idx = {a: i for i, a in enumerate(all_ages)}
    for _, row in preds.iterrows():
        t = age_idx.get(row["target_age"])
        p = age_idx.get(row["predicted_age"])
        if t is not None and p is not None:
            cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(max(6, len(all_ages) * 0.55), max(5, len(all_ages) * 0.5)))
    norm_cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_ages)))
    ax.set_yticks(range(len(all_ages)))
    ax.set_xticklabels(all_ages, fontsize=7)
    ax.set_yticklabels(all_ages, fontsize=7)
    ax.set_xlabel("Przewidywany wiek")
    ax.set_ylabel("Rzeczywisty wiek")
    ax.set_title("Macierz pomylek (wiersze znormalizowane)")
    for i in range(len(all_ages)):
        for j in range(len(all_ages)):
            if cm[i, j] > 0:
                txt = str(cm[i, j])
                col = "white" if norm_cm[i, j] > 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=col)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    parts.append("<h3>Macierz pomylek</h3>" + _img_tag(_fig_to_b64(fig)))

    # ---- scatter: actual vs predicted ----
    fig, ax = plt.subplots(figsize=(6, 6))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.3, 0.3, len(preds))
    ax.scatter(preds["target_age"] + jitter, preds["predicted_age"] + jitter,
               alpha=0.3, s=8, color="#2196F3")
    lo, hi = min(all_ages) - 0.5, max(all_ages) + 0.5
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="ideal y=x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Rzeczywisty wiek")
    ax.set_ylabel("Przewidywany wiek")
    ax.set_title("Rzeczywisty vs przewidywany wiek (z jitter)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    parts.append("<h3>Scatter: rzeczywisty vs przewidywany</h3>" + _img_tag(_fig_to_b64(fig)))

    # ---- MAE per age class + boxplot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    age_mae = preds.groupby("target_age")["abs_error"].mean()
    ax.bar(age_mae.index, age_mae.values, color="#FF9800", edgecolor="white")
    ax.axhline(mean_mae, color="#F44336", linestyle="--", linewidth=1, label=f"overall MAE={mean_mae:.2f}")
    ax.set_xlabel("Rzeczywisty wiek")
    ax.set_ylabel("Mean MAE")
    ax.set_title("MAE per klasa wiekowa")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    age_groups = [preds[preds["target_age"] == a]["abs_error"].values for a in all_ages]
    ax.boxplot(age_groups, positions=all_ages, widths=0.6, patch_artist=True,
               boxprops=dict(facecolor="#B3E5FC"), medianprops=dict(color="#0277BD", linewidth=2))
    ax.set_xlabel("Rzeczywisty wiek")
    ax.set_ylabel("Abs error (lata)")
    ax.set_title("Boxplot bledow per wiek")
    ax.set_xticks(all_ages)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    parts.append("<h3>MAE per wiek / Boxplot</h3>" + _img_tag(_fig_to_b64(fig)))

    # ---- error histogram + CDF ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    max_err = int(preds["abs_error"].max()) + 1
    ax.hist(preds["abs_error"], bins=np.arange(-0.5, max_err + 0.5),
            color="#4CAF50", edgecolor="white")
    ax.axvline(mean_mae, color="#F44336", linestyle="--", linewidth=1.5,
               label=f"mean={mean_mae:.2f}")
    ax.axvline(median_mae, color="#FF9800", linestyle="--", linewidth=1.5,
               label=f"median={median_mae:.2f}")
    ax.set_xlabel("Bezwzgledny blad (lata)")
    ax.set_ylabel("Liczba probek")
    ax.set_title("Rozklad bledow predykcji")
    ax.legend(fontsize=8)

    ax = axes[1]
    err_sorted = np.sort(preds["abs_error"].values)
    cdf = np.arange(1, len(err_sorted) + 1) / len(err_sorted) * 100
    ax.plot(err_sorted, cdf, color="#9C27B0", linewidth=2)
    ax.axhline(50, color="#9E9E9E", linestyle=":", linewidth=1)
    for thr in [1, 2, 3]:
        pct = (preds["abs_error"] <= thr).mean() * 100
        ax.scatter([thr], [pct], zorder=5, s=60)
        ax.text(thr + 0.1, pct - 3, f"{pct:.0f}%", fontsize=8)
    ax.set_xlabel("Prog bledu (lata)")
    ax.set_ylabel("% probek <= progu")
    ax.set_title("Dystrybuanta skumulowana bledu (CDF)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    parts.append("<h3>Rozklad bledu / CDF</h3>" + _img_tag(_fig_to_b64(fig)))

    return _section("Ewaluacja modelu", "\n".join(parts))


# ---------------------------------------------------------------------------
# Section 4: Interpretation samples
# ---------------------------------------------------------------------------

def build_interpretation_section(
    heatmaps_dir: Optional[Path],
    overlays_dir: Optional[Path],
    n: int = 12,
) -> str:
    if not heatmaps_dir or not overlays_dir:
        return _section("Interpretacja (heatmapy)", _notice("Katalogi heatmap / overlay nie podane."))
    if not heatmaps_dir.exists() or not overlays_dir.exists():
        return _section("Interpretacja (heatmapy)",
                        _notice(f"Katalog nie istnieje: {heatmaps_dir} lub {overlays_dir}"))

    hm_files  = sorted(heatmaps_dir.glob("*_heatmap.png"))
    ov_files  = sorted(overlays_dir.glob("*_overlay.png"))

    if not hm_files:
        return _section("Interpretacja (heatmapy)", _notice("Brak plikow heatmap."))

    rng = np.random.default_rng(0)
    sel_idx = rng.choice(len(hm_files), size=min(n, len(hm_files)), replace=False)

    cells = []
    for i in sorted(sel_idx):
        hm = hm_files[i]
        stem = hm.stem.replace("_heatmap", "")
        ov_match = overlays_dir / f"{stem}_overlay.png"
        b64_hm = _pil_to_b64(hm, max_px=300)
        cell_html = f'<div style="font-size:10px;text-align:center;word-break:break-all">{stem}</div>'
        if ov_match.exists():
            b64_ov = _pil_to_b64(ov_match, max_px=300)
            cell_html += (
                f'<div style="display:flex;gap:4px">'
                f'{_img_tag(b64_ov, alt="overlay", style="width:49%")}'
                f'{_img_tag(b64_hm, alt="heatmap", style="width:49%")}'
                f"</div>"
            )
        else:
            cell_html += _img_tag(b64_hm, alt="heatmap")
        cells.append(cell_html)

    content = (
        f"<p>Pokazano {len(cells)} z {len(hm_files)} próbek (losowo). "
        f"Lewo: overlay (heatmap nałożona na zdjęcie). Prawo: surowa heatmap.</p>"
        + _grid(cells, cols=min(3, len(cells)))
    )
    return _section("Interpretacja (heatmapy)", content)


# ---------------------------------------------------------------------------
# Section 5: Candidate increment markers
# ---------------------------------------------------------------------------

def build_candidates_section(
    cand_overlays_dir: Optional[Path],
    cand_json_dir: Optional[Path],
    n: int = 12,
) -> str:
    if not cand_overlays_dir or not cand_json_dir:
        return _section("Kandydaci przyrostowi", _notice("Katalogi kandydatow nie podane."))
    if not cand_overlays_dir.exists():
        return _section("Kandydaci przyrostowi",
                        _notice(f"Katalog nie istnieje: {cand_overlays_dir}"))

    ov_files   = sorted(cand_overlays_dir.glob("*_candidates_overlay.png"))
    json_files = sorted(cand_json_dir.glob("*_candidates.json")) if cand_json_dir.exists() else []

    if not ov_files:
        return _section("Kandydaci przyrostowi", _notice("Brak plikow overlay kandydatow."))

    parts: List[str] = []

    # Peak count statistics from JSON files
    if json_files:
        peak_counts = []
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                peak_counts.append(data.get("num_candidates", 0))
            except Exception:
                pass
        if peak_counts:
            fig, ax = plt.subplots(figsize=(6, 3))
            max_p = max(peak_counts)
            ax.hist(peak_counts, bins=np.arange(-0.5, max_p + 1.5),
                    color="#F44336", edgecolor="white")
            ax.set_xlabel("Liczba kandydatow per zdjecie")
            ax.set_ylabel("Liczba zdjecic")
            ax.set_title(f"Rozklad liczby kandydatow (N={len(peak_counts)}, "
                         f"srednia={np.mean(peak_counts):.1f})")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            fig.tight_layout()
            parts.append(_img_tag(_fig_to_b64(fig)))

    # Image grid
    rng = np.random.default_rng(0)
    sel_idx = rng.choice(len(ov_files), size=min(n, len(ov_files)), replace=False)

    # Build a map from stem to json data
    json_map = {}
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            key = jf.stem.replace("_candidates", "")
            json_map[key] = data
        except Exception:
            pass

    cells = []
    for i in sorted(sel_idx):
        ov = ov_files[i]
        stem = ov.stem.replace("_candidates_overlay", "")
        b64 = _pil_to_b64(ov, max_px=300)
        n_cand = json_map.get(stem, {}).get("num_candidates", "?")
        cell_html = (
            f'<div style="font-size:10px;text-align:center;word-break:break-all">'
            f'{stem}<br>kandydaci: {n_cand}</div>'
            + _img_tag(b64)
        )
        cells.append(cell_html)

    parts.append(
        f"<p>Pokazano {len(cells)} z {len(ov_files)} próbek (losowo). "
        f"Czerwone linie = kandydaci przyrostowi.</p>"
        + _grid(cells, cols=min(3, len(cells)))
    )

    return _section("Kandydaci przyrostowi", "\n".join(parts))


# ---------------------------------------------------------------------------
# CSS + HTML template
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; color: #212121; }
header { background: #1565C0; color: white; padding: 20px 40px; }
header h1 { font-size: 1.6rem; }
header p  { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }
nav { background: #1976D2; padding: 0 40px; display: flex; flex-wrap: wrap; gap: 4px; }
nav a { color: white; text-decoration: none; padding: 8px 14px; font-size: 0.85rem;
        border-radius: 2px 2px 0 0; }
nav a:hover { background: rgba(255,255,255,0.15); }
main { max-width: 1300px; margin: 0 auto; padding: 20px 24px; }
section { background: white; border-radius: 4px; padding: 24px 28px;
          margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
h2 { font-size: 1.3rem; color: #1565C0; border-bottom: 2px solid #1565C0;
     padding-bottom: 8px; margin-bottom: 16px; }
h3 { font-size: 1rem; color: #424242; margin: 20px 0 8px; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
th { background: #1976D2; color: white; padding: 7px 10px; text-align: left; }
td { padding: 6px 10px; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) td { background: #f5f7ff; }
.notice { background: #FFF3E0; border-left: 4px solid #FF9800; padding: 10px 14px;
          border-radius: 2px; font-size: 0.9rem; color: #E65100; margin: 10px 0; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.img-grid { display: grid; gap: 12px; margin-top: 12px; }
.grid-cell { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px;
             padding: 8px; font-size: 0.8rem; }
img { border-radius: 3px; }
footer { text-align: center; padding: 20px; font-size: 0.8rem; color: #757575; }
"""


def build_html_report(
    labels_csv: Optional[Path] = None,
    log_path: Optional[Path] = None,
    predictions_csv: Optional[Path] = None,
    heatmaps_dir: Optional[Path] = None,
    overlays_dir: Optional[Path] = None,
    cand_json_dir: Optional[Path] = None,
    cand_overlays_dir: Optional[Path] = None,
    n_image_samples: int = 12,
) -> str:
    """Build and return a complete self-contained HTML report string.

    Missing inputs are gracefully skipped — each section notes what was unavailable.
    """
    _SECTION_COUNTER[0] = 0  # reset counter

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        build_data_section(labels_csv) if labels_csv else
            _section("Dane wejsciowe", _notice("--labels nie podano")),
        build_training_section(log_path) if log_path else
            _section("Krzywe treningowe", _notice("--log nie podano")),
        build_eval_section(predictions_csv, labels_csv) if predictions_csv else
            _section("Ewaluacja modelu", _notice("--preds nie podano")),
        build_interpretation_section(heatmaps_dir, overlays_dir, n_image_samples),
        build_candidates_section(cand_overlays_dir, cand_json_dir, n_image_samples),
    ]

    nav_anchors = [
        ("dane_wejsciowe",          "Dane wejsciowe"),
        ("krzywe_treningowe",       "Krzywe treningowe"),
        ("ewaluacja_modelu",        "Ewaluacja"),
        ("interpretacja_(heatmapy)","Interpretacja"),
        ("kandydaci_przyrostowi",   "Kandydaci"),
    ]
    nav_html = "".join(f'<a href="#{a}">{label}</a>' for a, label in nav_anchors)

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OtolithDino — Raport treningu</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>OtolithDino — Raport treningu</h1>
  <p>Wygenerowano: {generated_at}</p>
</header>
<nav>{nav_html}</nav>
<main>
{"".join(sections)}
</main>
<footer>OtolithDino &bull; DINOv2-based otolith age prediction &bull; {generated_at}</footer>
</body>
</html>"""

    return html


def save_report(html: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"Raport zapisany: {path}  ({len(html) // 1024} KB)")
