"""28.07: side-by-side local comparison report — `22.07_reg` (518px) vs `23.07_hires966`
(966px), same otoliths, same panels, laid out next to each other so a human can visually
judge whether the higher-resolution run produces qualitatively different (cleaner/more
ring-like, or not) density maps and candidate placement — beyond the aggregate
pixel-distance metrics already measured (see outputs/DINO_proces.md §4.24).

Does NOT retrain or modify any src/ code. Reuses:
  - scripts.run_pipeline.load_merged_config / _compute_axis_data_for_samples / _parse_train_log
  - src.visualization.draw_reasoning_card / load_original_image (same wiring as
    save_reasoning_cards, just kept in memory and keyed explicitly by image_id instead of
    written to disk, so pairing between the two runs can't drift if a sample is skipped)
  - src.report_common helpers (fig_to_b64, img_tag, compute_metrics)
  - the already-computed 30-card localization_quality.json for both runs
    (outputs/23.07_reg_30cards, outputs/23.07_hires966_30cards) — no recomputation.

_compute_axis_data_for_samples() accepts an arbitrary caller-built `samples` list (bypassing
the ranking-only top-k best/worst selection used by the production pipeline), which is what
lets us force the SAME 12 otoliths through both checkpoints even though each model's own
best/worst-by-error selection would pick different fish.

Usage: python scripts/diagnostics/compare_22reg_vs_hires966.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from src.comparison_report import _style_ax, _INK, _MUTED, _GRID
from src.otolith_axis import apply_background_mask
from src.report_common import compute_metrics, fig_to_b64, img_tag
from src.utils import configure_attention
from src.visualization import draw_reasoning_card, load_original_image
from scripts.run_pipeline import (_compute_axis_data_for_samples, _parse_train_log,
                                  load_merged_config)

IMAGE_DIR = Path("Z:/Photo/Otolithes/HER/Processed")

RUN_A_DIR = PROJECT_ROOT / "outputs" / "22.07_reg"
RUN_B_DIR = PROJECT_ROOT / "outputs" / "23.07_hires966"
CKPT_A = RUN_A_DIR / "checkpoints" / "embedded" / "best.pt"
CKPT_B = RUN_B_DIR / "checkpoints" / "embedded" / "best.pt"
LABEL_A, LABEL_B = "22.07_reg (518px)", "23.07_hires966 (966px)"
COLOR_A, COLOR_B = "#2a78d6", "#eb6834"

OUT_DIR = PROJECT_ROOT / "outputs" / "28.07_compare_22reg_vs_hires966"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "comparison_report.html"

# Representative age-stratified sample: dense middle of the distribution (0-9, where most
# data lives) plus the sparse old-fish tail (10, 15) — the documented, recurring weak spot.
# One otolith per age, chosen deterministically (first image_id alphabetically) so re-running
# this script reproduces the exact same report.
TARGET_AGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
METHODS = ["density", "classical", "consensus", "dp"]


def pick_shared_image_ids(pred_csv_a: Path) -> list[str]:
    df = pd.read_csv(pred_csv_a)
    chosen = []
    for age in TARGET_AGES:
        pool = df[df["target_age"] == age].sort_values("image_id")
        if len(pool):
            chosen.append(str(pool.iloc[0]["image_id"]))
        else:
            print(f"  [uwaga] brak obrazu wieku {age} w zbiorze testowym — pomijam", flush=True)
    return chosen


def samples_for(pred_df: pd.DataFrame, shared_ids: list[str]) -> list[dict]:
    sub = pred_df[pred_df["image_id"].isin(shared_ids)]
    out = []
    for iid in shared_ids:
        rows = sub[sub["image_id"] == iid]
        if rows.empty:
            continue
        row = rows.iloc[0]
        out.append({"image_id": iid, "age": int(row["target_age"]),
                   "predicted_age": int(row["predicted_age"])})
    return out


def card_b64(image_id: str, sample: dict, grids: dict, axis_data: dict,
            mask_background: bool, inner_margin: float, target_w: int = 900) -> str | None:
    """Mirrors src.visualization.save_reasoning_cards' wiring exactly, but keeps the
    result in memory (base64), keyed explicitly by image_id — no risk of the two runs'
    card lists drifting out of alignment if a sample is silently skipped in one run."""
    grid = grids.get(image_id)
    if grid is None:
        return None
    try:
        original = load_original_image(image_id, IMAGE_DIR)
    except FileNotFoundError:
        return None
    axis = axis_data.get(image_id, {})
    card_mask = axis.get("mask")
    if mask_background and card_mask is not None:
        original = apply_background_mask(original, card_mask)
    card = draw_reasoning_card(
        original_rgb=original, importance_grid=grid,
        predicted_age=int(sample["predicted_age"]), true_age=int(sample["age"]),
        mask=axis.get("mask"), axis_info=axis.get("axis_info"),
        peak_indices=axis.get("peak_indices"), line_xy=axis.get("line_xy"),
        profile_1d=axis.get("profile_1d"), final_axis_pts=axis.get("final_axis_pts"),
        candidate_pts=axis.get("candidate_pts"), final_t=axis.get("final_t"),
        coral_gradcam=axis.get("coral_gradcam"), cls_attention=axis.get("cls_attention"),
        cls_is_fallback=axis.get("cls_is_fallback", False), ring_curves=axis.get("ring_curves"),
        classical_pts=axis.get("classical_pts"), image_name=str(image_id),
        inner_margin=inner_margin,
    )
    img = PILImage.fromarray(card, mode="RGB")
    if img.width > target_w:
        scale = target_w / img.width
        img = img.resize((target_w, max(1, int(round(img.height * scale)))), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    cfg_a = load_merged_config(PROJECT_ROOT / "configs" / "config.yaml",
                               PROJECT_ROOT / "configs" / "config_embedded.yaml")
    cfg_b = load_merged_config(PROJECT_ROOT / "configs" / "config_hires966.yaml",
                               PROJECT_ROOT / "configs" / "config_embedded.yaml")
    cfg_a.data.image_dir = str(IMAGE_DIR)
    cfg_b.data.image_dir = str(IMAGE_DIR)
    configure_attention(cfg_a.interpretation.disable_fused_attention)

    pred_a = pd.read_csv(RUN_A_DIR / "emb_on_emb" / "predictions.csv")
    pred_b = pd.read_csv(RUN_B_DIR / "emb_on_emb" / "predictions.csv")

    shared_ids = pick_shared_image_ids(RUN_A_DIR / "emb_on_emb" / "predictions.csv")
    print(f"Wspólne otolity ({len(shared_ids)}): {shared_ids}", flush=True)

    samples_a = samples_for(pred_a, shared_ids)
    samples_b = samples_for(pred_b, shared_ids)

    print("Liczenie run A (22.07_reg, 518px)...", flush=True)
    grids_a, axis_data_a, _ = _compute_axis_data_for_samples(
        samples_a, IMAGE_DIR, cfg_a, CKPT_A, OUT_DIR / "run_a")
    print("Liczenie run B (23.07_hires966, 966px)...", flush=True)
    grids_b, axis_data_b, _ = _compute_axis_data_for_samples(
        samples_b, IMAGE_DIR, cfg_b, CKPT_B, OUT_DIR / "run_b")

    by_id_a = {s["image_id"]: s for s in samples_a}
    by_id_b = {s["image_id"]: s for s in samples_b}

    # --- 1. wiek: tabelka metryk ---
    m_a = compute_metrics(pred_a["target_age"], pred_a["predicted_age"])
    m_b = compute_metrics(pred_b["target_age"], pred_b["predicted_age"])
    metrics_rows = "".join(
        f"<tr><td>{k}</td><td>{m_a[k]:.4f}</td><td>{m_b[k]:.4f}</td></tr>"
        for k in ("MAE", "RMSE", "Bias", "Acc1yr", "Acc2yr")
    )

    # --- 2. lokalizacja: wykres słupkowy z gotowych 30-kartowych localization_quality.json ---
    q_a = json.loads((PROJECT_ROOT / "outputs" / "23.07_reg_30cards"
                      / "localization_quality.json").read_text(encoding="utf-8"))["emb_on_emb"]
    q_b = json.loads((PROJECT_ROOT / "outputs" / "23.07_hires966_30cards"
                      / "localization_quality.json").read_text(encoding="utf-8"))["emb_on_emb"]
    vals_a = [q_a["per_method"][m]["mean_dist_final_to_classical_px"] for m in METHODS]
    vals_b = [q_b["per_method"][m]["mean_dist_final_to_classical_px"] for m in METHODS]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(METHODS))
    w = 0.35
    bars_a = ax.bar(x - w / 2, vals_a, width=w, color=COLOR_A, label=LABEL_A)
    bars_b = ax.bar(x + w / 2, vals_b, width=w, color=COLOR_B, label=LABEL_B)
    for bars in (bars_a, bars_b):
        for rect in bars:
            hgt = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, hgt + 2, f"{hgt:.0f}", ha="center",
                   va="bottom", fontsize=8, color=_INK)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("mean_dist_final_to_classical_px (niżej = lepiej)")
    ax.set_title("Lokalizacja przyrostów — 30 kart (własna kuratorska próba KAŻDEGO biegu)")
    ax.legend(fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    loc_chart_b64 = fig_to_b64(fig)

    norm_a = [vals_a[i] / vals_a[1] for i in range(len(METHODS))]   # /classical
    norm_b = [vals_b[i] / vals_b[1] for i in range(len(METHODS))]
    norm_rows = "".join(
        f"<tr><td>{m}</td><td>{norm_a[i]:.3f}</td><td>{norm_b[i]:.3f}</td>"
        f"<td>{(norm_b[i] / norm_a[i] - 1) * 100:+.1f}%</td></tr>"
        for i, m in enumerate(METHODS)
    )

    # --- 3. krzywe treningowe (val_mae per epoka, oba biegi) ---
    log_a = _parse_train_log(RUN_A_DIR / "logs" / "embedded" / "train.log")
    log_b = _parse_train_log(RUN_B_DIR / "logs" / "embedded" / "train.log")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot([r["epoch"] for r in log_a], [r["val_mae"] for r in log_a],
             color=COLOR_A, marker="o", markersize=3, label=LABEL_A)
    ax2.plot([r["epoch"] for r in log_b], [r["val_mae"] for r in log_b],
             color=COLOR_B, marker="o", markersize=3, label=LABEL_B)

    def _gate_epoch(log):
        for r in log:
            if r.get("density_active", 0) >= 1.0:
                return r["epoch"]
        return None

    ge_a, ge_b = _gate_epoch(log_a), _gate_epoch(log_b)
    if ge_a:
        ax2.axvline(ge_a, color=COLOR_A, linestyle="--", linewidth=1, alpha=0.6)
    if ge_b:
        ax2.axvline(ge_b, color=COLOR_B, linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_xlabel("epoka")
    ax2.set_ylabel("val_mae")
    ax2.set_title("Krzywe treningowe — val_mae per epoka (przerywana linia = otwarcie bramki density)")
    ax2.legend(fontsize=8)
    _style_ax(ax2)
    fig2.tight_layout()
    train_chart_b64 = fig_to_b64(fig2)

    # --- 4. per-otolith side-by-side ---
    otolith_html = ""
    for iid in shared_ids:
        sa, sb = by_id_a.get(iid), by_id_b.get(iid)
        if sa is None or sb is None:
            continue
        true_age = sa["age"]
        pa, pb = sa["predicted_age"], sb["predicted_age"]
        verdict_a = "trafne" if pa == true_age else f"pudło o {abs(pa - true_age)}"
        verdict_b = "trafne" if pb == true_age else f"pudło o {abs(pb - true_age)}"

        ca = card_b64(iid, sa, grids_a, axis_data_a,
                     cfg_a.data.mask_background, cfg_a.candidates.inner_margin)
        cb = card_b64(iid, sb, grids_b, axis_data_b,
                     cfg_b.data.mask_background, cfg_b.candidates.inner_margin)
        card_row = ""
        if ca and cb:
            card_row = f"""
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin:8px 0;">
              <div><p style="color:{COLOR_A};font-weight:bold;margin:2px 0;">{LABEL_A}</p>{img_tag(ca, style="width:640px;")}</div>
              <div><p style="color:{COLOR_B};font-weight:bold;margin:2px 0;">{LABEL_B}</p>{img_tag(cb, style="width:640px;")}</div>
            </div>"""
        else:
            card_row = "<p class='cap'>(karta rozumowania niedostępna — segmentacja/obraz nie powiodły się)</p>"

        ov_a = (axis_data_a.get(iid) or {}).get("localization_overlays") or {}
        ov_b = (axis_data_b.get(iid) or {}).get("localization_overlays") or {}
        method_blocks = ""
        for m in METHODS:
            oa, ob = ov_a.get(m), ov_b.get(m)
            if not oa or not ob:
                continue
            method_blocks += f"""
            <div style="margin:4px 8px 4px 0;">
              <p class="cap" style="margin:0 0 2px;font-weight:bold;">{m} (n={oa['n_final']} | {ob['n_final']})</p>
              <div style="display:flex;gap:4px;">
                <img src="{oa['b64']}" style="width:220px;max-width:100%;">
                <img src="{ob['b64']}" style="width:220px;max-width:100%;">
              </div>
            </div>"""

        otolith_html += f"""
        <div style="border-top:1px solid {_GRID};padding:16px 0;">
          <h3 style="margin:0 0 4px;">{iid}</h3>
          <p class="cap">prawdziwy wiek: <b>{true_age}</b> &nbsp;|&nbsp;
             <span style="color:{COLOR_A};">{LABEL_A}: {pa} ({verdict_a})</span> &nbsp;|&nbsp;
             <span style="color:{COLOR_B};">{LABEL_B}: {pb} ({verdict_b})</span></p>
          {card_row}
          <p class="cap" style="margin:10px 0 2px;">Nakładki lokalizacji per metoda (lewa: {LABEL_A}, prawa: {LABEL_B}):</p>
          <div style="display:flex;flex-wrap:wrap;">{method_blocks}</div>
        </div>"""

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>28.07 — 22.07_reg (518px) vs 23.07_hires966 (966px): te same otolity, obok siebie</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width:1400px; margin:24px auto; color:{_INK}; }}
h1,h2,h3 {{ color:{_INK}; }}
table {{ border-collapse:collapse; margin:12px 0; }}
td, th {{ border:1px solid {_GRID}; padding:6px 10px; font-size:90%; }}
th {{ background:#f4f3f0; }}
.cap {{ color:{_MUTED}; font-size:85%; }}
</style></head>
<body>
<h1>22.07_reg (518px) vs 23.07_hires966 (966px) — te same otolity, obok siebie</h1>
<p class="cap">Oba biegi: ten sam zbiór testowy (1131 obrazów, identyczny split), ten sam backbone
(<code>dinov2_vits14_reg</code>), ta sama receptura poza rozdzielczością wejścia (518px vs 966px, więc
też <code>candidates.density_image_size</code> 728→966 i inne pochodne). Checkpointy:
<code>{CKPT_A.relative_to(PROJECT_ROOT)}</code>, <code>{CKPT_B.relative_to(PROJECT_ROOT)}</code>.
Poniżej {len(shared_ids)} otolitów — jeden na koszyk wieku ({', '.join(str(a) for a in TARGET_AGES)}),
dobranych deterministycznie (bez cherry-pickingu po błędzie żadnego z modeli). Pełna analiza liczbowa:
<code>outputs/DINO_proces.md</code> §4.24.</p>

<h2>1. Wiek (CORAL) — test set, n=1131</h2>
<table><tr><th>Metryka</th><th>{LABEL_A}</th><th>{LABEL_B}</th></tr>{metrics_rows}</table>

<h2>2. Lokalizacja przyrostów — 30 kart (własna kuratorska próba KAŻDEGO biegu)</h2>
{img_tag(loc_chart_b64)}
<table><tr><th>Metoda / classical</th><th>{LABEL_A}</th><th>{LABEL_B}</th><th>zmiana względna</th></tr>
{norm_rows}</table>
<p class="cap">Kolumny 2-3: metoda podzielona przez <code>classical</code> TEGO SAMEGO biegu — częściowa
kontrola o to, że 30-kartowe próby obu biegów to inny, tylko częściowo pokrywający się zestaw ryb (każdy
model dobiera swoje własne najlepsze/najgorsze predykcje). Surowe wartości px: powyższy wykres.</p>

<h2>3. Krzywe treningowe</h2>
{img_tag(train_chart_b64)}

<h2>4. Otolity obok siebie ({len(shared_ids)})</h2>
{otolith_html}

</body></html>
"""

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\n=== DONE ===\nRaport: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
