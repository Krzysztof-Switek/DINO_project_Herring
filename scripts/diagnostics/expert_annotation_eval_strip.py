"""02.09 — dendrochronology-strip experiment: held-out ZEGAR evaluation for a dual-branch,
strip-trained checkpoint (plans and summaries/02.09_wycinki_plan.md).

Directly comparable to every prior ZEGAR number in this project — Run N (22.8px, 42 images),
Opcja A (12 held-out: seed42 31.45/20.66px, seed7 60.18/62.21px), classical (122.9px), human
noise floor (4.28px) — because it reuses, UNCHANGED, the exact same ground-truth loading,
matching, and distance-metric functions from ``expert_annotation_eval.py``
(``load_expert_annotations``, ``resolve_and_crop_target_otolith``, ``point_to_axis_t``,
``match_t_values``, ``axis_dist``). Only the model-side PREDICTION pipeline differs: instead of
going through ``scripts/run_pipeline.py``'s candidate-detection machinery (which is NOT
strip-aware — it builds its own square ``build_transforms`` independently of ``OtolithDataset``,
see the plan's risk section — running it on a strip-trained checkpoint would silently feed it
square-resized images the density head never trained on), this script builds the strip crop
directly (``src.strip_extraction.extract_strip``) and reads density peaks off it via
``model.get_density_probs(..., patch_grid=(h_p, w_p))`` — the same accessor the plan's
``src/model.py`` fix (``_resolve_patch_grid``) exists for.

Age is NOT part of this script at all — read it from the training run's own
``pipeline_summary.json``, exactly as for every prior checkpoint. The strip branch never touches
CORAL/MIL (see ``OtolithModel.forward``'s ``density_image`` param), so there is nothing
age-related for this script to compute.

Headline result: the FULL 42-image ZEGAR set (Track A never trains on any ZEGAR image, so this
is a fair, direct comparison to Run N's own original 42-image number). Secondary: the same 12
``held_out`` image IDs from ``data/zegar_semi_weak_split.json`` (free to compute, same script) —
an apples-to-apples cross-check against Opcja A's seed42/seed7 rows, which only ever scored those
12.

Usage:
    python scripts/diagnostics/expert_annotation_eval_strip.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DIAG_DIR = PROJECT_ROOT / "scripts" / "diagnostics"
if str(DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(DIAG_DIR))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage

import expert_annotation_eval as ev
from train_zegar_localization_head import decode_topk_peaks
from src.dataset import build_transforms
from src.inference import load_model_from_checkpoint
from src.otolith_axis import detect_axis
from src.strip_extraction import extract_strip
from scripts.run_pipeline import load_merged_config

# ---------------------------------------------------------------------------
# Constants — edit here, not inline below. Same temp-swap-and-revert workflow as
# expert_annotation_eval.py's own REFERENCE_CKPT/REFERENCE_CONFIG: point these at the
# strip-trained run to score, run this script (~similar order of CPU time as the base
# script — one forward pass per image, no full run_pipeline overhead), revert after —
# `git diff` on this file should stay clean.
# ---------------------------------------------------------------------------

REFERENCE_CONFIG = PROJECT_ROOT / "configs" / "config_strip_a.yaml"
REFERENCE_CKPT = (PROJECT_ROOT / "outputs" / "data" / "REPLACE_WITH_RUN_TAG" / "checkpoints"
                  / "embedded" / "best.pt")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "02.09_zegar_strip_a_eval"

ZEGAR_SPLIT_PATH = PROJECT_ROOT / "data" / "zegar_semi_weak_split.json"

# Same default as decode_topk_peaks's own signature elsewhere in this project.
MIN_DIST_PATCHES = 1.5


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ZEGAR — ewaluacja lokalizacji, wariant paskowy (dwugałęziowy, 02.09)")
    print("=" * 70)

    print("\n[1/6] Wczytywanie adnotacji...")
    ann = ev.load_expert_annotations()
    samples = sorted(ann["Sample"].unique())
    print(f"  {len(samples)} próbek, {len(ann)} zaadnotowanych przyrostów (KK+SS łącznie)")
    ev.validate_annotation_bounds(ann, ev.IMAGE_DIR)

    cfg = load_merged_config(REFERENCE_CONFIG, None)
    if not cfg.data.dual_branch_density:
        sys.exit(
            "REFERENCE_CONFIG nie ma data.dual_branch_density=true — to nie jest config "
            "biegu paskowego, sprawdź stałe na górze pliku."
        )
    seg_params = cfg.segmentation.as_params()
    strip_length_px = cfg.data.strip_length_px
    strip_width_px = cfg.data.strip_width_px
    h_patches = strip_width_px // cfg.data.patch_size
    w_patches = strip_length_px // cfg.data.patch_size
    max_gap_t = cfg.candidates.min_peak_distance / ev.N_SAMPLES_AXIS
    strip_transform = build_transforms(strip_length_px, split="test", strip=True)

    print("\n[2/6] Wczytanie modelu (checkpoint paskowy)...")
    if not REFERENCE_CKPT.exists():
        sys.exit(f"Brak checkpointu: {REFERENCE_CKPT}")
    model = load_model_from_checkpoint(cfg, REFERENCE_CKPT)
    model.eval()

    print("\n[3/6] Wczytanie podziału held-out ZEGAR (dla wtórnego porównania z Opcją A)...")
    heldout_ids: set[str] = set()
    if ZEGAR_SPLIT_PATH.exists():
        split_data = json.loads(ZEGAR_SPLIT_PATH.read_text(encoding="utf-8"))
        heldout_ids = set(split_data.get("held_out", []))
        print(f"  {len(heldout_ids)} obrazów held-out wczytanych z {ZEGAR_SPLIT_PATH.name}")
    else:
        print(f"  [ostrzeżenie] brak {ZEGAR_SPLIT_PATH} — wynik na 12 held-out nie zostanie "
              f"policzony, tylko pełne 42")

    print("\n[4/6] Przycinanie do pojedynczego otolitu, budowanie pasków, inferencja...")
    per_image_results = []
    n_seg_failed = 0
    for i, sample in enumerate(samples, 1):
        image_id = f"{sample}.jpg"
        sub = ann[ann.Sample == sample]
        mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
        raw_rgb = np.array(PILImage.open(ev.IMAGE_DIR / image_id).convert("RGB"), dtype=np.uint8)
        cropped, x0, y0, _used_second = ev.resolve_and_crop_target_otolith(raw_rgb, mean_xy, seg_params)
        if cropped is None:
            print(f"  [{i}/{len(samples)}] {sample}: segmentacja nieudana, pomijam")
            n_seg_failed += 1
            continue

        # axis_info liczone na NIEOBRÓCONEJ (przyciętej) klatce — ten sam wzorzec co
        # train_zegar_localization_head_canonical.py: point_to_axis_t/GT działają w tej samej
        # przestrzeni bez żadnych zmian, tylko M_inv (niżej) mapuje predykcje paska z powrotem.
        axis_info = detect_axis(
            cropped, seg_params=seg_params, nucleus_method=cfg.segmentation.nucleus_method,
            axis_method=cfg.segmentation.axis_method,
        )
        if axis_info is None:
            print(f"  [{i}/{len(samples)}] {sample}: oś nie policzona, pomijam")
            n_seg_failed += 1
            continue

        t_kk = [ev.point_to_axis_t(x, y, axis_info) for x, y in
               zip(sub[sub.annotator == "KK"].x - x0, sub[sub.annotator == "KK"].y - y0)]
        t_ss = [ev.point_to_axis_t(x, y, axis_info) for x, y in
               zip(sub[sub.annotator == "SS"].x - x0, sub[sub.annotator == "SS"].y - y0)]
        pairs, _u_kk, _u_ss = ev.match_t_values(t_kk, t_ss, max_gap_t)
        gt_t = [(t_kk[a] + t_ss[b]) / 2.0 for a, b in pairs]
        if not gt_t:
            print(f"  [{i}/{len(samples)}] {sample}: brak dopasowanych par GT (KK/SS), pomijam")
            continue

        strip_rgb, M = extract_strip(
            cropped, axis_info["mask"], axis_info["centroid"], axis_info["far_edge"],
            axis_info["length_px"], strip_length_px, strip_width_px,
        )
        strip_tensor = strip_transform(PILImage.fromarray(strip_rgb))
        with torch.no_grad():
            density_grid = model.get_density_probs(
                strip_tensor.unsqueeze(0), patch_grid=(h_patches, w_patches),
            )[0].numpy()

        k = len(gt_t)
        peaks_rc = decode_topk_peaks(density_grid, k, min_dist_patches=MIN_DIST_PATCHES)
        M_inv = cv2.invertAffineTransform(M)
        t_model = []
        for r, c in peaks_rc:
            x_strip = (c + 0.5) * cfg.data.patch_size
            y_strip = (r + 0.5) * cfg.data.patch_size
            x_crop, y_crop = (M_inv @ np.array([x_strip, y_strip, 1.0]))[:2]
            t_model.append(ev.point_to_axis_t(x_crop, y_crop, axis_info))

        length_px = axis_info["length_px"]
        model_vs_gt_axis = ev.axis_dist(t_model, gt_t, length_px)
        kk_vs_ss_axis = ev.axis_dist(t_kk, t_ss, length_px)

        per_image_results.append({
            "Sample": sample, "image_id": image_id,
            "is_held_out": sample in heldout_ids,
            "n_gt_rings": len(gt_t), "n_model_peaks": len(t_model),
            "model_vs_gt_axis_px": model_vs_gt_axis,
            "kk_vs_ss_axis_px": kk_vs_ss_axis,
        })
        print(f"  [{i}/{len(samples)}] {sample}: model_vs_gt={model_vs_gt_axis:.1f}px "
              f"(n_gt={len(gt_t)}, held_out={sample in heldout_ids})")

    print(f"\n  {len(per_image_results)}/{len(samples)} obrazów ocenionych "
          f"({n_seg_failed} pominiętych z powodu segmentacji/osi)")

    print("\n[5/6] Wyniki zbiorcze...")
    results_df = pd.DataFrame(per_image_results)
    results_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    def _agg(df: pd.DataFrame, col: str):
        v = df[col].dropna()
        return float(v.mean()) if len(v) else None

    aggregate_full = {
        "n_images_scored": len(results_df),
        "n_images_total": len(samples),
        "model_vs_gt_axis_px": _agg(results_df, "model_vs_gt_axis_px"),
        "kk_vs_ss_axis_px": _agg(results_df, "kk_vs_ss_axis_px"),
    }
    print(f"  Pełne 42 (nagłówkowe): model_vs_gt_axis_px = {aggregate_full['model_vs_gt_axis_px']}")

    aggregate_heldout = None
    if heldout_ids:
        heldout_df = results_df[results_df["is_held_out"]]
        aggregate_heldout = {
            "n_images_scored": len(heldout_df),
            "n_images_total": len(heldout_ids),
            "model_vs_gt_axis_px": _agg(heldout_df, "model_vs_gt_axis_px"),
            "kk_vs_ss_axis_px": _agg(heldout_df, "kk_vs_ss_axis_px"),
        }
        print(f"  12 held-out (porównanie z Opcją A): model_vs_gt_axis_px = "
              f"{aggregate_heldout['model_vs_gt_axis_px']}")

    print("\n[6/6] Zapisywanie wyników...")
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps({
            "aggregate_full_42": aggregate_full,
            "aggregate_heldout_12": aggregate_heldout,
            "per_image": per_image_results,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nZapisano: {OUTPUT_DIR / 'metrics.csv'}, {OUTPUT_DIR / 'metrics.json'}")
    print("\nDla porównania (z pamięci projektu, nie liczone tu ponownie):")
    print("  Run N (42 obrazy):            22.8px")
    print("  Opcja A seed42 (12 held-out):  31.45 / 20.66px (mean/median)")
    print("  Opcja A seed7  (12 held-out):  60.18 / 62.21px (mean/median)")
    print("  Klasyka (42 obrazy):           122.9px")
    print("  Podłoga ludzka (KK vs SS):     4.28px")


if __name__ == "__main__":
    main()
