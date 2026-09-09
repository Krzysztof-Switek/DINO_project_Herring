"""09.09 — wycinek kątowy (polar wedge) experiment: held-out ZEGAR localization evaluation for a
dual-branch, wedge-trained checkpoint (`configs/config_wedge_a.yaml`,
`C:\\Users\\kswitek\\.claude\\plans\\wise-moseying-rivest.md`).

This is the missing piece flagged at the end of the wedge implementation session (09.09): the
plan's own code (Fazas 0-7) was complete and tested, but there was no way to SCORE a wedge run
against Run N / the strip / the classical method once training finishes. This script closes that
gap — it does not exist to answer any localization question by itself (no wedge checkpoint has
been trained yet; `REFERENCE_CKPT` below is a placeholder, same convention as
`expert_annotation_eval_strip.py`'s own `REPLACE_WITH_RUN_TAG`).

Two things are DELIBERATELY different from `expert_annotation_eval_strip.py`, not oversights:

1. **Metric**: this project's methodology moved on since the strip script was written (08-09.09,
   `plans and summaries/08.09_metodyka_i_diagnoza_paska.md`) — the original `ev.axis_dist` is a
   one-directional nearest-neighbour metric that lets one true ring register a "hit" against
   several model points (or vice versa), silently inflating scores. The corrected methodology is
   a bounded Hungarian, one-to-one, order-respecting pairing on `t`-values (`ev.match_t_values`,
   already used for KK-vs-SS consensus, now applied to model-vs-consensus too) — the same
   `paired_eval`/`summarize` shape used by the (gitignored, session-scratch) `outputs/
   09.09_masked_wizualizacje/paired_metric_all_runs.py` for Run N/classical/Track A/B/masked, but
   redefined HERE from the tracked, permanent primitives (`ev.match_t_values`) rather than
   imported from a throwaway script, so this eval survives independently of that session's own
   temp files.
2. **Geometry**: no affine strip warp / `cv2.invertAffineTransform`. The wedge is a Daugman-style
   polar remap (`src.wedge_extraction`) — `wedge_geometry_from_axis_info` builds the geometry,
   `extract_polar_wedge` produces the canvas, `wedge_xy_to_point`/`point_to_wedge_xy` are the
   (already unit-tested, round-trip-verified) inverse mappings back to real image space. The
   wedge's own per-patch `(t, theta)` (`wedge_polar_coords`) are ALSO fed into
   `model.get_density_probs` as `polar_t`/`polar_theta` — required for `density_head_type=
   radial_attention` to reproduce training-time behaviour (`get_density_probs`'s own docstring);
   the strip never needed this since its head is `"mlp"`, which ignores polar args entirely.

Reuses, unchanged: `expert_annotation_eval.py` (annotation loading, `point_to_axis_t`,
`match_t_values`), `train_zegar_localization_head.py::decode_topk_peaks`, `src.wedge_extraction`
(all four geometry functions, all already unit-tested in `tests/test_wedge_extraction.py` and
exercised on real ZEGAR photos in `scripts/diagnostics/verify_wedge_end_to_end.py`).

Angular-resolution bands (09.09 follow-up, `plans and summaries/09.09_wycinek_pasma_katowe_plan.md`,
Faza 14): this script also serves `configs/config_wedge_b.yaml` (multi-band) transparently — it
detects `cfg.data.wedge_band_edges_t is not None` and switches to `wedge_band_geometries_from_
axis_info`/`extract_polar_wedge_band`/`model.get_density_probs_bands`/`decode_topk_peaks_real_t`
(peaks decoded directly in REAL (t, theta) space, since a sequence concatenated across bands of
different shapes has no single rectangular grid for the gridded `decode_topk_peaks`'s 2D NMS to
operate on) instead of the single-canvas path. Point REFERENCE_CONFIG at whichever config you are
scoring; no other change needed.

Usage (once a wedge checkpoint exists — edit REFERENCE_CKPT below first):
    python scripts/diagnostics/expert_annotation_eval_wedge.py
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

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage

import expert_annotation_eval as ev
from train_zegar_localization_head import decode_topk_peaks, decode_topk_peaks_real_t
from src.dataset import build_transforms
from src.inference import load_model_from_checkpoint
from src.otolith_axis import detect_axis
from src.wedge_extraction import (
    extract_polar_wedge, extract_polar_wedge_band, wedge_band_geometries_from_axis_info,
    wedge_band_polar_coords, wedge_band_xy_to_point, wedge_geometry_from_axis_info,
    wedge_polar_coords, wedge_xy_to_point,
)
from scripts.run_pipeline import load_merged_config

# ---------------------------------------------------------------------------
# Constants — edit here, not inline below. Same temp-swap-and-point-at-a-real-run workflow as
# expert_annotation_eval_strip.py's own REFERENCE_CKPT: point this at the wedge-trained run once
# main_wedge_a.py has finished on the server.
# ---------------------------------------------------------------------------

REFERENCE_CONFIG = PROJECT_ROOT / "configs" / "config_wedge_a.yaml"
REFERENCE_CKPT = (PROJECT_ROOT / "outputs" / "data" / "REPLACE_WITH_RUN_TAG" / "checkpoints"
                  / "embedded" / "best.pt")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "09.09_zegar_wedge_a_eval"

ZEGAR_SPLIT_PATH = PROJECT_ROOT / "data" / "zegar_semi_weak_split.json"

# Same default as decode_topk_peaks's own signature / expert_annotation_eval_strip.py's own
# MIN_DIST_PATCHES — no wedge-specific peak-spacing measurement exists yet, so this is the same
# "no special assumption" starting point, not a tuned value.
MIN_DIST_PATCHES = 1.5

N_SAMPLES_AXIS = 50  # matches ev.N_SAMPLES_AXIS / paired_metric_all_runs.py's own axis resolution


# ---------------------------------------------------------------------------
# One-to-one, order-respecting pairing (Hungarian on t-values) — the corrected methodology.
# Defined locally (not imported from the gitignored outputs/ scratch script) so this eval survives
# independently; the actual primitive it wraps, ev.match_t_values, IS a tracked, permanent function.
# ---------------------------------------------------------------------------

def paired_eval(t_model: list[float], gt_t: list[float], length_px: float, max_gap_t: float) -> dict:
    pairs, unmatched_model, unmatched_gt = ev.match_t_values(t_model, gt_t, max_gap_t)
    paired_px = [abs(t_model[i] - gt_t[j]) * length_px for i, j in pairs]
    return {
        "n_gt": len(gt_t), "n_model": len(t_model), "n_pairs": len(pairs),
        "n_unmatched_model": len(unmatched_model), "n_unmatched_gt": len(unmatched_gt),
        "paired_px": paired_px,
        "mean_paired_px": float(np.mean(paired_px)) if paired_px else None,
    }


def summarize(rows: list[dict]) -> dict:
    per_image_means = [r["mean_paired_px"] for r in rows if r["mean_paired_px"] is not None]
    all_paired = [d for r in rows for d in r["paired_px"]]
    total_gt = sum(r["n_gt"] for r in rows)
    total_pairs = sum(r["n_pairs"] for r in rows)
    return {
        "n_images": len(rows),
        "mean_of_per_image_means_px": float(np.mean(per_image_means)) if per_image_means else None,
        "median_of_per_image_means_px": float(np.median(per_image_means)) if per_image_means else None,
        "pooled_mean_over_all_pairs_px": float(np.mean(all_paired)) if all_paired else None,
        "pooled_median_over_all_pairs_px": float(np.median(all_paired)) if all_paired else None,
        "total_gt_rings": total_gt, "total_pairs": total_pairs,
        "pairing_coverage": total_pairs / total_gt if total_gt else None,
        "total_unmatched_model_spurious": sum(r["n_unmatched_model"] for r in rows),
        "total_unmatched_gt_missed": sum(r["n_unmatched_gt"] for r in rows),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ZEGAR — ewaluacja lokalizacji, wycinek kątowy (dwugałęziowy, 09.09)")
    print("=" * 70)

    print("\n[1/6] Wczytywanie adnotacji...")
    ann = ev.load_expert_annotations()
    samples = sorted(ann["Sample"].unique())
    print(f"  {len(samples)} próbek, {len(ann)} zaadnotowanych przyrostów (KK+SS łącznie)")
    ev.validate_annotation_bounds(ann, ev.IMAGE_DIR)

    cfg = load_merged_config(REFERENCE_CONFIG, None)
    if not cfg.data.dual_branch_wedge:
        sys.exit(
            "REFERENCE_CONFIG nie ma data.dual_branch_wedge=true — to nie jest config "
            "biegu wycinka kątowego, sprawdź stałe na górze pliku."
        )
    seg_params = cfg.segmentation.as_params()
    delta_theta_deg = cfg.data.wedge_delta_theta_deg
    max_gap_t = cfg.candidates.min_peak_distance / N_SAMPLES_AXIS
    is_radial_attention = cfg.model.density_head_type == "radial_attention"
    is_bands = cfg.data.wedge_band_edges_t is not None

    if is_bands:
        band_edges_t = cfg.data.wedge_band_edges_t
        band_canvas_w = [n * cfg.data.patch_size for n in cfg.data.wedge_band_n_angle_patches]
        band_canvas_h = [n * cfg.data.patch_size for n in cfg.data.wedge_band_n_radius_patches]
        # build_transforms(wedge=True) skips Resize entirely — size-agnostic, ONE shared
        # transform serves every band regardless of its own (different) canvas width.
        wedge_transform = build_transforms(1, split="test", wedge=True)
        print(f"  Tryb: PASMA promieniowe ({len(band_edges_t) - 1} pasm, kolumny="
              f"{cfg.data.wedge_band_n_angle_patches}, wiersze={cfg.data.wedge_band_n_radius_patches})")
    else:
        h_patches, w_patches = cfg.data.wedge_n_radius_patches, cfg.data.wedge_n_angle_patches
        canvas_h, canvas_w = h_patches * cfg.data.patch_size, w_patches * cfg.data.patch_size
        wedge_transform = build_transforms(canvas_h, split="test", wedge=True)
        print("  Tryb: pojedynczy wycinek (jedna kanwa)")

    print("\n[2/6] Wczytanie modelu (checkpoint wycinka kątowego)...")
    if not REFERENCE_CKPT.exists():
        sys.exit(
            f"Brak checkpointu: {REFERENCE_CKPT}\n"
            f"       Ten skrypt jest gotową infrastrukturą, ale trening jeszcze się nie odbył —\n"
            f"       podmień REFERENCE_CKPT na ścieżkę do best.pt z main_wedge_a.py, gdy bieg\n"
            f"       skończy się na serwerze."
        )
    model = load_model_from_checkpoint(cfg, REFERENCE_CKPT)
    model.eval()

    print("\n[3/6] Wczytanie podziału held-out ZEGAR (dla wtórnego porównania z Opcją A)...")
    heldout_ids: set[str] = set()
    if ZEGAR_SPLIT_PATH.exists():
        split_data = json.loads(ZEGAR_SPLIT_PATH.read_text(encoding="utf-8"))
        heldout_ids = set(split_data.get("held_out", []))
        print(f"  {len(heldout_ids)} obrazów held-out wczytanych z {ZEGAR_SPLIT_PATH.name}")
    else:
        print(f"  [ostrzeżenie] brak {ZEGAR_SPLIT_PATH} — wynik na held-out nie zostanie policzony")

    print("\n[4/6] Przycinanie do pojedynczego otolitu, budowanie wycinków, inferencja...")
    per_image_results = []
    paired_rows = []
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
        kkss_pairs, _u_kk, _u_ss = ev.match_t_values(t_kk, t_ss, max_gap_t)
        gt_t = [(t_kk[a] + t_ss[b]) / 2.0 for a, b in kkss_pairs]
        if not gt_t:
            print(f"  [{i}/{len(samples)}] {sample}: brak dopasowanych par GT (KK/SS), pomijam")
            continue

        mask = axis_info["mask"]
        k = len(gt_t)

        if is_bands:
            bands = wedge_band_geometries_from_axis_info(
                mask, axis_info, delta_theta_deg, band_edges_t, band_canvas_w, band_canvas_h)
            band_tensors, band_t, band_theta, band_valid = [], [], [], []
            t_flat_all: list[np.ndarray] = []
            patch_locations: list[tuple] = []   # (band, row_local, col_local), same flat order
            for band in bands:
                arr = extract_polar_wedge_band(cropped, mask, band)
                band_tensors.append(wedge_transform(PILImage.fromarray(arr)).unsqueeze(0))
                t_grid, theta_grid = wedge_band_polar_coords(band, cfg.data.patch_size)
                t_flat = torch.from_numpy(t_grid.reshape(1, -1).copy()).float()
                band_t.append(t_flat)
                band_theta.append(torch.from_numpy(theta_grid.reshape(1, -1).copy()).float())
                band_valid.append(torch.ones_like(t_flat, dtype=torch.bool))
                t_flat_all.append(t_grid.reshape(-1))
                h_p, w_p = t_grid.shape
                patch_locations.extend((band, r, c) for r in range(h_p) for c in range(w_p))

            density_flat = model.get_density_probs_bands(
                band_tensors, polar_t=band_t, polar_theta=band_theta, polar_valid=band_valid,
            )[0].numpy()
            t_flat_all = np.concatenate(t_flat_all)
            peak_idx = decode_topk_peaks_real_t(density_flat, t_flat_all, k)

            t_model = []
            for idx in peak_idx:
                band, r_local, c_local = patch_locations[idx]
                col = (c_local + 0.5) * cfg.data.patch_size
                row = (r_local + 0.5) * cfg.data.patch_size
                x_crop, y_crop = wedge_band_xy_to_point(col, row, band)
                t_model.append(ev.point_to_axis_t(x_crop, y_crop, axis_info))
        else:
            geom = wedge_geometry_from_axis_info(mask, axis_info, delta_theta_deg, canvas_w, canvas_h)
            wedge_rgb = extract_polar_wedge(cropped, mask, geom)
            wedge_tensor = wedge_transform(PILImage.fromarray(wedge_rgb)).unsqueeze(0)

            polar_t = polar_theta = polar_valid = None
            if is_radial_attention:
                t_grid, theta_grid = wedge_polar_coords(geom, cfg.data.patch_size)
                polar_t = torch.from_numpy(t_grid.reshape(1, -1).copy()).float()
                polar_theta = torch.from_numpy(theta_grid.reshape(1, -1).copy()).float()
                polar_valid = torch.ones_like(polar_t, dtype=torch.bool)

            with torch.no_grad():
                density_grid = model.get_density_probs(
                    wedge_tensor, polar_t=polar_t, polar_theta=polar_theta, polar_valid=polar_valid,
                    patch_grid=(h_patches, w_patches),
                )[0].numpy()

            peaks_rc = decode_topk_peaks(density_grid, k, min_dist_patches=MIN_DIST_PATCHES)
            t_model = []
            for r, c in peaks_rc:
                col = (c + 0.5) * cfg.data.patch_size
                row = (r + 0.5) * cfg.data.patch_size
                x_crop, y_crop = wedge_xy_to_point(col, row, geom)
                t_model.append(ev.point_to_axis_t(x_crop, y_crop, axis_info))

        length_px = axis_info["length_px"]
        result = paired_eval(t_model, gt_t, length_px, max_gap_t)
        kkss_result = paired_eval(t_kk, t_ss, length_px, max_gap_t)
        result["Sample"] = sample
        result["is_held_out"] = sample in heldout_ids
        paired_rows.append(result)

        per_image_results.append({
            "Sample": sample, "image_id": image_id,
            "is_held_out": sample in heldout_ids,
            **{f"model_{k_}": v_ for k_, v_ in result.items()
               if k_ not in ("paired_px", "Sample", "is_held_out")},
            "kkss_mean_paired_px": kkss_result["mean_paired_px"],
        })
        if result["mean_paired_px"] is not None:
            print(f"  [{i}/{len(samples)}] {sample}: {result['n_pairs']}/{result['n_gt']} sparowane, "
                  f"{result['mean_paired_px']:.1f}px")
        else:
            print(f"  [{i}/{len(samples)}] {sample}: 0 par (n_gt={result['n_gt']})")

    print(f"\n  {len(per_image_results)}/{len(samples)} obrazów ocenionych "
          f"({n_seg_failed} pominiętych z powodu segmentacji/osi)")

    print("\n[5/6] Wyniki zbiorcze (parowanie Hungarian 1:1, od jądra na zewnątrz)...")
    summary_full = summarize(paired_rows)
    print(json.dumps(summary_full, indent=2))

    summary_heldout = None
    if heldout_ids:
        heldout_rows = [r for r in paired_rows if r["is_held_out"]]
        if heldout_rows:
            summary_heldout = summarize(heldout_rows)
            print("\n  Held-out subset:")
            print(json.dumps(summary_heldout, indent=2))

    print("\n[6/6] Zapisywanie wyników...")
    results_df = pd.DataFrame(per_image_results)
    results_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({"full": summary_full, "held_out": summary_heldout}, indent=2), encoding="utf-8")
    print(f"\nZapisano: {OUTPUT_DIR / 'metrics.csv'}, {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
