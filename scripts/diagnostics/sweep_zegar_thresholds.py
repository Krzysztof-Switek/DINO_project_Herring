"""17.08 — Sweep progów post-processingu (`candidates.*`) na prawdziwym ground truth ZEGAR,
BEZ TRENINGU. Realizacja `plans and summaries/12.08_ZEGAR_ANOTACJE_TO_DO.md` §5 "Część B",
pierwsza (najbezpieczniejsza) opcja stamtąd — dokładnie ten sam wzorzec, który już raz zadziałał
dla `candidates.density_image_size` (Etap 25, `plans and summaries/6.08_PODSUMOWANIE_PROCESU.md`).

Metodologia: kroki [1/5]-[4/5] (adnotacje, przycinanie do pojedynczego otolitu, wiek, oś +
mapa density) są NIEZALEŻNE od progów post-processingu i liczone RAZ na obraz — model
liczy density_grid tylko raz per obraz, niezależnie od tego, ile wartości progu testujemy.
Tylko krok [5/5] (`src.ring_extraction.select_increments` — wykrywanie pików + klastrowanie
DP-owe) jest powtarzany per wartość progu; to jest tanie (bez sieci), więc cały sweep jest
szybki mimo wielu wartości. `select_increments` zwraca `final_t` (promień znormalizowany,
niezależny od układu współrzędnych obcięcia — patrz `sweep_forced_topk_peaks.py`), więc
porównanie z ground truth nie wymaga cofania przesunięcia obcięcia.

Referencyjny checkpoint/config: Run N (`configs/config_radial_attention.yaml`,
`outputs/11.08_radial_attention`) — obecny punkt odniesienia (Etap 24-26). Progi sweepowane
PO JEDNYM naraz (reszta na wartości produkcyjnej), zgodnie z zasadą projektu "nigdy nie
hardkoduj progu spoza cfg.candidates.*" (feedback 31.07) — wszystkie wartości bazowe czytane
z REFERENCE_CONFIG, nie wpisane na sztywno.

Usage: python scripts/diagnostics/sweep_zegar_thresholds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Constants — edit here, not inline below (project convention).
# ---------------------------------------------------------------------------

REFERENCE_CONFIG = PROJECT_ROOT / "configs" / "config_radial_attention.yaml"   # Run N
REFERENCE_CKPT = (PROJECT_ROOT / "outputs" / "11.08_radial_attention"
                   / "checkpoints" / "embedded" / "best.pt")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "17.08_zegar_threshold_sweep"

# Jeden próg naraz, reszta na wartości bazowej (czytanej z REFERENCE_CONFIG w main()).
SWEEP_GRID: dict[str, list[float]] = {
    "min_peak_distance": [3, 5, 8],
    "prominence_threshold": [0.05, 0.1, 0.2],
    "inner_margin": [0.05, 0.20, 0.35],
    "width_decay_weight": [0.3, 1.0, 3.0],
    "width_ceiling_weight": [1.0, 3.0, 6.0],
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from scripts.run_pipeline import load_merged_config
    from scripts.diagnostics.expert_annotation_eval import (
        load_expert_annotations, validate_annotation_bounds,
        resolve_and_crop_target_otolith, point_to_axis_t, match_t_values,
        ANNOTATION_FILES, IMAGE_DIR as ZEGAR_IMAGE_DIR, N_SAMPLES_AXIS,
    )
    from src.otolith_axis import (detect_axis, mask_bbox, shift_axis_info, apply_background_mask,
                                  compute_polar_grid)
    from src.inference import load_model_from_checkpoint, run_inference
    from src.dataset import OtolithDataset, build_transforms
    from src.ring_extraction import select_increments
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("ZEGAR — sweep progów post-processingu (candidates.*), bez treningu")
    print("=" * 70)

    print("\n[1/5] Wczytywanie adnotacji...")
    ann = load_expert_annotations()
    samples = sorted(ann["Sample"].unique())
    print(f"  {len(samples)} próbek.")
    validate_annotation_bounds(ann, ZEGAR_IMAGE_DIR)

    print("\n[2/5] Wybór właściwego otolitu z pary + przycinanie...")
    if not REFERENCE_CKPT.exists():
        sys.exit(f"Brak checkpointu: {REFERENCE_CKPT}")
    cfg = load_merged_config(REFERENCE_CONFIG, None)
    # Prywatny cache masek TYLKO dla tego sweepu — ten sam powód co w expert_annotation_eval.py:
    # przycięty pojedynczy otolit reużywa image_id oryginalnego (dwuotolitowego) zdjęcia, więc
    # współdzielony cache serwowałby złą, nieaktualną maskę.
    cfg.data.mask_cache_dir = str(OUTPUT_DIR / "dataset_masks_cache")
    seg_params = cfg.segmentation.as_params()
    CROPPED_DIR = OUTPUT_DIR / "cropped_singles"
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    crop_offsets: dict[str, tuple[int, int]] = {}
    for sample in samples:
        image_id = f"{sample}.jpg"
        sub = ann[ann.Sample == sample]
        mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
        raw_rgb = np.array(PILImage.open(ZEGAR_IMAGE_DIR / image_id).convert("RGB"), dtype=np.uint8)
        cropped, x0, y0, _used_second = resolve_and_crop_target_otolith(raw_rgb, mean_xy, seg_params)
        if cropped is None:
            cropped, x0, y0 = raw_rgb, 0, 0
        crop_offsets[sample] = (x0, y0)
        PILImage.fromarray(cropped).save(CROPPED_DIR / image_id)
    ann = ann.copy()
    ann["x"] = ann.apply(lambda r: r.x - crop_offsets[r.Sample][0], axis=1)
    ann["y"] = ann.apply(lambda r: r.y - crop_offsets[r.Sample][1], axis=1)

    print(f"\n[3/5] Wczytanie modelu ({REFERENCE_CKPT}) i wieku (na przyciętych otolitach)...")
    model = load_model_from_checkpoint(cfg, REFERENCE_CKPT)
    model.eval()
    device = next(model.parameters()).device

    age_by_sample = (ann[ann.annotator == "KK"].drop_duplicates("Sample").set_index("Sample")["age"])
    scratch_csv = OUTPUT_DIR / "scratch_labels.csv"
    pd.DataFrame([
        {"image_id": f"{s}.jpg", "age": int(age_by_sample.get(s, 0)), "split": "test"}
        for s in samples
    ]).to_csv(scratch_csv, index=False)
    ds = OtolithDataset(cfg, split="test", labels_csv=str(scratch_csv), image_dir=str(CROPPED_DIR))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    run_inference(cfg, model, loader, OUTPUT_DIR)
    preds = pd.read_csv(OUTPUT_DIR / "predictions.csv").set_index("image_id")["predicted_age"]

    print("\n[4/5] Oś, mapa density i ground truth (RAZ na obraz, niezależnie od progów)...")

    def _polar_tensors_for(mask_for_polar, centroid, patch_grid_size):
        """(17.08) Mirror of scripts.run_pipeline._compute_axis_data_for_samples's own
        closure of the same name — bez tego, checkpoint radial_attention widziałby
        niezgodność trening/inferencja (brak informacji pozycyjnej, nieograniczona uwaga),
        dokładnie ten sam problem, który ten mechanizm miał rozwiązać (patrz src/model.py::
        RadialAttentionDensityHead docstring: "gracefully degrades... no positional encoding,
        no masking" gdy polar_t/polar_theta nie są podane). Nie importowane z run_pipeline.py
        bo tam to domknięcie (closure) przechwytujące cfg/device z otaczającej funkcji, nie
        samodzielna funkcja modułowa.
        """
        if mask_for_polar is None or centroid is None:
            return None, None, None
        h_p = w_p = patch_grid_size // cfg.data.patch_size
        t_grid, valid_grid, theta_grid = compute_polar_grid(mask_for_polar, centroid, h_p, w_p)
        to_flat = lambda a, dt: torch.from_numpy(a.reshape(1, -1).copy()).to(dt).to(device)
        return (to_flat(t_grid, torch.float32), to_flat(theta_grid, torch.float32),
                to_flat(valid_grid, torch.bool))

    density_transform = build_transforms(cfg.candidates.density_image_size, "test")
    max_gap_t = cfg.candidates.min_peak_distance / N_SAMPLES_AXIS
    per_image: dict[str, dict] = {}
    n_seg_failed = 0
    for sample in samples:
        image_id = f"{sample}.jpg"
        img_path = CROPPED_DIR / image_id
        if not img_path.exists() or image_id not in preds.index:
            continue
        orig_rgb = np.array(PILImage.open(img_path).convert("RGB"), dtype=np.uint8)
        axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                                nucleus_method=cfg.segmentation.nucleus_method,
                                axis_method=cfg.segmentation.axis_method)
        if axis_info is None:
            n_seg_failed += 1
            continue
        mask_arr = axis_info["mask"]
        model_input_rgb = (apply_background_mask(orig_rgb, mask_arr)
                           if cfg.data.mask_background else orig_rgb)

        crop_x0, crop_y0, cw, ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)
        crop_rgb = model_input_rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
        d_axis_info = shift_axis_info(axis_info, -crop_x0, -crop_y0)
        mask_for_density = mask_arr[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
        density_tensor = density_transform(PILImage.fromarray(crop_rgb)).unsqueeze(0).to(device)
        p_t, p_th, p_v = _polar_tensors_for(
            mask_for_density, d_axis_info["centroid"], cfg.candidates.density_image_size)
        with torch.no_grad():
            density_grid = model.get_density_probs(
                density_tensor, polar_t=p_t, polar_theta=p_th, polar_valid=p_v,
            ).squeeze(0).cpu().numpy()

        sub = ann[ann.image_id == image_id]
        t_kk = sub[sub.annotator == "KK"].apply(
            lambda r: point_to_axis_t(r.x, r.y, axis_info), axis=1).tolist()
        t_ss = sub[sub.annotator == "SS"].apply(
            lambda r: point_to_axis_t(r.x, r.y, axis_info), axis=1).tolist()
        pairs, _unmatched_kk, _unmatched_ss = match_t_values(t_kk, t_ss, max_gap_t)
        gt_t = [(t_kk[i] + t_ss[j]) / 2.0 for i, j in pairs]

        per_image[sample] = {
            "density_grid": density_grid, "d_axis_info": d_axis_info,
            "ch": ch, "cw": cw, "length_px": axis_info["length_px"],
            "predicted_age": int(preds.loc[image_id]),
            "gt_t": gt_t,
        }
    print(f"  {len(per_image)}/{len(samples)} obrazów gotowych do sweepu "
         f"(segmentacja nieudana: {n_seg_failed}).")

    print("\n[5/5] Sweep progów (tanie — bez modelu, tylko peak-finding + DP-select)...")
    baseline = {
        "min_peak_distance": cfg.candidates.min_peak_distance,
        "prominence_threshold": cfg.candidates.prominence_threshold,
        "inner_margin": cfg.candidates.inner_margin,
        "width_decay_weight": cfg.candidates.width_decay_weight,
        "width_ceiling_weight": cfg.candidates.width_ceiling_weight,
    }

    def score(params: dict) -> tuple[float, int]:
        dists = []
        for d in per_image.values():
            res = select_increments(
                d["density_grid"], d["d_axis_info"], d["predicted_age"], d["ch"], d["cw"],
                min_distance=params["min_peak_distance"],
                prominence=params["prominence_threshold"],
                inner_margin=params["inner_margin"],
                width_decay_weight=params["width_decay_weight"],
                width_ceiling_weight=params["width_ceiling_weight"],
            )
            t_model = res["final_t"]
            gt_t = d["gt_t"]
            if not t_model or not gt_t:
                continue
            dd = np.abs(np.asarray(t_model)[:, None] - np.asarray(gt_t)[None, :]) * d["length_px"]
            dists.append(float(dd.min(axis=1).mean()))
        return (float(np.mean(dists)) if dists else float("nan")), len(dists)

    rows = []
    base_score, base_n = score(baseline)
    rows.append({"param": "(baseline)", "value": "produkcyjne", **baseline,
                 "model_vs_gt_axis_px": base_score, "n_scored": base_n})
    print(f"  baseline: model_vs_gt_axis_px={base_score:.2f}px (n={base_n})", flush=True)

    for param, values in SWEEP_GRID.items():
        for v in values:
            if v == baseline[param]:
                continue   # baseline już policzone wyżej
            params = dict(baseline)
            params[param] = v
            s, n = score(params)
            rows.append({"param": param, "value": v, **params,
                         "model_vs_gt_axis_px": s, "n_scored": n})
            print(f"  {param}={v}: model_vs_gt_axis_px={s:.2f}px (n={n})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
    print(f"\nZapisano: {OUTPUT_DIR / 'sweep_results.csv'}")
    print("\n=== Podsumowanie ===")
    print(df[["param", "value", "model_vs_gt_axis_px", "n_scored"]].to_string(index=False))
    print(f"\nPodłoga szumu ludzkiego (KK vs SS) z poprzednich biegów: ~4px — patrz "
         f"outputs/12.08_zegar_annotation_eval/metrics.json lub 17.08_zegar_wide_window.")


if __name__ == "__main__":
    main()
