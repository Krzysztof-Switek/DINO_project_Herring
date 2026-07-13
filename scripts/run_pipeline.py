"""Full Embedded vs NotEmbedded pipeline orchestrator.

Steps:
  1  scan     — scan image dir + build labels CSVs
  2  train_e  — train Embedded model
  3  train_n  — train NotEmbedded model
  4  infer_ee — infer Embedded model on Embedded test set
  5  infer_nn — infer NotEmbedded model on NotEmbedded test set
  6  infer_en — CROSS: infer Embedded model on NotEmbedded test set
  7  infer_ne — CROSS: infer NotEmbedded model on Embedded test set
  8  cards    — generate increment annotation cards
  9  report   — build comparison HTML report

Usage:
    python scripts/run_pipeline.py \\
        --image-dir "Z:/Photo/Otolithes/HER/Processed" \\
        --excel     data/analysisWithOtolithPhoto.xlsx \\
        --output-dir outputs/ \\
        [--base-config configs/config.yaml] \\
        [--config-embedded     configs/config_embedded.yaml] \\
        [--config-not-embedded configs/config_not_embedded.yaml] \\
        [--skip-scan]   \\
        [--skip-train]  \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time as _time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_merged_config(base_path: Path | None, override_path: Path | None):
    """Load base + override YAML files and return merged OtolithConfig."""
    from src.config import OtolithConfig

    base_raw: dict = {}
    if base_path and Path(base_path).exists():
        raw = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))
        if raw:
            base_raw = raw

    override_raw: dict = {}
    if override_path and Path(override_path).exists():
        raw = yaml.safe_load(Path(override_path).read_text(encoding="utf-8"))
        if raw:
            override_raw = raw

    merged = _deep_update(base_raw, override_raw)
    return OtolithConfig(**merged)


# Ordered pipeline steps (for the --dry-run plan printout only). There is no
# resume/skip-by-state mechanism any more: every run is full and from zero
# (11.07 TO-DO Punkt 3).
STEPS = [
    "scan", "train_e", "train_n",
    "infer_ee", "infer_nn", "infer_en", "infer_ne",
    "cards", "report",
]


# ---------------------------------------------------------------------------
# Individual step runners
# ---------------------------------------------------------------------------

def _step_scan(args, data_dir: Path) -> tuple[Path, Path]:
    """Scan image directory and produce labels_embedded.csv + labels_not_embedded.csv."""
    from src.scan_labels import build_combined_labels

    combined = build_combined_labels(
        image_dir=Path(args.image_dir),
        excel_path=Path(args.excel),
        train=args.train,
        val=args.val,
        seed=args.seed,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(data_dir / "labels_combined.csv", index=False)
    emb_path = data_dir / "labels_embedded.csv"
    notemb_path = data_dir / "labels_not_embedded.csv"
    combined[combined.preprocessing_type == "Embedded"].to_csv(emb_path, index=False)
    combined[combined.preprocessing_type == "NotEmbedded"].to_csv(notemb_path, index=False)
    print(f"  Saved: {emb_path}, {notemb_path}")
    return emb_path, notemb_path


def _parse_train_log(log_path: Path) -> list[dict]:
    """Parsuje train.log trainera → lista słowników per epoka.

    Format linii: [timestamp] epoch=  N  train_loss=X.XXXX  val_loss=X.XXXX  val_mae=X.XXX
    """
    if not log_path.exists():
        return []
    pattern = re.compile(
        r"epoch=\s*(\d+)\s+"
        r"train_loss=([\d.]+)\s+"
        r"val_loss=([\d.]+)\s+"
        r"val_mae=([\d.]+)"
        r"(?:\s+lr=([\d.eE+-]+))?"
    )
    # Optional Section-B diagnostics (only present in newer logs / with the
    # relevant heads active) — parsed independently so older logs still work.
    extra_keys = ("coral_loss", "mil_loss", "mil_active",
                  "density_loss", "density_active", "mean_age")
    extra_pat = {k: re.compile(rf"{k}=([\d.eE+-]+)") for k in extra_keys}
    rows: list[dict] = []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        m = pattern.search(line)
        if m:
            try:
                row = {
                    "epoch":      int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_loss":   float(m.group(3)),
                    "val_mae":    float(m.group(4)),
                }
                # lr is optional — only present in logs written by the newer
                # Trainer; keep the row shape unchanged for older logs.
                if m.group(5) is not None:
                    row["lr"] = float(m.group(5))
                for k, pat in extra_pat.items():
                    em = pat.search(line)
                    if em:
                        row[k] = float(em.group(1))
                rows.append(row)
            except ValueError:
                continue
    # Keep only the LAST training session. train.log is now written fresh per run
    # (Trainer truncates it), but a legacy/concatenated log (several appended runs)
    # must NOT inflate epoch counts — slice from the last epoch==1 so the summary and
    # the report reflect the real last run, not the sum (11.07 TO-DO Punkt 5).
    last_start = max((i for i, r in enumerate(rows) if r.get("epoch") == 1), default=0)
    return rows[last_start:]


def _step_train(cfg, labels_csv: Path) -> tuple[Path, list[dict]]:
    """Train model and return (path to best checkpoint, per-epoch training logs)."""
    from src.dataset import OtolithDataset
    from src.model import OtolithModel
    from src.trainer import Trainer
    from src.utils import seed_everything, seed_worker, make_loader_generator
    from torch.utils.data import DataLoader

    # Reproducibility: seed BEFORE model init (fixes head weights) and before the
    # DataLoader is built (fixes shuffle order). Without this two identical runs
    # land on different checkpoints by chance — see 13.07_wnioski_TO_DO.md.
    seed_everything(cfg.project.seed)

    cfg_copy = cfg.model_copy(deep=True)
    cfg_copy.data.labels_csv = str(labels_csv)

    train_ds = OtolithDataset(cfg_copy, split="train")
    val_ds = OtolithDataset(cfg_copy, split="val")

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.data.num_workers,
                              worker_init_fn=seed_worker,
                              generator=make_loader_generator(cfg.project.seed))
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=cfg.data.num_workers,
                            worker_init_fn=seed_worker)

    model = OtolithModel(cfg_copy)
    trainer = Trainer(cfg_copy, model, train_loader, val_loader)
    trainer.fit()

    # Parse training log (Trainer resolves log_dir relative to PROJECT_ROOT)
    log_dir = Path(cfg.training.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    training_log_data = _parse_train_log(log_dir / "train.log")

    best_ckpt = trainer.checkpoint_dir / "best.pt"
    if not best_ckpt.exists():
        # Fallback: checkpointy bez best.pt (wznowienie po starej wersji kodu)
        ckpt_files = sorted(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint saved in {trainer.checkpoint_dir}")
        best_ckpt_src = min(ckpt_files, key=lambda p: float(p.stem.split("_loss")[-1]))
        import shutil as _shutil
        _shutil.copy2(best_ckpt_src, best_ckpt)
    print(f"  Best checkpoint: {best_ckpt}")
    return best_ckpt, training_log_data


def _select_topk_image_ids(pred_csv: Path, k_best: int, k_worst: int) -> set[str]:
    """image_ids of the k_best closest + k_worst farthest-off predictions.

    ``run_inference`` writes the ground-truth column as ``target_age``; it is
    normalised to ``age`` here so the |predicted - true| ranking works. Returns
    an empty set when the file is missing or the required columns are absent.
    """
    import pandas as pd

    if not Path(pred_csv).exists():
        return set()
    try:
        df = pd.read_csv(pred_csv)
    except Exception:
        return set()
    if "target_age" in df.columns and "age" not in df.columns:
        df = df.rename(columns={"target_age": "age"})
    if "age" not in df.columns or "predicted_age" not in df.columns:
        return set()
    df = df.assign(_err=(df["predicted_age"] - df["age"]).abs()).sort_values("_err")
    return (
        set(df.head(k_best)["image_id"].astype(str))
        | set(df.tail(k_worst)["image_id"].astype(str))
    )


def _step_infer(cfg, ckpt_path: Path, labels_csv: Path, output_dir: Path) -> Path:
    """Run inference for all test samples; then interpretation+candidates for top-k only."""
    import pandas as pd
    from src.dataset import OtolithDataset
    from src.inference import run_inference, load_model_from_checkpoint
    from src.interpretation import run_interpretation
    from src.candidates import run_candidates
    from torch.utils.data import DataLoader, Subset

    cfg_copy = cfg.model_copy(deep=True)
    cfg_copy.data.labels_csv = str(labels_csv)

    # Resolve original image directory for full-resolution outputs
    image_dir = Path(cfg_copy.data.image_dir)
    if not image_dir.is_absolute():
        image_dir = PROJECT_ROOT / image_dir

    test_ds = OtolithDataset(cfg_copy, split="test")
    loader = DataLoader(test_ds, batch_size=cfg.training.batch_size,
                        shuffle=False, num_workers=cfg.data.num_workers)

    model = load_model_from_checkpoint(cfg_copy, ckpt_path)

    # --- Step A: inference on all test samples ---
    print("  Inferencja (wszystkie próbki)...")
    run_inference(cfg_copy, model, loader, output_dir)

    # --- Step B: choose which samples get interpretation + candidate dots ---
    # annotate_all=True → every test image (full gallery); else top-k best/worst.
    pred_csv = output_dir / "predictions.csv"
    if cfg.inference.increment_samples.annotate_all:
        print("  Interpretacja/kandydaci dla WSZYSTKICH próbek testowych (annotate_all=True)")
        interp_loader = loader
    else:
        k_best  = cfg.inference.increment_samples.top_k_best
        k_worst = cfg.inference.increment_samples.top_k_worst
        top_ids = _select_topk_image_ids(pred_csv, k_best, k_worst)
        if top_ids:
            print(f"  Interpretacja dla {len(top_ids)} próbek "
                  f"({k_best} najlepszych + {k_worst} najgorszych)")
            top_indices = [i for i, row in test_ds.df.iterrows()
                           if str(row["image_id"]) in top_ids]
            subset_ds = Subset(test_ds, top_indices)
            interp_loader = DataLoader(subset_ds, batch_size=cfg.training.batch_size,
                                       shuffle=False, num_workers=cfg.data.num_workers)
        else:
            interp_loader = loader

    print("  Heatmapy i nakładki (oryginalna rozdzielczość)...")
    run_interpretation(cfg_copy, model, interp_loader, output_dir, image_dir=image_dir)

    print("  Kandydaci przyrostów (oryginalna rozdzielczość)...")
    run_candidates(cfg_copy, model, interp_loader, output_dir, image_dir=image_dir)

    print(f"  Predictions: {pred_csv}")
    return pred_csv


def _compute_axis_data_for_samples(
    samples: list[dict],
    image_dir: Path,
    cfg,
    ckpt_path: Path,
    cond_dir: Path,
) -> tuple[dict, dict]:
    """Compute importance grids and reasoning-card axis data for the given samples.

    For each sample we run the model once (importance grid + last-ordinal sigmoid),
    then resolve the otolith mask (cached under ``cond_dir/masks/`` if present,
    otherwise segmented on-the-fly and saved there), compute the measurement
    axis, sample the importance profile along it and detect peaks.

    Args:
        samples:    list of prediction rows (each at least ``image_id``)
        image_dir:  directory holding the original-resolution photos
        cfg:        config to drive ``compute_patch_importance``
        ckpt_path:  checkpoint to load for this condition
        cond_dir:   ``output_dir / cond_key`` — mask cache lives here

    Returns:
        grids:     ``{image_id → (H_p, W_p) ndarray}``
        axis_data: ``{image_id → {mask, axis_info, peak_indices, line_xy, profile_1d}}``
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage

    from src.candidates import find_candidate_peaks
    from src.dataset import build_transforms
    from src.inference import load_model_from_checkpoint
    from src.interpretation import (compute_patch_importance, compute_coral_gradcam,
                                     compute_cls_attention)
    from src.otolith_axis import (
        detect_axis,
        find_centroid,
        find_farthest_edge,
        load_mask,
        sample_profile_along_axis,
        save_mask,
    )

    grids: dict = {}
    axis_data: dict = {}

    if not samples:
        return grids, axis_data

    mask_dir = cond_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint(cfg, ckpt_path)
    model.eval()
    device = next(model.parameters()).device
    transform = build_transforms(cfg.data.image_size, "test")

    min_dist = cfg.candidates.min_peak_distance
    prominence = cfg.candidates.prominence_threshold
    n_samples_axis = 50

    missing_images = 0
    failed_axes = 0
    for row in samples:
        iid = str(row["image_id"])
        img_path = image_dir / iid
        if not img_path.exists():
            print(f"    [cards] obraz nie znaleziony: {img_path}")
            missing_images += 1
            continue
        try:
            img_pil = PILImage.open(img_path).convert("RGB")
            tensor = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                # Localisation signal: prefer the DECOUPLED density map (Kierunek B) when
                # the model has one, else the MIL / L2 importance map.
                if getattr(model, "use_density_head", False) and hasattr(model, "density_head"):
                    grid = model.get_density_probs(tensor).squeeze(0).cpu().numpy()
                else:
                    grid = compute_patch_importance(model, tensor).cpu().numpy()
            grids[iid] = grid
            # CORAL-head attributions (age verdict): Grad-CAM + CLS attention.
            # Both are internally defensive → None on failure (11.07 Punkt 7).
            _gc = compute_coral_gradcam(model, tensor)
            _ca = compute_cls_attention(model, tensor)
            coral_gc_grid = _gc.cpu().numpy() if _gc is not None else None
            cls_attn_grid = _ca.cpu().numpy() if _ca is not None else None
            # Keep the CLS panel from ever being blank: when true CLS attention is
            # unavailable (fused attention), fall back to a labelled L2-norm proxy.
            cls_is_fallback = cls_attn_grid is None
            if cls_is_fallback:
                cls_attn_grid = compute_patch_importance(
                    model, tensor, method="patch_token_importance").cpu().numpy()
        except Exception as e:
            print(f"    [cards] błąd dla {iid}: {e}")
            continue

        # --- Resolve axis: cached mask → re-derive centroid/far_edge; else segment ---
        orig_rgb = np.array(img_pil, dtype=np.uint8)
        H_img, W_img = orig_rgb.shape[:2]
        stem = Path(iid).stem
        mask_path = mask_dir / f"{stem}_mask.png"

        import cv2
        axis_info = None
        mask_arr = load_mask(mask_path)
        if mask_arr is not None:
            cent = find_centroid(mask_arr)
            far = find_farthest_edge(mask_arr, cent) if cent else None
            if cent and far:
                contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    axis_info = {
                        "mask":      mask_arr,
                        "centroid":  cent,
                        "far_edge":  far,
                        "contour":   contour,
                        "length_px": float(np.hypot(far[0] - cent[0],
                                                     far[1] - cent[1])),
                    }
        if axis_info is None:
            axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params())
            if axis_info is not None:
                mask_arr = axis_info["mask"]
                save_mask(mask_arr, mask_path)

        if axis_info is None:
            failed_axes += 1
            # CORAL panels (Grad-CAM, uwaga CLS, werdykt) NIE potrzebują osi — pokaż je
            # nawet gdy segmentacja padła; tylko panele lokalizacji będą placeholderem.
            axis_data[iid] = {
                "mask": None, "axis_info": None,
                "peak_indices": None, "line_xy": None, "profile_1d": None,
                "coral_gradcam": coral_gc_grid, "cls_attention": cls_attn_grid,
                "cls_is_fallback": cls_is_fallback,
            }
            continue

        profile_1d, line_xy = sample_profile_along_axis(
            grid, axis_info["centroid"], axis_info["far_edge"],
            H_img, W_img, n_samples=n_samples_axis,
        )
        peak_indices = find_candidate_peaks(profile_1d, min_dist, prominence)

        # Multi-axis increment localisation with count = predicted_age (11.07 Punkt 7):
        # candidates from every ray + the top-`age` consensus increments on the axis.
        from src.ring_extraction import select_increments, extract_ring_curves
        increments = select_increments(
            grid, axis_info, int(row.get("predicted_age", 0)), H_img, W_img,
            min_distance=min_dist, prominence=prominence,
        )
        # Ring CURVES from the probability map (drawn on the MIL/density card panel).
        ring_curves = extract_ring_curves(grid, axis_info, H_img, W_img)

        # OpenCV reference (Kierunek A): downscaled image + axis + CLASSICAL intensity
        # profile along the axis (sampled from the original grayscale, not the model),
        # for the interactive report widget where a technician tunes classical detection.
        opencv_ref = None
        classical_pts: list = []   # classical (OpenCV) peak positions on the axis — card cross-check
        try:
            import base64 as _b64
            import io as _io
            disp_max = 360
            sc = min(1.0, disp_max / max(H_img, W_img))
            dw, dh = max(1, int(W_img * sc)), max(1, int(H_img * sc))
            _buf = _io.BytesIO()
            PILImage.fromarray(orig_rgb).resize((dw, dh)).save(_buf, format="PNG")
            img_b64 = "data:image/png;base64," + _b64.b64encode(_buf.getvalue()).decode("ascii")
            gray = orig_rgb.mean(axis=2)
            prof, line_disp = [], []
            for (px, py) in line_xy:
                xi = min(max(int(px), 0), W_img - 1)
                yi = min(max(int(py), 0), H_img - 1)
                prof.append(float(gray[yi, xi]))
                line_disp.append([int(px * sc), int(py * sc)])
            prof = np.asarray(prof, dtype=np.float32)
            rng = float(prof.max() - prof.min())
            if rng > 1e-6:
                prof = (prof - prof.min()) / rng
            # Classical peaks on the intensity profile → axis pixel positions (model↔classic).
            for _i in find_candidate_peaks(prof, min_dist, prominence):
                _i = int(_i)
                if 0 <= _i < len(line_xy):
                    classical_pts.append((int(line_xy[_i][0]), int(line_xy[_i][1])))
            opencv_ref = {
                "img": img_b64, "w": dw, "h": dh,
                "line": line_disp,
                "profile": [round(float(v), 4) for v in prof.tolist()],
                "true_age": int(row.get("age", 0)),
                "pred_age": int(row.get("predicted_age", 0)),
            }
        except Exception as e:
            print(f"    [cards] opencv_ref błąd dla {iid}: {e}")

        axis_data[iid] = {
            "mask":           mask_arr,
            "axis_info":      axis_info,
            "peak_indices":   peak_indices,
            "line_xy":        line_xy,
            "profile_1d":     profile_1d,
            "final_axis_pts": increments["final_axis_pts"],
            "candidate_pts":  increments["candidate_pts"],
            "final_t":        increments["final_t"],
            "opencv_ref":     opencv_ref,
            "coral_gradcam":  coral_gc_grid,
            "cls_attention":  cls_attn_grid,
            "cls_is_fallback": cls_is_fallback,
            "ring_curves":    ring_curves,
            "classical_pts":  classical_pts,
        }

    total = len(samples)
    summary_line = (f"[cards] {cond_dir.name}: gridy {len(grids)}/{total} "
                    f"(brak obrazu: {missing_images}, segmentacja nieudana: {failed_axes})")
    print(f"    {summary_line}")
    # Persist card diagnostics to a file too — stdout is otherwise easily lost (13.07).
    try:
        diag_path = cond_dir.parent / "cards_diagnostics.txt"
        with diag_path.open("a", encoding="utf-8") as _f:
            _f.write(summary_line + "\n")
    except OSError:
        pass
    if len(grids) == 0 and total > 0:
        print(f"    [cards] OSTRZEŻENIE: 0/{total} próbek przetworzonych — "
              f"sprawdź --image-dir={image_dir}")
    return grids, axis_data


def _reload_cards_from_disk(output_dir: Path) -> dict[str, list[Path]]:
    """Reconstruct ``{best, worst}`` card lists from ``output_dir/cards/**/*.png``.

    Called when the cards step is skipped (already completed in an earlier run)
    so the comparison report still gets card paths instead of an empty dict.
    """
    cards: dict[str, list[Path]] = {"best": [], "worst": []}
    cards_root = Path(output_dir) / "cards"
    if not cards_root.exists():
        return cards
    for label in ("best", "worst"):
        for png in sorted(cards_root.glob(f"**/{label}/*.png")):
            cards[label].append(png)
    return cards


_CONDITION_LABELS = {
    "emb_on_emb":          "Emb → Emb",
    "notemb_on_notemb":    "NotEmb → NotEmb",
    "cross_emb_on_notemb": "Emb → NotEmb (CROSS)",
    "cross_notemb_on_emb": "NotEmb → Emb (CROSS)",
}


def _collect_candidate_overlays(output_dir: Path, cond_keys) -> dict[str, list[Path]]:
    """Gather model-drawn increment-dot overlay PNGs per condition (report Section G).

    Reads ``output_dir/<cond_key>/candidates_overlays/*_candidates_overlay.png``.
    How many images are present depends on ``increment_samples.annotate_all``.
    """
    overlays: dict[str, list[Path]] = {}
    for cond_key in cond_keys:
        ov_dir = Path(output_dir) / cond_key / "candidates_overlays"
        if ov_dir.exists():
            pngs = sorted(ov_dir.glob("*_candidates_overlay.png"))
            if pngs:
                overlays[_CONDITION_LABELS.get(cond_key, cond_key)] = pngs
    return overlays


def _localization_quality(axis_data: dict, samples: list[dict]) -> list[dict]:
    """Per-sample localisation stats: model increments vs classical intensity peaks.

    ``mean_dist_final_to_classical_px`` = mean over final increments of the distance to
    the nearest classical (OpenCV) peak along the axis. Lower ⇒ the model's increments
    sit on the same structures a technician's classical detector finds. ``None`` when a
    side has no points. This is the quantitative localisation signal (not "na oko").
    """
    import numpy as np
    by_id = {str(s["image_id"]): s for s in samples}
    rows: list[dict] = []
    for iid, d in axis_data.items():
        s = by_id.get(iid, {})
        finals = d.get("final_axis_pts") or []
        classical = d.get("classical_pts") or []
        mean_dist = None
        if finals and classical:
            fa = np.asarray(finals, dtype=np.float32)
            ca = np.asarray(classical, dtype=np.float32)
            dists = np.sqrt(((fa[:, None, :] - ca[None, :, :]) ** 2).sum(-1))   # (F, C)
            mean_dist = round(float(dists.min(axis=1).mean()), 2)
        rows.append({
            "image_id": iid,
            "true_age": int(s.get("age", 0)),
            "predicted_age": int(s.get("predicted_age", 0)),
            "n_final": len(finals),
            "n_candidates": len(d.get("candidate_pts") or []),
            "n_classical": len(classical),
            "mean_dist_final_to_classical_px": mean_dist,
        })
    return rows


def _step_cards(
    pred_csvs: dict[str, Path],
    cfg_emb,
    image_dir: Path,
    output_dir: Path,
    cond_models: dict[str, tuple],
) -> dict[str, list[Path]]:
    """Generate 6-panel reasoning cards for best/worst predictions of each condition.

    cond_models: cond_key → (cfg, ckpt_path)
    """
    from src.visualization import select_top_k_samples, save_reasoning_cards

    if not Path(image_dir).exists():
        print(f"    [cards] OSTRZEŻENIE: --image-dir nie istnieje: {image_dir} "
              f"— karty zostaną pominięte")
        return {"best": [], "worst": []}, {}

    top_k_best = cfg_emb.inference.increment_samples.top_k_best
    top_k_worst = cfg_emb.inference.increment_samples.top_k_worst
    cards: dict[str, list[Path]] = {"best": [], "worst": []}
    opencv_ref_all: dict = {}   # image_id → interactive OpenCV-reference data (Kierunek A)
    loc_quality: dict[str, list] = {}   # cond_key → per-card localisation stats (Kierunek B)

    for cond_key, pred_csv in pred_csvs.items():
        if not Path(pred_csv).exists():
            continue
        try:
            best, worst = select_top_k_samples(pred_csv, top_k_best, top_k_worst)
        except Exception as e:
            print(f"    [cards] {cond_key}: nie można wczytać predictions.csv ({e})")
            continue

        cfg_cond, ckpt_cond = cond_models.get(cond_key, (None, None))
        if cfg_cond is None or ckpt_cond is None or not ckpt_cond.exists():
            print(f"    [cards] {cond_key}: brak checkpointu — pomijam")
            continue

        cond_dir = output_dir / cond_key
        all_samples = list(best) + list(worst)
        importance_grids, axis_data = _compute_axis_data_for_samples(
            all_samples, image_dir, cfg_cond, ckpt_cond, cond_dir,
        )

        cards_dir = output_dir / "cards" / cond_key
        best_saved = save_reasoning_cards(
            best, image_dir, importance_grids, axis_data,
            cards_dir / "best", "best",
        )
        worst_saved = save_reasoning_cards(
            worst, image_dir, importance_grids, axis_data,
            cards_dir / "worst", "worst",
        )
        cards["best"].extend(best_saved)
        cards["worst"].extend(worst_saved)

        loc_quality[cond_key] = _localization_quality(axis_data, all_samples)

        for iid, d in axis_data.items():
            ref = d.get("opencv_ref")
            if ref and iid not in opencv_ref_all:
                opencv_ref_all[iid] = ref

    # Localisation-quality summary (Kierunek B) — quantitative, saved next to the report.
    try:
        import json
        import numpy as np
        summary: dict = {}
        for ck, rows in loc_quality.items():
            dists = [r["mean_dist_final_to_classical_px"] for r in rows
                     if r["mean_dist_final_to_classical_px"] is not None]
            summary[ck] = {
                "n_cards": len(rows),
                "mean_dist_final_to_classical_px": (round(float(np.mean(dists)), 2)
                                                    if dists else None),
                "mean_n_candidates": (round(float(np.mean([r["n_candidates"] for r in rows])), 2)
                                      if rows else None),
                "per_card": rows,
            }
        (output_dir / "localization_quality.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Localization quality: {output_dir / 'localization_quality.json'}")
    except Exception as e:
        print(f"    [cards] localization_quality błąd: {e}")

    return cards, opencv_ref_all


def _compute_dataset_stats(
    labels_combined_csv: Path,
    active_ptypes: list[str] | None = None,
) -> dict:
    """Compute dataset statistics from labels_combined.csv for Section A.

    ``active_ptypes`` limits which preparation types the report focuses on
    (``["Embedded"]`` for an embedded-only run); ``None`` → both. Also reads a
    sibling ``scan_stats.json`` (written by ``scan_labels``) for the data funnel.
    """
    import json
    import pandas as pd

    SPLITS = ["train", "val", "test"]
    PTYPES = ["Embedded", "NotEmbedded"]

    # Try pipeline output dir first, then project data/ directory
    candidates = [
        labels_combined_csv,
        PROJECT_ROOT / "data" / "labels_combined.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)

    if csv_path is None:
        return {"counts": {}, "orphan_count": "N/A", "age_distributions": {},
                "fish_counts": {}, "age_by_split": {}, "active_ptypes": active_ptypes or PTYPES}

    df = pd.read_csv(csv_path)

    counts: dict = {}
    fish_counts: dict = {}
    age_by_split: dict = {}
    age_distributions: dict = {}
    for ptype in PTYPES:
        sub = df[df["preprocessing_type"] == ptype]
        counts[ptype] = {s: int((sub["split"] == s).sum()) for s in SPLITS}
        # fish (neutral_fish_key) counts per split — an image count double-counts
        # the two sides / preparations of one fish.
        fish_counts[ptype] = {
            s: int(sub.loc[sub["split"] == s, "neutral_fish_key"].nunique())
            for s in SPLITS
        }
        labeled = sub[(sub["split"].notna()) & (sub["age"] >= 0)]
        if not labeled.empty:
            age_distributions[ptype] = labeled["age"].dropna().astype(int).tolist()
            age_by_split[ptype] = {
                s: labeled.loc[labeled["split"] == s, "age"].dropna().astype(int).tolist()
                for s in SPLITS
            }

    orphan_count = int(df["orphan"].sum()) if "orphan" in df.columns else "N/A"

    # Data funnel from scan_stats.json (disk → parsed → labeled → orphans), if present.
    funnel = None
    for cand in (csv_path.parent / "scan_stats.json",
                 PROJECT_ROOT / "data" / "scan_stats.json"):
        if cand.exists():
            try:
                funnel = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                funnel = None
            break

    return {
        "counts": counts,
        "fish_counts": fish_counts,
        "orphan_count": orphan_count,
        "age_distributions": age_distributions,
        "age_by_split": age_by_split,
        "active_ptypes": active_ptypes or PTYPES,
        "funnel": funnel,
    }


def _build_split_lookup(labels_combined_csv: Path) -> dict:
    """Return {image_id -> split} from labels_combined.csv for Section G tile badges."""
    import pandas as pd
    for p in (labels_combined_csv, PROJECT_ROOT / "data" / "labels_combined.csv"):
        if Path(p).exists():
            try:
                df = pd.read_csv(p)
            except (OSError, ValueError):
                return {}
            if "image_id" in df.columns and "split" in df.columns:
                sub = df.dropna(subset=["split"])
                return dict(zip(sub["image_id"].astype(str), sub["split"].astype(str)))
            return {}
    return {}


def _write_pipeline_summary(
    output_dir: Path,
    training_logs: dict[str, list[dict]],
    results_dfs: dict,
    completed_steps: list[str],
) -> None:
    """Zapisuje pipeline_summary.json — szybka weryfikacja wyników bez otwierania HTML."""
    from src.comparison_report import compute_metrics

    training_summary: dict[str, dict] = {}
    for model_key, logs in training_logs.items():
        if not logs:
            training_summary[model_key] = {
                "epochs_completed": 0,
                "best_val_mae": None,
                "final_train_loss": None,
            }
        else:
            # epochs_completed = highest epoch NUMBER, not len(logs): robust even if a
            # legacy train.log ever concatenates runs (11.07 TO-DO Punkt 5).
            training_summary[model_key] = {
                "epochs_completed": int(max(r["epoch"] for r in logs)),
                "best_val_mae":     round(min(r["val_mae"] for r in logs), 4),
                "final_train_loss": round(logs[-1]["train_loss"], 4),
            }

    inference_summary: dict[str, dict] = {}
    for cond_key, df in results_dfs.items():
        if df is None or df.empty:
            inference_summary[cond_key] = None
            continue
        if "age" not in df.columns or "predicted_age" not in df.columns:
            inference_summary[cond_key] = {"n_samples": len(df), "error": "missing columns"}
            continue
        m = compute_metrics(df["age"].values, df["predicted_age"].values)
        inference_summary[cond_key] = {
            "n_samples": int(len(df)),
            "MAE":    round(m["MAE"],    4),
            "RMSE":   round(m["RMSE"],   4),
            "Acc1yr": round(m["Acc1yr"], 4),
            "Bias":   round(m["Bias"],   4),
        }

    summary = {
        "generated_at":    _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps_completed": completed_steps,
        "training":        training_summary,
        "inference":       inference_summary,
    }
    out_path = output_dir / "pipeline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Pipeline summary: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OtolithDino — Embedded vs NotEmbedded pipeline")
    p.add_argument("--image-dir", default="Z:/Photo/Otolithes/HER/Processed")
    p.add_argument("--excel", default="data/analysisWithOtolithPhoto.xlsx")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--base-config", default="configs/config.yaml")
    p.add_argument("--config-embedded", default="configs/config_embedded.yaml",
                   dest="config_embedded")
    p.add_argument("--config-not-embedded", default="configs/config_not_embedded.yaml",
                   dest="config_not_embedded")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rescan", action="store_true",
                   help="Rebuild data/labels_*.csv from scratch (default: reuse "
                        "data/labels_*.csv if present — splits are deterministic)")
    p.add_argument("--embedded-only", action="store_true", dest="embedded_only",
                   help="Only Embedded: train_e + infer_ee + cards + report "
                        "(skip NotEmbedded training and all cross-domain inference)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned steps without executing them")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)

    base_cfg_path = Path(args.base_config)
    cfg_emb = load_merged_config(base_cfg_path, Path(args.config_embedded))
    cfg_notemb = load_merged_config(base_cfg_path, Path(args.config_not_embedded))

    # Must run before the DINOv2 backbone is imported (torch.hub load in _step_train)
    # so XFORMERS_DISABLED takes effect and true CLS attention can be captured.
    from src.utils import configure_attention
    configure_attention(cfg_emb.interpretation.disable_fused_attention)

    # --image-dir is authoritative for EVERY step (scan, train, inference, cards).
    # Without this, train/inference fall back to cfg.data.image_dir from the YAML
    # (e.g. "Z:/..."), which on Linux resolves under the project root → FileNotFound
    # mid-training (see plans and summaries/błąd.md).
    if args.image_dir:
        cfg_emb.data.image_dir = args.image_dir
        cfg_notemb.data.image_dir = args.image_dir

    # Consolidate ALL run artifacts under the dated run dir (11.07 TO-DO Punkt 2):
    # checkpoints/ and logs/ otherwise land in the project root via cfg — force them
    # under output_dir so one folder holds the whole run.
    cfg_emb.training.checkpoint_dir = str((output_dir / "checkpoints" / "embedded").resolve())
    cfg_emb.training.log_dir        = str((output_dir / "logs" / "embedded").resolve())
    cfg_notemb.training.checkpoint_dir = str((output_dir / "checkpoints" / "not_embedded").resolve())
    cfg_notemb.training.log_dir        = str((output_dir / "logs" / "not_embedded").resolve())

    if args.dry_run:
        skip_set: set[str] = set()
        if args.embedded_only:
            skip_set.update({"train_n", "infer_nn", "infer_en", "infer_ne"})
        print("=== DRY RUN — pipeline steps ===")
        for i, step in enumerate(STEPS, 1):
            status = "SKIP" if step in skip_set else "RUN "
            print(f"  {i}. [{status}] {step}")
        return

    # Clean start (11.07 TO-DO Punkt 3): wipe the run dir so no result can be stale.
    if output_dir.exists():
        print(f"[fresh] Czyszczę katalog runu {output_dir} — pełny bieg od zera")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OtolithDino — pipeline Embedded vs NotEmbedded")
    print("=" * 60)

    # Canonical labels live in the stable project data/ dir and are reused across
    # runs (splits are deterministic at seed=42). --rescan forces a rebuild.
    canonical_dir = PROJECT_ROOT / "data"
    run_data_dir = output_dir / "data"
    run_data_dir.mkdir(parents=True, exist_ok=True)

    labels_present = all(
        (canonical_dir / n).exists()
        for n in ("labels_embedded.csv", "labels_not_embedded.csv", "labels_combined.csv")
    )
    if args.rescan or not labels_present:
        print("\n[1/9] SCAN — budowanie labels CSVs")
        _step_scan(args, canonical_dir)
    else:
        print("\n[1/9] SCAN — pominięty (używam istniejących data/labels_*.csv; "
              "--rescan wymusza skan)")

    # Copy canonical labels into the run dir (report lookups + provenance).
    for name in ("labels_combined.csv", "labels_embedded.csv",
                 "labels_not_embedded.csv", "scan_stats.json"):
        src_csv = canonical_dir / name
        if src_csv.exists():
            shutil.copy2(src_csv, run_data_dir / name)
    emb_labels = run_data_dir / "labels_embedded.csv"
    notemb_labels = run_data_dir / "labels_not_embedded.csv"

    # Checkpoint paths follow the (now run-local, absolute) config paths.
    ckpt_emb = Path(cfg_emb.training.checkpoint_dir) / "best.pt"
    ckpt_notemb = Path(cfg_notemb.training.checkpoint_dir) / "best.pt"

    logs_emb: list[dict] = []
    logs_notemb: list[dict] = []

    # --- Steps 2-3: training (always full, never skipped) ---
    print("\n[2/9] TRAIN — Embedded")
    ckpt_emb, logs_emb = _step_train(cfg_emb, emb_labels)
    if args.embedded_only:
        print("\n[3/9] TRAIN NotEmbedded — pominięty (--embedded-only)")
    else:
        print("\n[3/9] TRAIN — NotEmbedded")
        ckpt_notemb, logs_notemb = _step_train(cfg_notemb, notemb_labels)

    # --- Steps 4-7: inference + interpretation + candidates (4 conditions) ---
    conditions = [
        ("infer_ee", "emb_on_emb",          cfg_emb,    ckpt_emb,    emb_labels),
        ("infer_nn", "notemb_on_notemb",     cfg_notemb, ckpt_notemb, notemb_labels),
        ("infer_en", "cross_emb_on_notemb",  cfg_emb,    ckpt_emb,    notemb_labels),
        ("infer_ne", "cross_notemb_on_emb",  cfg_notemb, ckpt_notemb, emb_labels),
    ]
    if args.embedded_only:
        conditions = [c for c in conditions if c[0] == "infer_ee"]

    pred_csvs: dict[str, Path] = {}
    step_nums = {"infer_ee": 4, "infer_nn": 5, "infer_en": 6, "infer_ne": 7}

    for step_name, cond_key, cfg, ckpt, labels_csv in conditions:
        n = step_nums[step_name]
        infer_dir = output_dir / cond_key
        print(f"\n[{n}/9] INFER — {cond_key}")
        _step_infer(cfg, ckpt, labels_csv, infer_dir)
        pred_csvs[cond_key] = infer_dir / "predictions.csv"

    # --- Step 8: increment cards ---
    # Mapping: condition key → (cfg, ckpt_path) used by that model
    cond_models = {
        "emb_on_emb":          (cfg_emb,    ckpt_emb),
        "notemb_on_notemb":    (cfg_notemb, ckpt_notemb),
        "cross_emb_on_notemb": (cfg_emb,    ckpt_emb),
        "cross_notemb_on_emb": (cfg_notemb, ckpt_notemb),
    }

    print("\n[8/9] CARDS — karty rozumowania")
    image_dir = Path(args.image_dir)
    increment_cards, opencv_reference = _step_cards(
        pred_csvs, cfg_emb, image_dir, output_dir, cond_models
    )

    # --- Prepare results_dfs (needed for both report and summary) ---
    import pandas as pd
    results_dfs: dict[str, pd.DataFrame | None] = {}
    for cond_key, csv_path in pred_csvs.items():
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            if "target_age" in df.columns and "age" not in df.columns:
                df = df.rename(columns={"target_age": "age"})
            results_dfs[cond_key] = df
        else:
            results_dfs[cond_key] = None

    # --- Step 9: comparison report ---
    print("\n[9/9] REPORT — raport porównawczy")
    from src.comparison_report import build_comparison_report

    model_info = {
        "backbone": cfg_emb.model.backbone,
        "num_age_classes": cfg_emb.model.num_age_classes,
        "use_metadata": cfg_emb.model.use_metadata,
        "ckpt_embedded": str(ckpt_emb),
        "ckpt_not_embedded": str(ckpt_notemb),
    }

    active_ptypes = ["Embedded"] if args.embedded_only else ["Embedded", "NotEmbedded"]
    dataset_stats = _compute_dataset_stats(
        run_data_dir / "labels_combined.csv", active_ptypes=active_ptypes,
    )

    report_path = output_dir / "comparison_report.html"
    training_logs = {"embedded": logs_emb, "not_embedded": logs_notemb}
    candidate_overlays = _collect_candidate_overlays(output_dir, pred_csvs.keys())
    split_lookup = _build_split_lookup(run_data_dir / "labels_combined.csv")
    build_comparison_report(
        results=results_dfs,
        training_logs=training_logs,
        increment_cards=increment_cards,
        dataset_stats=dataset_stats,
        output_path=report_path,
        model_info=model_info,
        candidate_overlays=candidate_overlays,
        split_lookup=split_lookup,
        opencv_reference=opencv_reference,
    )
    print(f"  Report: {report_path}")

    # --- Pipeline summary (always written) ---
    completed_steps = (["scan", "train_e"]
                       + ([] if args.embedded_only else ["train_n"])
                       + [c[0] for c in conditions] + ["cards", "report"])
    _write_pipeline_summary(
        output_dir=output_dir,
        training_logs={"embedded": logs_emb, "not_embedded": logs_notemb},
        results_dfs=results_dfs,
        completed_steps=completed_steps,
    )

    print("\n=== Pipeline zakończony ===")
    print(f"Raport:          {output_dir / 'comparison_report.html'}")
    print(f"Pipeline summary: {output_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()