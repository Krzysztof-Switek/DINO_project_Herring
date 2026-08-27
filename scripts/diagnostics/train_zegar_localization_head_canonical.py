"""26.08 -- Axis-canonicalized variant of train_zegar_localization_head.py, testing the single
most promising untested lever identified after that script's negative result (see plans and
summaries/26.08_zegar_localization_head_negative_result.md).

Diagnosis that motivated this script: the plain per-patch head's decoded peaks clustered at a
near-fixed SCREEN position (patch rows ~31-34/37, regardless of each image's own axis orientation)
-- consistent with the head exploiting ViT positional-embedding leakage ("where in the frame")
instead of ring texture, given only 28 training images/fold. Every crop so far has been fed to the
backbone in whatever orientation `resolve_and_crop_target_otolith` happened to produce -- "near the
axis" is a DIFFERENT absolute screen region in every image, so a position-based shortcut can never
generalize.

Fix tested here: rotate each crop around its own centroid so the (centroid -> far_edge) reading
axis always points along +x (canonical, angle=0) BEFORE resizing and feeding the backbone. If the
head's shortcut really is "content at a fixed screen position", canonicalizing what "fixed screen
position" MEANS (now consistently = near the reading axis, in every image) turns the same shortcut
from harmful into actively useful, instead of trying to suppress it (which standardization/
regularization/spatial-conv context in the base script's 6 tried configs all failed to do).

Implementation notes:
- Only build_dataset/evaluate_sample are overridden. Everything else (ZegarLocalizationHead,
  centernet_focal_loss, decode_topk_peaks, train_one_fold, make_folds, random_control_baseline) is
  imported unchanged from train_zegar_localization_head.py -- same head architecture, same loss,
  same decode, same training loop, ONLY the input image's orientation differs. Keeps this a clean
  single-variable ablation.
- Rotation is computed via cv2.getRotationMatrix2D(center=centroid, angle=atan2(dy,dx) degrees,
  scale=1.0) on the CROP-LOCAL image (before resize to IMAGE_SIZE), applied with cv2.warpAffine
  (borderValue=0, matching the already-zeroed masked background). The exact same affine matrix is
  applied to annotation points (for building the training target) via cv2.transform. At eval time,
  decoded peaks are mapped back through the INVERSE affine (cv2.invertAffineTransform) before
  calling ev.point_to_axis_t -- axis_info itself is never rotated, so the existing GT/matching code
  in expert_annotation_eval.py needs zero changes (point_to_axis_t's scalar projection is rotation-
  invariant as long as points and axis stay in the same frame, which the inverse mapping restores).
- Augmentations become 2, not 4: identity and vertical-flip-only. Horizontal flip (or the h+v
  combo) would reverse the just-canonicalized axis direction, undoing the whole point of this
  experiment -- dropped deliberately, not an oversight. Vertical flip mirrors top/bottom around the
  now-horizontal axis, which stays a valid, axis-direction-preserving augmentation.

Usage:
    python scripts/diagnostics/train_zegar_localization_head_canonical.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DIAG_DIR = PROJECT_ROOT / "scripts" / "diagnostics"
if str(DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(DIAG_DIR))

import json
import math
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage

import expert_annotation_eval as ev
import train_zegar_localization_head as base
from src.otolith_axis import detect_axis, apply_background_mask
from src.inference import load_model_from_checkpoint
from src.model import EMBED_DIMS
from scripts.run_pipeline import load_merged_config

OUTPUT_DIR = base.PROJECT_ROOT / "outputs" / "26.08_zegar_localization_head_canonical"

# Only identity + vertical flip -- see module docstring for why horizontal flip is dropped here.
_AUGMENTATIONS = [(False, False), (False, True)]


def _canonicalizing_matrix(centroid: tuple[float, float], far_edge: tuple[float, float]) -> np.ndarray:
    """2x3 affine that rotates around ``centroid`` so (far_edge - centroid) points along +x."""
    cx, cy = centroid
    fx, fy = far_edge
    angle_deg = math.degrees(math.atan2(fy - cy, fx - cx))
    return cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)


def build_dataset_canonical(model, ann: pd.DataFrame, seg_params: dict) -> dict:
    """Same contract/return shape as train_zegar_localization_head.build_dataset, but the image
    (and the points used to build training targets) are rotated to the canonical axis frame before
    resizing. ``axis_info`` stored per sample is the ORIGINAL (unrotated) one, unchanged -- needed
    intact for evaluate_sample_canonical's inverse-mapping + ev.point_to_axis_t.
    """
    samples = sorted(ann["Sample"].unique())
    data: dict = {}
    n_failed = 0
    for i, sample in enumerate(samples, 1):
        image_id = f"{sample}.jpg"
        sub = ann[ann.Sample == sample]
        mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
        raw_rgb = np.array(PILImage.open(ev.IMAGE_DIR / image_id).convert("RGB"), dtype=np.uint8)
        cropped, x0, y0, _used_second = ev.resolve_and_crop_target_otolith(raw_rgb, mean_xy, seg_params)
        if cropped is None:
            print(f"  [{i}/{len(samples)}] {sample}: crop failed, skipping")
            n_failed += 1
            continue
        axis_info = detect_axis(cropped, seg_params)
        if axis_info is None:
            print(f"  [{i}/{len(samples)}] {sample}: axis detection failed, skipping")
            n_failed += 1
            continue
        masked = apply_background_mask(cropped, axis_info["mask"])
        crop_h, crop_w = masked.shape[:2]

        M = _canonicalizing_matrix(axis_info["centroid"], axis_info["far_edge"])
        rotated = cv2.warpAffine(masked, M, (crop_w, crop_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

        # Sanity check (cheap, always on): far_edge should now project to ~(cx + length, cy).
        cx, cy = axis_info["centroid"]
        fx_rot, fy_rot = (M @ np.array([axis_info["far_edge"][0], axis_info["far_edge"][1], 1.0]))[:2]
        assert abs(fy_rot - cy) < 1.0, f"{sample}: canonicalization sanity check failed (dy={fy_rot - cy:.2f})"

        resized = PILImage.fromarray(rotated).resize((base.IMAGE_SIZE, base.IMAGE_SIZE), PILImage.BILINEAR)
        scale_x = base.IMAGE_SIZE / crop_w
        scale_y = base.IMAGE_SIZE / crop_h

        # Rebase annotations: original crop-local -> rotated crop-local (via M) -> resized-pixel.
        pts_resized = []
        for _, row in sub.iterrows():
            xl, yl = row.x - x0, row.y - y0
            xr, yr = (M @ np.array([xl, yl, 1.0]))[:2]
            pts_resized.append((xr * scale_x, yr * scale_y))

        tokens_by_aug: dict[tuple[bool, bool], torch.Tensor] = {}
        heat_by_aug: dict[tuple[bool, bool], np.ndarray] = {}
        base_tensor = base._NORMALIZE(resized)
        for flip_h, flip_v in _AUGMENTATIONS:
            t = base_tensor
            pts = pts_resized
            if flip_h:
                t = torch.flip(t, dims=[-1])
                pts = [((base.IMAGE_SIZE - 1) - x, y) for x, y in pts]
            if flip_v:
                t = torch.flip(t, dims=[-2])
                pts = [(x, (base.IMAGE_SIZE - 1) - y) for x, y in pts]
            with torch.no_grad():
                tokens = model.get_patch_tokens(t.unsqueeze(0).to(base.DEVICE))[0].cpu()
            peaks_rc = [(y / base.PATCH_SIZE, x / base.PATCH_SIZE) for x, y in pts]
            heat = base.gaussian_heatmap(peaks_rc, base.HP, base.WP, base.GAUSSIAN_SIGMA_PATCHES)
            tokens_by_aug[(flip_h, flip_v)] = tokens
            heat_by_aug[(flip_h, flip_v)] = heat

        data[sample] = {
            "axis_info": axis_info, "x0": x0, "y0": y0,
            "scale_x": scale_x, "scale_y": scale_y,
            "M": M, "M_inv": cv2.invertAffineTransform(M),
            "tokens": tokens_by_aug, "heatmaps": heat_by_aug,
        }
        print(f"  [{i}/{len(samples)}] {sample}: OK ({len(pts_resized)} points)")
    print(f"Built {len(data)}/{len(samples)} samples ({n_failed} failed)")
    return data


def evaluate_sample_canonical(head, sample_data: dict, ann: pd.DataFrame, sample: str,
                               max_gap_t: float) -> Optional[dict]:
    axis_info = sample_data["axis_info"]

    sub = ann[ann.Sample == sample]
    t_kk = sub[sub.annotator == "KK"].apply(lambda r: ev.point_to_axis_t(r.x, r.y, axis_info), axis=1).tolist()
    t_ss = sub[sub.annotator == "SS"].apply(lambda r: ev.point_to_axis_t(r.x, r.y, axis_info), axis=1).tolist()
    pairs, _u_kk, _u_ss = ev.match_t_values(t_kk, t_ss, max_gap_t)
    gt_t = [(t_kk[i] + t_ss[j]) / 2.0 for i, j in pairs]
    if not gt_t:
        return None

    tokens = sample_data["tokens"][(False, False)].unsqueeze(0)   # identity (non-flipped) view
    with torch.no_grad():
        heat = head(tokens)[0].numpy()
    peaks_rc = base.decode_topk_peaks(heat, len(gt_t))

    scale_x, scale_y = sample_data["scale_x"], sample_data["scale_y"]
    M_inv = sample_data["M_inv"]
    t_model = []
    for r, c in peaks_rc:
        x_resized = (c + 0.5) * base.PATCH_SIZE
        y_resized = (r + 0.5) * base.PATCH_SIZE
        x_rot_crop = x_resized / scale_x
        y_rot_crop = y_resized / scale_y
        # Map back through the inverse canonicalization rotation into axis_info's original frame.
        x_crop, y_crop = (M_inv @ np.array([x_rot_crop, y_rot_crop, 1.0]))[:2]
        t_model.append(ev.point_to_axis_t(x_crop, y_crop, axis_info))
    if not t_model:
        return None
    length = axis_info["length_px"]
    d = np.abs(np.asarray(t_model)[:, None] - np.asarray(gt_t)[None, :]) * length
    return {
        "sample": sample, "n_model_peaks": len(t_model), "n_gt": len(gt_t),
        "model_vs_gt_axis_px": float(d.min(axis=1).mean()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(base.SEED)
    np.random.seed(base.SEED)

    print("=" * 70)
    print("ZEGAR localization head -- CANONICAL AXIS variant (26.08)")
    print("=" * 70)

    print("\n[1/5] Wczytywanie adnotacji...")
    ann = ev.load_expert_annotations()
    ev.validate_annotation_bounds(ann, ev.IMAGE_DIR)

    print("\n[2/5] Wczytanie zamrożonego backbone'u (Run N)...")
    cfg = load_merged_config(base.REFERENCE_CONFIG, None)
    if not base.REFERENCE_CKPT.exists():
        sys.exit(f"Brak checkpointu: {base.REFERENCE_CKPT}")
    model = load_model_from_checkpoint(cfg, base.REFERENCE_CKPT)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    embed_dim = EMBED_DIMS.get(cfg.model.backbone, 384)
    seg_params = cfg.segmentation.as_params()
    max_gap_t = cfg.candidates.min_peak_distance / ev.N_SAMPLES_AXIS

    print(f"\n[3/5] Ekstrakcja cech, KANONIZACJA OSI (42 obrazy x {len(_AUGMENTATIONS)} augmentacje)...")
    dataset = build_dataset_canonical(model, ann, seg_params)
    del model

    samples = sorted(dataset.keys())
    folds = base.make_folds(samples, base.N_FOLDS, base.SEED)

    print(f"\n[4/5] Trening {base.N_FOLDS}-fold cross-walidacji ({base.EPOCHS} epok/fold)...")
    all_results = []
    for fold_idx, held_out in enumerate(folds):
        train_samples = [s for s in samples if s not in held_out]
        print(f"\n  --- Fold {fold_idx + 1}/{base.N_FOLDS}: {len(train_samples)} trening, "
              f"{len(held_out)} held-out ({held_out}) ---")
        head = base.train_one_fold(dataset, train_samples, embed_dim)
        torch.save(head.state_dict(), OUTPUT_DIR / f"head_fold{fold_idx}.pt")
        for s in held_out:
            r = evaluate_sample_canonical(head, dataset[s], ann, s, max_gap_t)
            if r is not None:
                r["fold"] = fold_idx
                all_results.append(r)
                print(f"    {s}: model_vs_gt_axis_px={r['model_vs_gt_axis_px']:.2f}px "
                      f"(n_pred={r['n_model_peaks']}, n_gt={r['n_gt']})")

    print("\n[5/5] Wyniki zbiorcze...")
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "per_sample_results.csv", index=False)
    random_px = base.random_control_baseline(dataset, ann, samples, max_gap_t, seed=base.SEED)
    aggregate = {
        "n_samples_scored": int(len(df)),
        "n_folds": base.N_FOLDS,
        "model_vs_gt_axis_px_mean": float(df["model_vs_gt_axis_px"].mean()),
        "model_vs_gt_axis_px_median": float(df["model_vs_gt_axis_px"].median()),
        "model_vs_gt_axis_px_std": float(df["model_vs_gt_axis_px"].std()),
        "random_control_axis_px": random_px,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    rand_str = f"{random_px:.1f}px" if random_px is not None else "n/a"
    print(f"\nPorównanie: Run N = 27.88px, klasyka = 122.92px, podłoga szumu = 4.28px, "
          f"kontrola losowa = {rand_str}, wariant bez kanonizacji (26.08) = 503.2px.")
    print(f"Zapisano: {OUTPUT_DIR / 'metrics.json'}, {OUTPUT_DIR / 'per_sample_results.csv'}")


if __name__ == "__main__":
    main()
