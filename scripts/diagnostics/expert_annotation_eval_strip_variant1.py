"""04.09 — Wariant 1 (multi-wycinek plan, `plans and summaries/03.09_multi_wycinek_plan.md`):
inference-time selection among ``k`` candidate reading-axis strips, reusing the Track A checkpoint
UNCHANGED (no retraining). Directly follows up on `04.09_pasek_dendrologiczny_wyniki.md`, which
scored Track A's single-wycinek (``k=1``) result at 66.66px/42 images — this script asks how much of
the gap to Run N (22.8px) is recoverable purely by picking, per image, the best of several candidate
reading directions instead of committing to the one deterministic axis `find_reading_edge` returns.

Every variant below is scored against the SAME canonical ground-truth axis (``axis_candidates[0]`` —
verified in the multi-wycinek plan to be byte-identical to the single-result ``find_reading_edge``/
``detect_axis`` output), so peak positions decoded from any candidate strip are re-projected back onto
this one fixed measurement axis before computing ``axis_dist``. Concretely, every selection criterion
below just picks an index into an already-computed, per-candidate ``axis_dist_all`` array (the "true"
position quality of that candidate, always measured with ``k=n_gt`` peaks — the project's established
methodology throughout this whole ZEGAR-eval series) — so adding a new criterion never needs a new
forward pass, only a new way to pick an index.

History (same day, three rounds):

1. ``k1`` (today's deterministic single axis, reproduces the already-scored 66.66px/42 baseline) vs
   ``predicted``/``oracle`` (argmin ``|density integral - target|``, at ``k=CORAL-predicted age`` /
   ``k=true ring count`` respectively — ``select_density_candidate``'s own training-time criterion,
   `src/model.py`) vs a diagnostic-only ``position_oracle`` (always picks whichever candidate truly
   minimises ``axis_dist`` — unreachable at real inference, an upper bound). Result:
   `predicted`/`oracle` both WORSE than `k1` (statistically indistinguishable from a uniformly random
   pick, 86.2px), while `position_oracle` (24.4px) nearly ties Run N — proving the extra candidate axes
   carry strong position signal that raw integral-matching simply fails to use.
2. ``predicted_sharp``/``oracle_sharp`` — same targets, but scored by `concentration_score` (this
   file), the same top-k "sharp, well-separated peaks" formula `density_count_loss`'s `L_conc` term
   already trains toward, evaluated here as a selection score instead of a loss. Result: clearly better
   than count-matching (`predicted_sharp` 60.0px beats `k1`), but the improvement is concentrated in
   avoiding the count criterion's worst misses, not in genuinely better candidate identification (still
   only ~26% hit rate against `position_oracle`'s actual pick).
3. **This round** — every other locally-testable criterion/combination, plus a small trained ranker:
   - ``predicted_gap``/``oracle_gap`` — `gap_regularity_score`: coefficient of variation of the spacing
     between decoded peaks. A geometric prior independent of both count and per-cell sharpness.
   - ``predicted_combo``/``oracle_combo`` — `rank_combine` of all three criteria (count + sharpness +
     gap regularity), scale-free (average rank, not a weighted sum of incompatible units).
   - ``predicted_weighted``/``oracle_weighted`` — the LITERAL `density_count_loss` formula
     (`smooth_l1(integral - target) + conc_weight * concentration_score`, using the run's own
     `cfg.model.density_conc_weight`) evaluated as a selection score — the 2-way combination closest to
     what the density head was actually optimised against during training.
   - ``ranker_predicted``/``ranker_oracle`` — a tiny ridge-regression ranker (`fit_ridge`/
     `predict_ridge`, plain numpy — no sklearn dependency), trained via leave-one-IMAGE-out
     cross-validation (only ~42 images / ~200 candidate rows total — anything heavier would just
     memorise) to predict `log1p(axis_dist)` from per-candidate features, then picks `argmin` per image.
     `ranker_predicted` uses only features available at real inference (integral, `concentration_score`/
     `gap_regularity_score` at `k=predicted_age`, density-map max/mean/std, candidate index — the last
     one matters because `detect_axis_candidates` already ranks candidates by ring-richness, so index 0
     carries real prior information `k1` alone already exploits); `ranker_oracle` additionally gets
     `k=n_gt`-based features, the learned analogue of the other `oracle_*` variants.

Reuses everything already built/verified for the k=1 script (`expert_annotation_eval_strip.py`) and
for Track B training (`detect_axis_candidates`, `src/model.py::select_density_candidate`'s selection
formula) — no new forward-pass primitives, only new post-hoc scoring of already-computed density maps.

Usage:
    python scripts/diagnostics/expert_annotation_eval_strip_variant1.py
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
from src.dataset import build_transforms, decode_age_ordinal
from src.inference import load_model_from_checkpoint
from src.otolith_axis import apply_background_mask, detect_axis_candidates
from src.strip_extraction import extract_strip_from_axis_info
from scripts.run_pipeline import load_merged_config

# ---------------------------------------------------------------------------
# Same temp-swap-and-revert workflow as every other REFERENCE_CKPT/REFERENCE_CONFIG script in
# this project — point at the run to score, run this script, revert after (`git diff` should stay
# clean).
# ---------------------------------------------------------------------------

REFERENCE_CONFIG = PROJECT_ROOT / "configs" / "config_strip_a.yaml"
REFERENCE_CKPT = (PROJECT_ROOT / "outputs" / "data" / "REPLACE_WITH_RUN_TAG" / "checkpoints"
                  / "embedded" / "best.pt")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "04.09_zegar_strip_a_variant1_eval"

ZEGAR_SPLIT_PATH = PROJECT_ROOT / "data" / "zegar_semi_weak_split.json"

MIN_DIST_PATCHES = 1.5

# Number of candidate reading axes per image + minimum angular separation between them — same
# values as Track B's own `configs/config_strip_b.yaml` (`multi_wycinek_k`/
# `multi_wycinek_min_angle_sep_deg`), reused here rather than re-guessed so this eval-time
# mechanism and the (separately built) training-time mechanism explore the same candidate geometry.
K_CANDIDATES = 5
MIN_ANGLE_SEP_DEG = 8.0

# Ridge-regression regularisation for the trainable ranker — a single unvalidated first guess
# (standardized features, ~40 training images per fold), not swept/tuned. This whole ranker is an
# exploratory diagnostic, not a candidate for deployment as-is.
RIDGE_ALPHA = 1.0


# ---------------------------------------------------------------------------
# Per-candidate scoring criteria — each takes a density grid (+ a target `k` where relevant) and
# returns a scalar where LOWER is always better, so every criterion/combination is selected via a
# plain argmin over candidates.
# ---------------------------------------------------------------------------

def concentration_score(density: np.ndarray, k: float) -> float:
    """Lower is better. Same top-k concentration formula as ``src.model.density_count_loss``'s
    ``L_conc`` term (the ``k`` strongest cells should be near 1, the rest near 0 — the same
    "sharp, well-separated peaks" objective the density head was actually trained against),
    evaluated here as a per-candidate SELECTION score instead of a differentiable training loss.

    Unlike raw-integral matching (04.09's first-round `predicted`/`oracle` variants, both scored
    worse than no selection at all), this asks about the SHAPE of the map, not just whether its sum
    happens to equal ``k`` — a diffuse blob that integrates to the right count still scores poorly
    here, because none of its cells are actually close to 0 or 1.
    """
    flat = density.ravel()
    k_int = int(round(np.clip(k, 0, flat.size)))
    if k_int <= 0:
        return float(np.mean(flat ** 2))
    sorted_d = np.sort(flat)[::-1]
    on, off = sorted_d[:k_int], sorted_d[k_int:]
    l_on = float(np.mean((1.0 - on) ** 2)) if len(on) else 0.0
    l_off = float(np.mean(off ** 2)) if len(off) else 0.0
    return l_on + l_off


def gap_regularity_score(peaks_rc: list[tuple[int, int]]) -> float:
    """Lower is better. Coefficient of variation (std/mean) of the gaps between consecutive
    decoded peaks, sorted by patch column along the strip's OWN length axis — needs no projection
    onto the canonical measurement axis, since regularity is invariant to which axis measures it.

    0.0 (neutral — neither rewarded nor penalised) when fewer than 2 peaks are available, since
    "irregular spacing" isn't a meaningful concept for 0-1 peaks and young fish with few rings
    shouldn't be penalised just for having little to compare.

    A geometric prior orthogonal to the other two criteria: real growth rings are drawn from a
    roughly consistent (if age-decreasing) radial spacing, so wildly uneven gaps are a signal of a
    bad candidate even when its total count is right (satisfies `predicted`/`oracle`) and even when
    its top-k values are individually sharp (satisfies `*_sharp`).
    """
    if len(peaks_rc) < 2:
        return 0.0
    cols = sorted(c for _, c in peaks_rc)
    gaps = np.diff(cols).astype(float)
    mean_gap = gaps.mean()
    if mean_gap <= 0:
        return 0.0
    return float(gaps.std() / mean_gap)


def smooth_l1(x: float) -> float:
    """Scalar version of ``torch.nn.functional.smooth_l1_loss`` — matches
    ``density_count_loss``'s own count term exactly, so `predicted_weighted`/`oracle_weighted`
    below reproduce that training loss's full two-term formula as a selection score."""
    ax = abs(x)
    return 0.5 * ax * ax if ax < 1.0 else ax - 0.5


def rank_combine(*score_lists: list[float]) -> list[float]:
    """Average rank (1=best) across one or more per-candidate score lists (each lower=better) —
    a scale-free way to combine criteria measured in incompatible units (count mismatch in
    ring-count units, concentration in mean-squared-probability units, gap regularity as a
    coefficient of variation) without needing to calibrate relative weights between them."""
    n = len(score_lists[0])
    ranks_sum = np.zeros(n)
    for scores in score_lists:
        order = np.argsort(scores)
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        ranks_sum += ranks
    return list(ranks_sum / len(score_lists))


# ---------------------------------------------------------------------------
# Tiny trainable ranker — plain numpy ridge regression (no sklearn dependency), leave-one-IMAGE-out
# cross-validation. Only ~42 images / ~200 candidate rows total exist for this — CV must hold out
# whole images (not individual rows), or a candidate would leak information about its own image's
# sibling candidates into training.
# ---------------------------------------------------------------------------

def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    reg = alpha * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0  # never regularise the bias term
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)


def predict_ridge(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ w


def loo_rank_select(features: list[list[float]], image_of_row: list[int],
                     labels: list[float], candidates_by_image: dict[int, list[int]],
                     alpha: float = RIDGE_ALPHA) -> dict[int, int]:
    """For each image, fit ridge on every OTHER image's candidate rows (features standardised
    using only those training rows — no leakage), predict ``log1p(axis_dist)`` for this image's
    own candidates, and pick the argmin. Returns ``{image_idx: chosen candidate's LOCAL index}``
    (0-based within that image's own candidate list, directly comparable to every other variant's
    ``idx_*``)."""
    X_all = np.array(features, dtype=float)
    y_all = np.array(labels, dtype=float)
    img_arr = np.array(image_of_row)
    chosen: dict[int, int] = {}
    for img_idx, positions in candidates_by_image.items():
        train_mask = img_arr != img_idx
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
        sigma[sigma == 0] = 1.0
        w = fit_ridge((X_train - mu) / sigma, y_train, alpha)
        X_test = (X_all[positions] - mu) / sigma
        preds = predict_ridge(w, X_test)
        chosen[img_idx] = int(np.argmin(preds))  # LOCAL index — `positions` is candidate-ordered
    return chosen


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ZEGAR — wariant paskowy, Wariant 1: wybór przy inferencji (04.09)")
    print("=" * 70)

    print("\n[1/7] Wczytywanie adnotacji...")
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
    square_transform = build_transforms(cfg.data.image_size, split="test")
    conc_weight = float(getattr(cfg.model, "density_conc_weight", 1.0))

    print("\n[2/7] Wczytanie modelu (checkpoint paskowy, Track A)...")
    if not REFERENCE_CKPT.exists():
        sys.exit(f"Brak checkpointu: {REFERENCE_CKPT}")
    model = load_model_from_checkpoint(cfg, REFERENCE_CKPT)
    model.eval()

    print("\n[3/7] Wczytanie podziału held-out ZEGAR (dla wtórnego porównania z Opcją A)...")
    heldout_ids: set[str] = set()
    if ZEGAR_SPLIT_PATH.exists():
        split_data = json.loads(ZEGAR_SPLIT_PATH.read_text(encoding="utf-8"))
        heldout_ids = set(split_data.get("held_out", []))
        print(f"  {len(heldout_ids)} obrazów held-out wczytanych z {ZEGAR_SPLIT_PATH.name}")
    else:
        print(f"  [ostrzeżenie] brak {ZEGAR_SPLIT_PATH} — wynik na 12 held-out nie zostanie "
              f"policzony, tylko pełne 42")

    print(f"\n[4/7] Generowanie {K_CANDIDATES} kandydackich pasków/obraz, inferencja, cechy...")
    records: list[dict] = []
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

        axis_candidates = detect_axis_candidates(
            cropped, seg_params=seg_params, nucleus_method=cfg.segmentation.nucleus_method,
            axis_method=cfg.segmentation.axis_method, k=K_CANDIDATES,
            min_angle_sep_deg=MIN_ANGLE_SEP_DEG,
        )
        if not axis_candidates:
            print(f"  [{i}/{len(samples)}] {sample}: oś nie policzona, pomijam")
            n_seg_failed += 1
            continue
        axis0 = axis_candidates[0]  # canonical axis — identical to find_reading_edge's own result
        mask = axis0["mask"]

        # GT positions live in axis0's t-space — the ONE fixed measurement axis every candidate's
        # decoded peaks get re-projected onto below, so every variant stays directly comparable.
        t_kk = [ev.point_to_axis_t(x, y, axis0) for x, y in
               zip(sub[sub.annotator == "KK"].x - x0, sub[sub.annotator == "KK"].y - y0)]
        t_ss = [ev.point_to_axis_t(x, y, axis0) for x, y in
               zip(sub[sub.annotator == "SS"].x - x0, sub[sub.annotator == "SS"].y - y0)]
        pairs, _u_kk, _u_ss = ev.match_t_values(t_kk, t_ss, max_gap_t)
        gt_t = [(t_kk[a] + t_ss[b]) / 2.0 for a, b in pairs]
        if not gt_t:
            print(f"  [{i}/{len(samples)}] {sample}: brak dopasowanych par GT (KK/SS), pomijam")
            continue
        n_gt = len(gt_t)
        length_px = axis0["length_px"]
        kk_vs_ss_axis = ev.axis_dist(t_kk, t_ss, length_px)

        # CORAL-predicted age from the model's own square branch — ZEGAR images aren't part of the
        # main labelled dataset, so there is no pre-computed prediction to read off anywhere.
        masked_square = apply_background_mask(cropped, mask)
        square_tensor = square_transform(PILImage.fromarray(masked_square))
        with torch.no_grad():
            square_out = model(square_tensor.unsqueeze(0))
        predicted_age = int(decode_age_ordinal(square_out["coral_logits"])[0].item())

        density_grids, integrals, M_list = [], [], []
        for axis_info in axis_candidates:
            strip_rgb, M = extract_strip_from_axis_info(
                cropped, mask, axis_info, strip_length_px, strip_width_px,
            )
            strip_tensor = strip_transform(PILImage.fromarray(strip_rgb))
            with torch.no_grad():
                density_grid = model.get_density_probs(
                    strip_tensor.unsqueeze(0), patch_grid=(h_patches, w_patches),
                )[0].numpy()
            density_grids.append(density_grid)
            integrals.append(float(density_grid.sum()))
            M_list.append(M)

        # Per-candidate features/criteria — no further forward passes, everything below is
        # cheap numpy post-processing of the already-computed density maps.
        conc_pred = [concentration_score(g, predicted_age) for g in density_grids]
        conc_oracle = [concentration_score(g, n_gt) for g in density_grids]
        peaks_pred_list = [decode_topk_peaks(g, predicted_age, min_dist_patches=MIN_DIST_PATCHES)
                            for g in density_grids]
        peaks_oracle_list = [decode_topk_peaks(g, n_gt, min_dist_patches=MIN_DIST_PATCHES)
                              for g in density_grids]
        gap_pred = [gap_regularity_score(p) for p in peaks_pred_list]
        gap_oracle = [gap_regularity_score(p) for p in peaks_oracle_list]
        max_d = [float(g.max()) for g in density_grids]
        mean_d = [float(g.mean()) for g in density_grids]
        std_d = [float(g.std()) for g in density_grids]

        # The one "true quality" measurement, unchanged from the first two rounds: decode
        # k=n_gt peaks, project back onto the canonical axis0, compare to gt_t. Every selection
        # criterion below only ever picks an INDEX into this precomputed array.
        def _true_quality(idx: int) -> float:
            M_inv = cv2.invertAffineTransform(M_list[idx])
            t_model = []
            for r, c in peaks_oracle_list[idx]:
                x_strip = (c + 0.5) * cfg.data.patch_size
                y_strip = (r + 0.5) * cfg.data.patch_size
                x_crop, y_crop = (M_inv @ np.array([x_strip, y_strip, 1.0]))[:2]
                t_model.append(ev.point_to_axis_t(x_crop, y_crop, axis0))
            return ev.axis_dist(t_model, gt_t, length_px)

        axis_dist_all = [_true_quality(k) for k in range(len(axis_candidates))]

        records.append({
            "Sample": sample, "image_id": image_id,
            "is_held_out": sample in heldout_ids,
            "n_gt": n_gt, "n_candidates": len(axis_candidates), "predicted_age": predicted_age,
            "kk_vs_ss_axis_px": kk_vs_ss_axis,
            "integrals": integrals, "conc_pred": conc_pred, "conc_oracle": conc_oracle,
            "gap_pred": gap_pred, "gap_oracle": gap_oracle,
            "max_d": max_d, "mean_d": mean_d, "std_d": std_d,
            "axis_dist_all": axis_dist_all,
        })
        print(f"  [{i}/{len(samples)}] {sample}: n_candidates={len(axis_candidates)} "
              f"n_gt={n_gt} pred_age={predicted_age} "
              f"axis_dist_all=[{', '.join(f'{v:.0f}' for v in axis_dist_all)}] "
              f"(held_out={sample in heldout_ids})")

    print(f"\n  {len(records)}/{len(samples)} obrazów ocenionych "
          f"({n_seg_failed} pominiętych z powodu segmentacji/osi)")

    # -----------------------------------------------------------------------
    print("\n[5/7] Warianty bez uczenia: count / sharp / gap / combo / ważona strata...")
    for rec in records:
        integrals, conc_pred, conc_oracle = rec["integrals"], rec["conc_pred"], rec["conc_oracle"]
        gap_pred, gap_oracle = rec["gap_pred"], rec["gap_oracle"]
        n_gt, predicted_age = rec["n_gt"], rec["predicted_age"]
        axis_dist_all = rec["axis_dist_all"]

        def pick(scores: list[float]) -> tuple[int, float]:
            idx = int(np.argmin(scores))
            return idx, axis_dist_all[idx]

        count_pred_scores = [abs(v - predicted_age) for v in integrals]
        count_oracle_scores = [abs(v - n_gt) for v in integrals]
        idx_pred, axis_pred = pick(count_pred_scores)
        idx_oracle, axis_oracle = pick(count_oracle_scores)
        idx_pred_sharp, axis_pred_sharp = pick(conc_pred)
        idx_oracle_sharp, axis_oracle_sharp = pick(conc_oracle)
        idx_pred_gap, axis_pred_gap = pick(gap_pred)
        idx_oracle_gap, axis_oracle_gap = pick(gap_oracle)

        combo_pred_scores = rank_combine(count_pred_scores, conc_pred, gap_pred)
        combo_oracle_scores = rank_combine(count_oracle_scores, conc_oracle, gap_oracle)
        idx_pred_combo, axis_pred_combo = pick(combo_pred_scores)
        idx_oracle_combo, axis_oracle_combo = pick(combo_oracle_scores)

        weighted_pred_scores = [smooth_l1(iv - predicted_age) + conc_weight * cp
                                 for iv, cp in zip(integrals, conc_pred)]
        weighted_oracle_scores = [smooth_l1(iv - n_gt) + conc_weight * co
                                   for iv, co in zip(integrals, conc_oracle)]
        idx_pred_weighted, axis_pred_weighted = pick(weighted_pred_scores)
        idx_oracle_weighted, axis_oracle_weighted = pick(weighted_oracle_scores)

        idx_pos_oracle = int(np.argmin(axis_dist_all))

        rec.update({
            "idx_k1": 0, "idx_predicted": idx_pred, "idx_oracle": idx_oracle,
            "idx_predicted_sharp": idx_pred_sharp, "idx_oracle_sharp": idx_oracle_sharp,
            "idx_predicted_gap": idx_pred_gap, "idx_oracle_gap": idx_oracle_gap,
            "idx_predicted_combo": idx_pred_combo, "idx_oracle_combo": idx_oracle_combo,
            "idx_predicted_weighted": idx_pred_weighted, "idx_oracle_weighted": idx_oracle_weighted,
            "idx_position_oracle": idx_pos_oracle,
            "axis_dist_k1_px": axis_dist_all[0],
            "axis_dist_predicted_px": axis_pred, "axis_dist_oracle_px": axis_oracle,
            "axis_dist_predicted_sharp_px": axis_pred_sharp, "axis_dist_oracle_sharp_px": axis_oracle_sharp,
            "axis_dist_predicted_gap_px": axis_pred_gap, "axis_dist_oracle_gap_px": axis_oracle_gap,
            "axis_dist_predicted_combo_px": axis_pred_combo, "axis_dist_oracle_combo_px": axis_oracle_combo,
            "axis_dist_predicted_weighted_px": axis_pred_weighted,
            "axis_dist_oracle_weighted_px": axis_oracle_weighted,
            "axis_dist_position_oracle_px": axis_dist_all[idx_pos_oracle],
        })

    # -----------------------------------------------------------------------
    print("\n[6/7] Trenowalny ranker (grzbietowa regresja, leave-one-obraz-out)...")
    feature_rows_pred: list[list[float]] = []
    feature_rows_oracle: list[list[float]] = []
    labels: list[float] = []
    image_of_row: list[int] = []
    candidates_by_image: dict[int, list[int]] = {}
    row = 0
    for img_idx, rec in enumerate(records):
        positions = []
        for c in range(rec["n_candidates"]):
            feat_pred = [rec["integrals"][c], rec["conc_pred"][c], rec["gap_pred"][c],
                         rec["max_d"][c], rec["mean_d"][c], rec["std_d"][c], float(c)]
            feat_oracle = feat_pred + [rec["conc_oracle"][c], rec["gap_oracle"][c],
                                        abs(rec["integrals"][c] - rec["n_gt"])]
            feature_rows_pred.append(feat_pred)
            feature_rows_oracle.append(feat_oracle)
            labels.append(float(np.log1p(rec["axis_dist_all"][c])))
            image_of_row.append(img_idx)
            positions.append(row)
            row += 1
        candidates_by_image[img_idx] = positions

    chosen_pred = loo_rank_select(feature_rows_pred, image_of_row, labels, candidates_by_image)
    chosen_oracle = loo_rank_select(feature_rows_oracle, image_of_row, labels, candidates_by_image)
    for img_idx, rec in enumerate(records):
        idx_rp, idx_ro = chosen_pred[img_idx], chosen_oracle[img_idx]
        rec["idx_ranker_predicted"] = idx_rp
        rec["idx_ranker_oracle"] = idx_ro
        rec["axis_dist_ranker_predicted_px"] = rec["axis_dist_all"][idx_rp]
        rec["axis_dist_ranker_oracle_px"] = rec["axis_dist_all"][idx_ro]
    print(f"  Ranker wytrenowany/oceniony na {len(records)} obrazach "
          f"(leave-one-image-out, {row} wierszy kandydatów łącznie, alpha={RIDGE_ALPHA})")

    # -----------------------------------------------------------------------
    print("\n[7/7] Wyniki zbiorcze i zapis...")
    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    VARIANT_COLS = (
        "axis_dist_k1_px",
        "axis_dist_predicted_px", "axis_dist_oracle_px",
        "axis_dist_predicted_sharp_px", "axis_dist_oracle_sharp_px",
        "axis_dist_predicted_gap_px", "axis_dist_oracle_gap_px",
        "axis_dist_predicted_combo_px", "axis_dist_oracle_combo_px",
        "axis_dist_predicted_weighted_px", "axis_dist_oracle_weighted_px",
        "axis_dist_ranker_predicted_px", "axis_dist_ranker_oracle_px",
        "axis_dist_position_oracle_px",
    )

    def _agg(df: pd.DataFrame, col: str):
        v = df[col].dropna()
        return (float(v.mean()), float(v.median())) if len(v) else (None, None)

    def _summary(df: pd.DataFrame) -> dict:
        out = {"n_images_scored": len(df)}
        for col in VARIANT_COLS:
            mean_v, median_v = _agg(df, col)
            out[col] = {"mean": mean_v, "median": median_v}
        return out

    def _print_block(label: str, agg: dict) -> None:
        print(f"\n  {label}:")
        for col in VARIANT_COLS:
            name = col.replace("axis_dist_", "").replace("_px", "")
            print(f"    {name:22s} mean={agg[col]['mean']:7.2f}px  median={agg[col]['median']:7.2f}px")

    aggregate_full = _summary(results_df)
    aggregate_full["n_images_total"] = len(samples)
    _print_block("Pełne 42", aggregate_full)

    aggregate_heldout = None
    if heldout_ids:
        heldout_df = results_df[results_df["is_held_out"]]
        aggregate_heldout = _summary(heldout_df)
        aggregate_heldout["n_images_total"] = len(heldout_ids)
        _print_block("12 held-out", aggregate_heldout)

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps({
            "k_candidates": K_CANDIDATES, "min_angle_sep_deg": MIN_ANGLE_SEP_DEG,
            "ridge_alpha": RIDGE_ALPHA,
            "aggregate_full_42": aggregate_full,
            "aggregate_heldout_12": aggregate_heldout,
            "per_image": records,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nZapisano: {OUTPUT_DIR / 'metrics.csv'}, {OUTPUT_DIR / 'metrics.json'}")
    print("\nDla porównania (z pamięci projektu, nie liczone tu ponownie):")
    print("  Run N (42 obrazy):                    22.8px")
    print("  Track A, k=1 (42 obrazy, 04.09):       66.66px / mediana 53.35px")
    print("  Metoda tradycyjna (42 obrazy):        122.9px")
    print("  Zgodność odczytów eksperckich (KK/SS):  4.28px")


if __name__ == "__main__":
    main()
