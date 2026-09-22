"""Peak decoding — turning a density map into a discrete set of increment positions.

(22.09) Moved here from ``scripts/diagnostics/train_zegar_localization_head.py``, unchanged, so
PRODUCTION code (the report's wedge decision-path section, ``src/wedge_cards.py``) can decode peaks
without importing from ``scripts/diagnostics/``. That script re-exports both functions, so every
existing caller (`from train_zegar_localization_head import decode_topk_peaks`) keeps working.

Two decoders, for two geometries:

* :func:`decode_topk_peaks` — a rectangular ``(H_p, W_p)`` grid (single wedge canvas, strip, or the
  square image), NMS in grid-index space.
* :func:`decode_topk_peaks_real_t` — a FLAT sequence of patches with known real axis-``t``, for the
  multi-band wedge whose concatenated bands have no single rectangular grid.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def decode_topk_peaks(heat: np.ndarray, k: int, min_dist_patches: float = 1.5) -> list[tuple[int, int]]:
    """Greedy top-k local maxima with minimum-separation NMS, count fixed to the sample's own
    known GT ring count (k). Threshold-free by construction.

    Replaces an earlier fixed-threshold decode (PEAK_THRESHOLD=0.3), removed 26.08. Diagnosed:
    this head's actual held-out sigmoid outputs never exceed ~0.01-0.15, so the fixed threshold
    returned ZERO peaks for 41/42 held-out evaluations in the original 24.08 run --
    evaluate_sample() silently treated each as "no match" (empty t_model, not a crash), and the
    run's own aggregate (outputs/24.08_zegar_localization_head/metrics.json) ended up scoring
    only 1/42 samples without anyone noticing at the time."""
    if k <= 0:
        return []
    local_max = ndi.maximum_filter(heat, size=3) == heat
    cand_r, cand_c = np.nonzero(local_max)
    cand_v = heat[cand_r, cand_c]
    order = np.argsort(-cand_v)
    chosen: list[tuple[int, int]] = []
    for idx in order:
        r, c = int(cand_r[idx]), int(cand_c[idx])
        if all((r - rr) ** 2 + (c - cc) ** 2 >= min_dist_patches ** 2 for rr, cc in chosen):
            chosen.append((r, c))
        if len(chosen) >= k:
            break
    return chosen


# Same default separation as decode_topk_peaks's own MIN_DIST_PATCHES=1.5, translated to axis-t
# units via the average radial-patch t-width (1/44, src.wedge_extraction's own n_radius_patches) —
# not independently measured, same "reasonable unmeasured default" status MIN_DIST_PATCHES itself
# already has, just re-expressed in real units instead of grid-index units.
DEFAULT_MIN_GAP_T = 1.5 / 44


def decode_topk_peaks_real_t(
    density: np.ndarray, t_vals: np.ndarray, k: int, min_gap_t: float = DEFAULT_MIN_GAP_T,
) -> list[int]:
    """Greedy top-k selection over a FLAT (ungridded) sequence of per-patch density values,
    enforcing a minimum REAL separation in axis-t (not grid-index distance) between any two
    chosen patches — the multi-band angular-resolution-bands equivalent of
    :func:`decode_topk_peaks`, needed because a sequence concatenated across bands of DIFFERENT
    shapes (09.09, `plans and summaries/09.09_wycinek_pasma_katowe_plan.md`, Faza 12) has no single
    rectangular ``(H_p, W_p)`` grid for :func:`decode_topk_peaks`'s 2D local-maximum-filter/NMS to
    operate on. Dropping the local-maximum pre-filter changes nothing about the FINAL set of
    peaks — greedy selection by value, rejecting anything within ``min_gap_t`` of an already-
    accepted (necessarily higher-density) peak, already IS local-max-plus-NMS; the pre-filter in
    the gridded version is a thinning optimisation, not a correctness requirement.

    Args:
        density: (N,) per-patch density values (any real geometry — bands or a single grid,
            flattened either way).
        t_vals:  (N,) each patch's real axis-t position (same convention as `wedge_polar_coords`/
            `wedge_band_polar_coords` — 0=nucleus, ~1=edge).
        k:       number of peaks to select (typically the sample's own known/predicted ring count).
        min_gap_t: minimum |t_i - t_j| required between any two selected peaks.

    Returns indices into ``density``/``t_vals`` (not sorted by position — same convention as
    :func:`decode_topk_peaks`'s own ``(row, col)`` list, caller sorts if order matters).
    """
    if k <= 0:
        return []
    order = np.argsort(-density)
    chosen: list[int] = []
    chosen_t: list[float] = []
    for idx in order:
        t = float(t_vals[idx])
        if all(abs(t - ct) >= min_gap_t for ct in chosen_t):
            chosen.append(int(idx))
            chosen_t.append(t)
        if len(chosen) >= k:
            break
    return chosen
