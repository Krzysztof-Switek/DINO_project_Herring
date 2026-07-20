"""Tests for src/visualization.py."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synth_image(tmp_path: Path, name: str = "img.png", h: int = 56, w: int = 42) -> Path:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    p = tmp_path / name
    PILImage.fromarray(arr, mode="RGB").save(p)
    return p


def _make_predictions_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "predictions.csv"
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return p


# ---------------------------------------------------------------------------
# test_load_original_image
# ---------------------------------------------------------------------------

def test_load_original_image(tmp_path):
    from src.visualization import load_original_image
    img_path = _make_synth_image(tmp_path, "otolith.png", h=64, w=48)
    result = load_original_image("otolith.png", tmp_path)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape == (64, 48, 3)


# ---------------------------------------------------------------------------
# test_select_top_k
# ---------------------------------------------------------------------------

def test_select_top_k(tmp_path):
    from src.visualization import select_top_k_samples
    rows = [
        {"image_id": f"img_{i}.png", "age": 5, "predicted_age": 5 + i}
        for i in range(10)
    ]
    csv_path = _make_predictions_csv(tmp_path, rows)
    best, worst = select_top_k_samples(csv_path, k_best=3, k_worst=3)
    assert len(best) == 3
    assert len(worst) == 3
    # best: errors 0,1,2; worst: errors 9,8,7
    best_errors = [abs(r["predicted_age"] - r["age"]) for r in best]
    worst_errors = [abs(r["predicted_age"] - r["age"]) for r in worst]
    assert best_errors == sorted(best_errors)
    assert worst_errors == sorted(worst_errors, reverse=True)


# ---------------------------------------------------------------------------
# Reasoning-card pipeline (6 panels)
# ---------------------------------------------------------------------------

def _make_axis_payload(H: int, W: int):
    """Synthetic mask + axis_info + peaks for a horizontal rectangle."""
    import cv2
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(mask, (W // 5, H // 3), (4 * W // 5, 2 * H // 3), 255, -1)
    centroid = (W // 4, H // 2)
    far_edge = (3 * W // 4, H // 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    axis_info = {
        "mask": mask, "centroid": centroid, "far_edge": far_edge,
        "contour": contour, "length_px": float(W // 2),
    }
    n_samples = 20
    xs = np.linspace(centroid[0], far_edge[0], n_samples).astype(np.int64)
    ys = np.full(n_samples, centroid[1], dtype=np.int64)
    line_xy = np.stack([xs, ys], axis=1)
    profile_1d = np.linspace(0.0, 1.0, n_samples).astype(np.float32)
    peak_indices = np.array([5, 12, 17], dtype=np.int64)
    return mask, axis_info, line_xy, profile_1d, peak_indices


def test_draw_reasoning_card_shape_with_axis():
    from src.visualization import draw_reasoning_card
    H, W = 120, 200
    original = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    mask, axis_info, line_xy, profile_1d, peak_indices = _make_axis_payload(H, W)
    grid = np.random.rand(8, 14).astype(np.float32)
    card = draw_reasoning_card(
        original_rgb=original,
        importance_grid=grid,
        predicted_age=3,
        true_age=3,
        mask=mask,
        axis_info=axis_info,
        peak_indices=peak_indices,
        line_xy=line_xy,
        profile_1d=profile_1d,
    )
    assert card.ndim == 3 and card.shape[2] == 3
    # 3 columns × 2 rows + title bar per panel
    assert card.shape[1] == 3 * W
    assert card.shape[0] > 2 * H   # extra rows for title bars


def test_draw_reasoning_card_fallback_no_axis():
    """When axis_info/mask are None, the function must still produce a card."""
    from src.visualization import draw_reasoning_card
    H, W = 80, 100
    original = np.zeros((H, W, 3), dtype=np.uint8)
    grid = np.random.rand(5, 7).astype(np.float32)
    card = draw_reasoning_card(
        original_rgb=original,
        importance_grid=grid,
        predicted_age=2,
        true_age=4,
        mask=None,
        axis_info=None,
        peak_indices=None,
        line_xy=None,
        profile_1d=None,
    )
    assert card.shape[1] == 3 * W
    assert card.shape[0] > 2 * H


def test_save_reasoning_cards_writes_png(tmp_path):
    from src.visualization import save_reasoning_cards
    H, W = 80, 120
    img_name = "fish99.png"
    (tmp_path / "images").mkdir()
    _make_synth_image(tmp_path / "images", img_name, h=H, w=W)

    mask, axis_info, line_xy, profile_1d, peak_indices = _make_axis_payload(H, W)
    grid = np.random.rand(5, 8).astype(np.float32)

    samples = [{"image_id": img_name, "age": 3, "predicted_age": 3}]
    saved = save_reasoning_cards(
        samples=samples,
        image_dir=tmp_path / "images",
        importance_grids={img_name: grid},
        axis_data={img_name: {
            "mask":         mask,
            "axis_info":    axis_info,
            "peak_indices": peak_indices,
            "line_xy":      line_xy,
            "profile_1d":   profile_1d,
        }},
        output_dir=tmp_path / "cards",
        label="best",
    )
    assert len(saved) == 1
    assert saved[0].exists()
    img = PILImage.open(saved[0])
    assert img.mode == "RGB"


def test_select_increments_count_le_age_and_graceful():
    """select_increments returns <= predicted_age finals and degrades gracefully (Punkt 7)."""
    from src.ring_extraction import select_increments
    H, W = 120, 200
    _, axis_info, _, _, _ = _make_axis_payload(H, W)
    grid = np.random.rand(8, 14).astype(np.float32)
    out = select_increments(grid, axis_info, predicted_age=3, image_h=H, image_w=W)
    assert len(out["final_axis_pts"]) <= 3
    assert isinstance(out["candidate_pts"], list)
    empty = select_increments(grid, None, 3, H, W)   # no axis_info → empty, no crash
    assert empty["final_axis_pts"] == [] and empty["candidate_pts"] == []


def test_draw_reasoning_card_new_increment_mode():
    """draw_reasoning_card accepts final/candidate points (count=age overlay mode)."""
    from src.visualization import draw_reasoning_card
    H, W = 120, 200
    original = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    mask, axis_info, line_xy, profile_1d, _ = _make_axis_payload(H, W)
    grid = np.random.rand(8, 14).astype(np.float32)
    card = draw_reasoning_card(
        original_rgb=original, importance_grid=grid,
        predicted_age=2, true_age=2,
        mask=mask, axis_info=axis_info, line_xy=line_xy, profile_1d=profile_1d,
        final_axis_pts=[(W // 3, H // 2), (W // 2, H // 2)],
        candidate_pts=[(W // 3, H // 2 - 5), (W // 2, H // 2 + 5)],
        final_t=[0.3, 0.6],
    )
    assert card.ndim == 3 and card.shape[2] == 3
    assert card.shape[1] == 3 * W


def test_draw_reasoning_card_two_head_bars():
    """Two-row layout: CORAL (blue) title bars in row 1, MIL (orange) in row 2 (Punkt 7)."""
    from src.visualization import draw_reasoning_card, _HEAD_CORAL_BAR, _HEAD_MIL_BAR
    H, W = 120, 200
    original = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    mask, axis_info, line_xy, _profile, _peaks = _make_axis_payload(H, W)
    grid = np.random.rand(8, 14).astype(np.float32)
    card = draw_reasoning_card(
        original_rgb=original, importance_grid=grid, predicted_age=3, true_age=3,
        mask=mask, axis_info=axis_info,
        coral_gradcam=np.random.rand(8, 14).astype(np.float32),
        final_axis_pts=[(W // 3, H // 2)], candidate_pts=[(W // 2, H // 2)], final_t=[0.3],
    )

    def _has(color):
        return int((np.abs(card.astype(int) - np.array(color)).sum(2) < 20).sum()) > 0

    assert _has(_HEAD_CORAL_BAR) and _has(_HEAD_MIL_BAR)


def test_draw_reasoning_card_p0_enrichments():
    """P0 (13.07): CLS-fallback label + ring curves + classical cross-check points render."""
    from src.visualization import draw_reasoning_card
    H, W = 120, 200
    original = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    mask, axis_info, line_xy, profile_1d, _ = _make_axis_payload(H, W)
    grid = np.random.rand(8, 14).astype(np.float32)
    ring = np.array([[W // 4, H // 2], [W // 3, H // 2 - 8],
                     [W // 2, H // 2], [W // 3, H // 2 + 8]], dtype=np.int32)
    card = draw_reasoning_card(
        original_rgb=original, importance_grid=grid, predicted_age=2, true_age=2,
        mask=mask, axis_info=axis_info, line_xy=line_xy, profile_1d=profile_1d,
        final_axis_pts=[(W // 3, H // 2)], candidate_pts=[(W // 2, H // 2)], final_t=[0.4],
        cls_attention=np.random.rand(8, 14).astype(np.float32), cls_is_fallback=True,
        ring_curves=[ring], classical_pts=[(W // 3, H // 2), (W // 2, H // 2)],
    )
    assert card.ndim == 3 and card.shape[1] == 3 * W and card.shape[0] > 2 * H


def test_classical_increments_multi_ray_on_gray():
    """classical_increments (E1): piki intensywności na tych samych 48 promieniach; graceful bez osi."""
    import numpy as np
    from src.ring_extraction import classical_increments
    H, W = 120, 200
    _, axis_info, _, _, _ = _make_axis_payload(H, W)
    cx, cy = axis_info["centroid"]
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    gray = (128 + 100 * np.cos(r / 6.0)).astype(np.float32)   # koncentryczne pierścienie
    out = classical_increments(gray, axis_info, smooth_sigma=0.0, prominence=0.02, min_distance=1)
    assert len(out["candidate_pts"]) > 0
    assert all(len(t) == 3 for t in out["clusters"])          # (mean_t, support, mean_strength)
    assert classical_increments(gray, None)["candidate_pts"] == []   # bez osi → pusto, bez crasha


def test_density_peaks_shape():
    import numpy as np
    from src.ring_extraction import density_peaks
    H, W = 120, 200
    _, axis_info, _, _, _ = _make_axis_payload(H, W)
    grid = np.random.rand(8, 14).astype(np.float32)
    peaks, cand = density_peaks(grid, axis_info, H, W)
    assert isinstance(peaks, list) and isinstance(cand, list)
    if peaks:
        assert len(peaks[0]) == 4          # (t, strength, x, y)


def test_cluster_by_radius_no_collapse_on_dense_peaks():
    """E1 (16.07): dense near-uniform peaks must NOT collapse into 1–2 mega-clusters
    (the greedy-chaining bug), and tight groups must still form distinct clusters."""
    import numpy as np
    from src.ring_extraction import _cluster_by_radius
    # ~270 near-uniform peaks across [0.05, 0.95] (spacing ~0.003 ≪ t_tol) — old code → 1 cluster.
    dense = [(float(t), 1.0) for t in np.linspace(0.05, 0.95, 270)]
    cl = _cluster_by_radius(dense, t_tol=0.06)
    assert len(cl) >= 8, f"dense peaks collapsed into {len(cl)} clusters (E1 regression)"
    assert all(0.0 <= c[0] <= 1.0 for c in cl)
    assert sum(c[1] for c in cl) <= len(dense)          # each peak counted at most once
    # Two tight, well-separated groups → exactly 2 clusters at ~0.3 and ~0.7.
    two = [(0.30, 1.0)] * 5 + [(0.31, 1.0)] * 4 + [(0.70, 1.0)] * 6
    cl2 = _cluster_by_radius(two, t_tol=0.06)
    assert len(cl2) == 2
    ts = sorted(c[0] for c in cl2)
    assert abs(ts[0] - 0.305) < 0.03 and abs(ts[1] - 0.70) < 0.03
    # Degenerate: single peak → single cluster.
    assert _cluster_by_radius([(0.5, 2.0)], t_tol=0.06) == [(0.5, 1, 2.0)]


def test_fuse_increments_three_methods():
    """fuse_increments: density/classical/consensus → <= age finals; consensus preferuje zgodność."""
    from src.ring_extraction import fuse_increments
    H, W = 120, 200
    _, axis_info, _, _, _ = _make_axis_payload(H, W)
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]

    def pk(t):
        return (t, 1.0, int(cx + t * (fx - cx)), int(cy + t * (fy - cy)))

    density = [pk(0.30), pk(0.31), pk(0.60), pk(0.61)]
    classical = [pk(0.29), pk(0.30), pk(0.60), pk(0.90), pk(0.91)]
    for m in ("density", "classical", "consensus"):
        out = fuse_increments(density, classical, 2, axis_info, method=m)
        assert len(out["final_t"]) <= 2
        assert len(out["final_axis_pts"]) == len(out["final_t"])
    ts = fuse_increments(density, classical, 2, axis_info, method="consensus")["final_t"]
    assert len(ts) == 2
    assert min(abs(t - 0.30) for t in ts) < 0.05
    assert min(abs(t - 0.60) for t in ts) < 0.05          # 0.9 (tylko klasyka) NIE wybrane
    assert fuse_increments(density, classical, 2, None, method="consensus")["final_t"] == []


def test_fuse_increments_dp_enforces_spacing():
    """method='dp': gdy dwa najmocniejsze piki są zbyt blisko (< dp_min_gap), DP rozsuwa
    wybór (bierze jeden z nich + odległy), zamiast 'skupiać' oba obok siebie jak top-k."""
    from src.ring_extraction import fuse_increments
    H, W = 120, 200
    _, axis_info, _, _, _ = _make_axis_payload(H, W)
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]

    def pk(t):
        return (t, 1.0, int(cx + t * (fx - cx)), int(cy + t * (fy - cy)))

    # Klastry: t=0.20 (support 4, najmocniejszy), t=0.27 (support 3), t=0.80 (support 2).
    # 0.20 i 0.27 są > t_tol (osobne klastry), ale < dp_min_gap → nie mogą być wybrane oba.
    density = [pk(0.20)] * 4 + [pk(0.27)] * 3 + [pk(0.80)] * 2
    out = fuse_increments(density, [], 2, axis_info, method="dp", dp_min_gap=0.12)
    ts = out["final_t"]
    assert len(ts) == 2
    assert ts == sorted(ts)                                  # inner→outer
    assert min(b - a for a, b in zip(ts, ts[1:])) >= 0.12    # rozstaw wymuszony
    assert min(abs(t - 0.80) for t in ts) < 0.06            # odległy pik wybrany (rozłożenie)
    assert min(abs(t - 0.27) for t in ts) > 0.06            # 0.27 odrzucone na rzecz rozstawu
    assert len(out["final_axis_pts"]) == 2
    # k większe niż liczba klastrów → zwraca tyle, ile jest (bez wysypki)
    assert len(fuse_increments(density, [], 9, axis_info, method="dp")["final_t"]) == 3


def test_merge_clusters_consensus_sums_scores():
    """_merge_clusters: pierścień widziany przez density I klasykę na tym samym promieniu →
    jeden pierścień 'consensus' o zsumowanym score; pierścienie solo zachowują źródło."""
    from src.ring_extraction import _merge_clusters
    H, W = 120, 200
    _, axis_info, _, _, _ = _make_axis_payload(H, W)
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]

    def pk(t):
        return (t, 1.0, int(cx + t * (fx - cx)), int(cy + t * (fy - cy)))

    density = [pk(0.30)] * 3 + [pk(0.70)] * 2          # 0.30 (support3), 0.70 (support2, tylko density)
    classical = [pk(0.31)] * 4                          # 0.31 ~ 0.30 → konsensus
    merged = {round(t, 2): (score, src) for (t, score, src) in _merge_clusters(density, classical)}
    # pierścień ~0.30: konsensus, score = density(3×1) + klasyka(4×1) = 7
    key = min(merged, key=lambda k: abs(k - 0.30))
    assert merged[key][1] == "consensus"
    assert merged[key][0] == 7.0
    # pierścień 0.70: tylko density
    key2 = min(merged, key=lambda k: abs(k - 0.70))
    assert merged[key2][1] == "density"


def test_dp_walkthrough_data_keys_and_consistency():
    """dp_walkthrough_data: zwraca komplet artefaktów, a chosen_t == to, co wybrałaby metoda dp."""
    import numpy as np
    from src.ring_extraction import dp_walkthrough_data, fuse_increments, density_peaks, classical_increments
    H, W = 140, 220
    _, axis_info, _, _, _ = _make_axis_payload(H, W)
    rng = np.random.default_rng(0)
    grid = rng.random((10, 16)).astype(np.float32)
    gray = rng.random((H, W)).astype(np.float32)
    age = 3
    wd = dp_walkthrough_data(grid, gray, axis_info, H, W, age)
    for key in ("density_peaks", "classical_peaks", "density_pts", "classical_pts",
                "density_clusters", "classical_clusters", "merged", "chosen_t",
                "final_axis_pts", "sample_profiles"):
        assert key in wd, f"brak klucza {key}"
    assert len(wd["sample_profiles"]) >= 1
    p0 = wd["sample_profiles"][0]
    assert len(p0["t"]) == len(p0["raw"]) == len(p0["norm"])
    assert 0.0 <= min(p0["norm"]) and max(p0["norm"]) <= 1.0
    # chosen_t z walkthrough == final_t z prawdziwej fuzji dp (ta sama logika)
    dpk, _ = density_peaks(grid, axis_info, H, W)
    cpk = classical_increments(gray, axis_info)["peaks"]
    ref = fuse_increments(dpk, cpk, age, axis_info, method="dp")["final_t"]
    assert wd["chosen_t"] == ref


def test_dp_interactive_data_profiles_match_server_peaks():
    """dp_interactive_data's raw per-ray profiles must reproduce EXACTLY the peaks that
    density_peaks/classical_increments (the real server pipeline) find at the same
    prominence/min-distance — this is the contract the Krok-4 JS slider widget relies
    on (it reruns peak-finding on these profiles client-side; if they drifted from the
    server's own profiles, the live widget would silently show wrong ring counts)."""
    import numpy as np
    from scipy.signal import find_peaks
    from src.ring_extraction import dp_interactive_data, density_peaks, classical_increments

    H, W = 140, 220
    _, axis_info, _, _, _ = _make_axis_payload(H, W)
    rng = np.random.default_rng(1)
    grid = rng.random((10, 16)).astype(np.float32)
    gray = rng.random((H, W)).astype(np.float32)
    age = 3

    interactive = dp_interactive_data(grid, gray, axis_info, H, W, age)
    for key in ("predicted_age", "n_samples", "inner_margin", "edge_margin",
                "density_min_distance", "classical_min_distance", "centroid",
                "contour_pts", "density_profiles", "classical_profiles"):
        assert key in interactive
    n_dirs = len(interactive["contour_pts"])
    assert n_dirs > 0
    assert len(interactive["density_profiles"]) == n_dirs
    assert len(interactive["classical_profiles"]) == n_dirs

    prom, min_d = 0.1, interactive["density_min_distance"]
    n_samples = interactive["n_samples"]
    js_t = []
    for prof in interactive["density_profiles"]:
        if prof is None:
            continue
        arr = np.asarray(prof)
        idxs, _ = find_peaks(arr, distance=max(1, min_d), prominence=prom)
        for idx in idxs:
            t = idx / max(1, n_samples - 1)
            if interactive["inner_margin"] <= t <= 1.0 - interactive["edge_margin"]:
                js_t.append(round(t, 6))

    server_peaks, _ = density_peaks(grid, axis_info, H, W, prominence=prom, min_distance=min_d)
    server_t = sorted(round(p[0], 6) for p in server_peaks)
    assert sorted(js_t) == server_t


def test_render_single_ray_draws_highlighted_line():
    """render_single_ray (20.07, Krok 2 companion): same shape as input, and pixels
    along the highlighted jądro→contour_pt line actually change."""
    from src.visualization import render_single_ray
    H, W = 140, 220
    original = np.full((H, W, 3), 200, dtype=np.uint8)
    _, axis_info, _, _, _ = _make_axis_payload(H, W)
    contour_pt = axis_info["far_edge"]
    out = render_single_ray(original, axis_info, contour_pt, peak_ts=[0.3, 0.7])
    assert out.shape == original.shape
    assert not np.array_equal(out, original)
    # midpoint of the ray should no longer be the flat background colour
    cx, cy = axis_info["centroid"]
    fx, fy = contour_pt
    mx, my = int((cx + fx) / 2), int((cy + fy) / 2)
    assert not np.array_equal(out[my, mx], original[my, mx])


def test_render_single_ray_none_axis_info_returns_copy():
    from src.visualization import render_single_ray
    H, W = 60, 90
    original = np.zeros((H, W, 3), dtype=np.uint8)
    out = render_single_ray(original, None, (10, 10))
    assert out.shape == original.shape
