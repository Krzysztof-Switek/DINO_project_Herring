"""31.07 (cd. 3) — Fuzja pewnych fragmentów łuku w pełny pierścień (RANSAC).

Kontynuacja `ring_shortest_path_report.py` (raporty v1-v5): tam pełny mechanizm (Faza A,
prior szerokości, hires966, korekta kwartału, K=wiek vs K=wiek-1 na 3 otolitach) jest już
opisany i NIE jest tu powtarzany — patrz `outputs/31.07_ring_shortest_path/report_v5.html`
oraz pamięć projektu. **Ten raport pokazuje WYŁĄCZNIE kolejne podejście do fragmentów**:
złożenie kilku "pewnych otwartych fragmentów łuku" (już istniejący mechanizm) w JEDEN pełny,
gładki pierścień metodą RANSAC — dokładnie krok "dopasuj/wybierz w postprodukcji" z przeglądu
literatury (RANSAC-fitting elipsy/okręgu z fragmentów łuku; Rodin & Troadec; CS-TRD "spider
web"), dotąd tylko OPISANY, nie wdrożony.

Świadoma decyzja projektowa: RANSAC dopasowuje gładki, OKRESOWY model `t(θ)` (obcięty szereg
Fouriera) w przestrzeni (kąt, znormalizowany promień t) — NIE surowy okrąg/elipsę w pikselach
x/y, jak w klasycznym podręcznikowym RANSAC (np. wykrywanie granicy tęczówki). Powód: otolit
śledzia nie jest kołowy/eliptyczny (ma "ogon"/wcięcie) — per-promieniowa normalizacja `t`
(dokładnie ta, której już używa reszta tego prototypu i `otolith_axis.compute_polar_grid`)
już koduje tę geometrię; dopasowanie w tej przestrzeni jest więc geometrycznie poprawne,
podczas gdy dopasowanie surowej elipsy w pikselach byłoby błędne dla tego kształtu (ten sam
argument, który uzasadnił per-promieniową normalizację wszędzie indziej w projekcie).

src/ CAŁKOWICIE niedotknięte — dalej Faza A, dowód w diagnostyce.

Usage: PYTHONIOENCODING=utf-8 python scripts/diagnostics/ring_fragment_fusion_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from PIL import Image as PILImage

from src.report_common import fig_to_b64, img_tag
from src.visualization import _CONTOUR_COLOR
from src.otolith_axis import detect_axis, apply_background_mask, mask_bbox, shift_axis_info
from src.dataset import build_transforms, decode_age_ordinal
from src.inference import load_model_from_checkpoint
from scripts.run_pipeline import load_merged_config

from scripts.diagnostics.ring_shortest_path_report import (
    ray_cast_R_theta, polar_unwrap, apply_margins, frangi_cost_field,
    find_confident_open_arcs, _b64_from_rgb, MODEL_VARIANTS,
    N_ANGLE, N_RADIUS, WINDOW, JUMP_PENALTY, INNER_MARGIN, EDGE_MARGIN,
    ARC_STRIDE_DEG, _RING_COLORS,
)

IMAGE_ID = "2023_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg"
ARC_LEN_DEG = 60
FUSION_ARC_TOP_K = 12      # więcej niż w v1-v5 (6) — RANSAC potrzebuje materiału do odróżnienia inlierów/outlierów
FUSION_ARC_MIN_GAP_DEG = 12

RANSAC_ORDER = 2           # rząd obciętego szeregu Fouriera (2*ORDER+1 współczynników)
RANSAC_ITERS = 800
RANSAC_TOL = 0.035         # tolerancja inliera w jednostkach t
RANSAC_SAMPLE_FRAGS = 2    # ile fragmentów losowo próbkowanych na iterację RANSAC

OUT_DIR = PROJECT_ROOT / "outputs" / "31.07_ring_shortest_path"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION = "v6"
OUT_PATH = OUT_DIR / f"report_{VERSION}_fragment_fusion.html"

_INLIER_COLOR = (30, 200, 30)
_OUTLIER_COLOR = (150, 150, 150)
_FIT_COLOR = (230, 30, 200)


def _fourier_basis(theta: np.ndarray, order: int) -> np.ndarray:
    cols = [np.ones_like(theta)]
    for k in range(1, order + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    return np.stack(cols, axis=1)


def ransac_fit_ring(fragments: list[dict], angles: np.ndarray, n_radius: int,
                    order: int = RANSAC_ORDER, n_iters: int = RANSAC_ITERS,
                    tol: float = RANSAC_TOL, sample_frags: int = RANSAC_SAMPLE_FRAGS,
                    seed: int = 0) -> dict | None:
    """RANSAC: dopasuj gładki, okresowy model t(theta) do zbioru PEWNYCH fragmentów łuku
    (``find_confident_open_arcs``), odporny na fragmenty spoza tego samego pierścienia
    (odrzucane jako outliery). Patrz docstring modułu — dopasowanie w przestrzeni (kąt, t),
    nie w surowych pikselach x/y.

    Zwraca dict {"coeffs", "predict", "inlier_frag_idx", "n_inlier_pts", "n_total_pts"}
    albo None, jeśli fragmentów jest za mało (<2) do dopasowania.
    """
    n_angle = len(angles)
    pts_theta_l, pts_t_l, frag_id_l = [], [], []
    for fi, frag in enumerate(fragments):
        for j, p in enumerate(frag["path"]):
            col = (frag["start_col"] + j) % n_angle
            pts_theta_l.append(angles[col])
            pts_t_l.append(float(p) / max(1, n_radius - 1))
            frag_id_l.append(fi)
    pts_theta = np.array(pts_theta_l)
    pts_t = np.array(pts_t_l)
    frag_id = np.array(frag_id_l)
    n_frags = len(fragments)
    if n_frags < 2 or len(pts_theta) < 2 * order + 1:
        return None

    B_all = _fourier_basis(pts_theta, order)
    rng = np.random.default_rng(seed)
    best_score = -1
    best_coeffs = None
    for _ in range(n_iters):
        k = min(sample_frags, n_frags)
        sample_ids = rng.choice(n_frags, size=k, replace=False)
        mask = np.isin(frag_id, sample_ids)
        if int(mask.sum()) < 2 * order + 1:
            continue
        coeffs, *_ = np.linalg.lstsq(B_all[mask], pts_t[mask], rcond=None)
        pred = B_all @ coeffs
        resid = np.abs(pred - pts_t)
        n_inliers = int((resid < tol).sum())
        if n_inliers > best_score:
            best_score = n_inliers
            best_coeffs = coeffs
    if best_coeffs is None:
        return None

    pred = B_all @ best_coeffs
    inlier_mask = np.abs(pred - pts_t) < tol
    coeffs_refined, *_ = np.linalg.lstsq(B_all[inlier_mask], pts_t[inlier_mask], rcond=None)
    inlier_frag_idx = set(frag_id[inlier_mask].tolist())

    def predict(theta_query: np.ndarray) -> np.ndarray:
        return _fourier_basis(theta_query, order) @ coeffs_refined

    return {"coeffs": coeffs_refined, "predict": predict,
            "inlier_frag_idx": inlier_frag_idx, "n_inlier_pts": int(inlier_mask.sum()),
            "n_total_pts": len(pts_theta)}


def render_fusion_unwrap(polar_field: np.ndarray, arcs: list[dict], fit: dict | None,
                         angles: np.ndarray, title: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(polar_field, aspect="auto", cmap="jet", origin="lower", extent=[0, 360, 0, 1])
    n_angle = len(angles)
    inlier_idx = fit["inlier_frag_idx"] if fit else set()
    for i, arc in enumerate(arcs):
        t = arc["path"].astype(np.float32) / max(1, N_RADIUS - 1)
        deg = np.array([(arc["start_col"] + j) % n_angle for j in range(arc["arc_len"])]) * 360.0 / n_angle
        is_inlier = i in inlier_idx
        color = np.array(_INLIER_COLOR if is_inlier else _OUTLIER_COLOR) / 255.0
        breaks = np.where(np.diff(deg) < 0)[0]
        segs = [(0, breaks[0] + 1), (breaks[0] + 1, len(deg))] if len(breaks) else [(0, len(deg))]
        for lo, hi in segs:
            ax.plot(deg[lo:hi], t[lo:hi], color=color, linewidth=2.0,
                   linestyle="-" if is_inlier else "--", alpha=1.0 if is_inlier else 0.6)
    if fit is not None:
        theta_full = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        t_full = np.clip(fit["predict"](theta_full), 0.0, 1.0)
        deg_full = (theta_full + np.pi) * 180.0 / np.pi
        order = np.argsort(deg_full)
        color = np.array(_FIT_COLOR) / 255.0
        ax.plot(deg_full[order], t_full[order], color=color, linewidth=2.6)
    ax.set_xlabel("kąt θ (stopnie)")
    ax.set_ylabel("znormalizowany promień t (0=jądro, 1=brzeg)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return fig_to_b64(fig)


def render_fusion_on_photo(crop_rgb: np.ndarray, axis_info: dict, arcs: list[dict],
                           fit: dict | None, R_theta: np.ndarray, angles: np.ndarray,
                           cx: float, cy: float) -> np.ndarray:
    out = np.ascontiguousarray(crop_rgb[..., :3]).copy()
    H = out.shape[0]
    if axis_info.get("contour") is not None:
        cv2.drawContours(out, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    n_angle = len(angles)
    inlier_idx = fit["inlier_frag_idx"] if fit else set()
    for i, arc in enumerate(arcs):
        cols = np.array([(arc["start_col"] + j) % n_angle for j in range(arc["arc_len"])])
        sub_R = R_theta[cols]
        sub_angles = angles[cols]
        t = arc["path"].astype(np.float32) / max(1, N_RADIUS - 1)
        radius_px = t * sub_R
        x = cx + radius_px * np.cos(sub_angles)
        y = cy + radius_px * np.sin(sub_angles)
        xy = np.stack([x, y], axis=1).astype(np.int32).reshape(-1, 1, 2)
        is_inlier = i in inlier_idx
        color = _INLIER_COLOR if is_inlier else _OUTLIER_COLOR
        cv2.polylines(out, [xy], isClosed=False, color=color,
                     thickness=max(3, H // 350) if is_inlier else max(1, H // 700),
                     lineType=cv2.LINE_AA)
    if fit is not None:
        theta_full = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        t_full = np.clip(fit["predict"](theta_full), 0.0, 1.0)
        R_interp = np.interp(theta_full, angles, R_theta, period=2 * np.pi)
        radius_px = t_full * R_interp
        x = cx + radius_px * np.cos(theta_full)
        y = cy + radius_px * np.sin(theta_full)
        xy = np.stack([x, y], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [xy], isClosed=True, color=_FIT_COLOR,
                     thickness=max(2, H // 400), lineType=cv2.LINE_AA)
    return out


def main() -> None:
    base = MODEL_VARIANTS[0]
    cfg = load_merged_config(base["cfg_path"], None)
    device = torch.device(cfg.training.device if cfg.training.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Wczytywanie modelu ({base['name']})...", flush=True)
    model = load_model_from_checkpoint(cfg, base["run_dir"] / "checkpoints" / "embedded" / "best.pt")
    model.eval()
    device = next(model.parameters()).device

    image_dir = Path(cfg.data.image_dir)
    orig_rgb = np.array(PILImage.open(image_dir / IMAGE_ID).convert("RGB"), dtype=np.uint8)
    axis_info = detect_axis(orig_rgb, seg_params=cfg.segmentation.as_params(),
                            nucleus_method=cfg.segmentation.nucleus_method)
    if axis_info is None:
        raise RuntimeError("Segmentacja nie powiodła się")
    mask_arr = axis_info["mask"]
    model_input_rgb = (apply_background_mask(orig_rgb, mask_arr)
                       if cfg.data.mask_background else orig_rgb)
    crop_x0, crop_y0, cw, ch = mask_bbox(mask_arr, cfg.candidates.density_crop_pad_frac)
    crop_rgb = model_input_rgb[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    mask_cropped = mask_arr[crop_y0:crop_y0 + ch, crop_x0:crop_x0 + cw]
    d_axis_info = shift_axis_info(axis_info, -crop_x0, -crop_y0)
    cx, cy = d_axis_info["centroid"]

    gray_crop = crop_rgb.mean(axis=2).astype(np.float32)
    R_theta, angles = ray_cast_R_theta(mask_cropped, cx, cy, N_ANGLE)

    def _normalize(field: np.ndarray) -> np.ndarray:
        rng = float(field.max() - field.min())
        return (field - field.min()) / rng if rng > 1e-6 else field * 0.0

    polar_classical = _normalize(polar_unwrap(gray_crop, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))
    polar_classical_grad = _normalize(np.abs(np.gradient(polar_classical, axis=0)))
    print("Liczenie filtru Frangiego (skimage)...", flush=True)
    frangi_field = frangi_cost_field(gray_crop)
    polar_frangi = _normalize(polar_unwrap(frangi_field, float(cw), float(ch), R_theta, angles, cx, cy, N_RADIUS))

    sections = []
    for field_name, polar_norm in [
        ("classical (gradient promieniowy)", polar_classical_grad),
        ("classical (Frangi)", polar_frangi),
    ]:
        print(f"\n=== {field_name} ===", flush=True)
        cost = apply_margins(1.0 - polar_norm, N_RADIUS, INNER_MARGIN, EDGE_MARGIN)
        arcs = find_confident_open_arcs(cost, ARC_LEN_DEG, ARC_STRIDE_DEG, WINDOW, JUMP_PENALTY,
                                        FUSION_ARC_TOP_K, FUSION_ARC_MIN_GAP_DEG)
        print(f"  {len(arcs)} pewnych fragmentów, avg_cost={[round(a['avg_cost'], 4) for a in arcs]}", flush=True)

        fit = ransac_fit_ring(arcs, angles, N_RADIUS)
        if fit is None:
            print("  RANSAC: za mało danych do dopasowania.", flush=True)
        else:
            print(f"  RANSAC: {len(fit['inlier_frag_idx'])}/{len(arcs)} fragmentów jako inliery "
                 f"({fit['n_inlier_pts']}/{fit['n_total_pts']} punktów), współczynniki={np.round(fit['coeffs'], 4)}",
                 flush=True)

        unwrap_b64 = render_fusion_unwrap(polar_norm, arcs, fit, angles,
                                         f"Fuzja RANSAC: {field_name}")
        photo = render_fusion_on_photo(crop_rgb, d_axis_info, arcs, fit, R_theta, angles, cx, cy)

        fit_note = (
            f"<p style=\"font-size:90%;\"><b>RANSAC:</b> {len(fit['inlier_frag_idx'])}/{len(arcs)} "
            f"fragmentów uznanych za inliery (spójne z jednym gładkim modelem t(&theta;), rząd "
            f"{RANSAC_ORDER} szeregu Fouriera), {fit['n_inlier_pts']}/{fit['n_total_pts']} punktów. "
            "Różowa gruba krzywa = dopasowany, KOMPLETNY pierścień (ekstrapolowany przez odcinki, "
            "gdzie żaden fragment nie był wystarczająco pewny). Zielone (grube, ciągłe) fragmenty = "
            "inliery; szare (cienkie, przerywane) = odrzucone jako niespójne z resztą.</p>"
            if fit else "<p style=\"font-size:90%;color:#a00;\">Za mało pewnych fragmentów do "
            "dopasowania RANSAC.</p>")

        sections.append(f"""<section>
<h2>Fuzja fragmentów &mdash; <code>{field_name}</code></h2>
<p style="font-size:90%;">{len(arcs)} pewnych otwartych fragmentów ({ARC_LEN_DEG}&deg; każdy,
top {FUSION_ARC_TOP_K} po średnim koszcie/kolumnę) — WEJŚCIE do RANSAC.</p>
{fit_note}
<h3>Rozwinięcie biegunowe: fragmenty (zielone=inlier/szare=outlier) + dopasowany pierścień (różowy)</h3>
{img_tag(unwrap_b64, style="width:100%;max-width:820px;")}
<h3>Na zdjęciu (odwrotna transformata biegunowa)</h3>
<div style="text-align:center;">{img_tag(_b64_from_rgb(photo, 480), style="width:480px;")}</div>
</section>""")

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Fuzja fragmentów łuku RANSAC ({VERSION})</title>
<style>
body {{font-family:sans-serif;max-width:1000px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
h1 {{color:#1a237e;}} h2 {{color:#1a237e;}}
code {{background:#f0f0f5;padding:1px 4px;border-radius:3px;}}
</style>
</head>
<body>
<h1>Fuzja pewnych fragmentów łuku w pełny pierścień (RANSAC)</h1>
<p>Kontynuacja raportów v1-v5 (<code>ring_shortest_path_report.py</code>) — TYLKO nowe podejście,
bez powtarzania ustaleń o density/hires966/korekcie kwartału/priorze szerokości (patrz
<code>outputs/31.07_ring_shortest_path/report_v5.html</code> i pamięć projektu). Otolit
<code>{IMAGE_ID}</code>. Zamiast wymuszać PEŁNĄ zamkniętą pętlę (v1-v5) ANI zostawiać same
otwarte fragmenty (v3-v5), tu fragmenty ("pewne otwarte łuki") są SKŁADANE w jeden pełny
pierścień metodą RANSAC — dopasowanie gładkiego, okresowego modelu <code>t(&theta;)</code> w
przestrzeni (kąt, znormalizowany promień), odpornego na fragmenty spoza tego samego pierścienia
(odrzucane jako outliery). To wdrożenie kroku "dopasuj/wybierz w postprodukcji" z przeglądu
literatury (RANSAC-fitting z fragmentów łuku; Rodin &amp; Troadec; CS-TRD "spider web"), dotąd
tylko opisanego. Nic w <code>src/</code> nie zostało zmienione.</p>
{"".join(sections)}
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
