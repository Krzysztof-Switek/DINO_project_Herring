"""30.07 — Krok po kroku: jak DZIAŁA agregacja sektorowa (naprawa problemu z
ray_patch_voting_report.py / polar_sector_coverage_report.py), na jednym realnym otolicie,
plus walidacja ilościowa na 30 kartach.

Ten raport NIE powtarza wyjaśnienia klastrowania arc-aware / wyboru DP (Krok 4a-4d
w density_candidate_mechanism_report.py, v8) — TA CZĘŚĆ PIPELINE'U JEST NIEZMIENIONA,
więc odsyła tam zamiast duplikować. Skupia się WYŁĄCZNIE na tym, co jest NOWE: jak z
mapy density i maski otolitu powstaje 48 GĘSTYCH profili sektorowych (zamiast 48
próbek-linii), krok po kroku, z prawdziwymi liczbami z tego samego otolitu co reszta
sesji.

Kroki:
  0. Przypomnienie problemu: ile patchy stara metoda w ogóle widzi.
  1. compute_polar_grid: każdy patch dostaje sektor kątowy (0-47) i promień t.
  2. Budowa profilu JEDNEGO przykładowego sektora: surowe patche -> agregacja MAX per bin
     -> wypełnienie pustych binów.
  3. Wykrywanie piku na tym profilu + przesunięcie na falling-edge (identyczna logika co
     dawniej, inny profil wejściowy).
  4. Reszta pipeline'u (klastrowanie arc-aware + DP) — odsyłacz, bez zmian.
  5. Finalni kandydaci na zdjęciu: STARA vs NOWA metoda.
  6. Walidacja na 30 kartach (liczby z sweep_polar_sector_method.py, wklejone na stałe —
     NIE przeliczane przy każdym uruchomieniu, ten sam powód co Krok 6 gdzie indziej:
     30 kart x 2 warianty to zbyt wolne dla iteracji nad resztą raportu).

Nie trenuje ani nie modyfikuje żadnego kodu w src/.

Usage: PYTHONIOENCODING=utf-8 python scripts/diagnostics/polar_sector_mechanism_report.py
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
import torch
import cv2
from PIL import Image as PILImage
from scipy.signal import find_peaks

from src.report_common import fig_to_b64, img_tag, png_to_b64
from src.visualization import _CONTOUR_COLOR, _AXIS_COLOR
from src.otolith_axis import (detect_axis, apply_background_mask, mask_bbox, shift_axis_info,
                              compute_polar_grid)
from src.dataset import build_transforms
from src.inference import load_model_from_checkpoint
from src.ring_extraction import (_shift_peak_to_falling_edge, _cluster_by_radius_with_arcs,
                                 _cluster_score, _dp_select_t, _project_to_axis, density_peaks)
from scripts.run_pipeline import load_merged_config

RUN_DIR = PROJECT_ROOT / "outputs" / "28.07_e9_w0.1"
CFG_PATH = PROJECT_ROOT / "configs" / "config_e9_w0.1.yaml"
CKPT = RUN_DIR / "checkpoints" / "embedded" / "best.pt"
IMAGE_ID = "2023_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex95_Single2_Right.jpg"

OUT_DIR = PROJECT_ROOT / "outputs" / "29.07_candidate_selection_walkthrough"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION = "v1"
OUT_PATH = OUT_DIR / f"polar_sector_mechanism_{VERSION}.html"
SWEEP_DIR = OUT_DIR / "polar_sector_experiment"

N_DIRS, N_SAMPLES = 48, 64
N_T_BINS = 12          # patrz sweep_polar_sector_method.py — zmierzone dla TEJ SAMEJ siatki 52x52
EXAMPLE_SECTOR_IDX = 12   # ten sam ustalony indeks co EXAMPLE_RAY_IDX w ray_patch_voting_report.py


def _b64_from_rgb(arr: np.ndarray, target_w: int = 480) -> str:
    import base64, io
    img = PILImage.fromarray(np.ascontiguousarray(arr[..., :3]).astype(np.uint8))
    if img.width > target_w:
        scale = target_w / img.width
        img = img.resize((target_w, max(1, int(round(img.height * scale)))), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bin_idx_grid(H_p, W_p, H, W, cx, cy, n_dirs):
    py = (np.arange(H_p, dtype=np.float32) + 0.5) * (H / H_p)
    px = (np.arange(W_p, dtype=np.float32) + 0.5) * (W / W_p)
    grid_x, grid_y = np.meshgrid(px, py)
    dx = grid_x - cx
    dy = grid_y - cy
    theta = np.arctan2(dy, dx)
    return np.clip(((theta + np.pi) / (2.0 * np.pi) * n_dirs).astype(np.int64), 0, n_dirs - 1)


def _old_ray_cells(H_p, W_p, cx, cy, contour, cw, ch, n_dirs=N_DIRS, n_samples=N_SAMPLES):
    idx_sel = np.linspace(0, len(contour) - 1, min(n_dirs, len(contour)), dtype=int)
    cells: set[tuple[int, int]] = set()
    for ci in idx_sel:
        fx, fy = float(contour[ci][0]), float(contour[ci][1])
        xs = np.linspace(cx, fx, n_samples)
        ys = np.linspace(cy, fy, n_samples)
        px = np.clip((xs / max(cw, 1) * W_p).astype(np.int64), 0, W_p - 1)
        py = np.clip((ys / max(ch, 1) * H_p).astype(np.int64), 0, H_p - 1)
        cells |= set(zip(px.tolist(), py.tolist()))
    return cells


def render_masked_overlay(crop_rgb, rgb_small, valid_grid) -> np.ndarray:
    """Wspólny szkielet dla map sektora/t: koloruje CAŁY panel wg ``rgb_small`` (H_p,W_p,3)
    tam gdzie ``valid_grid`` True, przyciemnia resztę do szarości — ta sama "wygaszona
    reszta" konwencja co reszta tej serii raportów."""
    H, W = crop_rgb.shape[:2]
    rgb_up = cv2.resize(rgb_small.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    valid_up = cv2.resize(valid_grid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    blended = 0.6 * rgb_up + 0.4 * crop_rgb.astype(np.float32)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dimmed = np.stack([gray, gray, gray], axis=-1) * 0.30
    return np.where(valid_up[..., None], blended, dimmed).clip(0, 255).astype(np.uint8)


def render_final_comparison(crop_rgb, axis_info, old_pts, new_pts) -> np.ndarray:
    out = np.ascontiguousarray(crop_rgb[..., :3]).copy()
    H = out.shape[0]
    if axis_info.get("contour") is not None:
        cv2.drawContours(out, [axis_info["contour"]], -1, _CONTOUR_COLOR, max(2, H // 300))
    cx, cy = axis_info["centroid"]
    fx, fy = axis_info["far_edge"]
    cv2.line(out, (int(cx), int(cy)), (int(fx), int(fy)), _AXIS_COLOR, max(1, H // 400))
    r = max(4, H // 110)
    for (x, y) in old_pts:
        cv2.circle(out, (int(x), int(y)), r, (230, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), r, (0, 0, 0), 1, cv2.LINE_AA)
    for (x, y) in new_pts:
        cv2.circle(out, (int(x), int(y)), max(2, r // 2), (30, 200, 30), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), max(2, r // 2), (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _section_krok6_sweep() -> str:
    """Krok 6 — 30-kartowa walidacja ilościowa (sweep_polar_sector_method.py, 30.07).
    Liczby WKLEJONE NA STAŁE (nie przeliczane przy każdym uruchomieniu tego raportu —
    30 kart x 2 warianty to zbyt wolne). Jeśli sweep się zmieni, trzeba je tu ręcznie
    odświeżyć — patrz FORCED_TOPK-owy komentarz w density_candidate_mechanism_report.py
    dla dokładnie tego samego wzorca."""
    best_b64 = png_to_b64(SWEEP_DIR / "card_best_improvement.png")
    median_b64 = png_to_b64(SWEEP_DIR / "card_median.png")
    imgs_html = ""
    if best_b64 and median_b64:
        imgs_html = (
            '<div style="display:inline-block;vertical-align:top;width:400px;margin-right:16px;">'
            '<b>Najlepszy przypadek dla NOWEJ metody (rzadki)</b><br>'
            f'{img_tag(best_b64, style="width:380px;")}<br>'
            '<span style="font-size:88%;">dist_old=275px &rarr; dist_new=170px (<b>-38%</b>)</span></div>'
            '<div style="display:inline-block;vertical-align:top;width:400px;">'
            '<b>Karta MEDIANOWA (bliżej typowego wyniku)</b><br>'
            f'{img_tag(median_b64, style="width:380px;")}<br>'
            '<span style="font-size:88%;">dist_old=138px &rarr; dist_new=172px (<b>+25%, pogorszenie</b>)</span></div>'
            '<p style="font-size:88%;">Czerwone kropki = STARA metoda (48 linii). Zielone '
            '(mniejsze) = NOWA metoda (sektory). Żółta linia = oś pomiaru, obie metody '
            'rzutowane na tę samą oś.</p>'
        )
    return f"""<section><h2>Krok 6 — walidacja ilościowa na 30 kartach: WYNIK NEGATYWNY</h2>
<p><code>scripts/diagnostics/sweep_polar_sector_method.py</code> — 30 kart (best+worst wg
trafności wieku, <code>outputs/28.07_e9_w0.1</code>), ta sama metryka co reszta sesji
(<code>mean_dist</code> do klasycznej referencji jednoosiowej).</p>
<div style="background:#fdecea;padding:10px;border-left:4px solid #c62828;margin:10px 0;">
<b>Hipoteza z Kroku 0-5 (pełne pokrycie patchy = lepsza lokalizacja) NIE potwierdza się na
szerszej próbie.</b> Na jednym otolicie (Krok 5) wyniki obu metod wyglądały blisko siebie —
na 30 kartach NOWA metoda jest śr. o <b>15% GORSZA</b>, nie lepsza.</p>
<table>
<tr><th>Metryka</th><th>STARA (48 linii)</th><th>NOWA (sektory)</th></tr>
<tr><td>Śr. pokrycie patchy</td><td>37.4%</td><td><b>63.3%</b> (blisko teoretycznego maksimum)</td></tr>
<tr><td>mean_dist do klasyki (n=21/30)</td><td><b>260.71px</b></td><td>299.67px (<b>+14.9%, gorzej</b>)</td></tr>
<tr><td>Kart z niedoborem finałów (mniej niż wiek)</td><td>2/30</td><td>5/30 (gorzej)</td></tr>
</table>
<p>Kart, gdzie finalny wybór (<code>chosen_t</code>) się ZMIENIŁ: <b>30/30 (100%)</b> — NOWA
metoda nie jest subtelną poprawką nad tym samym wynikiem, tylko realnie inną pulą
kandydatów na KAŻDEJ karcie.</p>
<p><b>Poprawa (dist_old − dist_new) wg trafności wieku CORAL:</b> trafny wiek (|błąd|≤1, 13
kart z ważną metryką): <b style="color:#c62828;">-41.8px</b> (NOWA gorsza — nawet tam, gdzie
CORAL miał się najbardziej "zgadzać" z lokalizacją); nietrafny wiek (|błąd|≥2, 8 kart):
<b style="color:#c62828;">-34.3px</b> (też gorsza). Innymi słowy: pogorszenie jest
konsekwentne, nie ograniczone do jednej podgrupy kart.</p>
<p><b>Wg klasy wieku:</b> wiek 0-6 (17/21 kart z ważną metryką) — STARA 260.9px / NOWA
312.6px (gorzej o 20%); wiek 7+ (n=4, mała próba) — STARA 260.0px / NOWA 244.8px (nieznaczna
poprawa, ale przy n=4 to szum, nie sygnał).</p>
{imgs_html}
<h3 style="margin-top:1.2em;">Dlaczego pełne pokrycie NIE pomogło — hipotezy (niezweryfikowane
dalszymi eksperymentami)</h3>
<ol>
<li><b>Agregacja MAX wzmacnia szum, nie tylko sygnał.</b> Pokrycie skoczyło z ~37% do ~63%
patchy na bin — to ~2x więcej "kandydatów" na każdy bin profilu. Im więcej patchy bierze
udział w <code>max()</code>, tym większa szansa, że wygra pojedynczy odstający patch
(artefakt brzegowy/register-token, już wcześniej udokumentowany w tym projekcie — patrz
<code>ray_patch_voting_v4.html</code>, wyjaśnienie przy panelu (a)), a nie prawdziwy
pierścień. Więcej pokrycia mogło więc wstrzyknąć więcej fałszywych, silnych pików, nie
mniej.</li>
<li><b>Niższa rozdzielczość promieniowa.</b> Stary profil miał 64 próbki/promień; nowy —
tylko <code>n_t_bins=12</code> (dobrane, żeby biny były w większości zapełnione, nie pod
kątem precyzji pozycji). Nawet przy pełnym pokryciu, 12 binów to dużo grubszy krok
kwantyzacji pozycji niż 64 próbki — mogło to obniżyć precyzję końcowego położenia mimo
lepszego pokrycia danych wejściowych.</li>
<li><b>Inna definicja "t" niż w reszcie pipeline'u.</b> <code>compute_polar_grid</code>
liczy R(θ) przez rzucanie promieni po SUROWEJ masce binarnej (360 kierunków), podczas gdy
stara metoda mierzy odległość do 48 KONKRETNYCH punktów konturu. To blisko spokrewnione,
ale NIE identyczne parametryzacje promienia — mogły systematycznie przesunąć wybrane
pozycje względem niezależnie liczonej klasycznej referencji, niezależnie od szumu.</li>
</ol>
<p style="font-size:88%;background:#fff3cd;padding:8px;border-left:3px solid #b8860b;">
<b>Werdykt:</b> na podstawie tych 30 kart <b>NIE rekomenduję promocji agregacji sektorowej
do <code>src/ring_extraction.py</code></b> w obecnej formie (surowy MAX, n_t_bins dobrany
pod kątem zapełnienia binów, nie precyzji). Gdyby wracać do tego pomysłu, dwa najbardziej
obiecujące następne kroki to: (a) mniej podatna na odstające wartości agregacja (np. drugi
najwyższy patch zamiast dosłownego MAX, albo średnia z top-k), (b) większe
<code>n_t_bins</code> kosztem akceptacji części pustych binów, zamiast optymalizować pod
zapełnienie. Zastrzeżenie: 30 kart to wciąż niewielka próba — kod eksperymentu
(<code>sweep_polar_sector_method.py</code>) jest samodzielny, nic w produkcyjnym pipeline
nie zostało zmienione.</p>
</section>"""


def main() -> None:
    print("Wczytywanie configu i modelu...", flush=True)
    cfg = load_merged_config(CFG_PATH, None)
    image_dir = Path(cfg.data.image_dir)
    device = torch.device(cfg.training.device if cfg.training.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_from_checkpoint(cfg, CKPT)
    model.eval()
    device = next(model.parameters()).device

    print(f"Segmentacja + oś: {IMAGE_ID}", flush=True)
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
    density_transform = build_transforms(cfg.candidates.density_image_size, "test")
    density_tensor = density_transform(PILImage.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        density_grid = model.get_density_probs(density_tensor).squeeze(0).cpu().numpy()
    H_p, W_p = density_grid.shape
    total = H_p * W_p
    print(f"  Siatka: {H_p}x{W_p} = {total} patchy", flush=True)

    min_distance = cfg.candidates.min_peak_distance
    prominence = cfg.candidates.prominence_threshold
    inner_margin = cfg.candidates.inner_margin
    width_decay_weight = cfg.candidates.width_decay_weight
    width_ceiling_weight = cfg.candidates.width_ceiling_weight
    cx, cy = d_axis_info["centroid"]
    contour = d_axis_info["contour"].reshape(-1, 2)

    # --- Krok 0: przypomnienie problemu ---
    old_cells = _old_ray_cells(H_p, W_p, cx, cy, contour, cw, ch)
    old_pct = 100 * len(old_cells) / total

    # --- Krok 1: compute_polar_grid — sektor + t dla KAŻDEGO patcha ---
    print("Krok 1: compute_polar_grid...", flush=True)
    t_grid, valid_grid = compute_polar_grid(mask_cropped, (cx, cy), H_p, W_p, n_angle_bins=N_DIRS)
    bin_idx = _bin_idx_grid(H_p, W_p, ch, cw, cx, cy, N_DIRS)
    n_valid = int(valid_grid.sum())
    new_pct = 100 * n_valid / total

    sector_rgb_small = cv2.applyColorMap(
        (bin_idx.astype(np.float32) / N_DIRS * 255).astype(np.uint8), cv2.COLORMAP_HSV)
    sector_rgb_small = cv2.cvtColor(sector_rgb_small, cv2.COLOR_BGR2RGB)
    sector_map_img = render_masked_overlay(crop_rgb, sector_rgb_small, valid_grid)

    t_uint8_small = (np.clip(t_grid, 0, 1) * 255).astype(np.uint8)
    t_rgb_small = cv2.cvtColor(cv2.applyColorMap(t_uint8_small, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    t_map_img = render_masked_overlay(crop_rgb, t_rgb_small, valid_grid)

    # --- Krok 2: profil JEDNEGO przykładowego sektora — surowe patche + agregacja MAX ---
    print("Krok 2/3: przykładowy sektor...", flush=True)
    sel = bin_idx == EXAMPLE_SECTOR_IDX
    sel_valid = sel & valid_grid
    raw_t = t_grid[sel_valid]
    raw_v = density_grid[sel_valid]

    max_val = np.full(N_T_BINS, -np.inf, dtype=np.float64)
    has_val = np.zeros(N_T_BINS, dtype=bool)
    pos_row = np.full(N_T_BINS, -1, dtype=np.int64)
    pos_col = np.full(N_T_BINS, -1, dtype=np.int64)
    rows_idx, cols_idx = np.nonzero(sel_valid)
    for r, c in zip(rows_idx.tolist(), cols_idx.tolist()):
        t = float(t_grid[r, c])
        if t < 0.0:
            continue
        tb = min(int(t * N_T_BINS), N_T_BINS - 1)
        v = float(density_grid[r, c])
        if v > max_val[tb]:
            max_val[tb] = v
            has_val[tb] = True
            pos_row[tb] = r
            pos_col[tb] = c
    filled = max_val.copy()
    idxs_has = np.nonzero(has_val)[0]
    for tb in range(N_T_BINS):
        if not has_val[tb] and idxs_has.size:
            nearest = idxs_has[np.argmin(np.abs(idxs_has - tb))]
            filled[tb] = max_val[nearest]
            pos_row[tb] = pos_row[nearest]
            pos_col[tb] = pos_col[nearest]
    bin_centers = (np.arange(N_T_BINS) + 0.5) / N_T_BINS

    fig, (axr, axb) = plt.subplots(1, 2, figsize=(11, 3.6))
    axr.scatter(raw_t, raw_v, s=22, color="#2a78d6", alpha=0.75, edgecolor="black", linewidth=0.3)
    axr.set_title(f"Surowe patche w sektorze #{EXAMPLE_SECTOR_IDX} ({int(sel_valid.sum())} "
                 f"patchy w masce)", fontsize=9)
    axr.set_xlabel("t (0=jądro, 1=kontur)", fontsize=9)
    axr.set_ylabel("wartość density (surowa)", fontsize=9)
    axr.set_xlim(0, 1)

    axb.bar(bin_centers, max_val, width=1.0 / N_T_BINS * 0.85, color="#888888", alpha=0.5,
           label="MAX per bin (przed wypełnieniem)")
    empty_mask = ~has_val
    axb.bar(bin_centers[empty_mask], filled[empty_mask], width=1.0 / N_T_BINS * 0.5,
           color="#e01e1e", alpha=0.85, label="wypełnione (bin był pusty)")
    axb.set_title(f"Profil sektora #{EXAMPLE_SECTOR_IDX}: {N_T_BINS} binów, "
                 f"{int(empty_mask.sum())} pustych przed wypełnieniem", fontsize=9)
    axb.set_xlabel("t (środek bina)", fontsize=9)
    axb.set_ylabel("wartość density (MAX w binie)", fontsize=9)
    axb.legend(fontsize=7, loc="upper right")
    axb.set_xlim(0, 1)
    fig.tight_layout()
    sector_profile_b64 = fig_to_b64(fig)

    # --- Krok 3: normalizacja + wykrywanie piku + falling-edge na TYM profilu ---
    rng = float(filled.max() - filled.min())
    norm = ((filled - filled.min()) / rng).astype(np.float32) if rng > 1e-6 else None
    example_peak_b64 = ""
    if norm is not None:
        idxs, props = find_peaks(norm, distance=max(1, int(min_distance)), prominence=float(prominence))
        prominences = props.get("prominences", [])
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.axvspan(0, inner_margin, color="#dc0000", alpha=0.10, zorder=0)
        ax.plot(bin_centers, norm, color="#2a78d6", lw=1.8, marker="o", ms=4)
        ax.fill_between(bin_centers, 0, norm, color="#2a78d6", alpha=0.15)
        for pi, pidx in enumerate(idxs):
            prom = prominences[pi] if pi < len(prominences) else 0.0
            edge_idx = _shift_peak_to_falling_edge(norm, int(pidx))
            ax.plot([bin_centers[pidx]], [norm[pidx]], "o", color="#e01e1e", ms=8)
            ax.annotate(f"prom={prom:.3f}", (bin_centers[pidx], norm[pidx]),
                       textcoords="offset points", xytext=(4, 6), fontsize=7, color="#e01e1e")
            if edge_idx != pidx:
                ax.plot([bin_centers[edge_idx]], [norm[edge_idx]], "s", color="black", ms=8, mfc="none", mew=2)
                ax.annotate("", xy=(bin_centers[edge_idx], norm[edge_idx]),
                           xytext=(bin_centers[pidx], norm[pidx]),
                           arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax.axhline(prominence, color="#888", ls=":", lw=1.0)
        ax.set_title(f"Sektor #{EXAMPLE_SECTOR_IDX} po normalizacji + wykryte piki "
                    f"(prominence≥{prominence}) — kwadrat = po przesunięciu na falling-edge",
                    fontsize=9)
        ax.set_xlabel("t", fontsize=9)
        ax.set_ylabel("wartość znorm.", fontsize=9)
        ax.set_xlim(0, 1)
        fig.tight_layout()
        example_peak_b64 = fig_to_b64(fig)

    # --- Krok 4/5: pełen mechanizm na CAŁYM otolicie (STARA vs NOWA) ---
    print("Krok 4/5: pełny mechanizm (klastrowanie + DP) obiema metodami...", flush=True)
    age_transform = build_transforms(cfg.data.image_size, "test")
    from src.dataset import decode_age_ordinal
    age_tensor = age_transform(PILImage.fromarray(model_input_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        coral_logits = model(age_tensor)["coral_logits"]
    predicted_age = int(decode_age_ordinal(coral_logits).item())
    print(f"  Przewidziany wiek (CORAL): {predicted_age}", flush=True)

    dpk_old, _ = density_peaks(density_grid, d_axis_info, ch, cw, n_dirs=N_DIRS, n_samples=N_SAMPLES,
                               min_distance=min_distance, prominence=prominence, inner_margin=inner_margin)
    clusters_old = _cluster_by_radius_with_arcs(dpk_old, t_tol=0.06, n_dirs=N_DIRS)
    cands_old = [(c[0], _cluster_score(c)) for c in clusters_old]
    chosen_old = _dp_select_t(cands_old, predicted_age, 0.04, 1.5, width_decay_weight, width_ceiling_weight)
    finals_old = _project_to_axis(chosen_old, d_axis_info)

    from scripts.diagnostics.sweep_polar_sector_method import build_sector_profiles, sector_peaks
    profiles, pos_row_full, pos_col_full = build_sector_profiles(
        density_grid, t_grid, valid_grid, bin_idx, N_DIRS, N_T_BINS)
    dpk_new, _ = sector_peaks(profiles, pos_row_full, pos_col_full, cw, ch, H_p, W_p, N_T_BINS,
                              min_distance, prominence, inner_margin)
    clusters_new = _cluster_by_radius_with_arcs(dpk_new, t_tol=0.06, n_dirs=N_DIRS)
    cands_new = [(c[0], _cluster_score(c)) for c in clusters_new]
    chosen_new = _dp_select_t(cands_new, predicted_age, 0.04, 1.5, width_decay_weight, width_ceiling_weight)
    finals_new = _project_to_axis(chosen_new, d_axis_info)

    final_cmp_img = render_final_comparison(crop_rgb, d_axis_info, finals_old, finals_new)

    print("Składanie strony HTML...", flush=True)
    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<title>Mechanizm agregacji sektorowej krok po kroku ({VERSION}, {IMAGE_ID})</title>
<style>
body {{font-family:sans-serif;max-width:1300px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
h1 {{color:#1a237e;}} h2 {{color:#1a237e;}} h3 {{color:#283593;}}
table {{border-collapse:collapse;margin:0.6em 0;}}
td,th {{padding:4px 10px;border:1px solid #ddd;font-size:90%;text-align:center;}}
th {{background:#f0f0f5;}}
code {{background:#f0f0f5;padding:1px 4px;border-radius:3px;}}
</style>
</head>
<body>
<h1>Jak działa agregacja sektorowa — krok po kroku</h1>
<p>Otolit <code>{IMAGE_ID}</code>. Ten raport pokazuje MECHANIZM naprawy problemu z
<code>ray_patch_voting_report.py</code> (48 linii dotyka tylko ~{old_pct:.0f}% patchy).
Klastrowanie arc-aware i wybór DP (Krok 4a-4d w <code>density_mechanism_v8.html</code>)
są TU NIEZMIENIONE — nie powtarzamy ich wyjaśnienia, tylko odsyłamy. Ten raport skupia
się WYŁĄCZNIE na tym, co jest nowe: jak powstaje 48 GĘSTYCH profili sektorowych.</p>
<p style="background:#fdecea;padding:10px;border-left:4px solid #c62828;">
<b>Wynik z góry (szczegóły w Kroku 6):</b> mechanizm opisany w Krokach 0-5 działa dokładnie
tak, jak zaprojektowany — pokrycie patchy realnie skacze z ~37% do ~63% (teoretyczne
maksimum). Ale na 30-kartowej walidacji lokalizacja wychodzi śr. <b>o 15% GORZEJ</b>, nie
lepiej, niż dzisiejsza metoda. To NEGATYWNY wynik eksperymentu — patrz Krok 6 po
uzasadnienie i hipotezy dlaczego.</p>

<section><h2>Krok 0 — przypomnienie problemu</h2>
<p>Stara metoda (48 dosłownych linii, 64 próbki/linia) dotyka <b>{len(old_cells)}/{total}
({old_pct:.1f}%)</b> patchy na tym otolicie — reszta nigdy nie wchodzi do żadnego sygnału.
Szczegóły i wizualne dowody: <code>ray_patch_voting_v4.html</code>.</p>
</section>

<section><h2>Krok 1 — <code>compute_polar_grid</code>: sektor + promień dla KAŻDEGO patcha</h2>
<p>Zamiast rzucać 48 nieskończenie cienkich linii, KAŻDY z {total} patchy w siatce dostaje
własny <b>sektor kątowy</b> (0-47, wg kąta od jądra) i własny <b>znormalizowany promień
t = r/R(θ)</b> (0=jądro, 1=kontur W TYM konkretnym kierunku) — funkcja już istniejąca w
<code>src/otolith_axis.py::compute_polar_grid</code>, dotąd używana WYŁĄCZNIE przez trening
(stratę E9), nigdy przez wykrywanie kandydatów. <b>{n_valid}/{total} ({new_pct:.1f}%)</b>
patchy leży w masce — to WSZYSTKIE z nich dostają sektor+t, nie tylko te na 48 liniach.</p>
<div>
<div style="display:inline-block;vertical-align:top;width:460px;margin:0 14px 14px 0;">
<b>Sektor kątowy (0-47, kolor=kierunek)</b><br>
{img_tag(_b64_from_rgb(sector_map_img, 440), style="width:440px;")}</div>
<div style="display:inline-block;vertical-align:top;width:460px;margin:0 14px 14px 0;">
<b>Promień t (0=jądro fiolet, 1=kontur żółty)</b><br>
{img_tag(_b64_from_rgb(t_map_img, 440), style="width:440px;")}</div>
</div>
</section>

<section><h2>Krok 2 — budowa profilu JEDNEGO sektora: agregacja MAX</h2>
<p>Przykładowy sektor <b>#{EXAMPLE_SECTOR_IDX}/{N_DIRS}</b> (ten sam ustalony indeks co
"promień #{EXAMPLE_SECTOR_IDX}" w <code>ray_patch_voting_report.py</code>). Po lewej:
KAŻDY patch tego sektora jako punkt (t, wartość density) — to jest CAŁY materiał, z
którego budujemy profil (nie tylko próbki na jednej linii). Po prawej: dzielimy t na
{N_T_BINS} binów i w każdym bierzemy <b>MAX</b> (nie średnią — uśrednianie rozmyłoby
pojedynczy silny patch) wartości patchy, które w niego trafiły. Czerwone słupki = biny,
które były puste (żaden patch nie trafił) — wypełnione najbliższym niepustym binem wzdłuż
t.</p>
<div style="text-align:center;">{img_tag(sector_profile_b64, style="width:900px;")}</div>
</section>

<section><h2>Krok 3 — wykrywanie piku na profilu sektora + falling-edge</h2>
<p>Dokładnie ta sama logika co dawniej (<code>find_peaks</code> + <code>_shift_peak_to_
falling_edge</code>, zaimportowane wprost z <code>src.ring_extraction</code>, BEZ ZMIAN)
— jedyna różnica to WEJŚCIE: gęsty profil z Kroku 2, nie 64 próbki linii.</p>
{f'<div style="text-align:center;">{img_tag(example_peak_b64, style="width:640px;")}</div>' if example_peak_b64 else '<p><i>Sektor płaski/pusty na tym otolicie — brak wykrytych pików.</i></p>'}
</section>

<section><h2>Krok 4 — klastrowanie arc-aware + wybór DP (BEZ ZMIAN)</h2>
<p>Piki ze WSZYSTKICH 48 sektorów (dokładnie jak dawniej z 48 linii) trafiają do TEJ SAMEJ,
niezmienionej funkcji <code>_cluster_by_radius_with_arcs</code> (nagradza sektory
SĄSIADUJĄCE ze sobą, nie samą liczbę głosów) i <code>_dp_select_t</code> (wybiera dokładnie
<code>wiek</code> finałów, z biologicznym priorem szerokości przyrostu). Pełne wyjaśnienie
wzoru i przykład krok-po-kroku: <code>density_mechanism_v8.html</code>, Krok 4a-4d — TU
nieduplikowane, bo mechanizm jest identyczny.</p>
</section>

<section><h2>Krok 5 — finalni kandydaci: STARA vs NOWA (wiek={predicted_age})</h2>
<p><span style="color:#e01e1e;font-weight:bold;">&#9679; czerwone</span> = STARA metoda
(48 linii) — <b>{len(finals_old)}</b> punktów. <span style="color:#1ec81e;font-weight:bold;">
&#9679; zielone (mniejsze)</span> = NOWA metoda (sektory) — <b>{len(finals_new)}</b>
punktów. Oba rzutowane na tę samą oś pomiaru.</p>
<div style="text-align:center;">{img_tag(_b64_from_rgb(final_cmp_img, 480), style="width:480px;")}</div>
</section>

{_section_krok6_sweep()}
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
