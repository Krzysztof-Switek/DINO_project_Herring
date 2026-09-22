"""22.09 — weryfikacja paneli raportu dla gałęzi wycinka kątowego, na REALNYCH zdjęciach.

Po co: do 22.09 karty raportu liczyły density na kwadratowym obrazie 518 px — czyli na geometrii,
której głowica wycinka nigdy nie widziała w treningu (oba configi wycinka nosiły o tym ostrzeżenie
w nagłówku, a bieg `outputs/09.09_wedge_b` i tak trafił do raportu jako "wynik"). `src/wedge_cards.
py` + nowe rendery w `src/visualization.py` + sekcja G2 w `src/comparison_report.py` naprawiają to.
Ten skrypt pokazuje efekt na prawdziwych otolitach ZEGAR, ZANIM skończy się kolejny trening.

Co jest prawdziwe, a co nie — wprost, bez udawania:

* PRAWDZIWE: zdjęcia, segmentacja, oś, geometria klina/pasm, ekstrakcja kanw, backbone DINOv2
  (`dinov2_vits14_reg`, ten sam co w configu), rzutowanie pików z powrotem na piksele, wszystkie
  rysunki.
* NIEPRAWDZIWE (dopóki nie ma wytrenowanego checkpointu wycinka): WAGI GŁOWICY density są losowe.
  Wartości `density` będą więc płaskie i bliskie 0,5, a wybrane piki są w praktyce arbitralne.
  To celowe: sprawdzamy GEOMETRIĘ i PANELE, nie jakość lokalizacji. Panel podpisuje to sam.

Użycie:
    python scripts/diagnostics/verify_wedge_report_panels.py            # 3 zdjęcia, config_wedge_b
    python scripts/diagnostics/verify_wedge_report_panels.py --config configs/config_wedge_a.yaml
    python scripts/diagnostics/verify_wedge_report_panels.py --checkpoint outputs/.../best.pt
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DIAG_DIR = PROJECT_ROOT / "scripts" / "diagnostics"
if str(DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(DIAG_DIR))

import numpy as np
from PIL import Image as PILImage

import expert_annotation_eval as ev
from scripts.run_pipeline import load_merged_config
from src import wedge_cards as wc
from src.otolith_axis import detect_axis
from src.visualization import render_wedge_canvas_panel, render_wedge_sector_on_photo

DEFAULT_SAMPLES = ["Z20", "Z38", "Z37"]      # the three this project keeps coming back to
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "22.09_wedge_report_panels"


def _b64(arr: np.ndarray, target_w: int) -> str:
    h0, w0 = arr.shape[:2]
    s = min(1.0, target_w / max(w0, 1))
    im = PILImage.fromarray(arr)
    if s < 1.0:
        im = im.resize((max(1, int(w0 * s)), max(1, int(h0 * s))))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _build_model(cfg, checkpoint: Path | None):
    if checkpoint is not None:
        from src.inference import load_model_from_checkpoint
        print(f"  checkpoint: {checkpoint}")
        model = load_model_from_checkpoint(cfg, checkpoint)
        model.eval()
        return model, True
    from src.model import OtolithModel
    print("  checkpoint: BRAK — głowica density z losową inicjalizacją (patrz docstring)")
    model = OtolithModel(cfg)
    model.eval()
    return model, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config_wedge_b.yaml"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--samples", nargs="*", default=DEFAULT_SAMPLES)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_merged_config(Path(args.config), None)
    if not wc.wedge_enabled(cfg):
        sys.exit(f"{args.config} nie ma data.dual_branch_wedge=true — to nie jest config wycinka.")

    mode = "pasma" if wc.wedge_bands_enabled(cfg) else "pojedyncza kanwa"
    print(f"Config: {Path(args.config).name}  |  tryb: {mode}  |  backbone: {cfg.model.backbone}")

    model, trained = _build_model(cfg, Path(args.checkpoint) if args.checkpoint else None)
    seg_params = cfg.segmentation.as_params()

    blocks = []
    for sample in args.samples:
        img_path = ev.IMAGE_DIR / f"{sample}.jpg"
        if not img_path.exists():
            print(f"  [{sample}] brak pliku {img_path}, pomijam")
            continue
        raw = np.array(PILImage.open(img_path).convert("RGB"), dtype=np.uint8)

        ann = ev.load_expert_annotations()
        sub = ann[ann.Sample == sample]
        mean_xy = (float(sub.x.mean()), float(sub.y.mean())) if len(sub) else None
        cropped, x0, y0, _ = ev.resolve_and_crop_target_otolith(raw, mean_xy, seg_params)
        if cropped is None:
            print(f"  [{sample}] segmentacja nieudana, pomijam")
            continue
        axis_info = detect_axis(cropped, seg_params=seg_params,
                                nucleus_method=cfg.segmentation.nucleus_method,
                                axis_method=cfg.segmentation.axis_method)
        if axis_info is None:
            print(f"  [{sample}] oś nie policzona, pomijam")
            continue

        k = max(1, len(sub) // 2) if len(sub) else 4
        payload = wc.build_wedge_card_data(model, cropped, axis_info["mask"], axis_info, cfg, k=k)
        peaks = payload["peaks"]

        import cv2
        contours, _ = cv2.findContours(axis_info["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            axis_info["contour"] = max(contours, key=cv2.contourArea)

        photo = render_wedge_sector_on_photo(cropped, axis_info, payload["bands"][0].geom,
                                             peaks=peaks)
        raw_panel = render_wedge_canvas_panel(payload["canvases"], None, None,
                                              patch_size=cfg.data.patch_size)
        dens_panel = render_wedge_canvas_panel(payload["canvases"], payload["band_density"],
                                               peaks, patch_size=cfg.data.patch_size)

        d_all = np.concatenate([d.reshape(-1) for d in payload["band_density"]])
        print(f"  [{sample}] pasm={len(payload['bands'])} patchy={d_all.size} "
              f"density min/mean/max = {d_all.min():.4f}/{d_all.mean():.4f}/{d_all.max():.4f} "
              f"| pików={len(peaks)}")

        rows = "".join(
            f"<tr><td>{i}</td><td>{p['band']}</td><td>{p['t']:.3f}</td>"
            f"<td>{np.degrees(p['theta']):.1f}°</td><td>{p['score']:.4f}</td>"
            f"<td>({p['x']:.0f}, {p['y']:.0f})</td></tr>"
            for i, p in enumerate(peaks, start=1))

        blocks.append(f"""<section>
<h2>{sample}</h2>
<p class="cap">Pasm: <b>{len(payload['bands'])}</b> · patchy density: <b>{d_all.size}</b> ·
density min/śr/max: <b>{d_all.min():.4f} / {d_all.mean():.4f} / {d_all.max():.4f}</b> ·
dekodowanych pików: <b>{len(peaks)}</b></p>
<div style="display:inline-block;vertical-align:top;margin:0 12px 12px 0;">
  <h3>Krok 1 — sektor na zdjęciu</h3>
  <img src="{_b64(photo, 520)}" style="width:520px;">
</div>
<div style="display:inline-block;vertical-align:top;margin:0 12px 12px 0;">
  <h3>Krok 4 — piki liczbowo</h3>
  <table border="1" style="font-size:90%;">
  <tr><th>#</th><th>pasmo</th><th>t</th><th>θ</th><th>density</th><th>(x, y)</th></tr>
  {rows}</table>
</div>
<h3>Krok 2 — surowe kanwy (to widzi backbone)</h3>
<img src="{_b64(raw_panel, 1200)}" style="width:1200px;">
<h3>Krok 3 — density na kanwach + zdekodowane piki</h3>
<img src="{_b64(dens_panel, 1200)}" style="width:1200px;">
</section>""")

    warn = "" if trained else (
        '<p style="background:#fff3cd;padding:10px;border-left:4px solid #e0a800;">'
        '<b>Uwaga:</b> brak wytrenowanego checkpointu wycinka — wagi głowicy density są '
        '<b>losowe</b>. Geometria, kanwy, backbone i rysunki są prawdziwe; same wartości density '
        'i wybór pików — nie. Ten dokument weryfikuje PANELE, nie jakość lokalizacji.</p>')

    html = f"""<!DOCTYPE html><html lang="pl"><head><meta charset="UTF-8">
<title>Panele raportu — wycinek kątowy (weryfikacja 22.09)</title>
<style>
body {{font-family:sans-serif;max-width:1400px;margin:auto;padding:16px;}}
section {{margin-bottom:2em;border-top:2px solid #ccc;padding-top:1em;}}
table {{border-collapse:collapse;}} td,th {{padding:3px 7px;}}
h2 {{color:#1a237e;}} h3 {{color:#283593;}}
p.cap {{font-size:88%;color:#555;}}
</style></head><body>
<h1>Panele raportu — wycinek kątowy</h1>
<p class="cap">Config: <b>{Path(args.config).name}</b> · tryb: <b>{mode}</b> ·
backbone: <b>{cfg.model.backbone}</b> · Δθ = <b>{cfg.data.wedge_delta_theta_deg}°</b></p>
{warn}
{"".join(blocks)}
</body></html>"""

    out = OUTPUT_DIR / "wedge_report_panels.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nZapisano: {out}")


if __name__ == "__main__":
    main()
