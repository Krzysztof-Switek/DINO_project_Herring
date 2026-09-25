"""Strip-approach figures for the process report (geometry + real expert annotations, no backbone).

Produces, per sample:
  <S>_strip_corridor.jpg — photo + contour + axis + the +-49px strip corridor + real KK/SS points,
                            so the geometric ceiling (GT outside the corridor) is visible directly.
  <S>_strip_flat.jpg     — the straightened strip the model actually reads, with the patch grid,
                            the warped GT points, and the background-fill zone marked.
Reuses production code only (src.strip_extraction, src.otolith_axis, expert_annotation_eval).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(r"C:\Users\kswitek\Documents\DINO_project_Herring")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

import expert_annotation_eval as ev
from PIL import Image as PILImage
from src.config import load_config
from src.otolith_axis import detect_axis, MASK_FILL_RGB
from src.strip_extraction import (strip_transform_matrix, extract_strip,
                                  extract_strip_validity)
from src.visualization import _put_text_pl

cfg = load_config(str(ROOT / "configs" / "config_strip_a.yaml"))
seg = cfg.segmentation.as_params()
LEN, WID, PS = cfg.data.strip_length_px, cfg.data.strip_width_px, cfg.data.patch_size
print("pasek:", LEN, "x", WID, "px =", LEN // PS, "x", WID // PS, "patchy")

KK = (220, 60, 255)      # BGR magenta
SS = (255, 170, 60)      # BGR blue-ish
ann = ev.load_expert_annotations()
meta = {}

for sample in ["Z20", "Z38"]:
    sub = ann[ann.Sample == sample]
    mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
    raw = np.array(PILImage.open(ev.IMAGE_DIR / f"{sample}.jpg").convert("RGB"), dtype=np.uint8)
    crop, x0, y0, _ = ev.resolve_and_crop_target_otolith(raw, mean_xy, seg)
    ax = detect_axis(crop, seg_params=seg, nucleus_method=cfg.segmentation.nucleus_method,
                     axis_method=cfg.segmentation.axis_method)
    cx, cy = ax["centroid"]; fx, fy = ax["far_edge"]; L = ax["length_px"]
    M = strip_transform_matrix((cx, cy), (fx, fy), L, LEN, WID)
    M3 = np.vstack([M, [0, 0, 1]]); Minv = np.linalg.inv(M3)

    pts = [(float(r.x) - x0, float(r.y) - y0, r.annotator) for r in sub.itertuples()]

    # ---------- A: corridor on the photo ----------
    vis = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    H, W = vis.shape[:2]
    vis = np.where((ax["mask"] > 0)[:, :, None], vis, (vis * 0.45).astype(np.uint8))
    cv2.drawContours(vis, [ax["contour"]], -1, (0, 255, 255), max(2, W // 700))
    corners = np.array([[0, 0, 1], [LEN, 0, 1], [LEN, WID, 1], [0, WID, 1]], dtype=float).T
    img_c = (Minv @ corners)[:2].T.astype(np.int32)
    ov = vis.copy(); cv2.fillPoly(ov, [img_c], (90, 220, 90))
    vis = cv2.addWeighted(ov, 0.18, vis, 0.82, 0)
    cv2.polylines(vis, [img_c], True, (90, 220, 90), max(2, W // 700))
    cv2.line(vis, (int(cx), int(cy)), (int(fx), int(fy)), (0, 0, 255), max(3, W // 520))
    cv2.circle(vis, (int(cx), int(cy)), max(6, W // 190), (255, 255, 255), -1)
    cv2.circle(vis, (int(cx), int(cy)), max(6, W // 190), (0, 0, 0), 2)
    off = []
    for px, py, who in pts:
        col = KK if who == "KK" else SS
        cv2.circle(vis, (int(px), int(py)), max(9, W // 130), col, -1)
        cv2.circle(vis, (int(px), int(py)), max(9, W // 130), (20, 20, 20), 2)
        sx, sy = (M @ np.array([px, py, 1.0]))
        off.append(abs(sy - WID / 2.0))
    s = 1080.0 / max(H, W)
    if s < 1:
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    _put_text_pl(vis, "korytarz paska (±49 px)", (12, 10), size=18, color=(90, 220, 90), bg=(0, 0, 0))
    _put_text_pl(vis, "oś odczytu", (12, 36), size=18, color=(120, 120, 255), bg=(0, 0, 0))
    _put_text_pl(vis, "przyrosty ekspertów (KK / SS)", (12, 62), size=18, color=(230, 120, 255), bg=(0, 0, 0))
    cv2.imwrite(str(OUT / f"{sample}_strip_corridor.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ---------- B: straightened strip ----------
    strip, _ = extract_strip(crop, ax["mask"], (cx, cy), (fx, fy), L, LEN, WID)
    SC, BAND = 4, 52
    body = cv2.resize(cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), (LEN, WID * SC),
                      interpolation=cv2.INTER_NEAREST)
    for x in range(0, LEN + 1, PS):
        cv2.line(body, (x, 0), (x, body.shape[0]), (255, 255, 255), 1)
    for r in range(0, WID // PS + 1):
        cv2.line(body, (0, r * PS * SC), (LEN, r * PS * SC), (255, 255, 255), 1)
    val = extract_strip_validity(ax["mask"], (cx, cy), (fx, fy), L, LEN, WID, PS)
    hp, wp = val.shape
    nbad = 0
    for r in range(hp):
        for c in range(wp):
            if val[r, c] < 0.5:
                nbad += 1
                cv2.rectangle(body, (c * PS, r * PS * SC),
                              (c * PS + PS - 1, (r + 1) * PS * SC - 1), (60, 60, 240), 2)
    st = np.full((WID * SC + 2 * BAND, LEN, 3), 24, np.uint8)
    st[BAND:BAND + WID * SC] = body
    yc = BAND + int(WID / 2 * SC)
    for x in range(0, LEN, 18):
        cv2.line(st, (x, yc), (x + 9, yc), (0, 0, 255), 2)
    out_hi = []
    for px, py, who in pts:
        sx, sy = (M @ np.array([px, py, 1.0]))
        col = KK if who == "KK" else SS
        if 0 <= sy < WID:
            cv2.drawMarker(st, (int(sx), BAND + int(sy * SC)), col, cv2.MARKER_TILTED_CROSS, 24, 3)
        else:
            yy = BAND - 20 if sy < 0 else BAND + WID * SC + 20
            cv2.drawMarker(st, (int(sx), yy), col, cv2.MARKER_TILTED_CROSS, 24, 3)
            cv2.arrowedLine(st, (int(sx), BAND + (4 if sy < 0 else WID * SC - 4)),
                            (int(sx), yy + (8 if sy < 0 else -8)), col, 2, tipLength=.4)
            out_hi.append(abs(sy - WID / 2.0))
    _put_text_pl(st, f"{sample} · pasek {LEN}×{WID} px = {LEN//PS}×{WID//PS} patchy · "
                     f"jądro po lewej, brzeg po prawej", (6, 6), size=21,
                 color=(255, 255, 255), bg=(0, 0, 0))
    if nbad:
        _put_text_pl(st, f"patche poza tkanką: {nbad} z {hp*wp}", (LEN - 300, st.shape[0] - 30),
                     size=19, color=(120, 120, 255), bg=(0, 0, 0))
    if out_hi:
        _put_text_pl(st, f"wszystkie punkty ekspertów poza korytarzem "
                         f"({min(out_hi):.0f}–{max(out_hi):.0f} px od osi)",
                     (6, st.shape[0] - 30), size=21, color=(230, 120, 255), bg=(0, 0, 0))
    else:
        _put_text_pl(st, "punkty ekspertów mieszczą się w korytarzu, przy samej jego krawędzi",
                     (6, st.shape[0] - 30), size=21, color=(230, 120, 255), bg=(0, 0, 0))
    cv2.imwrite(str(OUT / f"{sample}_strip_flat.jpg"), st, [cv2.IMWRITE_JPEG_QUALITY, 90])

    meta[sample] = {"axis_len_px": L, "gt_offsets_px": [round(o, 1) for o in off],
                    "corridor_half_px": WID / 2, "patches_off_tissue": int(nbad), "patches_total": int(hp*wp)}
    print(sample, "| odchylenia GT od osi [px]:", [round(o) for o in off],
          "| korytarz ±", WID / 2, "| patche poza tkanką:", meta[sample]["patches_off_tissue"], "/", meta[sample]["patches_total"])

(OUT / "strip_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("DONE")
