"""Render wedge-geometry figures for the process summary (geometry only, no backbone).

23.09 FIX: the sector/arc radii are now looked up with the project's own
``wedge_extraction._r_at_angle``. The first version of this script re-derived the lookup by hand
and forgot that ``compute_R_theta`` bins span ``(-pi, pi]`` (bin i = -pi + i*2pi/n), not [0, 2pi)
— a half-turn offset, so every drawn radius came from the OPPOSITE direction and the sector
overshot the contour on tailed otoliths. Production code was never affected (it uses
``_r_at_angle``/``_remap_maps``); only this drawing script was.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(r"C:\Users\kswitek\Documents\DINO_project_Herring")
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

from src.config import load_config
from src.otolith_axis import (segment_otolith, resolve_centroid, _largest_contour,
                              find_reading_edge, find_farthest_edge, compute_R_theta)
from src.wedge_extraction import (wedge_band_geometries_from_axis_info,
                                  extract_polar_wedge_band, _r_at_angle)
from src.visualization import _put_text_pl

cfg = load_config(str(ROOT / "configs" / "config_wedge_b.yaml"))
seg = cfg.segmentation.as_params() if hasattr(cfg.segmentation, "as_params") else {}

SAMPLES = ["ZEGAR_Z20.jpg", "ZEGAR_Z38.jpg"]
BAND_EDGES = list(cfg.data.wedge_band_edges_t)
BAND_W = [n * cfg.data.patch_size for n in cfg.data.wedge_band_n_angle_patches]
BAND_H = [n * cfg.data.patch_size for n in cfg.data.wedge_band_n_radius_patches]
DTH = cfg.data.wedge_delta_theta_deg
print("bands", BAND_EDGES, BAND_W, BAND_H, DTH)

meta = {}
for name in SAMPLES:
    p = ROOT / "data" / "ZEGAR" / "Processed" / name
    bgr = cv2.imread(str(p))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = segment_otolith(rgb, **seg)
    if mask is None:
        print("seg failed", name); continue
    cen = resolve_centroid(rgb, mask, "geometric")
    cont = _largest_contour(mask)
    tail = find_farthest_edge(mask, cen)
    edge = find_reading_edge(rgb, mask, cen)
    cx, cy = cen; fx, fy = edge
    axis_ang = float(np.arctan2(fy - cy, fx - cx))
    R_theta = compute_R_theta(mask, (cx, cy), 720).astype(np.float32)
    length = float(np.hypot(fx - cx, fy - cy))
    print(name, "centroid", cen, "edge", edge, "tail", tail, "len", round(length, 1))

    # ---------- Fig A: sector on the photo ----------
    vis = bgr.copy()
    H, W = vis.shape[:2]
    dim = (vis * 0.45).astype(np.uint8)
    vis = np.where((mask > 0)[:, :, None], vis, dim)
    cv2.drawContours(vis, [cont], -1, (0, 255, 255), max(2, W // 700))

    half = np.radians(DTH) / 2.0
    angs = np.linspace(axis_ang - half, axis_ang + half, 160)
    # SAME lookup the production remap uses — do not re-derive (see module docstring).
    pts = [(cx, cy)] + [(cx + _r_at_angle(R_theta, a) * np.cos(a),
                         cy + _r_at_angle(R_theta, a) * np.sin(a)) for a in angs]
    poly = np.array(pts, dtype=np.int32)
    ov = vis.copy()
    cv2.fillPoly(ov, [poly], (255, 0, 255))
    vis = cv2.addWeighted(ov, 0.22, vis, 0.78, 0)
    cv2.polylines(vis, [poly], True, (255, 0, 255), max(2, W // 700))

    for t, col in zip(BAND_EDGES[1:-1], [(80, 220, 80), (0, 190, 255), (0, 110, 255)]):
        arc = np.array([[int(cx + t * _r_at_angle(R_theta, a) * np.cos(a)),
                         int(cy + t * _r_at_angle(R_theta, a) * np.sin(a))] for a in angs],
                       dtype=np.int32)
        cv2.polylines(vis, [arc], False, col, max(2, W // 800))

    cv2.line(vis, (int(cx), int(cy)), (int(fx), int(fy)), (0, 0, 255), max(3, W // 500))
    cv2.circle(vis, (int(cx), int(cy)), max(6, W // 180), (255, 255, 255), -1)
    cv2.circle(vis, (int(cx), int(cy)), max(6, W // 180), (0, 0, 0), 2)
    cv2.circle(vis, (int(fx), int(fy)), max(6, W // 200), (0, 0, 255), -1)

    scale = 1100.0 / max(H, W)
    if scale < 1:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sx = scale if scale < 1 else 1.0

    # Labels ON the picture — the first version relied on a prose caption alone and was unreadable.
    def lab(text, xy, color):
        _put_text_pl(vis, text, (int(xy[0]), int(xy[1])), size=17, color=color, bg=(0, 0, 0))
    lab("jądro (nucleus)", (cx * sx + 14, cy * sx - 8), (255, 255, 255))
    lab("oś odczytu (nucleus → brzeg)", ((cx + (fx - cx) * 0.55) * sx + 12, (cy + (fy - cy) * 0.55) * sx), (120, 120, 255))
    lab("WYCINEK — to trafia do modelu", (12, 10), (255, 120, 235))
    lab("granice pasm t=0,6 / 0,8 / 0,9", (12, 36), (120, 220, 255))
    cv2.imwrite(str(OUT / f"{name[:-4]}_sector.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ---------- Fig B: band canvases with patch grid ----------
    axis_info = {"centroid": (cx, cy), "far_edge": (fx, fy)}
    bands = wedge_band_geometries_from_axis_info(mask, axis_info, DTH, BAND_EDGES, BAND_W, BAND_H)
    panels, info = [], []
    ps = cfg.data.patch_size
    for i, b in enumerate(reversed(bands)):          # edge band on top
        k = len(bands) - 1 - i
        img = cv2.cvtColor(extract_polar_wedge_band(rgb, mask, b), cv2.COLOR_RGB2BGR)
        for x in range(0, img.shape[1] + 1, ps):
            cv2.line(img, (x, 0), (x, img.shape[0]), (255, 255, 255), 1)
        for y in range(0, img.shape[0] + 1, ps):
            cv2.line(img, (0, y), (img.shape[1], y), (255, 255, 255), 1)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 255, 255), 2)
        _put_text_pl(img, f"pasmo {k} · t={BAND_EDGES[k]:g}–{BAND_EDGES[k+1]:g} · "
                          f"{BAND_W[k]//ps}×{BAND_H[k]//ps} patchy",
                     (6, 5), size=22, color=(255, 255, 255), bg=(0, 0, 0))
        panels.append((k, img))
        info.append({"band": k, "t": [BAND_EDGES[k], BAND_EDGES[k + 1]],
                     "cols": BAND_W[k] // ps, "rows": BAND_H[k] // ps,
                     "px": [BAND_H[k], BAND_W[k]], "patches": (BAND_W[k] // ps) * (BAND_H[k] // ps)})

    target_w = max(p.shape[1] for _, p in panels)
    rows = []
    for k, p in panels:
        s = target_w / p.shape[1]
        rows.append(cv2.resize(p, (target_w, max(1, int(round(p.shape[0] * s)))),
                               interpolation=cv2.INTER_NEAREST))
        rows.append(np.full((10, target_w, 3), 30, np.uint8))
    comb = np.vstack(rows[:-1])
    s = 1400.0 / comb.shape[1]
    if s < 1:
        comb = cv2.resize(comb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / f"{name[:-4]}_bands_stack.jpg"), comb, [cv2.IMWRITE_JPEG_QUALITY, 88])

    meta[name] = {"centroid": [float(cx), float(cy)], "edge": [float(fx), float(fy)],
                  "tail": [float(tail[0]), float(tail[1])], "axis_len_px": length,
                  "bands": info, "img_shape": [int(H), int(W)]}

(OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("DONE")
