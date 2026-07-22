"""Generate a synthetic Krok-4 dataset + Python reference result, dump both as JSON so
krok4_crosscheck.js can cross-check the JS reimplementation (_KROK4_JS) against the real
Python ring_extraction math. Usage: python gen_krok4_crosscheck.py <out.json> [seed]"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2

from src.ring_extraction import (
    dp_interactive_data, density_peaks, classical_increments,
    _cluster_by_radius, _merge_clusters, _dp_select_t,
)

# Synthetic ellipse otolith so axis_info is realistic (non-trivial contour_pts/centroid).
H, W = 400, 300
mask = np.zeros((H, W), dtype=np.uint8)
cv2.ellipse(mask, (150, 200), (80, 150), 0, 0, 360, 255, -1)
centroid = (150, 200)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour = max(contours, key=cv2.contourArea)
axis_info = {"mask": mask, "centroid": centroid, "far_edge": (150, 350), "contour": contour, "length_px": 150.0}

seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
rng = np.random.default_rng(seed)
grid = rng.random((20, 16)).astype(np.float32)   # fake density map (H_p, W_p)
gray = rng.random((H, W)).astype(np.float32)      # fake grayscale image
age = 5

interactive = dp_interactive_data(grid, gray, axis_info, H, W, age)

PROM = 0.12
GAP = 0.05
TOL = 0.07

dpk, _ = density_peaks(grid, axis_info, H, W, prominence=PROM,
                       min_distance=interactive["density_min_distance"])
cinc = classical_increments(gray, axis_info, prominence=PROM,
                            min_distance=interactive["classical_min_distance"])
cpk = cinc["peaks"]
merged = _merge_clusters(dpk, cpk, TOL, len(interactive["contour_pts"]))
chosen = _dp_select_t([(t, s) for (t, s, _src) in merged], age, GAP)

out = {
    "interactive": interactive,
    "params": {"prom": PROM, "gap": GAP, "tol": TOL, "age": age},
    "python_chosen": chosen,
}
Path(sys.argv[1]).write_text(json.dumps(out), encoding="utf-8")
print(f"wrote {sys.argv[1]}; python_chosen={chosen}")
