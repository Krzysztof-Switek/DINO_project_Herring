"""Diagnostic tool — verify otolith segmentation on a sample of real images.

For each sampled image, writes a 3-panel grid PNG:
    [ original | mask | original + contour + centroid + axis ]

User reviews these visually to confirm the segmentation pipeline works on the
real data BEFORE running the full training/inference pipeline. If segmentation
fails on too many images, parameters in ``src/otolith_axis.py:segment_otolith``
need tuning.

Usage:
    python scripts/check_segmentation.py \
        --image-dir "Z:/Photo/Otolithes/HER/Processed" \
        --labels-csv outputs/demo/data/labels_combined.csv \
        --output-dir outputs/_segcheck \
        --n 20 \
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.otolith_axis import detect_axis  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Otolith segmentation diagnostic")
    p.add_argument("--image-dir",  required=True,
                   help="Directory with original-resolution otolith images")
    p.add_argument("--labels-csv", default=None,
                   help="Optional CSV with image_id (and optional age). "
                        "If not given or missing, samples directly from image-dir.")
    p.add_argument("--output-dir", default="outputs/_segcheck",
                   help="Where to write diagnostic grids")
    p.add_argument("--n",     type=int, default=20, help="number of images to sample")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--stratify-by-age", action="store_true",
                   help="Stratify by age (requires labels CSV with 'age' column)")
    return p.parse_args()


def _list_image_files(image_dir: Path) -> list[str]:
    """Return relative filenames of all otolith images in image_dir."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def _sample_images(df: pd.DataFrame, n: int, seed: int,
                   stratify: bool) -> pd.DataFrame:
    if "age" in df.columns and stratify and df["age"].notna().any():
        # Equal-ish counts per integer age bucket
        ages = df["age"].dropna().astype(int)
        buckets = sorted(ages.unique())
        per_bucket = max(1, n // max(len(buckets), 1))
        chosen: list[int] = []
        rng = np.random.default_rng(seed)
        for a in buckets:
            idx = df.index[df["age"] == a].tolist()
            rng.shuffle(idx)
            chosen.extend(idx[:per_bucket])
            if len(chosen) >= n:
                break
        chosen = chosen[:n]
        return df.loc[chosen].copy()
    return df.sample(n=min(n, len(df)), random_state=seed).copy()


def _draw_overlay(rgb: np.ndarray, info: dict,
                  contour_color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Draw contour + centroid cross + axis line on a copy of rgb."""
    img = rgb.copy()
    H = img.shape[0]
    lw = max(2, H // 250)
    cv2.drawContours(img, [info["contour"]], -1, contour_color, lw)
    cx, cy = info["centroid"]
    fx, fy = info["far_edge"]
    cv2.line(img, (cx, cy), (fx, fy), (255, 220, 0), lw)             # yellow axis
    cross = max(8, H // 80)
    cv2.line(img, (cx - cross, cy), (cx + cross, cy), (0, 100, 255), lw)  # blue cross
    cv2.line(img, (cx, cy - cross), (cx, cy + cross), (0, 100, 255), lw)
    return img


def _jaggedness(contour: np.ndarray) -> float:
    """Isoperimetric index perimeter²/(4π·area): 1=circle, higher=more squiggly."""
    area = cv2.contourArea(contour)
    per = cv2.arcLength(contour, True)
    return (per * per) / (4.0 * np.pi * area + 1e-9)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def _label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    """Black bar with white text on top of panel."""
    H, W = panel.shape[:2]
    bar = np.zeros((30, W, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def _fail_panel(rgb: np.ndarray, label: str) -> np.ndarray:
    warning = rgb.copy()
    W = rgb.shape[1]
    cv2.putText(warning, "FAILED", (W // 4, rgb.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
    return _label_panel(warning, label)


def _make_grid(rgb: np.ndarray, info_thr: dict | None,
               info_rad: dict | None) -> np.ndarray:
    """4-panel grid: original | threshold (old, cyan) | radial (new, green) | radial mask.

    Lets you eyeball whether the radial outline reaches the faint rim and is
    smoother than the old threshold contour (jaggedness index in the label).
    """
    if info_thr is not None:
        j = _jaggedness(info_thr["contour"])
        p_thr = _label_panel(_draw_overlay(rgb, info_thr, (0, 255, 255)),
                             f"2. threshold OLD (jag={j:.2f})")
    else:
        p_thr = _fail_panel(rgb, "2. threshold OLD")

    if info_rad is not None:
        j = _jaggedness(info_rad["contour"])
        p_rad = _label_panel(_draw_overlay(rgb, info_rad, (0, 255, 0)),
                             f"3. radial NEW (jag={j:.2f})")
        p_mask = _label_panel(_mask_to_rgb(info_rad["mask"]), "4. radial mask")
    else:
        p_rad = _fail_panel(rgb, "3. radial NEW")
        p_mask = _label_panel(np.zeros_like(rgb), "4. radial mask (FAILED)")

    return np.hstack([_label_panel(rgb, "1. original"), p_thr, p_rad, p_mask])


def main() -> None:
    args = _parse_args()

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    # Build DataFrame: from CSV if available, otherwise scan image_dir
    labels_csv = Path(args.labels_csv) if args.labels_csv else None
    if labels_csv and labels_csv.exists():
        df = pd.read_csv(labels_csv)
        if "image_id" not in df.columns:
            raise ValueError("CSV must contain an 'image_id' column")
        source = f"CSV {labels_csv}"
    else:
        if labels_csv:
            print(f"labels CSV not found ({labels_csv}); scanning image dir instead.")
        filenames = _list_image_files(image_dir)
        if not filenames:
            raise FileNotFoundError(f"no image files in {image_dir}")
        df = pd.DataFrame({"image_id": filenames})
        source = f"image dir {image_dir} ({len(filenames)} files)"

    sample = _sample_images(df, args.n, args.seed, stratify=args.stratify_by_age)
    print(f"Sampling {len(sample)} images from {source}")

    n_success = 0
    n_failed  = 0
    lengths: list[float] = []

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        iid = str(row["image_id"])
        img_path = image_dir / iid
        if not img_path.exists():
            print(f"  [{i:3d}] SKIP — file missing: {iid}")
            continue
        try:
            rgb = np.array(PILImage.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"  [{i:3d}] SKIP — read error ({e}): {iid}")
            continue

        info_thr = detect_axis(rgb, seg_params={"method": "threshold"})
        info_rad = detect_axis(rgb, seg_params={"method": "radial"})
        info = info_rad if info_rad is not None else info_thr
        if info is None:
            n_failed += 1
            status = "FAIL"
        else:
            n_success += 1
            lengths.append(info["length_px"])
            jr = _jaggedness(info_rad["contour"]) if info_rad is not None else float("nan")
            jt = _jaggedness(info_thr["contour"]) if info_thr is not None else float("nan")
            status = f"OK  axis={info['length_px']:.0f}px  jag old={jt:.2f}→new={jr:.2f}"

        grid = _make_grid(rgb, info_thr, info_rad)
        out_path = output_dir / f"grid_{i:03d}_{Path(iid).stem}.png"
        PILImage.fromarray(grid, mode="RGB").save(out_path)
        print(f"  [{i:3d}] {status}  → {out_path.name}")

    print()
    print(f"Summary: {n_success} OK  /  {n_failed} FAIL")
    if lengths:
        print(f"Axis length px — mean: {np.mean(lengths):.1f}, "
              f"min: {np.min(lengths):.1f}, max: {np.max(lengths):.1f}")
    print(f"\nReview the grids: {output_dir}")


if __name__ == "__main__":
    main()
