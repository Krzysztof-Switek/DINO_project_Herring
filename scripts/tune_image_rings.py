"""Tune image-based ring detection (candidates.detect_image_rings) on REAL otoliths.

Model-INDEPENDENT — run it while a model is still training. For each image it
segments the otolith, then sweeps ``prominence`` × ``polarity`` for
``ring_detection.detect_and_draw_rings`` and writes a side-by-side montage
[original | variant | variant | …] so you can eyeball which settings land the
cyan rings on the real annual bands. Pick the winning values and put them into
``configs`` / the detection call.

Usage:
    python scripts/tune_image_rings.py --image-dir "Z:/Photo/Otolithes/HER/Processed" --n 6
    python scripts/tune_image_rings.py --image-dir <dir> --images a.png,b.png \
        --prominence 0.03,0.06,0.1 --polarity bright,dark --output-dir outputs/ring_tuning
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _label_tile(img_rgb: np.ndarray, text: str) -> np.ndarray:
    """Add a dark caption bar above an RGB tile."""
    H, W = img_rgb.shape[:2]
    bar_h = max(22, H // 20)
    bar = np.full((bar_h, W, 3), (28, 28, 32), np.uint8)
    scale = max(0.4, bar_h / 34.0)
    cv2.putText(bar, text, (6, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (240, 240, 240), max(1, int(scale * 1.6)), cv2.LINE_AA)
    return np.concatenate([bar, img_rgb], axis=0)


def build_tuning_montage(image_rgb: np.ndarray, axis_info: dict,
                         prominences: list[float], polarities: list[str],
                         thickness: int = 2) -> tuple[np.ndarray, dict]:
    """Return (montage_rgb, {label: n_rings}) for one image across the param sweep."""
    from src.ring_detection import detect_and_draw_rings

    tiles = [_label_tile(image_rgb.copy(), "original")]
    counts: dict = {}
    for pol in polarities:
        for prom in prominences:
            scales, overlay = detect_and_draw_rings(
                image_rgb, axis_info, prominence=prom, polarity=pol, thickness=thickness)
            label = f"{pol} p={prom} n={len(scales)}"
            counts[label] = int(len(scales))
            tiles.append(_label_tile(overlay, label))
    montage = np.concatenate(tiles, axis=1)
    return montage, counts


def _pick_images(image_dir: Path, images_arg: str | None, n: int, seed: int) -> list[Path]:
    if images_arg:
        return [image_dir / name.strip() for name in images_arg.split(",") if name.strip()]
    all_imgs = sorted(p for p in image_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    if not all_imgs:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_imgs), size=min(n, len(all_imgs)), replace=False)
    return [all_imgs[i] for i in sorted(idx)]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strojenie detekcji pierścieni z obrazu (real data)")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--images", default=None, help="Lista nazw plików po przecinku (zamiast losowania)")
    p.add_argument("--n", type=int, default=6, help="Ile losowych obrazów, gdy --images nie podano")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prominence", default="0.03,0.06,0.1",
                   help="Progi prominencji do sprawdzenia (po przecinku)")
    p.add_argument("--polarity", default="bright,dark", help="bright,dark (po przecinku)")
    p.add_argument("--output-dir", default="outputs/ring_tuning")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from src.otolith_axis import detect_axis

    args = parse_args(argv)
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"[błąd] Katalog nie istnieje: {image_dir}")
        return 1

    prominences = [float(x) for x in args.prominence.split(",") if x.strip()]
    polarities = [x.strip() for x in args.polarity.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _pick_images(image_dir, args.images, args.n, args.seed)
    if not paths:
        print(f"[błąd] Brak obrazów w {image_dir}")
        return 1

    ok = 0
    for path in paths:
        if not path.exists():
            print(f"  pominięto (brak): {path.name}")
            continue
        rgb = np.array(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)) \
            if cv2.imread(str(path)) is not None else None
        if rgb is None:
            print(f"  pominięto (nie wczytano): {path.name}")
            continue
        axis_info = detect_axis(rgb)
        if axis_info is None:
            print(f"  segmentacja nieudana: {path.name}")
            continue
        montage, counts = build_tuning_montage(rgb, axis_info, prominences, polarities)
        out_path = out_dir / f"{path.stem}_ring_tuning.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        print(f"  {path.name}: {counts} -> {out_path.name}")
        ok += 1

    print(f"\nGotowe: {ok}/{len(paths)} obrazów. Montaże w: {out_dir}")
    print("Wybierz kombinację, gdzie cyjanowe pierścienie pokrywają się z widocznymi prążkami,")
    print("i ustaw ją w wywołaniu detect_and_draw_rings / domyślnych parametrach.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
