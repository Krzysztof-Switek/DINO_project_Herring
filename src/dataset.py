"""Otolith dataset loader with ordinal age encoding and optional metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from scripts.prepare_labels import extract_campaign_token
from src.config import OtolithConfig
from src.otolith_axis import (apply_background_mask, compute_polar_grid, get_or_compute_mask,
                              mask_bbox, resolve_centroid)

REQUIRED_COLUMNS = {"image_id", "age", "split"}
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# ImageNet stats — DINOv2 was pretrained with these
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_SEX_MAP: Dict[str, float] = {
    "m": 1.0, "male": 1.0,
    "f": 0.0, "female": 0.0,
}


def _wrap_angle(theta: np.ndarray) -> np.ndarray:
    """Wrap radians to (-pi, pi] — used when a flip transforms compute_polar_grid's
    theta_grid (see _load_image_with_polar), since pi - theta / -theta can land
    just outside that range."""
    return ((theta + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


# ---------------------------------------------------------------------------
# Ordinal encoding helpers
# ---------------------------------------------------------------------------

def encode_age_ordinal(age: int, num_classes: int) -> torch.Tensor:
    """Encode integer age as ordinal binary vector of length (num_classes - 1).

    Encoding: vec[i] = 1  iff  age > i   (CORAL / cumulative convention)
    Age 0  → all zeros
    Age k  → first k positions are 1, rest 0
    """
    k = num_classes - 1
    vec = torch.zeros(k, dtype=torch.float32)
    fill = min(age, k)
    if fill > 0:
        vec[:fill] = 1.0
    return vec


def decode_age_ordinal(logits: torch.Tensor) -> torch.Tensor:
    """Convert ordinal logit vector (before sigmoid) to predicted age integer.

    Works on batches (any leading dims) — last dim is the ordinal positions.
    """
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=-1).long()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(image_size: int, split: str, include_flips: bool = True) -> transforms.Compose:
    """``include_flips=False`` drops RandomHorizontalFlip/RandomVerticalFlip from the
    train pipeline — used when a caller needs to apply an IDENTICAL random flip
    decision to a second, non-image tensor (e.g. the E9 polar-coordinate grid,
    ``OtolithDataset._load_image_with_polar``) that this Compose has no hook to
    synchronise with, since torchvision's random transforms make their own private
    random choice with no way to inspect or replay it."""
    if split == "train":
        ops = [transforms.Resize((image_size, image_size))]
        if include_flips:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        ops += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
        return transforms.Compose(ops)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _encode_sex(val) -> float:
    if pd.isna(val):
        return 0.0
    return _SEX_MAP.get(str(val).strip().lower(), 0.0)


def _try_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OtolithDataset(Dataset):
    """PyTorch Dataset for otolith images.

    Returns per sample:
        image       : FloatTensor (3, H, W)  — normalized
        age_ordinal : FloatTensor (K-1,)     — ordinal binary target
        age         : LongTensor ()          — raw integer age
        image_id    : str
        metadata    : FloatTensor (M,)       — only if use_metadata=True
    """

    def __init__(
        self,
        cfg: OtolithConfig,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        labels_csv: Optional[str] = None,
        image_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.num_age_classes = cfg.model.num_age_classes
        self.use_metadata = cfg.model.use_metadata
        self.metadata_cols: List[str] = list(cfg.data.metadata_cols)

        root = Path(__file__).resolve().parents[1]
        csv_path = Path(labels_csv) if labels_csv else root / cfg.data.labels_csv
        self.img_dir = Path(image_dir) if image_dir else root / cfg.data.image_dir

        self.mask_background = cfg.data.mask_background
        self.mask_cache_dir: Optional[Path] = None
        if self.mask_background:
            # Project-relative by default (NOT under img_dir, which may be a
            # read-only network share) — one cache shared across every run, keyed
            # only by image_id since a mask depends solely on the raw image.
            self.mask_cache_dir = (Path(cfg.data.mask_cache_dir) if cfg.data.mask_cache_dir
                                   else root / "data" / "masks_cache")
            self.mask_cache_dir.mkdir(parents=True, exist_ok=True)

        if not csv_path.exists():
            raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

        full_df = pd.read_csv(csv_path)
        self._validate_columns(full_df)

        # Build population map from entire file for consistent encoding
        self._pop_map: Dict[str, int] = {}
        if "population" in full_df.columns:
            for val in full_df["population"].dropna().unique():
                key = str(val).strip()
                if key not in self._pop_map:
                    self._pop_map[key] = len(self._pop_map) + 1

        self.df = full_df[full_df["split"] == split].reset_index(drop=True)
        self.df = self._maybe_demo_subsample(self.df, split)
        # Change C (05.08): materialized "campaign" column (from scan_labels.py /
        # prepare_labels.py) is optional/additive — REQUIRED_COLUMNS deliberately
        # does NOT include it, so this works on CSVs generated before the label
        # pipeline was updated too (see _effective_age's re-parsing fallback).
        self._has_campaign_col = "campaign" in self.df.columns

        # E9: the polar-coordinate grid (needed only when the concentricity loss is
        # actually on) requires an EXPLICIT, shared flip decision between the image
        # and the geometry grid — build_transforms' built-in random flips can't be
        # synchronised with a second tensor, so they're disabled here and redone
        # manually in _load_image_with_polar(). Off by default (density_concentricity_
        # weight=0.0): zero behaviour change for every existing config.
        self._need_polar = (
            bool(getattr(cfg.model, "use_density_head", False))
            and getattr(cfg.model, "density_concentricity_weight", 0.0) > 0.0
        )
        self.transform = transform or build_transforms(
            cfg.data.image_size, split, include_flips=not self._need_polar)

    # ------------------------------------------------------------------
    # Demo mode — limit dataset right at the source
    # ------------------------------------------------------------------

    def _maybe_demo_subsample(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        """Limit dataset to cfg.demo.max_{split}_samples when demo mode is on.

        Sampling is deterministic (uses cfg.project.seed) so re-runs and the
        4 cross-condition inference passes see a consistent subset.
        """
        demo = getattr(self.cfg, "demo", None)
        if demo is None or not getattr(demo, "enabled", False):
            return df
        limit_map = {
            "train": demo.max_train_samples,
            "val":   demo.max_val_samples,
            "test":  demo.max_test_samples,
        }
        limit = limit_map.get(split)
        if limit is None or len(df) <= limit:
            return df
        seed = getattr(self.cfg.project, "seed", 42)
        return df.sample(n=limit, random_state=seed).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Labels CSV missing required columns: {missing}")
        if not pd.api.types.is_numeric_dtype(df["age"]):
            raise ValueError("Column 'age' must be numeric")
        # Wiersze z split=None to sieroty i age=-9 — wykluczone z treningu, pomijamy je
        split_rows = df[df["split"].notna()]
        if split_rows["age"].lt(0).any():
            raise ValueError("Column 'age' contains negative values")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def _effective_age(self, row: "pd.Series", image_id: str, recorded_age: int) -> int:
        """Change C (05.08, opt-in, default OFF): fish caught in BITS1q/BITS2q
        hauls (Jan-June) may not have finished forming that year's ring — the TRUE
        number of visible, complete increments can be recorded_age - 1. Prefers the
        materialized "campaign" column when present, falls back to re-parsing
        image_id (works even on CSVs generated before the label pipeline added the
        column). Feeds BOTH CORAL's ordinal target and MIL/density/val-MAE (all of
        which read the single "age" value returned from here) — per the user's
        explicit choice, not a density-only adjustment.
        """
        if not self.cfg.data.quarter_age_adjustment_enabled:
            return recorded_age
        campaign = None
        if self._has_campaign_col:
            val = row.get("campaign")
            if pd.notna(val):
                campaign = str(val)
        if campaign is None:
            campaign = extract_campaign_token(image_id)
        if campaign in set(self.cfg.data.quarter_age_adjustment_campaigns):
            return max(recorded_age - 1, 0)
        return recorded_age

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        recorded_age = int(row["age"])
        age = self._effective_age(row, image_id, recorded_age)

        if self._need_polar:
            image_tensor, polar_grid, polar_valid, polar_theta = self._load_image_with_polar(image_id)
        else:
            image_tensor = self._load_image(image_id)
            polar_grid = polar_valid = polar_theta = None
        age_ordinal = encode_age_ordinal(age, self.num_age_classes)

        sample: Dict = {
            "image": image_tensor,
            "age_ordinal": age_ordinal,
            "age": torch.tensor(age, dtype=torch.long),
            "age_original": torch.tensor(recorded_age, dtype=torch.long),
            "image_id": image_id,
        }
        if polar_grid is not None:
            sample["polar_grid"] = polar_grid
            sample["polar_valid"] = polar_valid
            sample["polar_theta"] = polar_theta

        if self.use_metadata and self.metadata_cols:
            sample["metadata"] = self._encode_metadata(row)

        return sample

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self, image_id: str) -> torch.Tensor:
        path = self.img_dir / image_id
        if not path.exists():
            for ext in IMAGE_EXTENSIONS:
                candidate = self.img_dir / (image_id + ext)
                if candidate.exists():
                    path = candidate
                    break
        image = Image.open(path).convert("RGB")
        if self.mask_background:
            image = self._mask_background(image, image_id)
        return self.transform(image)

    def _load_image_with_polar(self, image_id: str) -> tuple:
        """Like ``_load_image``, but also returns the (H_p, W_p) polar-coordinate
        grid (E9 concentricity loss) computed from the SAME cached mask used for
        background-masking — with an EXPLICIT, shared random flip decision applied
        identically to the image and the polar grid (see ``build_transforms``'s
        ``include_flips`` docstring for why this can't just reuse the normal
        Compose-embedded random flips on the train split).

        Returns (image_tensor, polar_t_grid, polar_valid_grid, polar_theta_grid); the
        latter three are all-zero / all-False when segmentation fails, matching the
        rest of this module's "never crash on a bad photo" fallback philosophy.
        polar_theta_grid (05.08, windowed/"local" E9) carries a signed direction, so
        unlike the other two it needs an actual VALUE transform (not just an array
        reversal) under a flip — see the wrap_angle calls below.
        """
        path = self.img_dir / image_id
        if not path.exists():
            for ext in IMAGE_EXTENSIONS:
                candidate = self.img_dir / (image_id + ext)
                if candidate.exists():
                    path = candidate
                    break
        image = Image.open(path).convert("RGB")
        rgb = np.array(image, dtype=np.uint8)

        mask = None
        centroid = None
        if self.mask_background:
            cache_path = self.mask_cache_dir / f"{Path(image_id).stem}_mask.png"
            mask = get_or_compute_mask(rgb, cache_path, seg_params=self.cfg.segmentation.as_params())
            if mask is not None:
                centroid = resolve_centroid(rgb, mask, self.cfg.segmentation.nucleus_method)

        h_patches = w_patches = self.cfg.data.image_size // self.cfg.data.patch_size
        if mask is not None and centroid is not None:
            t_grid, valid_grid, theta_grid = compute_polar_grid(mask, centroid, h_patches, w_patches)
            rgb = apply_background_mask(rgb, mask)
        else:
            t_grid = np.zeros((h_patches, w_patches), dtype=np.float32)
            valid_grid = np.zeros((h_patches, w_patches), dtype=bool)
            theta_grid = np.zeros((h_patches, w_patches), dtype=np.float32)

        pil_img = Image.fromarray(rgb)
        if self.split == "train":
            do_hflip, do_vflip = self._decide_flip()
            if do_hflip:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                t_grid = np.ascontiguousarray(t_grid[:, ::-1])
                valid_grid = np.ascontiguousarray(valid_grid[:, ::-1])
                # theta is a signed DIRECTION, not a plain scalar field — mirroring the
                # image negates dx (theta=atan2(dy,dx)), so besides reversing the array
                # (which relocates values to their new spatial position, same as t_grid/
                # valid_grid above) the VALUE itself must transform: atan2(dy,-dx) =
                # wrap(pi - atan2(dy,dx)). Getting only the array-reversal half of this
                # right (as t_grid/valid_grid do) would silently attach each patch a
                # pre-flip direction that no longer matches its post-flip geometry.
                theta_grid = _wrap_angle(np.pi - theta_grid[:, ::-1])
            if do_vflip:
                pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
                t_grid = np.ascontiguousarray(t_grid[::-1, :])
                valid_grid = np.ascontiguousarray(valid_grid[::-1, :])
                # Mirrors negate dy instead: atan2(-dy,dx) = wrap(-atan2(dy,dx)).
                theta_grid = _wrap_angle(-theta_grid[::-1, :])

        image_tensor = self.transform(pil_img)
        return (
            image_tensor,
            torch.from_numpy(t_grid.copy()),
            torch.from_numpy(valid_grid.copy()),
            torch.from_numpy(np.ascontiguousarray(theta_grid).copy()),
        )

    def _decide_flip(self) -> tuple:
        """Explicit horizontal/vertical flip decision — split out to its own method
        (instead of inlining ``torch.rand`` calls) purely so tests can force a
        specific flip combination via monkeypatching, without touching the global
        RNG that ``ColorJitter``/other transforms also rely on."""
        return (torch.rand(()).item() < 0.5, torch.rand(()).item() < 0.5)

    def _mask_background(self, image: Image.Image, image_id: str) -> Image.Image:
        """Blank out everything outside the segmented otolith (MASK_FILL_RGB).

        Falls back to the unmasked image when segmentation fails (e.g. a uniform or
        unusual photo) — masking must never be able to crash training on a bad image.
        """
        rgb = np.array(image, dtype=np.uint8)
        cache_path = self.mask_cache_dir / f"{Path(image_id).stem}_mask.png"
        mask = get_or_compute_mask(rgb, cache_path, seg_params=self.cfg.segmentation.as_params())
        if mask is None:
            return image
        return Image.fromarray(apply_background_mask(rgb, mask))

    # ------------------------------------------------------------------
    # Metadata encoding
    # ------------------------------------------------------------------

    def _encode_metadata(self, row) -> torch.Tensor:
        values: List[float] = []
        for col in self.metadata_cols:
            if col not in row.index or pd.isna(row[col]):
                values.append(0.0)
            elif col == "sex":
                values.append(_encode_sex(row[col]))
            elif col == "population":
                key = str(row[col]).strip()
                values.append(float(self._pop_map.get(key, 0)))
            else:
                values.append(_try_float(row[col]))
        return torch.tensor(values, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def metadata_dim(self) -> int:
        """Number of metadata features returned per sample."""
        return len(self.metadata_cols) if self.use_metadata else 0


# ---------------------------------------------------------------------------
# Density-head fine-tuning dataset (22.07)
# ---------------------------------------------------------------------------

class DensityFineTuneDataset(OtolithDataset):
    """Like ``OtolithDataset``, but resizes to ``density_image_size`` (not
    ``cfg.data.image_size``) and optionally crops to the segmentation mask's bounding
    box FIRST — mirrors the inference-time density signal built in
    ``scripts/run_pipeline.py::_compute_axis_data_for_samples`` (Zmiana A+B, 22.07), so
    ``density_head`` can be fine-tuned on exactly the patch-grid distribution it will see
    at report-generation time (a measured train/inference resolution mismatch — see
    ``plans and summaries/22.07_TO_DO.MD``).

    Requires ``cfg.data.mask_background=True`` — reuses the same mask cache/segmentation
    as the main dataset (``OtolithDataset._mask_background``), just applied inline here
    since cropping needs the raw mask array, not only the masked image.
    """

    def __init__(
        self,
        cfg: OtolithConfig,
        split: str = "train",
        density_image_size: Optional[int] = None,
        crop_to_otolith: bool = False,
        pad_frac: float = 0.05,
        **kwargs,
    ) -> None:
        self.crop_to_otolith = crop_to_otolith
        self.pad_frac = pad_frac
        size = density_image_size or cfg.data.image_size
        transform = kwargs.pop("transform", None) or build_transforms(size, split)
        super().__init__(cfg, split=split, transform=transform, **kwargs)
        if not self.mask_background:
            raise ValueError("DensityFineTuneDataset requires cfg.data.mask_background=True")

    def _load_image(self, image_id: str) -> torch.Tensor:
        path = self.img_dir / image_id
        if not path.exists():
            for ext in IMAGE_EXTENSIONS:
                candidate = self.img_dir / (image_id + ext)
                if candidate.exists():
                    path = candidate
                    break
        image = Image.open(path).convert("RGB")
        rgb = np.array(image, dtype=np.uint8)
        cache_path = self.mask_cache_dir / f"{Path(image_id).stem}_mask.png"
        mask = get_or_compute_mask(rgb, cache_path, seg_params=self.cfg.segmentation.as_params())
        if mask is not None:
            rgb = apply_background_mask(rgb, mask)
            if self.crop_to_otolith:
                x0, y0, w, h = mask_bbox(mask, self.pad_frac)
                rgb = rgb[y0:y0 + h, x0:x0 + w]
        return self.transform(Image.fromarray(rgb))
