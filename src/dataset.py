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

from src.config import OtolithConfig
from src.otolith_axis import apply_background_mask, get_or_compute_mask, mask_bbox

REQUIRED_COLUMNS = {"image_id", "age", "split"}
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# ImageNet stats — DINOv2 was pretrained with these
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_SEX_MAP: Dict[str, float] = {
    "m": 1.0, "male": 1.0,
    "f": 0.0, "female": 0.0,
}


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

def build_transforms(image_size: int, split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
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
        self.transform = transform or build_transforms(cfg.data.image_size, split)

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

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        age = int(row["age"])

        image_tensor = self._load_image(image_id)
        age_ordinal = encode_age_ordinal(age, self.num_age_classes)

        sample: Dict = {
            "image": image_tensor,
            "age_ordinal": age_ordinal,
            "age": torch.tensor(age, dtype=torch.long),
            "image_id": image_id,
        }

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
