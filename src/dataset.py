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
from src.otolith_axis import (apply_background_mask, compute_polar_grid, detect_axis,
                              detect_axis_candidates, get_or_compute_mask, mask_bbox,
                              resolve_centroid)
from src.strip_extraction import (get_or_compute_strip, get_or_compute_strip_validity,
                                  load_strip, load_strip_validity)
from src.wedge_extraction import (WedgeBandGeometry, extract_polar_wedge_band,
                                  extract_polar_wedge_band_validity, extract_polar_wedge_validity,
                                  get_or_compute_wedge, load_wedge, save_wedge,
                                  wedge_band_geometries_from_axis_info, wedge_band_polar_coords,
                                  wedge_band_row_frac_edges, wedge_geometry_from_axis_info,
                                  wedge_polar_coords)

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

def build_transforms(
    image_size: int, split: str, include_flips: bool = True, strip: bool = False,
    wedge: bool = False,
) -> transforms.Compose:
    """``include_flips=False`` drops RandomHorizontalFlip/RandomVerticalFlip from the
    train pipeline — used when a caller needs to apply an IDENTICAL random flip
    decision to a second, non-image tensor (e.g. the E9 polar-coordinate grid,
    ``OtolithDataset._load_image_with_polar``) that this Compose has no hook to
    synchronise with, since torchvision's random transforms make their own private
    random choice with no way to inspect or replay it.

    ``strip=True`` (02.09, dendrochronology-strip experiment — see
    ``OtolithDataset._load_strip_image``): the input is already the exact target size
    (``src.strip_extraction.extract_strip``'s output), so ``Resize`` is skipped —
    ``image_size`` is then unused. ``RandomHorizontalFlip`` is also unconditionally
    dropped regardless of ``include_flips`` — flipping along the strip's length axis
    would reverse the just-canonicalized (nucleus -> far_edge) reading direction, the
    same problem (and fix) already documented in
    ``scripts/diagnostics/train_zegar_localization_head_canonical.py``.
    ``RandomVerticalFlip`` (mirrors across the width axis, meaning-preserving) is
    unaffected by ``strip`` and needs no cross-tensor sync here (unlike the polar-grid
    case above) — the strip branch has no second geometric tensor riding along with it.

    ``wedge=True`` (09.09, polar-wedge experiment — see
    ``OtolithDataset._load_wedge_image``): input is already the exact target size
    (``src.wedge_extraction.extract_polar_wedge``'s output), Resize skipped, same as
    ``strip``. The flip logic is the OPPOSITE of the strip's, because the two axes mean
    opposite things: a wedge's ROWS are radius (nucleus -> far edge, t=0..1) — reversing
    them would be the same reading-direction violation ``strip`` avoids by dropping
    horizontal flip, so ``RandomVerticalFlip`` is unconditionally dropped here. A
    wedge's COLUMNS are angle around the reading axis — mirroring them (angle -> -angle)
    is a genuine geometric reflection symmetry (a ring seen swept one angular direction
    looks structurally the same swept the other way), so ``RandomHorizontalFlip`` is
    kept, unlike the strip's own dropped horizontal flip.
    """
    resize_op = [] if (strip or wedge) else [transforms.Resize((image_size, image_size))]
    if split == "train":
        ops = list(resize_op)
        if include_flips:
            if not strip:
                ops.append(transforms.RandomHorizontalFlip())
            if not wedge:
                ops.append(transforms.RandomVerticalFlip())
        ops += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
        return transforms.Compose(ops)
    return transforms.Compose(resize_op + [
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
        image            : FloatTensor (3, H, W)  — normalized
        age_ordinal      : FloatTensor (K-1,)     — ordinal binary target
        age              : LongTensor ()          — raw integer age
        image_id         : str
        zegar_heatmap    : FloatTensor (Hp, Wp)   — semi-weak-supervision target
                           (26.08); zeros unless this sample is one of the 42
                           ZEGAR-annotated images (see cfg.data.zegar_* fields)
        has_zegar_target : BoolTensor ()          — True iff zegar_heatmap is real
        metadata         : FloatTensor (M,)       — only if use_metadata=True
        image_strip      : FloatTensor (3, Wp, Lp) — straightened nucleus->far_edge
                           crop (02.09, dendrochronology-strip experiment, see
                           src/strip_extraction.py); only present when
                           cfg.data.dual_branch_density=True. Fed to the density head
                           via a SEPARATE backbone forward pass (OtolithModel.forward's
                           density_image param) — the age (CORAL/MIL) heads never see
                           this tensor, only the ordinary square "image" above.
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

        # Semi-weak supervision (26.08, "Opcja A"): the 42-image "ZEGAR" ground-truth set
        # lives in its OWN small CSV/image dir, never in labels_csv/image_dir (which
        # scripts/prepare_labels.py regenerates from Z:/Photo/... on every RESCAN=True run
        # — writing ZEGAR rows directly into labels_csv would make them silently vanish on
        # the next rescan). Concatenated BEFORE the split filter below, exactly like a
        # normal extra chunk of labelled data — those rows only ever have split="train"
        # (see scripts/prepare_zegar_semi_weak_data.py), so they never touch val/test.
        self._zegar_extra_image_dir: Optional[Path] = None
        self._zegar_image_ids: set = set()
        self._zegar_targets: Dict[str, List[tuple]] = {}
        self._zegar_sigma = cfg.data.zegar_gaussian_sigma_patches
        if cfg.data.zegar_extra_labels_csv and cfg.data.zegar_extra_image_dir:
            zegar_labels_path = root / cfg.data.zegar_extra_labels_csv
            zegar_df = pd.read_csv(zegar_labels_path)
            self._validate_columns(zegar_df)
            full_df = pd.concat([full_df, zegar_df], ignore_index=True)
            self._zegar_extra_image_dir = root / cfg.data.zegar_extra_image_dir
            self._zegar_image_ids = set(zegar_df["image_id"].astype(str))
            if cfg.data.zegar_targets_csv:
                targets_df = pd.read_csv(root / cfg.data.zegar_targets_csv)
                for image_id, group in targets_df.groupby("image_id"):
                    self._zegar_targets[str(image_id)] = list(
                        zip(group["x"].astype(float), group["y"].astype(float))
                    )

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

        # E9: the polar-coordinate grid (needed when the concentricity loss is on, OR
        # (10.08) when density_head_type="radial_attention" — that head also consumes
        # per-patch (t, theta) directly, see model.RadialAttentionDensityHead) requires
        # an EXPLICIT, shared flip decision between the image and the geometry grid —
        # build_transforms' built-in random flips can't be synchronised with a second
        # tensor, so they're disabled here and redone manually in
        # _load_image_with_polar(). Off by default: zero behaviour change for every
        # existing config (density_concentricity_weight=0.0 and density_head_type!=
        # "radial_attention" both being the defaults).
        # (26.08) ZEGAR heatmap targets need the SAME explicit-flip synchronisation as
        # the polar grid, for the same reason (Compose's built-in random flip can't be
        # replayed on a second tensor) — so configuring ZEGAR data also routes every
        # sample through _load_image_with_polar, regardless of density_head_type. In
        # practice this is a no-op for the intended launch config (Run N's
        # radial_attention already implies _need_polar=True); it's a correctness
        # safety net for any other density_head_type this might be layered onto later.
        self._need_polar = bool(getattr(cfg.model, "use_density_head", False)) and (
            getattr(cfg.model, "density_concentricity_weight", 0.0) > 0.0
            or getattr(cfg.model, "density_head_type", "mlp") == "radial_attention"
            or bool(self._zegar_image_ids)
        )
        self.transform = transform or build_transforms(
            cfg.data.image_size, split, include_flips=not self._need_polar)

        # Dendrochronology-strip experiment (02.09, plans and summaries/02.09_wycinki_plan.md):
        # dual_branch_density builds a SECOND tensor per sample (image_strip) for the density
        # head, via a straightened crop along the reading axis — the age heads keep reading
        # "image" above, completely unaffected. False (default) = zero extra cost, mirrors
        # every other opt-in field in this class.
        self.dual_branch_density = cfg.data.dual_branch_density
        self.strips_cache_dir: Optional[Path] = None
        self.strip_transform: Optional[transforms.Compose] = None
        self.multi_wycinek_k: int = 1
        self.multi_wycinek_min_angle_sep_deg: float = cfg.data.multi_wycinek_min_angle_sep_deg
        # Background-activation penalty (08.09, plans and summaries/
        # 08.09_metodyka_i_diagnoza_paska.md) — False (default) = zero behaviour change.
        # config.py forbids this together with multi_wycinek_k>1.
        self.strip_mask_background_loss = cfg.data.strip_mask_background_loss
        self.strip_transform_no_flip: Optional[transforms.Compose] = None
        if self.dual_branch_density:
            base_strips_dir = (Path(cfg.data.strips_cache_dir) if cfg.data.strips_cache_dir
                               else root / "data" / "strips_cache")
            # Dimension-specific subdirectory (not just filename) so changing
            # strip_length_px/strip_width_px between configs can never silently serve a
            # stale, wrong-shape cached strip — same lesson as the documented mask-cache
            # collision bug (scripts/diagnostics/expert_annotation_eval.py, 12.08).
            self.strips_cache_dir = (
                base_strips_dir / f"{cfg.data.strip_length_px}x{cfg.data.strip_width_px}"
            )
            self.strips_cache_dir.mkdir(parents=True, exist_ok=True)
            self.strip_transform = build_transforms(
                cfg.data.strip_length_px, split, include_flips=True, strip=True)
            if self.strip_mask_background_loss:
                # include_flips=False so the Compose applies no RandomVerticalFlip of
                # its own — the flip decision instead happens explicitly in
                # _build_strip_tensor_and_valid_mask, applied identically to the strip
                # image AND its valid mask (see that method's docstring for why the
                # normal self.strip_transform can't be reused for this path).
                self.strip_transform_no_flip = build_transforms(
                    cfg.data.strip_length_px, split, include_flips=False, strip=True)
            # Multi-wycinek experiment (03.09, plans and summaries/
            # 03.09_multi_wycinek_plan.md) — 1 (default) = today's single-wycinek
            # behaviour, zero change.
            self.multi_wycinek_k = cfg.data.multi_wycinek_k

        # Polar-wedge experiment (09.09, plans and summaries/
        # 09.09_pasek_maskowanie_wyniki_i_literatura.md follow-up) — alternative
        # dual-branch geometry to the strip above (config.py forbids both at once).
        # False (default) = zero extra cost, mirrors dual_branch_density.
        self.dual_branch_wedge = cfg.data.dual_branch_wedge
        self.wedges_cache_dir: Optional[Path] = None
        self.wedge_transform_no_flip: Optional[transforms.Compose] = None
        if self.dual_branch_wedge:
            base_wedges_dir = (Path(cfg.data.wedge_cache_dir) if cfg.data.wedge_cache_dir
                               else root / "data" / "wedges_cache")
            wedge_w_px = cfg.data.wedge_n_angle_patches * cfg.data.patch_size
            wedge_h_px = cfg.data.wedge_n_radius_patches * cfg.data.patch_size
            # Dimension+angle-keyed subdirectory — same "never silently serve a stale,
            # wrong-shape cache" lesson as strips_cache_dir/strip_mask_cache_dir.
            self.wedges_cache_dir = (
                base_wedges_dir / f"{wedge_h_px}x{wedge_w_px}_{cfg.data.wedge_delta_theta_deg:g}deg"
            )
            self.wedges_cache_dir.mkdir(parents=True, exist_ok=True)
            self.wedge_delta_theta_deg = cfg.data.wedge_delta_theta_deg
            self.wedge_canvas_w = wedge_w_px
            self.wedge_canvas_h = wedge_h_px
            # include_flips=False: the horizontal (angular-mirror) flip decision must be
            # made explicitly and applied identically to the wedge image AND its
            # polar_t/theta/valid tensors — same reason self.strip_transform_no_flip
            # exists for the strip's masked variant (Compose's own random flip can't be
            # replayed on a second tensor).
            self.wedge_transform_no_flip = build_transforms(
                wedge_w_px, split, include_flips=False, wedge=True)

        # Angular-resolution bands (09.09 follow-up, plans and summaries/
        # 09.09_wycinek_pasma_katowe_plan.md) — opt-in ON TOP OF dual_branch_wedge above (config.py
        # requires wedge_band_* to be all-None or all-set together, and dual_branch_wedge=True).
        # None (default) = zero extra cost, mirrors every other experiment flag in this class.
        self.wedge_bands_enabled = cfg.data.wedge_band_edges_t is not None
        self.wedge_band_geoms_cache_dirs: list[Path] = []
        self.wedge_band_canvas_w: list[int] = []
        self.wedge_band_canvas_h: list[int] = []
        self.wedge_band_row_fracs: Optional[np.ndarray] = None
        self.wedge_band_transform_no_flip: Optional[transforms.Compose] = None
        if self.wedge_bands_enabled:
            self.wedge_band_edges_t = list(cfg.data.wedge_band_edges_t)
            self.wedge_band_row_fracs = wedge_band_row_frac_edges(self.wedge_band_edges_t)
            self.wedge_band_canvas_w = [n * cfg.data.patch_size
                                        for n in cfg.data.wedge_band_n_angle_patches]
            self.wedge_band_canvas_h = [n * cfg.data.patch_size
                                        for n in cfg.data.wedge_band_n_radius_patches]
            base_bands_dir = (Path(cfg.data.wedge_bands_cache_dir) if cfg.data.wedge_bands_cache_dir
                              else root / "data" / "wedge_bands_cache")
            n_bands = len(self.wedge_band_canvas_w)
            for i in range(n_bands):
                band_dir = (base_bands_dir /
                           f"band{i}_{self.wedge_band_canvas_h[i]}x{self.wedge_band_canvas_w[i]}_"
                           f"{cfg.data.wedge_delta_theta_deg:g}deg")
                band_dir.mkdir(parents=True, exist_ok=True)
                self.wedge_band_geoms_cache_dirs.append(band_dir)
            # build_transforms(wedge=True) skips Resize entirely (input already the exact target
            # size) — size-agnostic, so ONE shared transform serves every band regardless of its
            # own (different) canvas width; the first positional arg is unused in that path.
            self.wedge_band_transform_no_flip = build_transforms(
                1, split, include_flips=False, wedge=True)

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
            (image_tensor, polar_grid, polar_valid, polar_theta,
             zegar_heatmap, has_zegar_target) = self._load_image_with_polar(image_id)
        else:
            image_tensor = self._load_image(image_id)
            polar_grid = polar_valid = polar_theta = None
            h_patches = w_patches = self.cfg.data.image_size // self.cfg.data.patch_size
            zegar_heatmap = torch.zeros((h_patches, w_patches), dtype=torch.float32)
            has_zegar_target = False
        age_ordinal = encode_age_ordinal(age, self.num_age_classes)

        sample: Dict = {
            "image": image_tensor,
            "age_ordinal": age_ordinal,
            "age": torch.tensor(age, dtype=torch.long),
            "age_original": torch.tensor(recorded_age, dtype=torch.long),
            "image_id": image_id,
            # Semi-weak supervision (26.08): ALWAYS present (zeros/False when this
            # sample has no real ZEGAR annotation) — unlike polar_grid above, this
            # can't be a conditionally-omitted key, because only a SUBSET of samples
            # in any given batch have a target; default_collate requires every sample
            # dict in a batch to have the same keys, so a mixed batch (some ZEGAR, some
            # not) would fail to collate if the key were sometimes missing.
            "zegar_heatmap": zegar_heatmap,
            "has_zegar_target": torch.tensor(has_zegar_target, dtype=torch.bool),
        }
        if polar_grid is not None:
            sample["polar_grid"] = polar_grid
            sample["polar_valid"] = polar_valid
            sample["polar_theta"] = polar_theta

        if self.use_metadata and self.metadata_cols:
            sample["metadata"] = self._encode_metadata(row)

        if self.dual_branch_density:
            if self.strip_mask_background_loss:
                image_strip, strip_valid_mask = self._build_strip_tensor_and_valid_mask(image_id)
                sample["image_strip"] = image_strip
                sample["strip_valid_mask"] = strip_valid_mask
            else:
                sample["image_strip"] = self._load_strip_image(image_id)

        if self.dual_branch_wedge and not self.wedge_bands_enabled:
            # Bands REPLACE the single-canvas wedge (not additive) — computing both would waste
            # a full extra extraction/backbone-worth of tensors per sample for no benefit; a
            # config with wedge_band_edges_t set is asking for the band geometry, not both.
            (image_wedge, wedge_polar_t,
             wedge_polar_theta, wedge_polar_valid) = self._build_wedge_tensor_and_polar(image_id)
            sample["image_wedge"] = image_wedge
            sample["wedge_polar_t"] = wedge_polar_t
            sample["wedge_polar_theta"] = wedge_polar_theta
            sample["wedge_polar_valid"] = wedge_polar_valid

        if self.wedge_bands_enabled:
            for i, (band_img, band_t, band_theta, band_valid) in enumerate(
                    self._build_wedge_bands_tensors_and_polar(image_id)):
                sample[f"image_wedge_band{i}"] = band_img
                sample[f"wedge_band{i}_polar_t"] = band_t
                sample[f"wedge_band{i}_polar_theta"] = band_theta
                sample[f"wedge_band{i}_polar_valid"] = band_valid

        return sample

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _image_dir_for(self, image_id: str) -> Path:
        """(26.08) ZEGAR-derived rows live in a SEPARATE directory from every other
        image (see __init__) — everything else resolves to the normal img_dir,
        unchanged."""
        if image_id in self._zegar_image_ids and self._zegar_extra_image_dir is not None:
            return self._zegar_extra_image_dir
        return self.img_dir

    def _build_zegar_heatmap(self, image_id: str, crop_h: int, crop_w: int) -> tuple:
        """(26.08) Soft Gaussian target for zegar_position_loss, in the SAME raw-crop-
        pixel-to-patch-grid mapping convention as otolith_axis.compute_polar_grid
        (row = y/crop_h*h_patches, col = x/crop_w*w_patches — algebraically identical
        to "resize to image_size, then divide by patch_size" for a square target grid,
        so this matches the polar grid's own convention exactly rather than introducing
        a second, only-superficially-different mapping).

        Returns (heatmap (h_patches, w_patches) float32 ndarray, has_target bool) — an
        all-zero/False heatmap when image_id has no ZEGAR annotation (the overwhelming
        majority of samples), matching this module's "never crash, degrade gracefully"
        philosophy elsewhere (e.g. failed segmentation).
        """
        h_patches = w_patches = self.cfg.data.image_size // self.cfg.data.patch_size
        points = self._zegar_targets.get(image_id)
        if not points:
            return np.zeros((h_patches, w_patches), dtype=np.float32), False
        rr, cc = np.meshgrid(np.arange(h_patches), np.arange(w_patches), indexing="ij")
        heat = np.zeros((h_patches, w_patches), dtype=np.float32)
        sigma = self._zegar_sigma
        for x, y in points:
            row0 = y / crop_h * h_patches
            col0 = x / crop_w * w_patches
            g = np.exp(-((rr - row0) ** 2 + (cc - col0) ** 2) / (2.0 * sigma * sigma))
            heat = np.maximum(heat, g.astype(np.float32))
        return heat, True

    def _load_image(self, image_id: str) -> torch.Tensor:
        img_dir = self._image_dir_for(image_id)
        path = img_dir / image_id
        if not path.exists():
            for ext in IMAGE_EXTENSIONS:
                candidate = img_dir / (image_id + ext)
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

        Returns (image_tensor, polar_t_grid, polar_valid_grid, polar_theta_grid,
        zegar_heatmap, has_zegar_target); the polar three are all-zero / all-False when
        segmentation fails, matching the rest of this module's "never crash on a bad
        photo" fallback philosophy. polar_theta_grid (05.08, windowed/"local" E9)
        carries a signed direction, so unlike the other two it needs an actual VALUE
        transform (not just an array reversal) under a flip — see the wrap_angle calls
        below. zegar_heatmap (26.08) needs only the array-reversal treatment, like
        t_grid/valid_grid — a Gaussian blob has no signed direction to correct.
        """
        img_dir = self._image_dir_for(image_id)
        path = img_dir / image_id
        if not path.exists():
            for ext in IMAGE_EXTENSIONS:
                candidate = img_dir / (image_id + ext)
                if candidate.exists():
                    path = candidate
                    break
        image = Image.open(path).convert("RGB")
        rgb = np.array(image, dtype=np.uint8)
        crop_h, crop_w = rgb.shape[:2]
        zegar_heat, has_zegar_target = self._build_zegar_heatmap(image_id, crop_h, crop_w)

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
                zegar_heat = np.ascontiguousarray(zegar_heat[:, ::-1])
            if do_vflip:
                pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
                t_grid = np.ascontiguousarray(t_grid[::-1, :])
                valid_grid = np.ascontiguousarray(valid_grid[::-1, :])
                # Mirrors negate dy instead: atan2(-dy,dx) = wrap(-atan2(dy,dx)).
                theta_grid = _wrap_angle(-theta_grid[::-1, :])
                zegar_heat = np.ascontiguousarray(zegar_heat[::-1, :])

        image_tensor = self.transform(pil_img)
        return (
            image_tensor,
            torch.from_numpy(t_grid.copy()),
            torch.from_numpy(valid_grid.copy()),
            torch.from_numpy(np.ascontiguousarray(theta_grid).copy()),
            torch.from_numpy(np.ascontiguousarray(zegar_heat).copy()),
            has_zegar_target,
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

    def _load_raw_rgb(self, image_id: str) -> np.ndarray:
        """Load ``image_id`` from disk as an RGB uint8 array — shared by the single-
        and multi-wycinek strip loaders below (03.09)."""
        img_dir = self._image_dir_for(image_id)
        path = img_dir / image_id
        if not path.exists():
            for ext in IMAGE_EXTENSIONS:
                candidate = img_dir / (image_id + ext)
                if candidate.exists():
                    path = candidate
                    break
        image = Image.open(path).convert("RGB")
        return np.array(image, dtype=np.uint8)

    def _load_strip_image(self, image_id: str) -> torch.Tensor:
        """Build the density branch's second input.

        Returns a single ``(3, Wp, Lp)`` tensor when ``multi_wycinek_k<=1`` (02.09,
        today's default, unchanged behaviour), or a stacked ``(K, 3, Wp, Lp)`` tensor
        when ``multi_wycinek_k>1`` (03.09, multi-wycinek experiment — see ``plans and
        summaries/03.09_multi_wycinek_plan.md``). ``default_collate`` adds the batch
        dimension on top either way, so ``OtolithModel.forward``'s ``density_image``
        ends up either 4D (``B,3,Wp,Lp``) or 5D (``B,K,3,Wp,Lp``) — it branches on
        ``ndim`` to tell the two apart.
        """
        if self.multi_wycinek_k <= 1:
            return self._build_strip_tensor(image_id)
        return self._build_multi_strip_tensor(image_id)

    def _build_strip_tensor(self, image_id: str) -> torch.Tensor:
        """Single-wycinek path (02.09): a straightened rectangular crop along the
        (nucleus -> far_edge) reading axis, ``src.strip_extraction.get_or_compute_
        strip``.

        Checks the strip cache FIRST, before touching the raw photo at all — unlike
        the mask cache (whose caller always needs the raw pixels regardless, to draw
        the mask onto them), a cache-hit strip needs no raw-image load, no
        segmentation, no axis-finding at all. Only on a cache miss does this load the
        raw photo and pay the (documented, first-epoch-dominant) segmentation +
        ``find_reading_edge`` cost. Degrades gracefully — a plain anisotropic resize of
        the whole raw photo — when segmentation or axis-finding fails, the same "never
        crash training on a bad photo" philosophy as ``_mask_background``/
        ``_load_image_with_polar`` elsewhere in this module.
        """
        strip_length_px = self.cfg.data.strip_length_px
        strip_width_px = self.cfg.data.strip_width_px
        strip_cache_path = self.strips_cache_dir / f"{Path(image_id).stem}_strip.png"

        cached = load_strip(strip_cache_path)
        if cached is not None and cached.shape[:2] == (strip_width_px, strip_length_px):
            return self.strip_transform(Image.fromarray(cached))

        rgb = self._load_raw_rgb(image_id)

        mask_cache_path = self.mask_cache_dir / f"{Path(image_id).stem}_mask.png"
        mask = get_or_compute_mask(rgb, mask_cache_path, seg_params=self.cfg.segmentation.as_params())
        axis_info = None
        if mask is not None:
            axis_info = detect_axis(
                rgb, seg_params=self.cfg.segmentation.as_params(),
                nucleus_method=self.cfg.segmentation.nucleus_method,
                axis_method=self.cfg.segmentation.axis_method,
                mask=mask,
            )
        if mask is None or axis_info is None:
            fallback = Image.fromarray(rgb).resize(
                (strip_length_px, strip_width_px), Image.BILINEAR)
            return self.strip_transform(fallback)

        strip = get_or_compute_strip(
            rgb, mask, axis_info, strip_cache_path, strip_length_px, strip_width_px)
        return self.strip_transform(Image.fromarray(strip))

    def _decide_strip_vflip(self) -> bool:
        """Split out from _build_strip_tensor_and_valid_mask purely so tests can force
        a flip via monkeypatching, mirroring _decide_flip's rationale."""
        return torch.rand(()).item() < 0.5

    def _build_strip_tensor_and_valid_mask(self, image_id: str) -> tuple:
        """Single-wycinek strip + its per-patch tissue-validity mask (08.09, plans and
        summaries/08.09_metodyka_i_diagnoza_paska.md), built TOGETHER so the same
        random vertical flip (train split) is applied to both — keeping the density
        loss's valid_mask aligned with the density grid it gates.

        Cannot reuse self.strip_transform here (unlike the unmasked
        _build_strip_tensor path): that Compose's RandomVerticalFlip makes its own
        private random choice with no hook to replay on a second tensor — the same
        problem already solved once in this module for the E9 polar grid
        (_load_image_with_polar/_decide_flip). Horizontal flip stays unconditionally
        dropped for strips (would reverse the canonicalized reading direction), so
        only the vertical decision needs to be explicit here. Config validation
        (src/config.py) forbids strip_mask_background_loss with multi_wycinek_k>1, so
        this is only ever reached on the single-wycinek path.
        """
        strip_length_px = self.cfg.data.strip_length_px
        strip_width_px = self.cfg.data.strip_width_px
        patch_size = self.cfg.data.patch_size
        h_p, w_p = strip_width_px // patch_size, strip_length_px // patch_size
        stem = Path(image_id).stem
        strip_cache_path = self.strips_cache_dir / f"{stem}_strip.png"
        valid_cache_path = self.strips_cache_dir / f"{stem}_strip_valid.npy"

        cached_strip = load_strip(strip_cache_path)
        cached_valid = load_strip_validity(valid_cache_path)
        if (cached_strip is not None and cached_strip.shape[:2] == (strip_width_px, strip_length_px)
                and cached_valid is not None and cached_valid.shape == (h_p, w_p)):
            strip_arr, valid_arr = cached_strip, cached_valid
        else:
            rgb = self._load_raw_rgb(image_id)
            mask_cache_path = self.mask_cache_dir / f"{stem}_mask.png"
            mask = get_or_compute_mask(rgb, mask_cache_path, seg_params=self.cfg.segmentation.as_params())
            axis_info = None
            if mask is not None:
                axis_info = detect_axis(
                    rgb, seg_params=self.cfg.segmentation.as_params(),
                    nucleus_method=self.cfg.segmentation.nucleus_method,
                    axis_method=self.cfg.segmentation.axis_method,
                    mask=mask,
                )
            if mask is None or axis_info is None:
                # Never crash training on a bad photo — fall back to a plain resize
                # (matching _build_strip_tensor) and an all-valid mask (unknown
                # geometry means "don't penalise anything", the conservative default).
                fallback = Image.fromarray(rgb).resize(
                    (strip_length_px, strip_width_px), Image.BILINEAR)
                valid_t = torch.ones(h_p, w_p, dtype=torch.float32)
                if self.split == "train" and self._decide_strip_vflip():
                    fallback = fallback.transpose(Image.FLIP_TOP_BOTTOM)
                    valid_t = torch.flip(valid_t, dims=[0])
                return self.strip_transform_no_flip(fallback), valid_t

            strip_arr = get_or_compute_strip(
                rgb, mask, axis_info, strip_cache_path, strip_length_px, strip_width_px)
            valid_arr = get_or_compute_strip_validity(
                mask, axis_info, valid_cache_path, strip_length_px, strip_width_px, patch_size)

        strip_img = Image.fromarray(strip_arr)
        valid_t = torch.from_numpy(valid_arr.copy())
        if self.split == "train" and self._decide_strip_vflip():
            strip_img = strip_img.transpose(Image.FLIP_TOP_BOTTOM)
            valid_t = torch.flip(valid_t, dims=[0])
        return self.strip_transform_no_flip(strip_img), valid_t

    def _build_multi_strip_tensor(self, image_id: str) -> torch.Tensor:
        """Multi-wycinek path (03.09, ``multi_wycinek_k>1``): ``K`` candidate wycinki
        per sample, one per candidate reading axis from
        ``otolith_axis.detect_axis_candidates``.

        Always returns exactly ``(multi_wycinek_k, 3, Wp, Lp)`` — when segmentation or
        axis-finding yields fewer than ``K`` genuine candidates, the BEST one (index 0,
        ``detect_axis_candidates`` returns them ranked) is repeated to pad the stack to
        a uniform shape (``torch.stack``/``default_collate`` require it across a
        batch); the downstream selection mechanism then simply has fewer effective
        choices for that particular image, not a crash or a shape mismatch. Same total
        fallback (plain resize, repeated ``K`` times) as the single-wycinek path when
        segmentation fails entirely.
        """
        strip_length_px = self.cfg.data.strip_length_px
        strip_width_px = self.cfg.data.strip_width_px
        k = self.multi_wycinek_k
        stem = Path(image_id).stem
        cache_dir = self.strips_cache_dir / f"k{k}"

        cache_paths = [cache_dir / f"{stem}_strip_{i}.png" for i in range(k)]
        cached = [load_strip(p) for p in cache_paths]
        if all(c is not None and c.shape[:2] == (strip_width_px, strip_length_px) for c in cached):
            tensors = [self.strip_transform(Image.fromarray(c)) for c in cached]
            return torch.stack(tensors, dim=0)

        rgb = self._load_raw_rgb(image_id)

        mask_cache_path = self.mask_cache_dir / f"{stem}_mask.png"
        mask = get_or_compute_mask(rgb, mask_cache_path, seg_params=self.cfg.segmentation.as_params())
        axis_candidates: list = []
        if mask is not None:
            axis_candidates = detect_axis_candidates(
                rgb, seg_params=self.cfg.segmentation.as_params(),
                nucleus_method=self.cfg.segmentation.nucleus_method,
                axis_method=self.cfg.segmentation.axis_method,
                mask=mask, k=k, min_angle_sep_deg=self.multi_wycinek_min_angle_sep_deg,
            )

        if not axis_candidates:
            fallback = Image.fromarray(rgb).resize(
                (strip_length_px, strip_width_px), Image.BILINEAR)
            tensor = self.strip_transform(fallback)
            return tensor.unsqueeze(0).expand(k, -1, -1, -1).clone()

        cache_dir.mkdir(parents=True, exist_ok=True)
        strips = []
        for i in range(k):
            axis_info = axis_candidates[i] if i < len(axis_candidates) else axis_candidates[0]
            strip = get_or_compute_strip(
                rgb, mask, axis_info, cache_paths[i], strip_length_px, strip_width_px)
            strips.append(strip)

        tensors = [self.strip_transform(Image.fromarray(s)) for s in strips]
        return torch.stack(tensors, dim=0)

    # ------------------------------------------------------------------
    # Polar-wedge branch (09.09) — see src/wedge_extraction.py's module docstring for
    # the geometry and literature this is based on.
    # ------------------------------------------------------------------

    def _decide_wedge_hflip(self) -> bool:
        """Split out purely so tests can force a flip via monkeypatching — mirrors
        _decide_strip_vflip's rationale exactly, same reason."""
        return torch.rand(()).item() < 0.5

    def _build_wedge_tensor_and_polar(self, image_id: str) -> tuple:
        """Wedge image + its per-patch (t, theta, valid) polar coordinates, built
        TOGETHER so the same random horizontal (angular-mirror) flip is applied to
        all four — keeping RadialAttentionDensityHead's positional encoding aligned
        with the pixels it describes. Cannot reuse a plain Compose with its own
        RandomHorizontalFlip here for the same reason _build_strip_tensor_and_valid_
        mask can't: a second (here: three more) tensor needs the IDENTICAL flip
        decision replayed on it, which torchvision's built-in random transforms have
        no hook for.

        Checks the wedge-image cache FIRST (get_or_compute_wedge) — a cache hit still
        needs the real mask (cheap, itself cached by get_or_compute_mask) to compute
        the validity mask, but never needs to re-run axis-finding/extraction.
        """
        stem = Path(image_id).stem
        wedge_cache_path = self.wedges_cache_dir / f"{stem}_wedge.png"
        patch_size = self.cfg.data.patch_size

        rgb = self._load_raw_rgb(image_id)
        mask_cache_path = self.mask_cache_dir / f"{stem}_mask.png"
        mask = get_or_compute_mask(rgb, mask_cache_path, seg_params=self.cfg.segmentation.as_params())

        # Wedge-image cache checked BEFORE detect_axis: a cache hit already carries its
        # own WedgeGeometry (saved alongside the .png as .geom.npz), so re-running real
        # axis-finding is unnecessary — mirrors get_or_compute_strip's own cache-first
        # short-circuit (see test_dual_branch_wedge_cache_reused_on_second_access).
        cached = load_wedge(wedge_cache_path) if mask is not None else None
        if cached is not None and cached[0].shape[:2] == (self.wedge_canvas_h, self.wedge_canvas_w):
            wedge_arr, geom = cached
            h_p, w_p = self.cfg.data.wedge_n_radius_patches, self.cfg.data.wedge_n_angle_patches
            t_grid, theta_grid = wedge_polar_coords(geom, patch_size)
            valid_grid = extract_polar_wedge_validity(mask, geom, patch_size)
            wedge_img = Image.fromarray(wedge_arr)
            if self.split == "train" and self._decide_wedge_hflip():
                wedge_img = wedge_img.transpose(Image.FLIP_LEFT_RIGHT)
                t_grid, theta_grid, valid_grid = (np.flip(a, axis=1).copy()
                                                  for a in (t_grid, theta_grid, valid_grid))
            return (self.wedge_transform_no_flip(wedge_img),
                    torch.from_numpy(t_grid), torch.from_numpy(theta_grid),
                    torch.from_numpy(valid_grid))

        axis_info = None
        if mask is not None:
            axis_info = detect_axis(
                rgb, seg_params=self.cfg.segmentation.as_params(),
                nucleus_method=self.cfg.segmentation.nucleus_method,
                axis_method=self.cfg.segmentation.axis_method,
                mask=mask,
            )

        h_p = self.cfg.data.wedge_n_radius_patches
        w_p = self.cfg.data.wedge_n_angle_patches
        if mask is None or axis_info is None:
            # Never crash training on a bad photo — plain resize + an all-valid mask
            # (unknown geometry means "don't penalise anything"), same philosophy as
            # every other fallback in this module. Polar coords still make sense (they
            # come from the CANVAS geometry, not the failed segmentation) — reuse a
            # throwaway WedgeGeometry centred at the image's own centre, angle 0, so
            # wedge_polar_coords has something consistent to compute from.
            fallback = Image.fromarray(rgb).resize(
                (self.wedge_canvas_w, self.wedge_canvas_h), Image.BILINEAR)
            dummy_geom = wedge_geometry_from_axis_info(
                np.ones(rgb.shape[:2], dtype=np.uint8) * 255,
                {"centroid": (rgb.shape[1] / 2, rgb.shape[0] / 2),
                 "far_edge": (rgb.shape[1], rgb.shape[0] / 2)},
                self.wedge_delta_theta_deg, self.wedge_canvas_w, self.wedge_canvas_h,
            )
            t_grid, theta_grid = wedge_polar_coords(dummy_geom, patch_size)
            valid_grid = np.ones((h_p, w_p), dtype=np.float32)
            if self.split == "train" and self._decide_wedge_hflip():
                fallback = fallback.transpose(Image.FLIP_LEFT_RIGHT)
                t_grid, theta_grid, valid_grid = (np.flip(a, axis=1).copy()
                                                  for a in (t_grid, theta_grid, valid_grid))
            return (self.wedge_transform_no_flip(fallback),
                    torch.from_numpy(t_grid), torch.from_numpy(theta_grid),
                    torch.from_numpy(valid_grid))

        wedge_arr, geom = get_or_compute_wedge(
            rgb, mask, axis_info, wedge_cache_path,
            self.wedge_delta_theta_deg, self.wedge_canvas_w, self.wedge_canvas_h,
        )
        t_grid, theta_grid = wedge_polar_coords(geom, patch_size)
        valid_grid = extract_polar_wedge_validity(mask, geom, patch_size)

        wedge_img = Image.fromarray(wedge_arr)
        if self.split == "train" and self._decide_wedge_hflip():
            wedge_img = wedge_img.transpose(Image.FLIP_LEFT_RIGHT)
            t_grid, theta_grid, valid_grid = (np.flip(a, axis=1).copy()
                                              for a in (t_grid, theta_grid, valid_grid))
        return (self.wedge_transform_no_flip(wedge_img),
                torch.from_numpy(t_grid), torch.from_numpy(theta_grid),
                torch.from_numpy(valid_grid))

    # ------------------------------------------------------------------
    # Angular-resolution bands (09.09 follow-up) — plans and summaries/
    # 09.09_wycinek_pasma_katowe_plan.md. REPLACES the single-canvas wedge above (not additive,
    # see __getitem__) when cfg.data.wedge_band_edges_t is set.
    # ------------------------------------------------------------------

    def _finish_wedge_bands(
        self, wedge_arrs: list, bands: list, mask: np.ndarray,
        valid_override: Optional[float] = None,
    ) -> list:
        """Shared tail for all three code paths below (cache-hit / segmentation-failure fallback /
        freshly-extracted): computes each band's polar coords + validity, then applies ONE SHARED
        flip decision (not one per band) to every band's image and polar tensors together —
        mirrors _build_wedge_tensor_and_polar's own single-flip-per-sample discipline exactly, just
        replayed across N bands instead of one canvas."""
        patch_size = self.cfg.data.patch_size
        do_flip = self.split == "train" and self._decide_wedge_hflip()
        results = []
        for arr, band in zip(wedge_arrs, bands):
            t_grid, theta_grid = wedge_band_polar_coords(band, patch_size)
            if valid_override is not None:
                h_p = band.geom.canvas_h // patch_size
                w_p = band.geom.canvas_w // patch_size
                valid_grid = np.full((h_p, w_p), valid_override, dtype=np.float32)
            else:
                valid_grid = extract_polar_wedge_band_validity(mask, band, patch_size)
            img = Image.fromarray(arr)
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                t_grid, theta_grid, valid_grid = (np.flip(a, axis=1).copy()
                                                  for a in (t_grid, theta_grid, valid_grid))
            results.append((self.wedge_band_transform_no_flip(img),
                            torch.from_numpy(t_grid), torch.from_numpy(theta_grid),
                            torch.from_numpy(valid_grid)))
        return results

    def _build_wedge_bands_tensors_and_polar(self, image_id: str) -> list:
        """Returns a list of ``(image, polar_t, polar_theta, polar_valid)`` tuples, one per band —
        mirrors :meth:`_build_wedge_tensor_and_polar`'s three-path structure (cache-hit /
        segmentation-failure fallback / freshly-extracted), generalised to N bands sharing ONE
        mask/axis_info/ray-cast computation instead of duplicating it per band."""
        stem = Path(image_id).stem
        patch_size = self.cfg.data.patch_size
        n_bands = len(self.wedge_band_canvas_w)
        band_cache_paths = [self.wedge_band_geoms_cache_dirs[i] / f"{stem}_wedge_band{i}.png"
                            for i in range(n_bands)]

        rgb = self._load_raw_rgb(image_id)
        mask_cache_path = self.mask_cache_dir / f"{stem}_mask.png"
        mask = get_or_compute_mask(rgb, mask_cache_path, seg_params=self.cfg.segmentation.as_params())

        # Cache-hit path: ALL bands present at the right shape -> reconstruct WedgeBandGeometry
        # from each cached .geom.npz (row_frac_lo/hi are fixed config values, not per-image —
        # wedge_band_row_fracs was precomputed once in __init__) — no detect_axis needed, mirrors
        # the single-band wedge's own cache-first short-circuit.
        if mask is not None:
            loaded = [load_wedge(p) for p in band_cache_paths]
            if all(l is not None and l[0].shape[:2] == (self.wedge_band_canvas_h[i],
                                                          self.wedge_band_canvas_w[i])
                   for i, l in enumerate(loaded)):
                bands = [
                    WedgeBandGeometry(geom=geom,
                                      row_frac_lo=float(self.wedge_band_row_fracs[i]),
                                      row_frac_hi=float(self.wedge_band_row_fracs[i + 1]))
                    for i, (_arr, geom) in enumerate(loaded)
                ]
                wedge_arrs = [arr for arr, _geom in loaded]
                return self._finish_wedge_bands(wedge_arrs, bands, mask)

        axis_info = None
        if mask is not None:
            axis_info = detect_axis(
                rgb, seg_params=self.cfg.segmentation.as_params(),
                nucleus_method=self.cfg.segmentation.nucleus_method,
                axis_method=self.cfg.segmentation.axis_method,
                mask=mask,
            )

        if mask is None or axis_info is None:
            # Never crash training on a bad photo — same philosophy as _build_wedge_tensor_and_
            # polar's own fallback: plain per-band resize + all-valid mask (unknown geometry means
            # "don't penalise anything"). Reuses a throwaway centred axis so wedge_band_polar_
            # coords still has something consistent to compute from.
            dummy_mask = np.ones(rgb.shape[:2], dtype=np.uint8) * 255
            dummy_axis_info = {"centroid": (rgb.shape[1] / 2, rgb.shape[0] / 2),
                               "far_edge": (rgb.shape[1], rgb.shape[0] / 2)}
            bands = wedge_band_geometries_from_axis_info(
                dummy_mask, dummy_axis_info, self.cfg.data.wedge_delta_theta_deg,
                self.wedge_band_edges_t, self.wedge_band_canvas_w, self.wedge_band_canvas_h,
            )
            wedge_arrs = [
                np.array(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
                for w, h in zip(self.wedge_band_canvas_w, self.wedge_band_canvas_h)
            ]
            return self._finish_wedge_bands(wedge_arrs, bands, dummy_mask, valid_override=1.0)

        bands = wedge_band_geometries_from_axis_info(
            mask, axis_info, self.cfg.data.wedge_delta_theta_deg,
            self.wedge_band_edges_t, self.wedge_band_canvas_w, self.wedge_band_canvas_h,
        )
        wedge_arrs = []
        for i, band in enumerate(bands):
            arr = extract_polar_wedge_band(rgb, mask, band)
            save_wedge(arr, band.geom, band_cache_paths[i])
            wedge_arrs.append(arr)
        return self._finish_wedge_bands(wedge_arrs, bands, mask)

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
