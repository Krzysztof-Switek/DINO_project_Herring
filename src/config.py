"""Central config loader with pydantic validation."""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ProjectConfig(BaseModel):
    name: str = "OtolithDinoStandalone"
    version: str = "0.1.0"
    seed: int = 42


class ModelConfig(BaseModel):
    backbone: str = "dinov2_vits14"
    use_metadata: bool = False
    target_type: Literal["regression", "ordinal", "count_aware"] = "ordinal"
    num_age_classes: int = Field(17, ge=2, le=100)
    dropout: float = Field(0.1, ge=0.0, lt=1.0)
    # MIL weakly supervised localisation head
    head_type: Literal["coral", "mil", "both"] = "both"
    mil_count_weight:    float = Field(1.0,  ge=0.0)  # weight of the MIL concentration loss
    mil_sparsity_weight: float = Field(1.0,  ge=0.0)  # off-region (background) term weight
    mil_hidden_dim:      int   = Field(64,   ge=1)
    coral_loss_weight:   float = Field(0.5,  ge=0.0)
    # Kierunek B: decoupled density-map counting head (crowd-counting style). Reads
    # backbone patch tokens with STOP-GRADIENT (.detach()), so its count-consistency
    # loss can never flow into the backbone → the age (CORAL) head is safe by design
    # (hard lesson from the radial-spread experiment). Localisation peaks come from
    # the density map; integral(density) ≈ age. Off by default (age recipe untouched).
    use_density_head:      bool  = False
    density_count_weight:  float = Field(1.0, ge=0.0)  # weight of the density loss in the sum
    density_conc_weight:   float = Field(1.0, ge=0.0)  # concentration term (breaks diffuse solution)
    # P2: spatial-coherence (total-variation) prior on the density map — nudges peaks
    # to be blob-like rather than salt-and-pepper (a safe proxy for the along-axis
    # peak prior, which would need the reading axis at train time). 0 = off (default).
    density_tv_weight:     float = Field(0.0, ge=0.0)


class DataConfig(BaseModel):
    image_dir: str = "data/images"
    labels_csv: str = "data/labels.csv"
    image_size: int = Field(518, ge=14)
    patch_size: int = Field(14, ge=8)
    train_split: float = Field(0.7, gt=0.0, lt=1.0)
    val_split: float = Field(0.15, gt=0.0, lt=1.0)
    test_split: float = Field(0.15, gt=0.0, lt=1.0)
    num_workers: int = Field(2, ge=0)
    metadata_cols: List[str] = Field(default_factory=list)

    @field_validator("image_size")
    @classmethod
    def image_size_divisible(cls, v: int) -> int:
        # checked against patch_size in model_validator below
        return v

    @model_validator(mode="after")
    def splits_sum_to_one(self) -> "DataConfig":
        total = round(self.train_split + self.val_split + self.test_split, 6)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"train+val+test splits must sum to 1.0, got {total}")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        return self


class TrainingConfig(BaseModel):
    epochs: int = Field(50, ge=1)
    batch_size: int = Field(16, ge=1)
    lr: float = Field(1e-4, gt=0.0)
    weight_decay: float = Field(1e-4, ge=0.0)
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    freeze_backbone_epochs: int = Field(5, ge=0)
    # Discriminative LR: pretrained backbone is fine-tuned at lr * this factor,
    # the freshly-initialised heads at the full lr. 1.0 => uniform LR (old behaviour).
    backbone_lr_mult: float = Field(0.1, ge=0.0, le=1.0)
    early_stopping_patience: int = Field(10, ge=0)
    early_stopping_metric: Literal["val_mae", "val_loss"] = "val_mae"
    early_stopping_min_delta: float = Field(0.001, ge=0.0)
    # EMA smoothing of the monitored metric for model selection + early stopping.
    # The raw per-epoch val_mae is noisy (chunky integer-difference metric), so
    # best.pt otherwise lands on the single luckiest epoch (diagnosed 13.07). With
    # ema > 0 the monitored value is smoothed as v_ema = ema*v + (1-ema)*v_ema, and
    # best.pt / early-stop track v_ema. 0.0 = disabled (raw metric, old behaviour).
    early_stopping_ema: float = Field(0.0, ge=0.0, le=1.0)
    # Density-maturity gate for checkpoint selection (16.07). The density (localisation)
    # head trains SLOWER than the age head; on the 15.07_rx run best.pt was frozen on
    # val_mae at e17 while density_active was still 0 (density woke at e23). These decouple
    # best.pt / early-stopping from age alone so the saved model has a TRAINED density head:
    #   min_epochs         — early stopping cannot fire before this epoch (floor).
    #   min_density_active — when use_density_head, best.pt is only eligible / patience only
    #                        counts once mean density_active ≥ this (density is "alive").
    # Defaults 0 = old behaviour (no gating).
    min_epochs: int = Field(0, ge=0)
    min_density_active: float = Field(0.0, ge=0.0)
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    # Keep only best.pt in the run dir; delete per-epoch checkpoints as training goes
    # (each ~265 MB). False = keep every epoch's checkpoint (old behaviour).
    keep_only_best: bool = True


class IncrementSamplesConfig(BaseModel):
    top_k_best: int = Field(10, ge=1)
    top_k_worst: int = Field(10, ge=1)
    annotate_all: bool = False


class InferenceConfig(BaseModel):
    output_dir: str = "outputs"
    save_heatmaps: bool = True
    save_overlays: bool = True
    save_candidates: bool = True
    increment_samples: IncrementSamplesConfig = Field(
        default_factory=IncrementSamplesConfig)


class InterpretationConfig(BaseModel):
    # Only the two implemented signals are exposed:
    #   auto                   — MIL patch probabilities if a MIL head exists, else L2 norm
    #   patch_token_importance — always the L2 norm of DINOv2 patch tokens
    #   mil_patch_probs        — always the trained MIL patch probabilities (requires MIL head)
    method: Literal["auto", "patch_token_importance", "mil_patch_probs"] = "auto"
    heatmap_alpha: float = Field(0.5, ge=0.0, le=1.0)
    # DINOv2 uses memory-efficient (xFormers) attention by default, which does NOT
    # expose the softmax attention matrix → the CLS-attention card panel falls back
    # to a labelled L2-norm proxy. Set true to force vanilla attention so true
    # CLS→patch attention is captured (slower training/inference — read by the
    # entry points BEFORE the backbone is imported, via XFORMERS_DISABLED).
    disable_fused_attention: bool = False


class CandidatesConfig(BaseModel):
    min_peak_distance: int = Field(5, ge=1)
    prominence_threshold: float = Field(0.1, ge=0.0)
    # Experimental / diagnostic only. Classical image-based ring detection was
    # evaluated on the current photos (whole otoliths, reflected light) and does
    # NOT reliably recover annual rings — the ring signal is too weak/ambiguous
    # (see plans and summaries/7.07_TO_DO.md, "wynik negatywny"). Kept OFF; the
    # trained model is the ring signal. Revisit only with better imaging.
    detect_image_rings: bool = False


class SegmentationConfig(BaseModel):
    """Otolith outline detection (``src/otolith_axis.py::segment_otolith``).

    ``radial`` (default) casts rays from the nucleus and stops each where the
    otolith fades into the background, giving a SMOOTH outline that reaches the
    faint, thinning rim (where late increments live). ``threshold`` is the
    original Otsu + hysteresis method, kept as a fallback / for high-contrast
    light-background cases.
    """
    method: Literal["radial", "threshold"] = "radial"
    # per-ray reach: boundary = fraction of THAT ray's body brightness (↓ = capture
    # more faint margin, ↑ = tighter to the opaque body)
    frac: float = Field(0.18, ge=0.0, le=1.0)
    background_k: float = Field(3.0, ge=0.0)      # background floor = bg_mean + k·bg_std
    n_angles: int = Field(720, ge=32)             # rays cast from the nucleus
    smooth_sigma: float = Field(4.0, ge=0.0)      # periodic r(θ) smoothing: low = follow teeth, high = smooth envelope
    gap_tolerance: int = Field(8, ge=0)           # px below threshold tolerated before commit

    def as_params(self) -> dict:
        """Kwargs for ``segment_otolith`` / ``detect_axis(seg_params=...)``."""
        return self.model_dump()


class DemoConfig(BaseModel):
    """Demo mode — szybki test końcowy pipeline na ograniczonej próbce.

    Limity działają na poziomie OtolithDataset (po filtrze split), więc
    cały dalszy pipeline (train / val / inferencja / interpretacja / cards /
    raport) od razu pracuje na małej liczbie próbek.
    Tryb wyłączony (enabled=False) = brak jakichkolwiek ograniczeń.
    """
    enabled: bool = False
    max_train_samples: Optional[int] = Field(None, ge=1)
    max_val_samples:   Optional[int] = Field(None, ge=1)
    max_test_samples:  Optional[int] = Field(None, ge=1)


class OtolithConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    interpretation: InterpretationConfig = Field(default_factory=InterpretationConfig)
    candidates: CandidatesConfig = Field(default_factory=CandidatesConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)


def load_config(path: str | Path) -> OtolithConfig:
    """Load and validate config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    return OtolithConfig(**raw)


def get_default_config() -> OtolithConfig:
    """Return a default config (all defaults, no file needed)."""
    return OtolithConfig()
