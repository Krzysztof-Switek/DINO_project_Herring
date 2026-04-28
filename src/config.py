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
    num_age_classes: int = Field(20, ge=2, le=100)
    dropout: float = Field(0.1, ge=0.0, lt=1.0)


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
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


class InferenceConfig(BaseModel):
    output_dir: str = "outputs"
    save_heatmaps: bool = True
    save_overlays: bool = True
    save_candidates: bool = True


class InterpretationConfig(BaseModel):
    method: Literal[
        "patch_token_importance", "attention_rollout", "gradient_saliency"
    ] = "patch_token_importance"
    top_k_patches: int = Field(20, ge=1)
    heatmap_alpha: float = Field(0.5, ge=0.0, le=1.0)


class CandidatesConfig(BaseModel):
    min_peak_distance: int = Field(5, ge=1)
    prominence_threshold: float = Field(0.1, ge=0.0)


class OtolithConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    interpretation: InterpretationConfig = Field(default_factory=InterpretationConfig)
    candidates: CandidatesConfig = Field(default_factory=CandidatesConfig)


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