"""Stage 1 tests: config loading, entrypoint smoke test."""
from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def test_config_yaml_exists() -> None:
    assert CONFIG_PATH.exists(), f"config.yaml missing at {CONFIG_PATH}"


def test_config_loads_without_error() -> None:
    from src.config import load_config
    cfg = load_config(CONFIG_PATH)
    assert cfg.project.name == "OtolithDinoStandalone"


def test_config_model_fields() -> None:
    from src.config import load_config
    cfg = load_config(CONFIG_PATH)
    assert cfg.model.backbone == "dinov2_vits14"
    assert cfg.model.target_type == "ordinal"
    assert cfg.model.num_age_classes == 17
    assert cfg.model.use_metadata is False


def test_config_data_splits_sum() -> None:
    from src.config import load_config
    cfg = load_config(CONFIG_PATH)
    total = cfg.data.train_split + cfg.data.val_split + cfg.data.test_split
    assert abs(total - 1.0) < 1e-5, f"splits sum to {total}"


def test_config_image_size_divisible() -> None:
    from src.config import load_config
    cfg = load_config(CONFIG_PATH)
    assert cfg.data.image_size % cfg.data.patch_size == 0


def test_default_config_is_valid() -> None:
    from src.config import get_default_config
    cfg = get_default_config()
    assert cfg.training.epochs > 0
    assert cfg.inference.output_dir != ""


def test_invalid_splits_raise_error() -> None:
    from pydantic import ValidationError
    from src.config import DataConfig
    with pytest.raises((ValidationError, ValueError)):
        DataConfig(train_split=0.8, val_split=0.3, test_split=0.3)


def test_entrypoint_info_mode(tmp_path) -> None:
    from src.entrypoint import run
    code = run(["--config", str(CONFIG_PATH), "--mode", "info"])
    assert code == 0


def test_entrypoint_missing_config_raises(tmp_path) -> None:
    from src.entrypoint import run
    with pytest.raises(FileNotFoundError):
        run(["--config", str(tmp_path / "nonexistent.yaml"), "--mode", "info"])
