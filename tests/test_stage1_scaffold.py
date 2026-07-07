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


def test_entrypoint_demo_delegates_to_pipeline(monkeypatch) -> None:
    """--mode demo must funnel into the single run_pipeline demo (config_demo.yaml)."""
    import scripts.run_pipeline as rp
    from src.entrypoint import run

    captured = {}
    monkeypatch.setattr(rp, "main", lambda argv=None: captured.update(argv=argv))

    code = run(["--config", str(CONFIG_PATH), "--mode", "demo"])
    assert code == 0
    argv = captured.get("argv")
    assert argv is not None, "run_pipeline.main was not called by --mode demo"
    assert "--base-config" in argv and "--output-dir" in argv
    assert any("config_demo.yaml" in a for a in argv)


def test_entrypoint_inference_eval_without_checkpoint_return_1(tmp_path) -> None:
    """inference/eval must fail gracefully (return 1) when no checkpoint exists."""
    import yaml
    from src.config import get_default_config
    from src.entrypoint import run

    cfg = get_default_config()
    cfg.training.checkpoint_dir = (tmp_path / "no_ckpts").as_posix()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")

    assert run(["--config", str(cfg_path), "--mode", "inference"]) == 1
    assert run(["--config", str(cfg_path), "--mode", "eval"]) == 1
