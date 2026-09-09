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
    assert cfg.model.backbone.startswith("dinov2_vits14")   # vits14 lub wariant _reg (rejestry)
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


def test_density_image_size_must_be_divisible_by_patch_size() -> None:
    """22.07: candidates.density_image_size (separate, higher-res density-only forward
    pass) must be divisible by data.patch_size, same rule as data.image_size."""
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(candidates={"density_image_size": 701}, data={"patch_size": 14})
    # divisible value + None (off) both valid
    OtolithConfig(candidates={"density_image_size": 728}, data={"patch_size": 14})
    OtolithConfig(candidates={"density_image_size": None})


def test_dual_branch_density_default_is_off() -> None:
    """02.09: dendrochronology-strip experiment — dual_branch_density defaults False, zero
    behaviour change for every existing config."""
    from src.config import get_default_config
    cfg = get_default_config()
    assert cfg.data.dual_branch_density is False


def test_dual_branch_density_requires_mask_background() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_density": True, "mask_background": False},
                      model={"use_density_head": True})
    # with mask_background=True it's valid (given use_density_head too)
    OtolithConfig(data={"dual_branch_density": True, "mask_background": True},
                  model={"use_density_head": True})


def test_dual_branch_density_requires_density_head() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_density": True, "mask_background": True},
                      model={"use_density_head": False})


def test_strip_dims_must_be_divisible_only_when_dual_branch_enabled() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    # dual_branch_density=False: bad strip dims are inert, must NOT raise.
    OtolithConfig(data={"dual_branch_density": False, "strip_length_px": 1331,
                        "strip_width_px": 99})
    # dual_branch_density=True: same bad dims now DO raise.
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_density": True, "mask_background": True,
                            "strip_length_px": 1331, "strip_width_px": 98},
                      model={"use_density_head": True})
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_density": True, "mask_background": True,
                            "strip_length_px": 1330, "strip_width_px": 99},
                      model={"use_density_head": True})
    # divisible values succeed
    OtolithConfig(data={"dual_branch_density": True, "mask_background": True,
                        "strip_length_px": 1330, "strip_width_px": 98},
                  model={"use_density_head": True})


def test_strip_mask_background_loss_default_is_off() -> None:
    """08.09: strip_mask_background_loss defaults False, zero behaviour change."""
    from src.config import get_default_config
    cfg = get_default_config()
    assert cfg.data.strip_mask_background_loss is False


def test_strip_mask_background_loss_requires_dual_branch_density() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_density": False, "strip_mask_background_loss": True})
    # with dual_branch_density=True it's valid (given the usual strip prerequisites)
    OtolithConfig(data={"dual_branch_density": True, "mask_background": True,
                        "strip_mask_background_loss": True},
                  model={"use_density_head": True})


def test_strip_mask_background_loss_rejects_multi_wycinek() -> None:
    """08.09 fix is scoped to the single-wycinek path — must reject k>1 explicitly rather
    than silently ignoring the mask for the multi-candidate branch."""
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_density": True, "mask_background": True,
                            "strip_mask_background_loss": True, "multi_wycinek_k": 5},
                      model={"use_density_head": True})
    # k=1 (default) is fine
    OtolithConfig(data={"dual_branch_density": True, "mask_background": True,
                        "strip_mask_background_loss": True, "multi_wycinek_k": 1},
                  model={"use_density_head": True})


def test_dual_branch_wedge_default_is_off() -> None:
    """09.09: polar-wedge experiment — dual_branch_wedge defaults False, zero behaviour change."""
    from src.config import get_default_config
    cfg = get_default_config()
    assert cfg.data.dual_branch_wedge is False


def test_dual_branch_wedge_requires_mask_background_and_density_head() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_wedge": True, "mask_background": False},
                      model={"use_density_head": True})
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_wedge": True, "mask_background": True},
                      model={"use_density_head": False})
    OtolithConfig(data={"dual_branch_wedge": True, "mask_background": True},
                  model={"use_density_head": True})


def test_dual_branch_wedge_and_strip_are_mutually_exclusive() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_wedge": True, "dual_branch_density": True,
                            "mask_background": True},
                      model={"use_density_head": True})


def test_dual_branch_wedge_rejects_multi_wycinek() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_wedge": True, "mask_background": True,
                            "multi_wycinek_k": 5},
                      model={"use_density_head": True})
    OtolithConfig(data={"dual_branch_wedge": True, "mask_background": True,
                        "multi_wycinek_k": 1},
                  model={"use_density_head": True})


def test_wedge_band_fields_default_to_none() -> None:
    """09.09 follow-up: angular-resolution bands default to None (off) — zero behaviour change."""
    from src.config import get_default_config
    cfg = get_default_config()
    assert cfg.data.wedge_band_edges_t is None
    assert cfg.data.wedge_band_n_angle_patches is None
    assert cfg.data.wedge_band_n_radius_patches is None


def test_wedge_bands_require_dual_branch_wedge() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_wedge": False,
                            "wedge_band_edges_t": [0.0, 0.6, 1.0],
                            "wedge_band_n_angle_patches": [97, 161],
                            "wedge_band_n_radius_patches": [23, 21]},
                      model={"use_density_head": True})
    # with dual_branch_wedge=True it's valid
    OtolithConfig(data={"dual_branch_wedge": True, "mask_background": True,
                        "wedge_band_edges_t": [0.0, 0.6, 1.0],
                        "wedge_band_n_angle_patches": [97, 161],
                        "wedge_band_n_radius_patches": [23, 21]},
                  model={"use_density_head": True})


def test_wedge_bands_must_all_be_set_together() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_wedge": True, "mask_background": True,
                            "wedge_band_edges_t": [0.0, 0.6, 1.0]},
                      model={"use_density_head": True})
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_wedge": True, "mask_background": True,
                            "wedge_band_n_angle_patches": [97, 161]},
                      model={"use_density_head": True})


def test_wedge_bands_length_mismatch_rejected() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_wedge": True, "mask_background": True,
                            "wedge_band_edges_t": [0.0, 0.6, 1.0],
                            "wedge_band_n_angle_patches": [97],  # should have 2 entries
                            "wedge_band_n_radius_patches": [23, 21]},
                      model={"use_density_head": True})


def test_wedge_bands_edges_must_span_zero_to_one_and_be_increasing() -> None:
    from pydantic import ValidationError
    from src.config import OtolithConfig
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_wedge": True, "mask_background": True,
                            "wedge_band_edges_t": [0.1, 0.6, 1.0],  # doesn't start at 0.0
                            "wedge_band_n_angle_patches": [97, 161],
                            "wedge_band_n_radius_patches": [23, 21]},
                      model={"use_density_head": True})
    with pytest.raises((ValidationError, ValueError)):
        OtolithConfig(data={"dual_branch_wedge": True, "mask_background": True,
                            "wedge_band_edges_t": [0.0, 0.6, 0.6, 1.0],  # not strictly increasing
                            "wedge_band_n_angle_patches": [97, 0, 161],
                            "wedge_band_n_radius_patches": [23, 1, 21]},
                      model={"use_density_head": True})


def test_wedge_bands_production_numbers_from_faza8() -> None:
    """Pins the 09.09 Faza 8 measurement (analyze_real_otolith_size_distribution.py + analyze_
    wedge_angular_resolution.py) — a silent drift in these numbers should fail this test."""
    from src.config import OtolithConfig
    cfg = OtolithConfig(data={
        "dual_branch_wedge": True, "mask_background": True,
        "wedge_band_edges_t": [0.0, 0.6, 0.8, 0.9, 1.0],
        "wedge_band_n_angle_patches": [97, 129, 145, 161],
        "wedge_band_n_radius_patches": [11, 12, 9, 12],
    }, model={"use_density_head": True})
    total_patches = sum(a * b for a, b in zip(cfg.data.wedge_band_n_angle_patches,
                                               cfg.data.wedge_band_n_radius_patches))
    assert total_patches == 5852
    assert sum(cfg.data.wedge_band_n_radius_patches) == cfg.data.wedge_n_radius_patches


def test_wedge_geometry_defaults_match_measured_values() -> None:
    """Pins the 09.09 empirical measurement (analyze_zegar_wedge_geometry.py) as the config
    default — a silent change to these numbers should fail this test, not pass unnoticed."""
    from src.config import get_default_config
    cfg = get_default_config()
    assert cfg.data.wedge_delta_theta_deg == pytest.approx(98.7)
    assert cfg.data.wedge_n_angle_patches == 13
    assert cfg.data.wedge_n_radius_patches == 44


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
