"""Stage 6 tests: patch importance, heatmaps, overlays, run_interpretation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.dataset import encode_age_ordinal


# ---------------------------------------------------------------------------
# Mock backbone — non-zero, spatially varying patch tokens for richer tests
# ---------------------------------------------------------------------------

class _MockDinoBackbone(nn.Module):
    embed_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        mean_val = x.mean(dim=(1, 2, 3), keepdim=True).reshape(B, 1)
        return self.proj(mean_val)

    def forward_features(self, x: Tensor) -> Dict:
        B, C, H, W = x.shape
        H_p = H // 14
        W_p = W // 14
        num_patches = H_p * W_p

        cls = self.forward(x)

        # Spatially varying patches: each patch has a distinct L2 norm
        # based on its linear index, so importance is non-uniform
        idx = torch.arange(num_patches, dtype=torch.float32, device=x.device)
        scale = (idx + 1.0).reshape(1, num_patches, 1)            # (1, N, 1)
        patches = scale.expand(B, num_patches, self.embed_dim)    # (B, N, D)
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patches.contiguous(),
        }


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class _SyntheticDataset(Dataset):
    def __init__(self, n: int = 6, num_age_classes: int = 10):
        self.n = n
        self.num_age_classes = num_age_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict:
        age = (idx % (self.num_age_classes - 1)) + 1
        return {
            "image": torch.randn(3, 56, 56),
            "age_ordinal": encode_age_ordinal(age, self.num_age_classes),
            "age": torch.tensor(age, dtype=torch.long),
            "image_id": f"img_{idx:03d}.png",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path = None):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.num_age_classes = 10
    cfg.model.dropout = 0.0
    cfg.training.device = "cpu"
    cfg.interpretation.heatmap_alpha = 0.5
    if tmp_path:
        cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
        cfg.training.log_dir = str(tmp_path / "logs")
    return cfg


def _make_model(cfg):
    from src.model import OtolithModel
    return OtolithModel(cfg, backbone=_MockDinoBackbone())


def _make_loader(n: int = 6) -> DataLoader:
    ds = _SyntheticDataset(n=n)
    return DataLoader(ds, batch_size=2, shuffle=False)


def _make_single_image_tensor():
    """Return a (1, 3, 56, 56) tensor (batch of 1)."""
    return torch.randn(1, 3, 56, 56)


# ---------------------------------------------------------------------------
# compute_patch_importance
# ---------------------------------------------------------------------------

def test_patch_importance_shape():
    from src.interpretation import compute_patch_importance
    cfg = _make_cfg()
    model = _make_model(cfg)
    model.eval()
    img = _make_single_image_tensor()
    importance = compute_patch_importance(model, img)
    # 56 / 14 = 4 patches per side
    assert importance.shape == (4, 4)


def test_patch_importance_accepts_3d_input():
    """compute_patch_importance must accept (3, H, W) without batch dim."""
    from src.interpretation import compute_patch_importance
    cfg = _make_cfg()
    model = _make_model(cfg)
    model.eval()
    img_3d = torch.randn(3, 56, 56)
    importance = compute_patch_importance(model, img_3d)
    assert importance.shape == (4, 4)


def test_patch_importance_nonneg():
    """L2 norm is always >= 0."""
    from src.interpretation import compute_patch_importance
    cfg = _make_cfg()
    model = _make_model(cfg)
    model.eval()
    importance = compute_patch_importance(model, _make_single_image_tensor())
    assert (importance >= 0).all()


def test_patch_importance_no_grad():
    """Result must not require gradient (uses no_grad internally)."""
    from src.interpretation import compute_patch_importance
    cfg = _make_cfg()
    model = _make_model(cfg)
    model.eval()
    importance = compute_patch_importance(model, _make_single_image_tensor())
    assert not importance.requires_grad


def test_patch_importance_nonzero_for_nonzero_patches():
    """With spatially varying mock patches, importance must be non-uniform."""
    from src.interpretation import compute_patch_importance
    cfg = _make_cfg()
    model = _make_model(cfg)
    model.eval()
    importance = compute_patch_importance(model, _make_single_image_tensor())
    # all values > 0 since patch scale = index + 1 >= 1
    assert (importance > 0).all()
    # values are not all equal (non-uniform)
    assert importance.max() > importance.min()


# ---------------------------------------------------------------------------
# importance_to_heatmap
# ---------------------------------------------------------------------------

def test_heatmap_output_shape():
    from src.interpretation import importance_to_heatmap
    grid = torch.rand(4, 4)
    heatmap = importance_to_heatmap(grid, image_size=56)
    assert heatmap.shape == (56, 56)


def test_heatmap_values_in_0_1():
    from src.interpretation import importance_to_heatmap
    grid = torch.rand(4, 4) * 10   # values outside [0,1] before normalisation
    heatmap = importance_to_heatmap(grid, image_size=56)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_heatmap_nonuniform_input_has_nonzero_range():
    from src.interpretation import importance_to_heatmap
    grid = torch.tensor([[0.0, 0.5], [1.0, 0.2]])
    heatmap = importance_to_heatmap(grid, image_size=56)
    assert heatmap.max() > heatmap.min()


def test_heatmap_uniform_input_is_zero():
    """Uniform importance → no spatial information → heatmap all zeros."""
    from src.interpretation import importance_to_heatmap
    grid = torch.ones(4, 4) * 3.7   # any uniform value
    heatmap = importance_to_heatmap(grid, image_size=56)
    assert heatmap.max() == 0.0


def test_heatmap_dtype_is_float32():
    from src.interpretation import importance_to_heatmap
    heatmap = importance_to_heatmap(torch.rand(4, 4), image_size=56)
    assert heatmap.dtype == np.float32


def test_heatmap_different_output_sizes():
    from src.interpretation import importance_to_heatmap
    grid = torch.rand(4, 4)
    for size in [28, 56, 112]:
        h = importance_to_heatmap(grid, image_size=size)
        assert h.shape == (size, size)


# ---------------------------------------------------------------------------
# save_heatmap
# ---------------------------------------------------------------------------

def test_save_heatmap_creates_file(tmp_path):
    from src.interpretation import save_heatmap
    heatmap = np.random.rand(56, 56).astype(np.float32)
    path = tmp_path / "test_heatmap.png"
    save_heatmap(heatmap, path)
    assert path.exists()


def test_save_heatmap_is_jet_rgb(tmp_path):
    """save_heatmap now colourises via JET → RGB output."""
    from src.interpretation import save_heatmap
    heatmap = np.random.rand(56, 56).astype(np.float32)
    path = tmp_path / "heatmap.png"
    save_heatmap(heatmap, path)
    img = PILImage.open(path)
    assert img.mode == "RGB"


def test_save_heatmap_correct_size(tmp_path):
    from src.interpretation import save_heatmap
    heatmap = np.random.rand(56, 56).astype(np.float32)
    path = tmp_path / "heatmap.png"
    save_heatmap(heatmap, path)
    img = PILImage.open(path)
    assert img.size == (56, 56)


def test_save_heatmap_creates_parent_dir(tmp_path):
    from src.interpretation import save_heatmap
    heatmap = np.random.rand(56, 56).astype(np.float32)
    path = tmp_path / "nested" / "dir" / "heatmap.png"
    save_heatmap(heatmap, path)
    assert path.exists()


# ---------------------------------------------------------------------------
# save_overlay
# ---------------------------------------------------------------------------

def test_save_overlay_creates_file(tmp_path):
    from src.interpretation import save_overlay
    orig = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
    heatmap = np.random.rand(56, 56).astype(np.float32)
    path = tmp_path / "overlay.png"
    save_overlay(orig, heatmap, path, alpha=0.5)
    assert path.exists()


def test_save_overlay_is_rgb(tmp_path):
    from src.interpretation import save_overlay
    orig = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
    heatmap = np.random.rand(56, 56).astype(np.float32)
    path = tmp_path / "overlay.png"
    save_overlay(orig, heatmap, path)
    img = PILImage.open(path)
    assert img.mode == "RGB"


def test_save_overlay_correct_size(tmp_path):
    from src.interpretation import save_overlay
    H = 56
    orig = np.random.randint(0, 255, (H, H, 3), dtype=np.uint8)
    heatmap = np.random.rand(H, H).astype(np.float32)
    path = tmp_path / "overlay.png"
    save_overlay(orig, heatmap, path)
    img = PILImage.open(path)
    assert img.size == (H, H)


def test_save_overlay_alpha_zero_equals_original(tmp_path):
    """alpha=0 means heatmap invisible → overlay should equal original image."""
    from src.interpretation import save_overlay
    orig = np.full((56, 56, 3), 128, dtype=np.uint8)
    heatmap = np.random.rand(56, 56).astype(np.float32)
    path = tmp_path / "overlay.png"
    save_overlay(orig, heatmap, path, alpha=0.0)
    result = np.array(PILImage.open(path))
    np.testing.assert_array_equal(result, orig)


# ---------------------------------------------------------------------------
# run_interpretation
# ---------------------------------------------------------------------------

def test_run_interpretation_returns_list(tmp_path):
    from src.interpretation import run_interpretation
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    N = 6
    results = run_interpretation(cfg, model, _make_loader(n=N), tmp_path / "out")
    assert isinstance(results, list)
    assert len(results) == N


def test_run_interpretation_required_keys(tmp_path):
    from src.interpretation import run_interpretation
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_interpretation(cfg, model, _make_loader(n=4), tmp_path / "out")
    for rec in results:
        assert "image_id" in rec
        assert "heatmap_path" in rec
        assert "overlay_path" in rec


def test_run_interpretation_heatmap_files_exist(tmp_path):
    from src.interpretation import run_interpretation
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_interpretation(cfg, model, _make_loader(n=4), tmp_path / "out")
    for rec in results:
        assert Path(rec["heatmap_path"]).exists()


def test_run_interpretation_overlay_files_exist(tmp_path):
    from src.interpretation import run_interpretation
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_interpretation(cfg, model, _make_loader(n=4), tmp_path / "out")
    for rec in results:
        assert Path(rec["overlay_path"]).exists()


def test_run_interpretation_heatmaps_are_jet_rgb_png(tmp_path):
    from src.interpretation import run_interpretation
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_interpretation(cfg, model, _make_loader(n=2), tmp_path / "out")
    for rec in results:
        img = PILImage.open(rec["heatmap_path"])
        assert img.mode == "RGB"


def test_run_interpretation_overlays_are_rgb_png(tmp_path):
    from src.interpretation import run_interpretation
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_interpretation(cfg, model, _make_loader(n=2), tmp_path / "out")
    for rec in results:
        img = PILImage.open(rec["overlay_path"])
        assert img.mode == "RGB"


def test_run_interpretation_image_ids_match_loader(tmp_path):
    from src.interpretation import run_interpretation
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)
    loader = _make_loader(n=4)
    results = run_interpretation(cfg, model, loader, tmp_path / "out")
    expected_ids = [f"img_{i:03d}.png" for i in range(4)]
    returned_ids = [r["image_id"] for r in results]
    assert returned_ids == expected_ids


def test_importance_mil_method_requires_head():
    """method='mil_patch_probs' on a CORAL-only model must raise."""
    import pytest
    from src.interpretation import compute_patch_importance
    cfg = _make_cfg()
    cfg.model.head_type = "coral"
    model = _make_model(cfg)
    model.eval()
    with pytest.raises(RuntimeError):
        compute_patch_importance(model, _make_single_image_tensor(), method="mil_patch_probs")


def test_importance_patch_token_method_returns_grid():
    """method='patch_token_importance' works without a MIL head and returns (H_p, W_p)."""
    from src.interpretation import compute_patch_importance
    cfg = _make_cfg()
    cfg.model.head_type = "coral"
    model = _make_model(cfg)
    model.eval()
    imp = compute_patch_importance(
        model, _make_single_image_tensor(), method="patch_token_importance"
    )
    assert imp.dim() == 2
