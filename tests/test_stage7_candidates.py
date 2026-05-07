"""Stage 7 tests: radial profile, peak detection, coordinate conversion,
candidate JSON/overlay saving, and run_candidates batch runner."""
from __future__ import annotations

import json
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
# Mock backbone with spatially varying patch tokens
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
        # Spatially varying: each patch scaled by (index+1) → non-uniform importance
        idx   = torch.arange(num_patches, dtype=torch.float32, device=x.device)
        scale = (idx + 1.0).reshape(1, num_patches, 1)
        patches = scale.expand(B, num_patches, self.embed_dim).contiguous()
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patches,
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
    cfg.candidates.min_peak_distance = 1
    cfg.candidates.prominence_threshold = 0.0
    if tmp_path:
        cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
        cfg.training.log_dir = str(tmp_path / "logs")
    return cfg


def _make_model(cfg):
    from src.model import OtolithModel
    return OtolithModel(cfg, backbone=_MockDinoBackbone())


def _make_loader(n: int = 6) -> DataLoader:
    return DataLoader(_SyntheticDataset(n=n), batch_size=2, shuffle=False)


# ---------------------------------------------------------------------------
# extract_radial_profile
# ---------------------------------------------------------------------------

def test_radial_profile_shape():
    from src.candidates import extract_radial_profile
    grid = np.random.rand(4, 4).astype(np.float32)
    profile = extract_radial_profile(grid)
    assert profile.shape == (4,)


def test_radial_profile_shape_non_square():
    from src.candidates import extract_radial_profile
    grid = np.random.rand(3, 5).astype(np.float32)
    # horizontal: mean(axis=0) → (W_p,) = (5,)
    profile = extract_radial_profile(grid, axis="horizontal")
    assert profile.shape == (5,)


def test_radial_profile_accepts_tensor():
    from src.candidates import extract_radial_profile
    grid = torch.rand(4, 4)
    profile = extract_radial_profile(grid)
    assert isinstance(profile, np.ndarray)
    assert profile.shape == (4,)


def test_radial_profile_values_are_row_means():
    from src.candidates import extract_radial_profile
    grid = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    # horizontal axis: col means: col0=(1+3)/2=2.0, col1=(2+4)/2=3.0
    profile_h = extract_radial_profile(grid, axis="horizontal")
    np.testing.assert_allclose(profile_h, [2.0, 3.0])
    # vertical axis: row means: row0=(1+2)/2=1.5, row1=(3+4)/2=3.5
    profile_v = extract_radial_profile(grid, axis="vertical")
    np.testing.assert_allclose(profile_v, [1.5, 3.5])


def test_radial_profile_dtype():
    from src.candidates import extract_radial_profile
    grid = np.ones((4, 4), dtype=np.float64)
    profile = extract_radial_profile(grid)
    assert profile.dtype == np.float32


def test_vertical_profile_shape():
    from src.candidates import extract_radial_profile
    grid = np.random.rand(4, 8).astype(np.float32)
    profile = extract_radial_profile(grid, axis="vertical")
    assert profile.shape == (4,)


def test_horizontal_profile_shape():
    from src.candidates import extract_radial_profile
    grid = np.random.rand(4, 8).astype(np.float32)
    profile = extract_radial_profile(grid, axis="horizontal")
    assert profile.shape == (8,)


def test_pixel_positions_vertical():
    from src.candidates import peaks_to_pixel_positions
    image_height = 56
    num_patches_h = 4
    indices = np.arange(num_patches_h)
    pos = peaks_to_pixel_positions(indices, image_height, num_patches_h)
    assert (pos >= 0).all()
    assert (pos < image_height).all()


# ---------------------------------------------------------------------------
# find_candidate_peaks
# ---------------------------------------------------------------------------

def test_peaks_known_single_peak():
    from src.candidates import find_candidate_peaks
    profile = np.zeros(20, dtype=np.float32)
    profile[10] = 1.0
    peaks = find_candidate_peaks(profile, min_distance=1, prominence_threshold=0.5)
    assert 10 in peaks


def test_peaks_flat_profile_no_peaks():
    from src.candidates import find_candidate_peaks
    profile = np.ones(15, dtype=np.float32)
    peaks = find_candidate_peaks(profile, min_distance=1, prominence_threshold=0.0)
    assert len(peaks) == 0


def test_peaks_returns_ndarray():
    from src.candidates import find_candidate_peaks
    profile = np.random.rand(20).astype(np.float32)
    peaks = find_candidate_peaks(profile)
    assert isinstance(peaks, np.ndarray)


def test_peaks_indices_in_valid_range():
    from src.candidates import find_candidate_peaks
    N = 20
    profile = np.random.rand(N).astype(np.float32)
    peaks = find_candidate_peaks(profile)
    assert (peaks >= 0).all()
    assert (peaks < N).all()


def test_peaks_respects_min_distance():
    from src.candidates import find_candidate_peaks
    # Two peaks at positions 3 and 5 (distance=2); min_distance=5 → only higher kept
    profile = np.zeros(20, dtype=np.float32)
    profile[3] = 1.0    # higher peak
    profile[5] = 0.9    # close, lower peak
    peaks = find_candidate_peaks(profile, min_distance=5, prominence_threshold=0.0)
    assert len(peaks) == 1
    assert peaks[0] == 3


def test_peaks_multiple_well_separated():
    from src.candidates import find_candidate_peaks
    profile = np.zeros(30, dtype=np.float32)
    profile[5]  = 1.0
    profile[15] = 1.0
    profile[25] = 1.0
    peaks = find_candidate_peaks(profile, min_distance=3, prominence_threshold=0.5)
    assert set(peaks) == {5, 15, 25}


# ---------------------------------------------------------------------------
# peaks_to_pixel_positions
# ---------------------------------------------------------------------------

def test_pixel_positions_first_patch_center():
    from src.candidates import peaks_to_pixel_positions
    # image_size=56, 4 patches → patch_width=14; first patch center = 0.5*14 = 7
    pos = peaks_to_pixel_positions(np.array([0]), image_size=56, num_patches=4)
    assert pos[0] == 7


def test_pixel_positions_last_patch_center():
    from src.candidates import peaks_to_pixel_positions
    # index 3 (last of 4): center = 3.5 * 14 = 49
    pos = peaks_to_pixel_positions(np.array([3]), image_size=56, num_patches=4)
    assert pos[0] == 49


def test_pixel_positions_empty_input():
    from src.candidates import peaks_to_pixel_positions
    pos = peaks_to_pixel_positions(np.array([]), image_size=56, num_patches=4)
    assert len(pos) == 0


def test_pixel_positions_all_in_range():
    from src.candidates import peaks_to_pixel_positions
    indices = np.arange(4)
    pos = peaks_to_pixel_positions(indices, image_size=56, num_patches=4)
    assert (pos >= 0).all()
    assert (pos < 56).all()


def test_pixel_positions_monotone_increasing():
    from src.candidates import peaks_to_pixel_positions
    indices = np.array([0, 1, 2, 3])
    pos = peaks_to_pixel_positions(indices, image_size=56, num_patches=4)
    assert (np.diff(pos) > 0).all()


# ---------------------------------------------------------------------------
# save_candidates_json
# ---------------------------------------------------------------------------

def test_save_candidates_json_creates_file(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.array([0.1, 0.5, 0.3], dtype=np.float32)
    peaks   = np.array([1], dtype=np.int64)
    path    = tmp_path / "out.json"
    save_candidates_json("img_001.png", peaks, profile, path)
    assert path.exists()


def test_save_candidates_json_valid_json(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.array([0.1, 0.5, 0.3], dtype=np.float32)
    peaks   = np.array([1], dtype=np.int64)
    path    = tmp_path / "out.json"
    save_candidates_json("img_001.png", peaks, profile, path)
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


def test_save_candidates_json_required_fields(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.array([0.2, 0.8, 0.4], dtype=np.float32)
    peaks   = np.array([1], dtype=np.int64)
    path    = tmp_path / "out.json"
    save_candidates_json("img_001.png", peaks, profile, path)
    data = json.loads(path.read_text())
    for key in ("image_id", "num_candidates", "peak_pixel_positions", "radial_profile"):
        assert key in data


def test_save_candidates_json_num_candidates_matches_peaks(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.zeros(10, dtype=np.float32)
    peaks   = np.array([2, 5, 8], dtype=np.int64)
    path    = tmp_path / "out.json"
    save_candidates_json("x.png", peaks, profile, path)
    data = json.loads(path.read_text())
    assert data["num_candidates"] == 3
    assert len(data["peak_pixel_positions"]) == 3


def test_save_candidates_json_empty_peaks(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.ones(5, dtype=np.float32)
    peaks   = np.array([], dtype=np.int64)
    path    = tmp_path / "out.json"
    save_candidates_json("x.png", peaks, profile, path)
    data = json.loads(path.read_text())
    assert data["num_candidates"] == 0
    assert data["peak_pixel_positions"] == []


def test_save_candidates_json_profile_length(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.array([0.1, 0.2, 0.9, 0.3, 0.1], dtype=np.float32)
    path    = tmp_path / "out.json"
    save_candidates_json("x.png", np.array([2]), profile, path)
    data = json.loads(path.read_text())
    assert len(data["radial_profile"]) == 5


# ---------------------------------------------------------------------------
# save_candidates_overlay
# ---------------------------------------------------------------------------

def test_save_candidates_overlay_creates_file(tmp_path):
    from src.candidates import save_candidates_overlay
    orig    = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
    heatmap = np.random.rand(56, 56).astype(np.float32)
    peaks   = np.array([14, 28, 42], dtype=np.int64)
    path    = tmp_path / "overlay.png"
    save_candidates_overlay(orig, peaks, heatmap, path)
    assert path.exists()


def test_save_candidates_overlay_is_rgb(tmp_path):
    from src.candidates import save_candidates_overlay
    orig    = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
    heatmap = np.random.rand(56, 56).astype(np.float32)
    path    = tmp_path / "overlay.png"
    save_candidates_overlay(orig, np.array([20]), heatmap, path)
    img = PILImage.open(path)
    assert img.mode == "RGB"


def test_save_candidates_overlay_correct_size(tmp_path):
    from src.candidates import save_candidates_overlay
    H = 56
    orig    = np.random.randint(0, 255, (H, H, 3), dtype=np.uint8)
    heatmap = np.random.rand(H, H).astype(np.float32)
    path    = tmp_path / "overlay.png"
    save_candidates_overlay(orig, np.array([]), heatmap, path)
    img = PILImage.open(path)
    assert img.size == (H, H)


def test_save_candidates_overlay_no_peaks_no_crash(tmp_path):
    from src.candidates import save_candidates_overlay
    orig    = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
    heatmap = np.random.rand(56, 56).astype(np.float32)
    path    = tmp_path / "overlay.png"
    save_candidates_overlay(orig, np.array([], dtype=np.int64), heatmap, path)
    assert path.exists()


def test_save_candidates_overlay_red_line_visible(tmp_path):
    """With alpha=0 (no heatmap blend), a red line must appear at peak x-position."""
    from src.candidates import save_candidates_overlay
    H, W = 56, 56
    orig    = np.zeros((H, W, 3), dtype=np.uint8)   # black original
    heatmap = np.zeros((H, W), dtype=np.float32)    # all-zero heatmap
    x_peak  = 28
    path    = tmp_path / "overlay.png"
    save_candidates_overlay(orig, np.array([x_peak]), heatmap, path, alpha=0.0)
    result = np.array(PILImage.open(path))
    # Column x_peak should be predominantly red (R channel > others)
    col = result[:, x_peak, :]   # (H, 3) — all pixels on the line
    assert (col[:, 0] > 200).all()   # R channel high
    assert (col[:, 1] < 10).all()    # G channel low
    assert (col[:, 2] < 10).all()    # B channel low


# ---------------------------------------------------------------------------
# run_candidates
# ---------------------------------------------------------------------------

def test_run_candidates_returns_list(tmp_path):
    from src.candidates import run_candidates
    cfg   = _make_cfg(tmp_path)
    model = _make_model(cfg)
    N = 6
    results = run_candidates(cfg, model, _make_loader(n=N), tmp_path / "out")
    assert isinstance(results, list)
    assert len(results) == N


def test_run_candidates_required_keys(tmp_path):
    from src.candidates import run_candidates
    cfg   = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_candidates(cfg, model, _make_loader(n=4), tmp_path / "out")
    for rec in results:
        assert "image_id" in rec
        assert "num_candidates" in rec
        assert "candidate_markers_path" in rec
        assert "candidates_overlay_path" in rec


def test_run_candidates_json_files_exist(tmp_path):
    from src.candidates import run_candidates
    cfg   = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_candidates(cfg, model, _make_loader(n=4), tmp_path / "out")
    for rec in results:
        assert Path(rec["candidate_markers_path"]).exists()


def test_run_candidates_overlay_files_exist(tmp_path):
    from src.candidates import run_candidates
    cfg   = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_candidates(cfg, model, _make_loader(n=4), tmp_path / "out")
    for rec in results:
        assert Path(rec["candidates_overlay_path"]).exists()


def test_run_candidates_num_candidates_nonneg(tmp_path):
    from src.candidates import run_candidates
    cfg   = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_candidates(cfg, model, _make_loader(n=4), tmp_path / "out")
    for rec in results:
        assert rec["num_candidates"] >= 0


def test_run_candidates_json_content_valid(tmp_path):
    from src.candidates import run_candidates
    cfg   = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_candidates(cfg, model, _make_loader(n=2), tmp_path / "out")
    for rec in results:
        data = json.loads(Path(rec["candidate_markers_path"]).read_text())
        assert data["num_candidates"] == rec["num_candidates"]
        assert len(data["peak_pixel_positions"]) == rec["num_candidates"]
        assert len(data["radial_profile"]) > 0


def test_run_candidates_overlays_are_rgb(tmp_path):
    from src.candidates import run_candidates
    cfg   = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_candidates(cfg, model, _make_loader(n=2), tmp_path / "out")
    for rec in results:
        img = PILImage.open(rec["candidates_overlay_path"])
        assert img.mode == "RGB"
