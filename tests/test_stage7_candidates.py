"""Stage 7 tests: candidates pipeline — profile, peaks, JSON, overlay, run_candidates.

Updated for the biological-axis API:
  - candidates use sample_profile_along_axis from src/otolith_axis.py
  - JSON saves peak_profile_indices (not peak_pixel_positions)
  - save_candidates_overlay takes line_xy + axis_info, not heatmap
"""
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
    profile = np.zeros(20, dtype=np.float32)
    profile[3] = 1.0
    profile[5] = 0.9
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
# save_candidates_json — new schema with peak_profile_indices + axis
# ---------------------------------------------------------------------------

def test_save_candidates_json_creates_file(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.array([0.1, 0.5, 0.3], dtype=np.float32)
    peaks   = np.array([1], dtype=np.int64)
    path    = tmp_path / "out.json"
    save_candidates_json("img_001.png", peaks, profile, path)
    assert path.exists()


def test_save_candidates_json_required_fields(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.array([0.2, 0.8, 0.4], dtype=np.float32)
    peaks   = np.array([1], dtype=np.int64)
    path    = tmp_path / "out.json"
    save_candidates_json("img_001.png", peaks, profile, path)
    data = json.loads(path.read_text())
    for key in ("image_id", "num_candidates", "peak_profile_indices",
                "radial_profile", "axis"):
        assert key in data


def test_save_candidates_json_num_candidates_matches_peaks(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.zeros(10, dtype=np.float32)
    peaks   = np.array([2, 5, 8], dtype=np.int64)
    path    = tmp_path / "out.json"
    save_candidates_json("x.png", peaks, profile, path)
    data = json.loads(path.read_text())
    assert data["num_candidates"] == 3
    assert len(data["peak_profile_indices"]) == 3


def test_save_candidates_json_empty_peaks(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.ones(5, dtype=np.float32)
    peaks   = np.array([], dtype=np.int64)
    path    = tmp_path / "out.json"
    save_candidates_json("x.png", peaks, profile, path)
    data = json.loads(path.read_text())
    assert data["num_candidates"] == 0
    assert data["peak_profile_indices"] == []


def test_save_candidates_json_profile_length(tmp_path):
    from src.candidates import save_candidates_json
    profile = np.array([0.1, 0.2, 0.9, 0.3, 0.1], dtype=np.float32)
    path    = tmp_path / "out.json"
    save_candidates_json("x.png", np.array([2]), profile, path)
    data = json.loads(path.read_text())
    assert len(data["radial_profile"]) == 5


def test_save_candidates_json_axis_info_serialized(tmp_path):
    """When axis_info is provided, JSON axis block contains centroid/far_edge/length_px."""
    from src.candidates import save_candidates_json
    axis_info = {
        "centroid":  (100, 150),
        "far_edge":  (110, 400),
        "length_px": 250.5,
    }
    profile = np.zeros(5, dtype=np.float32)
    path    = tmp_path / "out.json"
    save_candidates_json("x.png", np.array([2]), profile, path, axis_info=axis_info)
    data = json.loads(path.read_text())
    assert data["axis"]["method"] == "centroid_to_farthest"
    assert data["axis"]["centroid"] == [100, 150]
    assert data["axis"]["far_edge"] == [110, 400]


# ---------------------------------------------------------------------------
# save_candidates_overlay
# ---------------------------------------------------------------------------

def test_save_candidates_overlay_creates_file_no_axis(tmp_path):
    """Without axis_info, fallback overlay (vertical centre line) is drawn."""
    from src.candidates import save_candidates_overlay
    orig    = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
    line_xy = np.stack([np.full(10, 28), np.linspace(28, 55, 10).astype(int)], axis=1)
    peaks   = np.array([3, 6], dtype=np.int64)
    path    = tmp_path / "overlay.png"
    save_candidates_overlay(orig, peaks, line_xy, path, axis_info=None)
    assert path.exists()


def test_save_candidates_overlay_is_rgb(tmp_path):
    from src.candidates import save_candidates_overlay
    orig    = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
    line_xy = np.array([[28, 30], [28, 40]], dtype=np.int64)
    path    = tmp_path / "overlay.png"
    save_candidates_overlay(orig, np.array([0]), line_xy, path)
    img = PILImage.open(path)
    assert img.mode == "RGB"


def test_save_candidates_overlay_correct_size(tmp_path):
    from src.candidates import save_candidates_overlay
    H = 56
    orig    = np.random.randint(0, 255, (H, H, 3), dtype=np.uint8)
    line_xy = np.array([[28, 30]], dtype=np.int64)
    path    = tmp_path / "overlay.png"
    save_candidates_overlay(orig, np.array([], dtype=np.int64), line_xy, path)
    img = PILImage.open(path)
    assert img.size == (H, H)


def test_save_candidates_overlay_no_peaks_no_crash(tmp_path):
    from src.candidates import save_candidates_overlay
    orig    = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
    line_xy = np.array([[28, 30]], dtype=np.int64)
    path    = tmp_path / "overlay.png"
    save_candidates_overlay(orig, np.array([], dtype=np.int64), line_xy, path)
    assert path.exists()


def test_save_candidates_overlay_dot_visible_at_peak(tmp_path):
    """A red dot should appear at the pixel (x, y) = line_xy[peak_idx]."""
    from src.candidates import save_candidates_overlay
    H, W = 200, 200
    orig    = np.zeros((H, W, 3), dtype=np.uint8)
    line_xy = np.array([[100, 50], [100, 150]], dtype=np.int64)
    path    = tmp_path / "overlay.png"
    save_candidates_overlay(orig, np.array([1]), line_xy, path, axis_info=None)
    result = np.array(PILImage.open(path))
    # Pixel near (100, 150) should be red-ish (dot is filled circle with radius ~H/60)
    px = result[150, 100]
    assert px[0] > 150 and px[1] < 80 and px[2] < 80


# ---------------------------------------------------------------------------
# run_candidates — batch runner end-to-end
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
        assert len(data["peak_profile_indices"]) == rec["num_candidates"]
        assert len(data["radial_profile"]) > 0


def test_run_candidates_overlays_are_rgb(tmp_path):
    from src.candidates import run_candidates
    cfg   = _make_cfg(tmp_path)
    model = _make_model(cfg)
    results = run_candidates(cfg, model, _make_loader(n=2), tmp_path / "out")
    for rec in results:
        img = PILImage.open(rec["candidates_overlay_path"])
        assert img.mode == "RGB"
