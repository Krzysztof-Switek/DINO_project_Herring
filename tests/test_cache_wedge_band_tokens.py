"""L5 (30.09): wedge-band token cache — what the density head reads must equal the cache."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.diagnostics.cache_wedge_band_tokens import (CacheWriter, check_identity,
                                                         fill_cache)

DIM = 16
BANDS = [(2, 3), (1, 4)]          # (rows, cols) in patches per band
P = sum(r * c for r, c in BANDS)


class _MockBackbone(nn.Module):
    """Patch tokens depend on pixel content, so any band/order mix-up shows up."""
    embed_dim = DIM

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, DIM)

    def forward_features(self, x):
        B, C, H, W = x.shape
        pooled = nn.functional.avg_pool2d(x, 14)                   # (B, C, H/14, W/14)
        tok = self.proj(pooled.flatten(2).transpose(1, 2))           # (B, N, D)
        return {"x_norm_patchtokens": tok}


def _bands(seed: int):
    g = torch.Generator().manual_seed(seed)
    out = []
    for r, c in BANDS:
        img = torch.randn(3, r * 14, c * 14, generator=g)
        t = torch.rand(r, c, generator=g)
        th = torch.rand(r, c, generator=g)
        v = torch.rand(r, c, generator=g)
        out.append((img, t, th, v))
    return out


def _model_input(backbone, bands):
    """What OtolithModel's density_image_bands path hands the density head."""
    with torch.no_grad():
        toks = [backbone.forward_features(b[0].unsqueeze(0))["x_norm_patchtokens"] for b in bands]
    return torch.cat(toks, dim=1)[0]


def test_cached_tokens_equal_model_path(tmp_path):
    torch.manual_seed(0)
    bb = _MockBackbone().eval()
    items = [(i, _bands(i)) for i in range(5)]
    w = CacheWriter(tmp_path, 5, P, DIM)
    n = fill_cache(w, items, lambda x: bb.forward_features(x)["x_norm_patchtokens"],
                   batch_size=2, log=lambda m: None)
    assert n == 5 and w.done.all()
    toks = np.load(tmp_path / "tokens.npy")
    for i, bands in items:
        ref = _model_input(bb, bands).numpy()
        np.testing.assert_allclose(toks[i].astype(np.float32), ref, rtol=2e-3, atol=2e-3)
        t = np.concatenate([b[1].reshape(-1).numpy() for b in bands])
        np.testing.assert_array_equal(np.load(tmp_path / "polar_t.npy")[i], t)


def test_resume_skips_done_rows(tmp_path):
    bb = _MockBackbone().eval()
    fn = lambda x: bb.forward_features(x)["x_norm_patchtokens"]
    w = CacheWriter(tmp_path, 4, P, DIM)
    fill_cache(w, [(0, _bands(0)), (1, _bands(1))], fn, log=lambda m: None)
    del w
    w2 = CacheWriter(tmp_path, 4, P, DIM)                    # reopen existing memmaps
    assert w2.done.tolist() == [True, True, False, False]
    todo = np.flatnonzero(~w2.done).tolist()
    fill_cache(w2, [(i, _bands(i)) for i in todo], fn, log=lambda m: None)
    assert w2.done.all()


def test_reopen_with_other_shape_is_refused(tmp_path):
    CacheWriter(tmp_path, 4, P, DIM)
    with pytest.raises(SystemExit):
        CacheWriter(tmp_path, 4, P + 1, DIM)


def test_identity_mismatch_is_refused(tmp_path):
    meta = {"model.backbone": "dinov2_vits14_reg", "backbone_from": "raw"}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    check_identity(tmp_path / "meta.json", meta)                       # same → ok
    with pytest.raises(SystemExit):
        check_identity(tmp_path / "meta.json", meta | {"backbone_from": "x.pt"})
