"""30.09 (L6) — density-head lab on cached wedge-band tokens: which recipe matures reliably?

WHY. wedge_b's density head collapsed to an absorbing all-zero state (logits ≈ −13, σ' ≈ 1e-6)
and one full run costs ~5 h per epoch. Four candidate mechanisms acted at once
(`plans and summaries/29.09_ewdge_b_przerwany_PLAN_TO_do.md` §3.3–3.6):
  1. start at p = 0.5 per cell → integral ≈ N/2 = 2926 vs age ≈ 4;
  2. the probability-space squared loss has a vanishing gradient at deep negative logits;
  3. input drift — band tokens come from a backbone being tuned for age;
  4. the 45° × 12-bin attention window holds ~395 patches on the wedge vs 13 on the square.
On cached tokens (`cache_wedge_band_tokens.py`) the backbone is frozen by construction, so
mechanism 3 is off, and a head epoch costs minutes. The arms switch one mechanism at a time:

  A0  wedge_b recipe as is                       (control; tokens frozen → tests mechanism 3)
  A1  A0 + output bias at the prior log(π/(1−π)), π = mean age / N     (mechanism 1)
  A2  A0 + on/off concentration as BCE on logits                        (mechanism 2)
  A3  A1 + A2                                                           (main candidate)
  A4  A3 + attention restricted to the canvas neighbourhood: same band, row ±1, col ±3
      = 21 patches everywhere                                           (mechanism 4)
  A5  A3 + row head: attention-pooling over the columns of each of the 44 canvas rows
      → 1-D density over t (N = 44)                                     (dilution + window)

01.10 — series B (`plans and summaries/30.09_plan_testow_laboratorium.md` §3), each one change
against A5 on the raw cache. The question is no longer maturity but whether the head reads the
individual otolith or only learns where rings usually lie (population prior):
  B1  A5 without the row's t                      (E1, pos_mode=none)
  B2  A5 with t' = a·t + b in training            (E1, pos_mode=jitter)
  B3a/b  A5 + left/right column-half profile consistency, λ 0.1 / 0.5   (E2, columns)
  B4a/b  A5 + left/right otolith of one fish consistency, λ 0.1 / 0.5   (E2, fish)
  B6  row similarity structure only, no t         (E3, rows_selfsim)
  B6c1/B6c5, B6f1/B6f5  B6 + columns / fish consistency, λ 0.1 / 0.5   (E2 on the B6 base)

Everything that exists in production is reused, not re-implemented: `RadialAttentionDensityHead`,
`density_count_loss`, `polar_fourier_features`. A4 runs the production head's own
TransformerEncoderLayer with a sparse neighbourhood (pinned equal to the dense masked layer by
a test); only A5's row pooling and the BCE form of the concentration term are new code.

zero_ratio is ALWAYS the production `density_count_loss` on sigmoid(logits), divided by the same
loss of an all-zero map on the same batches — so it is comparable across arms whatever each arm
trains on. Maturity criterion (plan §5): density_active ≥ 1 and zero_ratio < 0.5 before epoch 20,
in ≥ 4 of 5 seeds.

Cost per training sample on the 16-core CPU (batch 16, N = 5852): A5 0.023 s (~2 min/epoch),
A4 0.66 s (~56 min/epoch), A0–A3 1.35 s (production mask computed per radial bin, ~2 h/epoch).
Neither machine has a GPU; the server (128 cores) is the faster CPU lane.

    python scripts/diagnostics/density_head_lab.py --cache raw --arms A5 --seeds 0,1,2,3,4
    python scripts/diagnostics/density_head_lab.py --cache raw --arms A0,A1,A2,A3,A4 --device cpu
    python scripts/diagnostics/density_head_lab.py --cache wedge_b_best_age --arms A0 --device cpu

Writes experiments/density_head_lab/<cache>/<arm>_seed<s>/{metrics.csv, head.pt, summary.json}
and experiments/density_head_lab/<cache>/summary.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model import (RadialAttentionDensityHead, density_count_loss,
                       polar_fourier_features, polar_positional_feature_dim)

CACHE_ROOT = PROJECT_ROOT / "data" / "wedge_band_tokens"
OUT_ROOT = PROJECT_ROOT / "experiments" / "density_head_lab"


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    name: str
    prior_bias: bool          # output bias = log(π/(1−π)) instead of PyTorch's default init
    loss_form: str            # "prob_sq" (production) | "bce_logit"
    head: str                 # "radial" (production mask) | "canvas" (21-patch window) | "rows"
                              # | "rows_selfsim" (row similarities only, no t — B6)
    # 01.10 (plan 30.09 §2.4) — defaults keep every earlier head.pt loadable as it was trained.
    pos_mode: str = "absolute"    # row heads: "absolute" (t as Fourier feature) | "none" (no t)
                                  # | "jitter" (train on t' = a·t + b, evaluate on the true t)
    consistency: str = "none"     # row heads: "none" | "columns" (left vs right half of the
                                  # canvas columns) | "fish" (the two otoliths of one fish)
    cons_weight: float = 0.0      # λ of the Jensen–Shannon consistency term


ARMS = {
    "A0": Arm("A0", False, "prob_sq", "radial"),
    "A1": Arm("A1", True, "prob_sq", "radial"),
    "A2": Arm("A2", False, "bce_logit", "radial"),
    "A3": Arm("A3", True, "bce_logit", "radial"),
    "A4": Arm("A4", True, "bce_logit", "canvas"),
    "A5": Arm("A5", True, "bce_logit", "rows"),
    # Plan 30.09 §3 — one change against the base A5 (raw cache) each.
    "B1": Arm("B1", True, "bce_logit", "rows", pos_mode="none"),                    # E1
    "B2": Arm("B2", True, "bce_logit", "rows", pos_mode="jitter"),                  # E1
    "B3a": Arm("B3a", True, "bce_logit", "rows", consistency="columns", cons_weight=0.1),  # E2
    "B3b": Arm("B3b", True, "bce_logit", "rows", consistency="columns", cons_weight=0.5),
    "B4a": Arm("B4a", True, "bce_logit", "rows", consistency="fish", cons_weight=0.1),     # E2
    "B4b": Arm("B4b", True, "bce_logit", "rows", consistency="fish", cons_weight=0.5),
    "B6": Arm("B6", True, "bce_logit", "rows_selfsim"),                             # E3
    # E2 on the B6 base (01.10: B6 was the best E1/E3 variant) — columns / fish, λ 0.1 / 0.5
    "B6c1": Arm("B6c1", True, "bce_logit", "rows_selfsim", consistency="columns", cons_weight=0.1),
    "B6c5": Arm("B6c5", True, "bce_logit", "rows_selfsim", consistency="columns", cons_weight=0.5),
    "B6f1": Arm("B6f1", True, "bce_logit", "rows_selfsim", consistency="fish", cons_weight=0.1),
    "B6f5": Arm("B6f5", True, "bce_logit", "rows_selfsim", consistency="fish", cons_weight=0.5),
}
# B3/B4 sit on the base (absolute t) until E1 decides; if B1 wins, add B1-based copies here.
JITTER_SCALE = (0.7, 1.3)     # B2: a ~ U(0,7; 1,3)
JITTER_SHIFT = (-0.2, 0.2)    #     b ~ U(−0,2; 0,2), redrawn per image
SELFSIM_MAX_OFFSET = 8        # B6: similarities to rows i±2 … i±8 (±1 masked: adjacent rows
                              # are alike by image continuity, not by ring structure)

# Production values of the wedge_b run (configs/config_wedge_b.yaml) — the lab changes only
# what an arm names.
HEAD_KW = dict(hidden_dim=64, dropout=0.1, num_heads=4, num_layers=1,
               num_angle_freqs=4, num_radius_freqs=4, n_radial_bins=12, window_deg=45.0)
LR = 1e-4 * 2.0               # training.lr × training.density_lr_mult
WEIGHT_DECAY = 1e-4
CONC_WEIGHT = 1.0             # model.density_conc_weight
CANVAS_ROWS, CANVAS_COLS = 1, 3   # A4 neighbourhood half-extent → (2·1+1)·(2·3+1) = 21


# ---------------------------------------------------------------------------
# Canvas geometry
# ---------------------------------------------------------------------------

def band_shapes(meta: dict) -> list[tuple[int, int]]:
    """(rows, cols) per band in token order — DINOv2 flattens each canvas row-major."""
    rows = meta["data.wedge_band_n_radius_patches"]
    cols = meta["data.wedge_band_n_angle_patches"]
    return list(zip(rows, cols))


def canvas_neighbour_counts(shapes: list[tuple[int, int]], dr: int = CANVAS_ROWS,
                            dc: int = CANVAS_COLS) -> np.ndarray:
    """Neighbourhood size of every token (21 in the interior, fewer at band edges)."""
    out = []
    for r, c in shapes:
        rr = np.arange(r)[:, None]; cc = np.arange(c)[None, :]
        nr = np.minimum(rr + dr, r - 1) - np.maximum(rr - dr, 0) + 1
        nc = np.minimum(cc + dc, c - 1) - np.maximum(cc - dc, 0) + 1
        out.append((nr * nc).reshape(-1))
    return np.concatenate(out)


def canvas_dense_mask(shapes: list[tuple[int, int]], dr: int = CANVAS_ROWS,
                      dc: int = CANVAS_COLS) -> Tensor:
    """(N, N) bool, True = blocked — the dense equivalent of the A4 window (tests only)."""
    band, row, col = [], [], []
    for b, (r, c) in enumerate(shapes):
        rr, cc = np.meshgrid(np.arange(r), np.arange(c), indexing="ij")
        band.append(np.full(r * c, b)); row.append(rr.reshape(-1)); col.append(cc.reshape(-1))
    band, row, col = (torch.from_numpy(np.concatenate(a)) for a in (band, row, col))
    allowed = ((band[:, None] == band[None, :]) & ((row[:, None] - row[None, :]).abs() <= dr)
               & ((col[:, None] - col[None, :]).abs() <= dc))
    return ~allowed


def row_layout(shapes: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """(token offset, rows, cols) per band."""
    out, off = [], 0
    for r, c in shapes:
        out.append((off, r, c)); off += r * c
    return out


# ---------------------------------------------------------------------------
# A4: production encoder layer, sparse canvas-neighbourhood attention
# ---------------------------------------------------------------------------

def _shift2d(x: Tensor, dr: int, dc: int) -> tuple[Tensor, Tensor]:
    """x: (B, R, C, ...). Returns y with y[:, r, c] = x[:, r+dr, c+dc] and a (R, C) validity."""
    B, R, C = x.shape[:3]
    y = torch.zeros_like(x)
    valid = torch.zeros(R, C, dtype=torch.bool, device=x.device)
    r0, r1 = max(0, -dr), min(R, R - dr)
    c0, c1 = max(0, -dc), min(C, C - dc)
    if r0 < r1 and c0 < c1:
        y[:, r0:r1, c0:c1] = x[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        valid[r0:r1, c0:c1] = True
    return y, valid


def local_encoder_layer(layer: nn.TransformerEncoderLayer, x: Tensor,
                        shapes: list[tuple[int, int]], dr: int = CANVAS_ROWS,
                        dc: int = CANVAS_COLS) -> Tensor:
    """``layer(x, mask=canvas_dense_mask(shapes))`` computed without an N×N matrix.

    Reproduces nn.TransformerEncoderLayer(norm_first=True) term by term with the layer's own
    parameters. Each canvas row attends to the (2dr+1) rows around it as one batched matmul
    (C queries × (2dr+1)·C keys), with the column window |Δcol| ≤ dc and the rows beyond the
    band edge masked out — cost scales with N·(2dr+1)·C instead of N².
    """
    assert layer.norm_first, "only the norm_first layout used by RadialAttentionDensityHead"
    attn = layer.self_attn
    B, N, D = x.shape
    H = attn.num_heads
    hd = D // H
    h = layer.norm1(x)
    q, k, v = F.linear(h, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q = q * (hd ** -0.5)
    W = 2 * dr + 1
    outs = []
    for off, R, C in row_layout(shapes):
        # (B, H, R, C, hd)
        qb = q[:, off:off + R * C].reshape(B, R, C, H, hd).permute(0, 3, 1, 2, 4)
        kb = k[:, off:off + R * C].reshape(B, R, C, H, hd).permute(0, 3, 1, 2, 4)
        vb = v[:, off:off + R * C].reshape(B, R, C, H, hd).permute(0, 3, 1, 2, 4)
        kp = F.pad(kb, (0, 0, 0, 0, dr, dr))                             # zero rows at band edges
        vp = F.pad(vb, (0, 0, 0, 0, dr, dr))
        # rows r-dr..r+dr stacked along the key axis → (B, H, R, W·C, hd)
        k3 = torch.cat([kp[:, :, a:a + R] for a in range(W)], dim=3)
        v3 = torch.cat([vp[:, :, a:a + R] for a in range(W)], dim=3)
        s = qb @ k3.transpose(-1, -2)                                    # (B, H, R, C, W·C)
        rows = torch.arange(R, device=x.device)
        key_row = rows[:, None] + torch.arange(W, device=x.device)[None, :] - dr      # (R, W)
        row_ok = (key_row >= 0) & (key_row < R)
        col = torch.arange(C, device=x.device)
        col_ok = (col[:, None] - col[None, :]).abs() <= dc                            # (C, C)
        allowed = row_ok[:, None, :, None] & col_ok[None, :, None, :]                 # (R, C, W, C)
        s = s.masked_fill(~allowed.reshape(R, C, W * C), float("-inf"))
        w = torch.softmax(s, dim=-1)
        w = F.dropout(w, p=attn.dropout, training=layer.training)
        o = (w @ v3).permute(0, 2, 3, 1, 4)                              # (B, R, C, H, hd)
        outs.append(o.reshape(B, R * C, D))
    o = attn.out_proj(torch.cat(outs, dim=1))
    x = x + layer.dropout1(o)
    x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
    return x


def grouped_encoder_layer(layer: nn.TransformerEncoderLayer, x: Tensor,
                          groups: list[tuple[Tensor, Tensor]]) -> Tensor:
    """``layer(x, mask=M)`` for a block-diagonal M given as ``(indices, blocked (n, n))`` groups.

    Tokens only attend within their own group, so attention is computed per group on the
    gathered sub-sequence; every other term of the norm_first layer is per-token and unchanged.
    """
    assert layer.norm_first
    attn = layer.self_attn
    B, N, D = x.shape
    H = attn.num_heads
    hd = D // H
    q, k, v = F.linear(layer.norm1(x), attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q = q * (hd ** -0.5)
    o = torch.zeros_like(q)
    for idx, blocked in groups:
        n = idx.numel()
        qg, kg, vg = (a[:, idx].reshape(B, n, H, hd).transpose(1, 2) for a in (q, k, v))
        s = (qg @ kg.transpose(-1, -2)).masked_fill(blocked, float("-inf"))
        w = F.dropout(torch.softmax(s, dim=-1), p=attn.dropout, training=layer.training)
        o[:, idx] = (w @ vg).transpose(1, 2).reshape(B, n, D)
    x = x + layer.dropout1(attn.out_proj(o))
    x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
    return x


class BinBlockRadialHead(RadialAttentionDensityHead):
    """The production head, computed per radial bin (arms A0–A3; same weights, same output).

    The production mask (`radial_local_attention_mask`) only lets a token attend inside its own
    radial bin and angular window. On the band canvases every image has the same t per token and
    θ differing only by the axis angle, so the bins are identical across the batch and the mask is
    block-diagonal: attention costs Σ n_bin² (5.7 M pairs) instead of N² (34 M). When the batch
    does not share its geometry the production forward is used unchanged.
    """

    def forward(self, patches, polar_t=None, polar_theta=None, polar_valid=None):
        same_geometry = (polar_t is not None and polar_theta is not None
                         and bool((polar_t == polar_t[:1]).all())
                         and bool(((polar_theta - polar_theta[:, :1])
                                   - (polar_theta[:1] - polar_theta[:1, :1])).abs().max() < 1e-4)
                         and (polar_valid is None or bool(polar_valid.all())))
        if not same_geometry:
            return super().forward(patches, polar_t, polar_theta, polar_valid)
        x = self.input_norm(patches)
        x = x + self.pos_proj(polar_fourier_features(polar_t, polar_theta,
                                                     self.num_angle_freqs, self.num_radius_freqs))
        t0, th0 = polar_t[0], polar_theta[0]
        bins = torch.clamp((t0 * self.n_radial_bins).long(), 0, self.n_radial_bins - 1)
        groups = []
        for b in torch.unique(bins):
            idx = torch.nonzero(bins == b).squeeze(1)
            d = th0[idx].unsqueeze(1) - th0[idx].unsqueeze(0)
            d = torch.atan2(torch.sin(d), torch.cos(d))
            allowed = (d.abs() <= math.radians(self.window_deg) / 2.0) | torch.eye(
                idx.numel(), dtype=torch.bool, device=x.device)
            groups.append((idx, ~allowed))
        for layer in self.encoder.layers:
            x = grouped_encoder_layer(layer, x, groups)
        return self.out_head(x)


class CanvasWindowDensityHead(RadialAttentionDensityHead):
    """Production head with the attention window defined in canvas indices (arm A4)."""

    def __init__(self, embed_dim: int, shapes: list[tuple[int, int]], **kw):
        super().__init__(embed_dim=embed_dim, **kw)
        self.shapes = shapes

    def forward(self, patches, polar_t=None, polar_theta=None, polar_valid=None):
        x = self.input_norm(patches)
        x = x + self.pos_proj(polar_fourier_features(polar_t, polar_theta,
                                                     self.num_angle_freqs, self.num_radius_freqs))
        for layer in self.encoder.layers:
            x = local_encoder_layer(layer, x, self.shapes)
        return self.out_head(x)


# ---------------------------------------------------------------------------
# A5: row head
# ---------------------------------------------------------------------------

class RowDensityHead(nn.Module):
    """One density value per canvas row (44 rows over the 4 bands).

    A row of a band canvas is a fixed normalised radius t, i.e. one arc of the wedge. The row is
    summarised by attention pooling over its columns (so a ring that is not perfectly aligned with
    the row can still dominate it), tissue-weighted, then the row's t is added as a Fourier
    feature and an MLP gives one logit. The count target is unchanged: the rows' probabilities
    should sum to the age, and ⌈age⌉ rows should be "on".
    """

    def __init__(self, embed_dim: int, shapes: list[tuple[int, int]], hidden_dim: int = 64,
                 dropout: float = 0.1, num_radius_freqs: int = 4, pos_mode: str = "absolute"):
        super().__init__()
        if pos_mode not in ("absolute", "none", "jitter"):
            raise ValueError(pos_mode)
        self.shapes = shapes
        self.num_radius_freqs = num_radius_freqs
        self.pos_mode = pos_mode
        self.input_norm = nn.LayerNorm(embed_dim)
        self.score = nn.Linear(embed_dim, 1)
        # B1: no position module at all — the head cannot see t (the DINOv2 tokens still carry
        # their own canvas position embedding; that is the backbone's, not the head's).
        self.pos_proj = (None if pos_mode == "none" else
                         nn.Linear(polar_positional_feature_dim(0, num_radius_freqs), embed_dim))
        self.out_head = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                                      nn.Dropout(p=dropout), nn.Linear(hidden_dim, 1))

    def rows(self, t: Tensor, tissue: Tensor) -> tuple[Tensor, Tensor]:
        """Per-row mean t and tissue fraction, (B, n_rows) each."""
        rt, rv = [], []
        for off, R, C in row_layout(self.shapes):
            rt.append(t[:, off:off + R * C].reshape(-1, R, C).mean(-1))
            rv.append(tissue[:, off:off + R * C].reshape(-1, R, C).mean(-1))
        return torch.cat(rt, 1), torch.cat(rv, 1)

    def pool(self, patches: Tensor, tissue: Tensor, col_half: Optional[str] = None) -> Tensor:
        """(B, n_rows, D) attention pooling over each row's columns.

        ``col_half`` = "left" | "right" pools only that half of every band's columns (B3: two
        independent profiles of the same wedge)."""
        x = self.input_norm(patches)
        s = self.score(x).squeeze(-1) + torch.log(tissue.clamp(min=1e-4))      # (B, N)
        pooled = []
        for off, R, C in row_layout(self.shapes):
            xb = x[:, off:off + R * C].reshape(x.shape[0], R, C, -1)
            sb = s[:, off:off + R * C].reshape(-1, R, C)
            if col_half is not None:
                keep = torch.arange(C, device=x.device) < C // 2
                sb = sb.masked_fill((~keep if col_half == "left" else keep)[None, None, :],
                                    float("-inf"))
            wb = torch.softmax(sb, dim=-1)
            pooled.append((wb.unsqueeze(-1) * xb).sum(2))                       # (B, R, D)
        return torch.cat(pooled, 1)                                              # (B, 44, D)

    def forward(self, patches: Tensor, polar_t: Tensor, tissue: Tensor,
                col_half: Optional[str] = None) -> Tensor:
        z = self.pool(patches, tissue, col_half)
        if self.pos_proj is not None:
            row_t, _ = self.rows(polar_t, tissue)
            if self.pos_mode == "jitter" and self.training:
                B = row_t.shape[0]
                a = torch.empty(B, 1, device=row_t.device).uniform_(*JITTER_SCALE)
                b = torch.empty(B, 1, device=row_t.device).uniform_(*JITTER_SHIFT)
                row_t = a * row_t + b
            z = z + self.pos_proj(polar_fourier_features(row_t, torch.zeros_like(row_t),
                                                         0, self.num_radius_freqs))
        return self.out_head(z)


class RowSelfSimDensityHead(RowDensityHead):
    """B6: one logit per row from the row-to-row similarity structure only (RepNet-style).

    Rows are pooled as in A5, projected and L2-normalised; row i is described by its cosine
    similarity to rows i±2 … i±SELFSIM_MAX_OFFSET (relative offsets, so the descriptor does not
    say where row i is), and a small 1-D convolution along the rows gives the logit. No t input.
    Out-of-range offsets are 0 — the only positional cue left is the distance to the canvas ends
    within SELFSIM_MAX_OFFSET rows (plus whatever the DINOv2 tokens carry themselves).
    """

    def __init__(self, embed_dim: int, shapes: list[tuple[int, int]], hidden_dim: int = 64,
                 dropout: float = 0.1, proj_dim: int = 64, max_offset: int = SELFSIM_MAX_OFFSET):
        super().__init__(embed_dim, shapes, hidden_dim, dropout, pos_mode="none")
        self.offsets = [d for d in range(-max_offset, max_offset + 1) if abs(d) >= 2]
        self.proj = nn.Linear(embed_dim, proj_dim)
        self.out_head = nn.Sequential(
            nn.Conv1d(len(self.offsets), hidden_dim, kernel_size=3, padding=1), nn.GELU(),
            nn.Dropout(p=dropout), nn.Conv1d(hidden_dim, 1, kernel_size=1))

    def similarity_features(self, z: Tensor) -> Tensor:
        """(B, n_offsets, R): feature[:, j, i] = cos(row i, row i + offsets[j]), 0 off the canvas."""
        e = F.normalize(self.proj(z), dim=-1)
        S = e @ e.transpose(1, 2)                                               # (B, R, R)
        R = S.shape[1]
        feats = []
        for d in self.offsets:
            diag = torch.diagonal(S, offset=d, dim1=1, dim2=2)                  # (B, R − |d|)
            pad = (0, d) if d > 0 else (-d, 0)
            feats.append(F.pad(diag, pad) if abs(d) < R else torch.zeros_like(S[:, 0]))
        return torch.stack(feats, 1)

    def forward(self, patches: Tensor, polar_t: Tensor, tissue: Tensor,
                col_half: Optional[str] = None) -> Tensor:
        z = self.pool(patches, tissue, col_half)
        return self.out_head(self.similarity_features(z)).transpose(1, 2)       # (B, R, 1)


# ---------------------------------------------------------------------------
# Losses and initialisation
# ---------------------------------------------------------------------------

def prior_bias(mean_age: float, n_cells: float) -> float:
    """log(π/(1−π)), π = expected fraction of "on" cells (Lin et al. 2017, RetinaNet)."""
    pi = min(max(mean_age / n_cells, 1e-6), 0.5)
    return math.log(pi / (1.0 - pi))


def set_prior(head: nn.Module, mean_age: float, n_cells: float) -> float:
    b = prior_bias(mean_age, n_cells)
    with torch.no_grad():
        head.out_head[-1].bias.fill_(b)
    return b


def bce_logit_loss(logits: Tensor, age: Tensor, valid: Tensor,
                   conc_weight: float = CONC_WEIGHT) -> Tensor:
    """Production structure (count + top-⌈age⌉ on/off), concentration on logits.

    Same count term and same on/off split as ``density_count_loss`` (top-k by value among
    tissue cells, background always off, on and off averaged separately so the ~k vs ~N
    imbalance is already balanced). Only the per-cell penalty changes: (1−p)² → −log p and
    p² → −log(1−p), whose gradient w.r.t. the logit tends to −1 (not 0) for an "on" cell deep
    in the negative range — the absorbing state of the squared form cannot occur.
    """
    p = torch.sigmoid(logits)
    l_count = F.smooth_l1_loss((p * valid).sum(dim=1), age.float())
    B, N = logits.shape
    key = logits.masked_fill(valid < 0.5, float("-inf"))
    order = torch.argsort(key, dim=1, descending=True)
    z = torch.gather(logits, 1, order)
    sv = torch.gather(valid, 1, order)
    k = age.long().clamp(min=0, max=N).unsqueeze(1)
    on = (torch.arange(N, device=logits.device).unsqueeze(0) < k).float() * sv
    off = 1.0 - on
    l_on = (F.softplus(-z) * on).sum(1) / on.sum(1).clamp(min=1.0)
    l_off = (F.softplus(z) * off).sum(1) / off.sum(1).clamp(min=1.0)
    return l_count + conc_weight * (l_on + l_off).mean()


def row_profile(logits: Tensor, valid: Tensor, eps: float = 1e-8) -> Tensor:
    """Where along the radius the head puts its count: p·tissue normalised to sum 1 per image."""
    p = torch.sigmoid(logits) * valid
    return p / p.sum(1, keepdim=True).clamp(min=eps)


def js_divergence(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """Jensen–Shannon divergence (nats) between rows of two (B, M) distributions → (B,)."""
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + eps) - torch.log(b + eps))).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def fish_pairs(keys: np.ndarray) -> list[tuple[int, int]]:
    """Index pairs (i, j) of batch positions holding two images of the same fish."""
    first: dict = {}
    pairs = []
    for i, k in enumerate(keys):
        if k in first:
            pairs.append((first.pop(k), i))
        else:
            first[k] = i
    return pairs


def fish_batches(rows: np.ndarray, keys: np.ndarray, batch_size: int,
                 rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffled batches that never split the images of one fish (B4)."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(keys[r], []).append(r)
    order = [groups[k] for k in rng.permutation(list(groups))]
    out, cur = [], []
    for g in order:
        if cur and len(cur) + len(g) > batch_size:
            out.append(np.array(cur)); cur = []
        cur.extend(g)
    if cur:
        out.append(np.array(cur))
    return out


def consistency_loss(arm: Arm, head: nn.Module, b: dict, logits: Tensor, valid: Tensor) -> Tensor:
    """E2 terms. columns: JS between the profiles of the left and right column halves of the same
    wedge. fish: JS between the two otoliths of one fish, gradient through one of them only
    (asymmetric, chosen at random per pair — Count-level weak supervision, 2003.00164)."""
    if arm.consistency == "columns":
        lv = head_forward(arm, head, b, col_half="left")[0]
        rv = head_forward(arm, head, b, col_half="right")[0]
        return js_divergence(row_profile(lv, valid), row_profile(rv, valid)).mean()
    if arm.consistency == "fish":
        pairs = fish_pairs(b["fish"])
        if not pairs:
            return logits.new_zeros(())
        prof = row_profile(logits, valid)
        flip = torch.rand(len(pairs)) < 0.5
        i = torch.tensor([q if f else p for (p, q), f in zip(pairs, flip)])
        j = torch.tensor([p if f else q for (p, q), f in zip(pairs, flip)])
        return js_divergence(prof[i], prof[j].detach()).mean()
    return logits.new_zeros(())


def arm_loss(arm: Arm, logits: Tensor, age: Tensor, valid: Tensor) -> Tensor:
    if arm.loss_form == "prob_sq":
        return density_count_loss(torch.sigmoid(logits), age, CONC_WEIGHT, 0.0, valid_mask=valid)
    return bce_logit_loss(logits, age, valid)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TokenCache:
    """Memory-mapped view of one ``cache_wedge_band_tokens.py`` output."""

    def __init__(self, root: Path, label: str):
        self.root = root
        self.meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.index = pd.read_csv(root / "index.csv")
        done = np.load(root / "done.npy")
        if not done.all():
            raise SystemExit(f"{root}: cache incomplete ({int(done.sum())}/{len(done)}) — "
                             f"finish cache_wedge_band_tokens.py first.")
        self.tokens = np.load(root / "tokens.npy", mmap_mode="r")
        self.t = np.load(root / "polar_t.npy", mmap_mode="r")
        self.theta = np.load(root / "polar_theta.npy", mmap_mode="r")
        self.valid = np.load(root / "tissue_valid.npy", mmap_mode="r")
        self.age = self.index[{"recorded": "age_recorded", "quarter": "age_quarter"}[label]].to_numpy()
        self.fish = (self.index["fish_key"].astype(str).to_numpy() if "fish_key" in self.index
                     else self.index["image_id"].astype(str).to_numpy())
        self.shapes = band_shapes(self.meta)

    def rows_of(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.index["split"].to_numpy() == split)

    def batch(self, rows: np.ndarray, device) -> dict:
        rows = np.sort(rows)                      # sequential reads from the memmap
        to = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a[rows]).astype(dt)).to(device)
        return {"x": to(self.tokens, np.float32), "t": to(self.t, np.float32),
                "theta": to(self.theta, np.float32), "valid": to(self.valid, np.float32),
                "age": torch.from_numpy(self.age[rows]).to(device), "fish": self.fish[rows]}


# ---------------------------------------------------------------------------
# Head construction, forward, metrics
# ---------------------------------------------------------------------------

def is_row_head(arm: Arm) -> bool:
    return arm.head.startswith("rows")


def build_head(arm: Arm, dim: int, shapes: list[tuple[int, int]]) -> nn.Module:
    if not is_row_head(arm) and (arm.pos_mode != "absolute" or arm.consistency != "none"):
        raise NotImplementedError(f"{arm.name}: pos_mode/consistency only for row heads")
    if arm.head == "radial":
        return BinBlockRadialHead(embed_dim=dim, **HEAD_KW)
    if arm.head == "canvas":
        return CanvasWindowDensityHead(dim, shapes, **HEAD_KW)
    if arm.head == "rows_selfsim":
        return RowSelfSimDensityHead(dim, shapes, hidden_dim=HEAD_KW["hidden_dim"],
                                     dropout=HEAD_KW["dropout"])
    return RowDensityHead(dim, shapes, hidden_dim=HEAD_KW["hidden_dim"],
                          dropout=HEAD_KW["dropout"], num_radius_freqs=HEAD_KW["num_radius_freqs"],
                          pos_mode=arm.pos_mode)


def head_forward(arm: Arm, head: nn.Module, b: dict,
                 col_half: Optional[str] = None) -> tuple[Tensor, Tensor]:
    """(logits (B, M), per-cell validity (B, M)) — M = 5852 cells, or 44 rows for row heads."""
    if is_row_head(arm):
        logits = head(b["x"], b["t"], b["valid"], col_half=col_half).squeeze(-1)
        _, row_valid = head.rows(b["t"], b["valid"])
        return logits, row_valid
    ones = torch.ones_like(b["t"], dtype=torch.bool)       # production: positions always valid
    logits = head(b["x"], polar_t=b["t"], polar_theta=b["theta"], polar_valid=ones).squeeze(-1)
    return logits, b["valid"]


@torch.no_grad()
def evaluate(arm: Arm, head: nn.Module, cache: TokenCache, rows: np.ndarray, device,
             batch_size: int) -> dict:
    head.eval()
    s = {"loss": 0.0, "zero": 0.0, "active": 0.0, "max_logit": 0.0, "median_logit": 0.0,
         "sum_p_over_age": 0.0}
    n = 0
    for i in range(0, len(rows), batch_size):
        b = cache.batch(rows[i:i + batch_size], device)
        logits, valid = head_forward(arm, head, b)
        p = torch.sigmoid(logits)
        bs = logits.shape[0]
        s["loss"] += density_count_loss(p, b["age"], CONC_WEIGHT, 0.0, valid_mask=valid).item() * bs
        s["zero"] += density_count_loss(torch.zeros_like(p), b["age"], CONC_WEIGHT, 0.0,
                                        valid_mask=valid).item() * bs
        s["active"] += (p > 0.5).sum(1).float().sum().item()
        s["max_logit"] += logits.amax(1).sum().item()
        s["median_logit"] += logits.median(1).values.sum().item()
        s["sum_p_over_age"] += ((p * valid).sum(1) / b["age"].float().clamp(min=1)).sum().item()
        n += bs
    out = {k: v / n for k, v in s.items() if k != "zero"}
    out["zero_ratio"] = s["loss"] / max(s["zero"], 1e-12)
    return out


def matured(history: list[dict], by_epoch: int = 20) -> Optional[int]:
    """First epoch ≤ by_epoch with density_active ≥ 1 and zero_ratio < 0.5, else None."""
    for h in history:
        if 1 <= h["epoch"] <= by_epoch and h["active"] >= 1.0 and h["zero_ratio"] < 0.5:
            return h["epoch"]
    return None


def run_arm(arm: Arm, cache: TokenCache, seed: int, epochs: int, batch_size: int, device,
            out_dir: Path, max_train: Optional[int] = None, log=print,
            stop_after_mature: Optional[int] = 3) -> dict:
    """Train one arm/seed. Saves ``state.pt`` after every epoch and resumes from it, so a
    multi-day CPU run survives an interruption. With ``stop_after_mature=k`` a seed that has
    matured stops k epochs later (the gate only looks at epochs <= 20, the extra epochs show the
    head stays mature); None trains all ``epochs``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dim = int(cache.meta["dim"])
    head = build_head(arm, dim, cache.shapes).to(device)
    train_rows, val_rows = cache.rows_of("train"), cache.rows_of("val")
    if max_train:
        train_rows = rng.permutation(train_rows)[:max_train]
    n_cells = 44 if is_row_head(arm) else int(cache.meta["n_patches"])
    bias = set_prior(head, float(cache.age[train_rows].mean()), n_cells) if arm.prior_bias else None
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = epochs * (len(fish_batches(train_rows, cache.fish, batch_size, np.random.default_rng(0)))
                      if arm.consistency == "fish" else math.ceil(len(train_rows) / batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.pt"
    start = 1
    if state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        head.load_state_dict(st["head"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
        history, bias, start = st["history"], st["bias"], st["history"][-1]["epoch"] + 1
        log(f"  {arm.name} seed{seed}: wznawiam od epoki {start}")
    else:
        history = [{"epoch": 0, "train_loss": float("nan"),
                    **evaluate(arm, head, cache, val_rows, device, batch_size)}]
        log(f"  {arm.name} seed{seed} e0  zero_ratio={history[0]['zero_ratio']:.3f}  "
            f"active={history[0]['active']:.2f}  max_logit={history[0]['max_logit']:.2f}"
            + (f"  prior_bias={bias:.2f}" if bias is not None else ""))
    t0 = time.time()
    for epoch in range(start, epochs + 1):
        mat_epoch = matured(history)
        if stop_after_mature is not None and mat_epoch is not None                 and epoch > mat_epoch + stop_after_mature:
            log(f"  {arm.name} seed{seed}: dojrzało w e{mat_epoch}, stop po {stop_after_mature} epokach")
            break
        head.train()
        if arm.consistency == "fish":
            batches = fish_batches(train_rows, cache.fish, batch_size, rng)
        else:
            order = rng.permutation(train_rows)
            batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        tot, n = 0.0, 0
        for rows_b in batches:
            b = cache.batch(rows_b, device)
            logits, valid = head_forward(arm, head, b)
            loss = arm_loss(arm, logits, b["age"], valid)
            if arm.consistency != "none":
                loss = loss + arm.cons_weight * consistency_loss(arm, head, b, logits, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item() * logits.shape[0]; n += logits.shape[0]
        h = {"epoch": epoch, "train_loss": tot / n,
             **evaluate(arm, head, cache, val_rows, device, batch_size)}
        history.append(h)
        log(f"  {arm.name} seed{seed} e{epoch}  train={h['train_loss']:.4f}  "
            f"zero_ratio={h['zero_ratio']:.3f}  active={h['active']:.2f}  "
            f"max_logit={h['max_logit']:.2f}  median_logit={h['median_logit']:.2f}  "
            f"Σp/age={h['sum_p_over_age']:.2f}  ({(time.time() - t0) / (epoch - start + 1):.0f} s/ep)")
        pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)
        torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "history": history, "bias": bias}, state_path)

    pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)
    torch.save({"arm": asdict(arm), "seed": seed, "head_kw": HEAD_KW, "shapes": cache.shapes,
                "prior_bias": bias, "state_dict": head.state_dict(),
                "cache_meta": cache.meta}, out_dir / "head.pt")
    mat = matured(history)
    summary = {"arm": arm.name, "seed": seed, "matured_epoch": mat, "matured": mat is not None,
               "final_zero_ratio": history[-1]["zero_ratio"], "final_active": history[-1]["active"],
               "min_zero_ratio": min(h["zero_ratio"] for h in history)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    state_path.unlink(missing_ok=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="tag under data/wedge_band_tokens/")
    ap.add_argument("--arms", default="A4,A5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=20,
                    help="maximum; the maturity gate only looks at epochs <= 20")
    ap.add_argument("--stop-after-mature", type=int, default=3,
                    help="stop a seed this many epochs after it matures; -1 = never")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--label", choices=["quarter", "recorded"], default="quarter",
                    help="count target; the same for every arm so arms differ in the head only")
    ap.add_argument("--max-train", type=int, default=None, help="subsample train (smoke runs)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--skip-done", action="store_true",
                    help="skip arm/seed pairs that already have summary.json (resume)")
    args = ap.parse_args()

    from src.utils import resolve_device
    # Windows redirects stdout in the console code page (cp1250), which has no "Σ": the first
    # local A5 run crashed on its log line after epoch 1. UTF-8 everywhere.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    cache = TokenCache(CACHE_ROOT / args.cache, args.label)
    arms = [ARMS[a.strip()] for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if device.type == "cpu" and any(a.head == "radial" for a in arms):
        print("INFO: ramiona A0–A3 (maska produkcyjna liczona blokami binów) to ~1,35 s na próbkę "
              "na 16 rdzeniach — ~2 h/epokę; serwer (128 rdzeni, bez GPU) szybciej.")
    root = OUT_ROOT / args.cache
    print(f"LAB  cache={args.cache}  label={args.label}  device={device}  arms={[a.name for a in arms]}  "
          f"seeds={seeds}  epochs={args.epochs}  N={cache.meta['n_patches']}  "
          f"train={len(cache.rows_of('train'))}  val={len(cache.rows_of('val'))}", flush=True)
    for arm in arms:
        for seed in seeds:
            out_dir = root / f"{arm.name}_seed{seed}"
            if args.skip_done and (out_dir / "summary.json").exists():
                print(f"  {arm.name} seed{seed}: gotowe wcześniej — pomijam", flush=True)
                continue
            run_arm(arm, cache, seed, args.epochs, args.batch_size, device, out_dir,
                    args.max_train, log=lambda m: print(m, flush=True),
                    stop_after_mature=None if args.stop_after_mature < 0 else args.stop_after_mature)
            collect_summaries(root).to_csv(root / "summary.csv", index=False)
    df = collect_summaries(root)
    if not df.empty:
        print("\n" + df.groupby("arm").agg(matured=("matured", "sum"), seeds=("seed", "count"),
                                           median_epoch=("matured_epoch", "median"),
                                           final_zero_ratio=("final_zero_ratio", "median")).to_string())


def collect_summaries(root: Path) -> pd.DataFrame:
    """Every finished arm/seed under ``root`` (from any invocation), one row each."""
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*_seed*/summary.json"))]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
