"""OtolithModel: DINOv2 backbone + ordinal (CORAL) head + optional MIL patch head.

Two heads, selectable via ``cfg.model.head_type``:

  * ``coral`` — original behaviour: Dropout + Linear on the CLS token →
    K-1 ordinal logits (CORAL). Patch tokens are NOT supervised.
  * ``mil``   — weakly-supervised localisation: a shared MLP scores every patch
    independently; the sum of patch scores ≈ predicted age. Patches learn
    *where* the increments are because the gradient flows back from a count
    regression loss into each patch.
  * ``both``  — train both heads simultaneously with a weighted combined loss.
    CORAL stabilises age prediction; MIL provides localisation. Patch
    probabilities replace the L2-norm heuristic in interpretation.

Forward always returns a dict; callers select the head they need.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import OtolithConfig

EMBED_DIMS: Dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    # "with registers" variants — recommended for patch-level interpretation, as
    # register tokens suppress the high-norm artifact patches that otherwise
    # pollute L2-norm / MIL importance maps (Darcet et al. 2023).
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}


def load_dinov2(backbone_name: str) -> nn.Module:
    """Download and return a DINOv2 backbone from torch.hub (requires internet)."""
    return torch.hub.load(
        "facebookresearch/dinov2",
        backbone_name,
        pretrained=True,
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def ordinal_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """CORAL ordinal loss: independent BCE at each rank threshold.

    logits:  FloatTensor (B, K-1) — raw pre-sigmoid logits
    targets: FloatTensor (B, K-1) — binary ordinal targets (1 iff age > position)
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


def mil_count_loss(
    patch_probs: Tensor,
    age: Tensor,
    sparsity_weight: float = 0.01,
) -> Tensor:
    """MIL count-regression loss.

    L = MSE(sum_of_patch_probs, age) + sparsity_weight * mean(patch_probs)

    The MSE term forces the total patch activation to match the true age.
    The sparsity term encourages most patches to be near zero, so only a
    small number of patches (≈ age) end up with high probability — those
    are the candidate increment locations.

    Args:
        patch_probs     : (B, N) probabilities ∈ [0, 1]
        age             : (B,)   true integer ages (will be cast to float)
        sparsity_weight : non-negative weight on the mean-activation penalty

    Returns scalar.
    """
    count = patch_probs.sum(dim=1)                       # (B,)
    l_count  = F.mse_loss(count, age.float())
    l_sparse = patch_probs.mean()
    return l_count + sparsity_weight * l_sparse


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class OtolithModel(nn.Module):
    """DINOv2 backbone + optional metadata fusion + CORAL and/or MIL heads.

    ``forward(image, metadata=None)`` returns a dict whose keys depend on
    ``cfg.model.head_type``:

      head_type='coral': {"coral_logits": (B, K-1)}
      head_type='mil'  : {"patch_probs": (B, N),  "patch_count": (B,)}
      head_type='both' : all of the above

    A backbone can be injected at construction time (used for unit tests
    to avoid torch.hub network calls; pass None to load the real DINOv2).
    """

    def __init__(
        self,
        cfg: OtolithConfig,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_metadata = cfg.model.use_metadata
        self.head_type = cfg.model.head_type

        self.backbone = backbone if backbone is not None else load_dinov2(cfg.model.backbone)

        embed_dim: int = getattr(
            self.backbone,
            "embed_dim",
            EMBED_DIMS.get(cfg.model.backbone, 384),
        )
        num_outputs = cfg.model.num_age_classes - 1

        # Metadata fusion (used only by CORAL head)
        meta_hidden = 0
        if self.use_metadata:
            meta_dim = len(cfg.data.metadata_cols)
            meta_hidden = 32
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, meta_hidden),
                nn.ReLU(),
            )

        # CORAL head (CLS-based ordinal regression) — rank-consistent.
        # A single shared weight vector maps the feature to one scalar score g;
        # K-1 monotonically-increasing thresholds are then subtracted, so
        # P(age>0) >= P(age>1) >= ... is GUARANTEED for every sample (Cao et
        # al. 2020, "Rank consistent ordinal regression"). Thresholds are
        # parameterised as base + cumulative softplus gaps to stay increasing.
        if self.head_type in {"coral", "both"}:
            self.head = nn.Sequential(
                nn.Dropout(p=cfg.model.dropout),
                nn.Linear(embed_dim + meta_hidden, 1, bias=False),
            )
            self.coral_theta0 = nn.Parameter(torch.zeros(1))
            self.coral_gaps = nn.Parameter(torch.zeros(max(num_outputs - 1, 0)))

        # MIL head (per-patch increment scoring — shared MLP across patches)
        if self.head_type in {"mil", "both"}:
            self.patch_head = nn.Sequential(
                nn.Linear(embed_dim, cfg.model.mil_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=cfg.model.dropout),
                nn.Linear(cfg.model.mil_hidden_dim, 1),
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, image: Tensor, metadata: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Return a dict of head outputs. Patches participate in autograd."""
        # NOTE: forward_features WITHOUT torch.no_grad — patches must
        # backpropagate when the MIL head is active.
        feats = self.backbone.forward_features(image)
        cls = feats["x_norm_clstoken"]                # (B, D)
        patches = feats["x_norm_patchtokens"]         # (B, N, D)

        out: Dict[str, Tensor] = {}

        if self.head_type in {"coral", "both"}:
            feat = cls
            if self.use_metadata and metadata is not None:
                feat = torch.cat([cls, self.meta_proj(metadata)], dim=1)
            out["coral_logits"] = self._coral_logits(feat)   # (B, K-1)

        if self.head_type in {"mil", "both"}:
            patch_logits = self.patch_head(patches).squeeze(-1)   # (B, N)
            patch_probs  = torch.sigmoid(patch_logits)            # (B, N) ∈ [0,1]
            out["patch_probs"] = patch_probs
            out["patch_count"] = patch_probs.sum(dim=1)           # (B,)

        return out

    def _coral_logits(self, feat: Tensor) -> Tensor:
        """Rank-consistent ordinal logits: shared score g minus increasing thresholds.

        thetas = [theta0, theta0+softplus(gap_0), theta0+softplus(gap_0)+softplus(gap_1), …]
        are strictly increasing, so logits[:, k] = g - thetas[k] are non-increasing
        in k and the decoded age (# of sigmoids > 0.5) is always a consistent prefix.
        """
        g = self.head(feat)                              # (B, 1)
        gaps = F.softplus(self.coral_gaps)               # (K-2,) >= 0
        thetas = torch.cat([
            self.coral_theta0,
            self.coral_theta0 + torch.cumsum(gaps, dim=0),
        ])                                               # (K-1,)
        return g - thetas.unsqueeze(0)                   # (B, K-1)

    # ------------------------------------------------------------------
    # Patch access (interpretation / inference)
    # ------------------------------------------------------------------

    def get_patch_tokens(self, image: Tensor) -> Tensor:
        """Return patch tokens as spatial grid (B, H_p, W_p, embed_dim).

        No gradient tracking — for L2-norm interpretation of the CORAL-only
        variant. The MIL pathway uses get_patch_probs() instead.
        """
        with torch.no_grad():
            feats = self.backbone.forward_features(image)
        patch_tokens = feats["x_norm_patchtokens"]
        B, N, D = patch_tokens.shape
        H_p = W_p = int(N ** 0.5)
        return patch_tokens.reshape(B, H_p, W_p, D)

    def get_cls_and_patches(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (cls_token, patch_grid) without gradient tracking."""
        with torch.no_grad():
            feats = self.backbone.forward_features(image)
        cls = feats["x_norm_clstoken"]
        patch_tokens = feats["x_norm_patchtokens"]
        B, N, D = patch_tokens.shape
        H_p = W_p = int(N ** 0.5)
        return cls, patch_tokens.reshape(B, H_p, W_p, D)

    def get_patch_probs(self, image: Tensor) -> Tensor:
        """MIL patch probabilities as a spatial grid (B, H_p, W_p).

        Bez gradientów — używane przez interpretacji i candidates.
        Raises if model wasn't built with a MIL head.
        """
        if not hasattr(self, "patch_head"):
            raise RuntimeError(
                "Model has no MIL head (head_type='coral'); cannot return patch_probs"
            )
        with torch.no_grad():
            out = self.forward(image)
        probs = out["patch_probs"]                     # (B, N)
        B, N = probs.shape
        H_p = W_p = int(N ** 0.5)
        return probs.reshape(B, H_p, W_p)

    # ------------------------------------------------------------------
    # Backbone freeze control
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def backbone_is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.backbone.parameters())