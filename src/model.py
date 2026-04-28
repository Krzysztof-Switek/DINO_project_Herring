"""OtolithModel: DINOv2 backbone + ordinal regression head."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import OtolithConfig

# Known embedding dimensions for DINOv2 variants
EMBED_DIMS: Dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def load_dinov2(backbone_name: str) -> nn.Module:
    """Download and return a DINOv2 backbone from torch.hub (requires internet)."""
    return torch.hub.load(
        "facebookresearch/dinov2",
        backbone_name,
        pretrained=True,
    )


def ordinal_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """CORAL ordinal loss: independent BCE at each rank threshold.

    logits:  FloatTensor (B, K-1) — raw pre-sigmoid logits
    targets: FloatTensor (B, K-1) — binary ordinal targets (1 iff age > position)
    returns: scalar mean loss
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


class OtolithModel(nn.Module):
    """DINOv2 backbone + linear ordinal regression head.

    Forward:   image (B,3,H,W) → CLS token (B,D) → Dropout+Linear → logits (B, K-1)
    Interpret: patch tokens available via get_patch_tokens() for Stage 7 heatmaps.

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

        self.backbone = backbone if backbone is not None else load_dinov2(cfg.model.backbone)

        embed_dim: int = getattr(
            self.backbone,
            "embed_dim",
            EMBED_DIMS.get(cfg.model.backbone, 384),
        )
        num_outputs = cfg.model.num_age_classes - 1

        self.head = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(embed_dim, num_outputs),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, image: Tensor) -> Tensor:
        """Return ordinal logits (B, K-1)."""
        cls_token = self.backbone(image)   # (B, embed_dim)
        return self.head(cls_token)        # (B, K-1)

    # ------------------------------------------------------------------
    # Patch token access (for interpretation, Stage 7)
    # ------------------------------------------------------------------

    def get_patch_tokens(self, image: Tensor) -> Tensor:
        """Return patch tokens as spatial grid (B, H_p, W_p, embed_dim).

        No gradient tracking — intended for inference / heatmap generation.
        Assumes square input (H == W); num_patches = (H / patch_size)^2.
        """
        with torch.no_grad():
            feats = self.backbone.forward_features(image)
        patch_tokens = feats["x_norm_patchtokens"]   # (B, N, D)
        B, N, D = patch_tokens.shape
        H_p = W_p = int(N ** 0.5)
        return patch_tokens.reshape(B, H_p, W_p, D)

    def get_cls_and_patches(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (cls_token, patch_grid) without gradient tracking.

        Returns:
            cls:     (B, embed_dim)
            patches: (B, H_p, W_p, embed_dim)
        """
        with torch.no_grad():
            feats = self.backbone.forward_features(image)
        cls = feats["x_norm_clstoken"]               # (B, D)
        patch_tokens = feats["x_norm_patchtokens"]   # (B, N, D)
        B, N, D = patch_tokens.shape
        H_p = W_p = int(N ** 0.5)
        return cls, patch_tokens.reshape(B, H_p, W_p, D)

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
