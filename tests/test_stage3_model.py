"""Stage 3 tests: OtolithModel, ordinal loss, freeze/unfreeze, backward pass."""
from __future__ import annotations

from typing import Dict

import pytest
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Mock DINOv2 backbone — no network calls
# ---------------------------------------------------------------------------

class _MockDinoBackbone(nn.Module):
    """Minimal DINOv2-compatible backbone for unit tests.

    - has real parameters (self.proj) so freeze/unfreeze is meaningful
    - forward() uses those params so gradients flow to backbone when unfrozen
    - forward_features() mirrors the DINOv2 dict interface
    """
    embed_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, self.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        mean_val = x.mean(dim=(1, 2, 3), keepdim=True).reshape(B, 1)
        return self.proj(mean_val)   # (B, embed_dim) — gradients flow through proj

    def forward_features(self, x: Tensor) -> Dict:
        B, C, H, W = x.shape
        num_patches = (H // 14) * (W // 14)
        cls = self.forward(x)
        patches = torch.zeros(B, num_patches, self.embed_dim, device=x.device)
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patches,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(num_age_classes: int = 10) -> "OtolithModel":
    from src.config import OtolithConfig
    from src.model import OtolithModel
    cfg = OtolithConfig()
    cfg.model.num_age_classes = num_age_classes
    cfg.model.dropout = 0.0      # deterministic for tests
    return OtolithModel(cfg, backbone=_MockDinoBackbone())


def _dummy_batch(B: int = 2, H: int = 56, num_age_classes: int = 10):
    """Return (images, age_ordinal_targets)."""
    from src.dataset import encode_age_ordinal
    images = torch.randn(B, 3, H, H)
    targets = torch.stack([encode_age_ordinal(i + 1, num_age_classes) for i in range(B)])
    return images, targets


# ---------------------------------------------------------------------------
# ordinal_loss
# ---------------------------------------------------------------------------

def test_ordinal_loss_is_scalar():
    from src.model import ordinal_loss
    logits = torch.randn(4, 9)
    targets = (torch.rand(4, 9) > 0.5).float()
    assert ordinal_loss(logits, targets).shape == ()


def test_ordinal_loss_is_positive():
    from src.model import ordinal_loss
    logits = torch.randn(4, 9)
    targets = (torch.rand(4, 9) > 0.5).float()
    assert ordinal_loss(logits, targets).item() > 0


def test_ordinal_loss_low_for_perfect_prediction():
    from src.model import ordinal_loss
    # target=1 → large positive logit; target=0 → large negative logit
    targets = torch.tensor([[1., 1., 1., 0., 0., 0., 0., 0., 0.]])
    logits = targets * 20.0 - 10.0
    assert ordinal_loss(logits, targets).item() < 0.01


def test_ordinal_loss_has_gradient():
    from src.model import ordinal_loss
    logits = torch.randn(2, 9, requires_grad=True)
    targets = (torch.rand(2, 9) > 0.5).float()
    loss = ordinal_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

def test_model_instantiates():
    assert _make_model() is not None


def test_model_embed_dim_from_backbone():
    from src.model import OtolithModel
    model = _make_model()
    head_linear = model.head[1]
    assert head_linear.in_features == _MockDinoBackbone.embed_dim


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_forward_output_shape():
    model = _make_model(num_age_classes=10)
    images, _ = _dummy_batch(B=3)
    out = model(images)
    assert out["coral_logits"].shape == (3, 9)   # K-1 = 9


def test_forward_output_dtype():
    model = _make_model()
    images, _ = _dummy_batch()
    assert model(images)["coral_logits"].dtype == torch.float32


def test_forward_batch_size_one():
    model = _make_model()
    images = torch.randn(1, 3, 56, 56)
    out = model(images)
    assert out["coral_logits"].shape[0] == 1


def test_forward_output_changes_with_different_inputs():
    model = _make_model()
    model.eval()
    x1 = torch.randn(1, 3, 56, 56)
    x2 = torch.randn(1, 3, 56, 56)
    assert not torch.allclose(model(x1)["coral_logits"], model(x2)["coral_logits"])


def test_coral_logits_are_rank_monotonic():
    """CORAL rank consistency: P(age>0) >= P(age>1) >= ... for every sample."""
    model = _make_model(num_age_classes=10)
    model.eval()
    x = torch.randn(4, 3, 56, 56)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)["coral_logits"])   # (4, 9)
    diffs = probs[:, 1:] - probs[:, :-1]                   # must be <= 0
    assert torch.all(diffs <= 1e-6), "ordinal probabilities must be non-increasing"


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------

def test_freeze_makes_backbone_params_no_grad():
    model = _make_model()
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_unfreeze_restores_backbone_params_grad():
    model = _make_model()
    model.freeze_backbone()
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_freeze_does_not_affect_head():
    model = _make_model()
    model.freeze_backbone()
    assert all(p.requires_grad for p in model.head.parameters())


def test_backbone_is_frozen_flag():
    model = _make_model()
    model.freeze_backbone()
    assert model.backbone_is_frozen() is True
    model.unfreeze_backbone()
    assert model.backbone_is_frozen() is False


# ---------------------------------------------------------------------------
# Backward pass with frozen backbone
# ---------------------------------------------------------------------------

def test_backward_frozen_backbone_no_grad_on_backbone():
    """Frozen backbone params must not receive gradients."""
    from src.model import ordinal_loss
    model = _make_model()
    model.freeze_backbone()
    images, targets = _dummy_batch()

    loss = ordinal_loss(model(images)["coral_logits"], targets)
    loss.backward()

    for p in model.backbone.parameters():
        assert p.grad is None, "Frozen backbone param must not accumulate grad"


def test_backward_frozen_backbone_head_gets_grad():
    """Head params must still receive gradients when backbone is frozen."""
    from src.model import ordinal_loss
    model = _make_model()
    model.freeze_backbone()
    images, targets = _dummy_batch()

    loss = ordinal_loss(model(images)["coral_logits"], targets)
    loss.backward()

    for p in model.head.parameters():
        assert p.grad is not None, "Head param must have grad even with frozen backbone"


# ---------------------------------------------------------------------------
# Backward pass with unfrozen backbone
# ---------------------------------------------------------------------------

def test_backward_unfrozen_backbone_all_grads():
    """All parameters (backbone + head) must receive gradients."""
    from src.model import ordinal_loss
    model = _make_model()
    images, targets = _dummy_batch()

    loss = ordinal_loss(model(images)["coral_logits"], targets)
    loss.backward()

    for p in model.backbone.parameters():
        assert p.grad is not None
    for p in model.head.parameters():
        assert p.grad is not None


# ---------------------------------------------------------------------------
# Full train step
# ---------------------------------------------------------------------------

def test_train_step_completes():
    """Forward + loss + backward + optimizer.step() must not raise."""
    import torch.optim as optim
    from src.model import ordinal_loss

    model = _make_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    images, targets = _dummy_batch()

    optimizer.zero_grad()
    loss = ordinal_loss(model(images)["coral_logits"], targets)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_loss_decreases_over_multiple_steps():
    """Loss should decrease (on average) over several gradient steps on fixed data."""
    import torch.optim as optim
    from src.model import ordinal_loss

    torch.manual_seed(0)
    model = _make_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    images, targets = _dummy_batch(B=4)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = ordinal_loss(model(images)["coral_logits"], targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "Loss should decrease over training steps"


# ---------------------------------------------------------------------------
# Patch tokens
# ---------------------------------------------------------------------------

def test_get_patch_tokens_shape():
    model = _make_model()
    images = torch.randn(2, 3, 56, 56)   # 56/14 = 4 patches per side
    patches = model.get_patch_tokens(images)
    assert patches.shape == (2, 4, 4, 64)


def test_get_patch_tokens_no_grad():
    model = _make_model()
    images = torch.randn(2, 3, 56, 56)
    patches = model.get_patch_tokens(images)
    assert not patches.requires_grad


def test_get_cls_and_patches_shapes():
    model = _make_model()
    images = torch.randn(2, 3, 56, 56)
    cls, patches = model.get_cls_and_patches(images)
    assert cls.shape == (2, 64)
    assert patches.shape == (2, 4, 4, 64)


# ---------------------------------------------------------------------------
# MIL head (weakly supervised localisation)
# ---------------------------------------------------------------------------

def _make_model_with_head(head_type: str):
    from src.config import OtolithConfig
    from src.model import OtolithModel
    cfg = OtolithConfig()
    cfg.model.num_age_classes = 10
    cfg.model.dropout = 0.0
    cfg.model.head_type = head_type
    return OtolithModel(cfg, backbone=_MockDinoBackbone())


def test_forward_dict_both_heads():
    model = _make_model_with_head("both")
    images, _ = _dummy_batch(B=2)
    out = model(images)
    assert "coral_logits" in out
    assert "patch_probs" in out
    assert "patch_count" in out
    # 56 / 14 = 4 patches per side → N = 16
    assert out["patch_probs"].shape == (2, 16)
    assert ((out["patch_probs"] >= 0) & (out["patch_probs"] <= 1)).all()
    assert out["coral_logits"].shape == (2, 9)


def test_forward_coral_only_no_patch_probs():
    model = _make_model_with_head("coral")
    images, _ = _dummy_batch()
    out = model(images)
    assert "coral_logits" in out
    assert "patch_probs" not in out
    assert "patch_count" not in out


def test_forward_mil_only_no_coral_logits():
    model = _make_model_with_head("mil")
    images, _ = _dummy_batch()
    out = model(images)
    assert "coral_logits" not in out
    assert "patch_probs" in out
    assert "patch_count" in out


def test_mil_count_equals_sum_of_probs():
    model = _make_model_with_head("mil")
    images, _ = _dummy_batch(B=2)
    out = model(images)
    assert torch.allclose(out["patch_count"], out["patch_probs"].sum(dim=1))


def test_mil_count_loss_concentrates_to_age():
    """MIL top-k loss: exactly ~age patches converge to high prob, the rest to ~0.

    Regression for F11 (the diffuse-map bug): the old sum-MSE + weak sparsity
    left every patch at ~age/N (nothing to localise). The top-k loss must break
    the symmetry so #active(prob>0.5) == age.
    """
    from src.model import mil_count_loss
    torch.manual_seed(0)
    logits = (torch.randn(1, 100) * 0.5).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.1)
    target_age = torch.tensor([7.0])
    for _ in range(400):
        opt.zero_grad()
        probs = torch.sigmoid(logits)
        loss = mil_count_loss(probs, target_age, sparsity_weight=1.0)
        loss.backward()
        opt.step()
    probs = torch.sigmoid(logits).detach()[0]
    n_active = int((probs > 0.5).sum())
    assert n_active == 7                                   # exactly age patches fire
    top = probs.sort(descending=True).values
    assert top[6] > 0.5 and top[7] < 0.2                  # 7 on, 8th is background


def test_mil_radial_spread_loss_lower_for_spread():
    """Radial-spread loss is lower for a radially-spread map than a single blob (Punkt 7)."""
    from src.model import mil_radial_spread_loss
    Hp = Wp = 8
    age = torch.tensor([6])
    nucleus = torch.tensor([[0.5, 0.5]])
    blob = torch.zeros(Hp, Wp); blob[3:5, 3:5] = 1.0
    spread = torch.zeros(Hp, Wp)
    spread[0, 4] = spread[7, 4] = spread[4, 0] = spread[4, 7] = spread[2, 2] = spread[5, 5] = 1.0
    l_blob = mil_radial_spread_loss(blob.reshape(1, -1), age, nucleus, (Hp, Wp))
    l_spread = mil_radial_spread_loss(spread.reshape(1, -1), age, nucleus, (Hp, Wp))
    assert float(l_spread) < float(l_blob)


def test_mil_radial_spread_loss_has_gradient():
    from src.model import mil_radial_spread_loss
    Hp = Wp = 8
    logits = torch.randn(2, Hp * Wp, requires_grad=True)
    age = torch.tensor([3, 8])
    nucleus = torch.tensor([[0.5, 0.5], [0.4, 0.6]])
    loss = mil_radial_spread_loss(torch.sigmoid(logits), age, nucleus, (Hp, Wp))
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_mil_count_loss_age_zero_is_empty():
    """age 0 → no patch should fire (k=0, background suppression only)."""
    from src.model import mil_count_loss
    torch.manual_seed(0)
    logits = (torch.randn(1, 100) * 0.5).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        probs = torch.sigmoid(logits)
        loss = mil_count_loss(probs, torch.tensor([0.0]), sparsity_weight=1.0)
        loss.backward()
        opt.step()
    probs = torch.sigmoid(logits).detach()[0]
    assert int((probs > 0.5).sum()) == 0


def test_get_patch_probs_raises_when_no_mil_head():
    model = _make_model_with_head("coral")
    images = torch.randn(1, 3, 56, 56)
    with pytest.raises(RuntimeError):
        model.get_patch_probs(images)


def test_get_patch_probs_shape():
    model = _make_model_with_head("mil")
    images = torch.randn(1, 3, 56, 56)
    probs = model.get_patch_probs(images)
    assert probs.shape == (1, 4, 4)
    assert ((probs >= 0) & (probs <= 1)).all()
