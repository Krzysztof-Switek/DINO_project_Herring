"""Stage 3 tests: OtolithModel, ordinal loss, freeze/unfreeze, backward pass."""
from __future__ import annotations

import contextlib
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


# ---------------------------------------------------------------------------
# Kierunek B — decoupled density-map head (13.07)
# ---------------------------------------------------------------------------

class _GradPatchBackbone(nn.Module):
    """Backbone whose patch tokens DEPEND on a real parameter, so a gradient CAN
    reach the backbone unless it is explicitly stopped. Used to prove stop-gradient."""
    embed_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, self.embed_dim)

    def forward_features(self, x: Tensor) -> Dict:
        B, C, H, W = x.shape
        npatch = (H // 14) * (W // 14)
        mean_val = x.mean(dim=(1, 2, 3), keepdim=True).reshape(B, 1)
        feat = self.proj(mean_val)                                   # (B, D) depends on proj
        patches = feat.unsqueeze(1).expand(B, npatch, self.embed_dim)
        return {"x_norm_clstoken": feat, "x_norm_patchtokens": patches}


def _make_density_model():
    from src.config import OtolithConfig
    from src.model import OtolithModel
    cfg = OtolithConfig()
    cfg.model.num_age_classes = 10
    cfg.model.dropout = 0.0
    cfg.model.head_type = "coral"          # isolate: only the density head reads patches
    cfg.model.use_density_head = True
    return OtolithModel(cfg, backbone=_GradPatchBackbone())


def test_density_head_forward_keys_and_shape():
    model = _make_density_model()
    out = model(torch.randn(2, 3, 56, 56))
    assert "density" in out and "density_count" in out
    assert out["density"].shape[0] == 2
    assert out["density_count"].shape == (2,)
    # density ∈ [0, 1]
    d = out["density"].detach()
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0


def test_density_count_loss_positive_scalar_with_grad():
    from src.model import density_count_loss
    model = _make_density_model()
    out = model(torch.randn(2, 3, 56, 56))
    loss = density_count_loss(out["density"], torch.tensor([2, 4]), conc_weight=1.0)
    assert loss.ndim == 0 and float(loss.detach()) >= 0.0
    assert loss.requires_grad


def test_density_head_starts_with_layer_norm():
    """22.07: density_head's first layer must be LayerNorm(embed_dim) — normalises
    patch-token scale before the head's own Linear (regression guard for the
    architecture change, since density_head is accessed generically elsewhere via
    .parameters()/.state_dict(), which wouldn't catch a missing/reordered layer)."""
    model = _make_density_model()
    first = model.density_head[0]
    assert isinstance(first, torch.nn.LayerNorm)
    assert first.normalized_shape == (model.density_head[1].in_features,)


def test_density_head_stop_gradient_blocks_backbone():
    """CRITICAL: density loss must NOT update the backbone (age head stays safe)."""
    from src.model import density_count_loss
    model = _make_density_model()
    out = model(torch.randn(2, 3, 56, 56))
    loss = density_count_loss(out["density"], torch.tensor([2, 4]), conc_weight=1.0)
    model.zero_grad(set_to_none=True)
    loss.backward()
    bb_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.backbone.parameters())
    dh_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.density_head.parameters())
    assert not bb_grad, "density loss leaked gradient into the backbone (stop-gradient broken)"
    assert dh_grad, "density head received no gradient"


def test_density_tv_prior_executes_and_nonnegative():
    """P2: TV spatial-coherence prior path runs and only adds a non-negative term."""
    from src.model import density_count_loss
    model = _make_density_model()
    out = model(torch.randn(2, 3, 56, 56))
    ages = torch.tensor([2, 4])
    base = density_count_loss(out["density"], ages, conc_weight=1.0, tv_weight=0.0)
    with_tv = density_count_loss(out["density"], ages, conc_weight=1.0, tv_weight=1.0)
    assert float(with_tv.detach()) >= float(base.detach())
    assert with_tv.requires_grad


def test_density_tv_prior_penalises_scattered_more_than_coherent():
    """N+1 (15.07 Tor B): the TV term is a spatial-coherence lever toward localisable rings.

    Count and concentration are permutation-invariant (sum / sorted density), so two maps
    holding the SAME values differ ONLY in the TV term — a scattered map must be penalised
    more than a coherent blob.
    """
    from src.model import density_count_loss
    ages = torch.tensor([4])                       # 4×4 grid (N=16), 4 hot cells
    coherent = torch.zeros(1, 16)
    coherent[0, [0, 1, 4, 5]] = 0.9                # 2×2 blob in the top-left corner → low TV
    scattered = torch.zeros(1, 16)
    scattered[0, [0, 2, 8, 10]] = 0.9             # same 4 values, maximally separated → high TV

    base_c = density_count_loss(coherent, ages, conc_weight=1.0, tv_weight=0.0)
    base_s = density_count_loss(scattered, ages, conc_weight=1.0, tv_weight=0.0)
    assert torch.allclose(base_c, base_s), "count+concentration must be identical for a spatial permutation"

    tv_c = float(density_count_loss(coherent, ages, conc_weight=1.0, tv_weight=1.0) - base_c)
    tv_s = float(density_count_loss(scattered, ages, conc_weight=1.0, tv_weight=1.0) - base_s)
    assert tv_s > tv_c, f"scattered TV penalty ({tv_s:.4f}) must exceed coherent blob ({tv_c:.4f})"


# ---------------------------------------------------------------------------
# Change A (05.08): attention-based density head (density_head_type="attention")
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _isolated_rng():
    """Snapshot/restore torch's global RNG state around a test.

    This module doesn't seed globally, so tests implicitly share one RNG
    stream in file order — constructing AttentionDensityHead's extra
    nn.TransformerEncoderLayer parameters consumes more random draws than the
    old plain-MLP head, which would otherwise silently shift which "random"
    weights every LATER test in this file gets (observed: it flipped
    test_density_concentricity_loss_stop_gradient_safe's density_head init
    into a degenerate all-equal-output case with a genuinely zero gradient).
    """
    state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(state)


def _make_attention_density_model():
    from src.config import OtolithConfig
    from src.model import OtolithModel
    cfg = OtolithConfig()
    cfg.model.num_age_classes = 10
    cfg.model.dropout = 0.0
    cfg.model.head_type = "coral"          # isolate: only the density head reads patches
    cfg.model.use_density_head = True
    cfg.model.density_head_type = "attention"
    cfg.model.density_attn_num_heads = 4   # divides _GradPatchBackbone.embed_dim=64
    cfg.model.density_attn_num_layers = 1
    return OtolithModel(cfg, backbone=_GradPatchBackbone())


def test_attention_density_head_forward_keys_and_shape():
    with _isolated_rng():
        model = _make_attention_density_model()
        out = model(torch.randn(2, 3, 56, 56))
        assert "density" in out and "density_count" in out
        assert out["density"].shape == (2, 16)
        assert out["density_count"].shape == (2,)
        d = out["density"].detach()
        assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0


def test_attention_density_head_stop_gradient_blocks_backbone():
    """Same CRITICAL guarantee as the plain-MLP head: attention density loss must
    NOT update the backbone — it is computed on the same STOP-GRADIENT tensor."""
    from src.model import density_count_loss
    with _isolated_rng():
        model = _make_attention_density_model()
        out = model(torch.randn(2, 3, 56, 56))
        loss = density_count_loss(out["density"], torch.tensor([2, 4]), conc_weight=1.0)
        model.zero_grad(set_to_none=True)
        loss.backward()
        bb_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.backbone.parameters())
        dh_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.density_head.parameters())
        assert not bb_grad, "attention density loss leaked gradient into the backbone"
        assert dh_grad, "attention density head received no gradient"


def test_attention_density_head_cross_patch_dependency():
    """The whole point of Change A: unlike the plain-MLP head (independent per
    patch), a patch's output CAN depend on OTHER patches' inputs — cross-patch
    attention is meant to bridge gaps between fragments rather than scoring every
    patch in total isolation. A real behavioural test, not just "it runs"."""
    from src.model import AttentionDensityHead

    with _isolated_rng():
        gen = torch.Generator().manual_seed(0)
        head = AttentionDensityHead(embed_dim=8, hidden_dim=16, dropout=0.0, num_heads=2, num_layers=1)
        head.eval()   # kill dropout stochasticity
        patches_a = torch.randn(1, 6, 8, generator=gen)
        patches_b = patches_a.clone()
        # A NON-uniform perturbation — adding the same constant to every dim of a
        # patch's vector would be exactly cancelled by the head's LayerNorm (which
        # subtracts the per-patch mean), so use per-dim random noise instead.
        patches_b[0, 0] += torch.randn(8, generator=gen) * 5.0

        with torch.no_grad():
            out_a = head(patches_a)
            out_b = head(patches_b)
        assert not torch.allclose(out_a[0, 1:], out_b[0, 1:]), \
            "attention head output for unperturbed patches should change when another patch changes"

        # Negative control: the plain-MLP head has no such dependency by construction —
        # contrasts the new behaviour against the old, unchanged default.
        mlp_head = nn.Sequential(
            nn.LayerNorm(8), nn.Linear(8, 16), nn.GELU(), nn.Dropout(p=0.0), nn.Linear(16, 1),
        )
        mlp_head.eval()
        with torch.no_grad():
            mlp_out_a = mlp_head(patches_a)
            mlp_out_b = mlp_head(patches_b)
        assert torch.allclose(mlp_out_a[0, 1:], mlp_out_b[0, 1:]), \
            "plain MLP head output for unperturbed patches must NOT change (independent per patch)"


# ---------------------------------------------------------------------------
# E9: density_concentricity_loss
# ---------------------------------------------------------------------------

def test_density_concentricity_loss_penalises_scattered_more_than_concentric():
    """The whole point of E9: density(r,θ)≈density(r) — a map that is CONSTANT
    within a radial bin (same value at every angle) must score a lower (here: zero)
    penalty than one with the same values scattered across angles within the bin."""
    from src.model import density_concentricity_loss

    # 4 radial bins, 4 patches per bin (bin = row, angle = column).
    polar_t = torch.tensor([[0.1] * 4 + [0.4] * 4 + [0.7] * 4 + [0.9] * 4])
    polar_valid = torch.ones(1, 16, dtype=torch.bool)

    concentric = torch.tensor([[0.9] * 4 + [0.2] * 4 + [0.5] * 4 + [0.1] * 4])
    scattered = torch.tensor([[0.9, 0.1, 0.9, 0.1, 0.2, 0.9, 0.2, 0.1,
                               0.5, 0.1, 0.9, 0.2, 0.1, 0.9, 0.1, 0.9]])

    l_concentric = density_concentricity_loss(concentric, polar_t, polar_valid, n_radial_bins=4)
    l_scattered = density_concentricity_loss(scattered, polar_t, polar_valid, n_radial_bins=4)
    assert float(l_concentric) == pytest.approx(0.0, abs=1e-6)
    assert float(l_scattered) > float(l_concentric)


def test_density_concentricity_loss_zero_when_no_valid_patches():
    """Segmentation-failure fallback: an all-invalid mask must not crash and must
    contribute zero loss (matches the rest of the codebase's graceful-degradation
    philosophy for failed segmentation)."""
    from src.model import density_concentricity_loss

    density = torch.rand(2, 16)
    polar_t = torch.rand(2, 16)
    polar_valid = torch.zeros(2, 16, dtype=torch.bool)
    loss = density_concentricity_loss(density, polar_t, polar_valid, n_radial_bins=4)
    assert float(loss) == 0.0


def test_density_concentricity_loss_stop_gradient_safe():
    """CRITICAL: like density_count_loss, this must never update the backbone —
    it is computed on the same STOP-GRADIENT density tensor."""
    from src.model import density_concentricity_loss

    model = _make_density_model()
    out = model(torch.randn(2, 3, 56, 56))
    polar_t = torch.rand(2, 16)
    polar_valid = torch.ones(2, 16, dtype=torch.bool)
    loss = density_concentricity_loss(out["density"], polar_t, polar_valid, n_radial_bins=4)
    model.zero_grad(set_to_none=True)
    loss.backward()
    bb_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.backbone.parameters())
    dh_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.density_head.parameters())
    assert not bb_grad, "concentricity loss leaked gradient into the backbone"
    assert dh_grad, "density head received no gradient"


# ---------------------------------------------------------------------------
# Change B (05.08): angle-windowed ("local") E9 concentricity loss
# ---------------------------------------------------------------------------

def test_density_concentricity_loss_backward_compat_when_window_none():
    """polar_theta supplied but window_deg=None must reproduce the OLD global
    behaviour byte-for-byte — the hard backward-compat guarantee."""
    from src.model import density_concentricity_loss

    polar_t = torch.tensor([[0.1] * 4 + [0.4] * 4 + [0.7] * 4 + [0.9] * 4])
    polar_valid = torch.ones(1, 16, dtype=torch.bool)
    scattered = torch.tensor([[0.9, 0.1, 0.9, 0.1, 0.2, 0.9, 0.2, 0.1,
                               0.5, 0.1, 0.9, 0.2, 0.1, 0.9, 0.1, 0.9]])
    polar_theta = torch.rand(1, 16) * 6.28 - 3.14   # arbitrary — must be IGNORED

    without_args = density_concentricity_loss(scattered, polar_t, polar_valid, n_radial_bins=4)
    with_theta_no_window = density_concentricity_loss(
        scattered, polar_t, polar_valid, n_radial_bins=4,
        polar_theta=polar_theta, window_deg=None,
    )
    assert torch.equal(without_args, with_theta_no_window)


def test_density_concentricity_loss_ignores_opposite_side_when_windowed():
    """The key differentiating behaviour: two clusters of patches on OPPOSITE
    sides of the same radial bin (~180deg apart) must NOT penalise each other
    under a small angular window, even though the OLD global loss (comparing the
    whole bin at once) would — real rings are typically visible on only part of
    the circumference, so this is what makes the loss stop punishing normal
    partial visibility as if it were noise."""
    from src.model import density_concentricity_loss
    import math as _math

    density = torch.tensor([[0.9, 0.85, 0.1, 0.15]])
    theta = torch.tensor([[0.0, _math.radians(5), _math.radians(175), _math.radians(180)]])
    polar_t = torch.full((1, 4), 0.5)          # irrelevant with n_radial_bins=1
    polar_valid = torch.ones(1, 4, dtype=torch.bool)

    global_loss = density_concentricity_loss(density, polar_t, polar_valid, n_radial_bins=1)
    windowed_loss = density_concentricity_loss(
        density, polar_t, polar_valid, n_radial_bins=1,
        polar_theta=theta, window_deg=30.0,
    )
    assert float(global_loss) == pytest.approx(0.14125, abs=1e-4)
    assert float(windowed_loss) == pytest.approx(0.000625, abs=1e-6)
    assert float(windowed_loss) < float(global_loss) * 0.05, \
        "windowed loss should be near-zero (each cluster is internally consistent) " \
        "while the global loss penalises the two clusters disagreeing with each other"


def test_density_concentricity_loss_wraparound_0_360():
    """Two patches straddling the +-180deg seam (theta=-179deg and +179deg) are
    only 2deg apart on the circle, not 358deg — a naive |theta_i - theta_j|
    difference (no wraparound) would wrongly treat them as maximally far apart
    and exclude them from each other's neighbourhood, silently zeroing the loss.
    """
    from src.model import density_concentricity_loss
    import math as _math

    density = torch.tensor([[0.9, 0.1]])
    theta = torch.tensor([[_math.radians(-179), _math.radians(179)]])
    polar_t = torch.full((1, 2), 0.5)
    polar_valid = torch.ones(1, 2, dtype=torch.bool)

    windowed_loss = density_concentricity_loss(
        density, polar_t, polar_valid, n_radial_bins=1,
        polar_theta=theta, window_deg=10.0,     # true circular distance (2deg) is well inside
    )
    assert float(windowed_loss) == pytest.approx(0.16, abs=1e-4), (
        "with correct wraparound the two patches ARE each other's neighbour "
        "(2deg apart) — a broken wraparound would silently return 0.0 instead"
    )
