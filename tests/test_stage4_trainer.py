"""Stage 4 tests: Trainer — train loop, validation, checkpointing, logging."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.dataset import encode_age_ordinal


# ---------------------------------------------------------------------------
# Mock backbone (same interface as DINOv2, no network calls)
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
        num_patches = (H // 14) * (W // 14)
        cls = self.forward(x)
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": torch.zeros(B, num_patches, self.embed_dim, device=x.device),
        }


# ---------------------------------------------------------------------------
# Synthetic in-memory dataset (no filesystem needed)
# ---------------------------------------------------------------------------

class _SyntheticDataset(Dataset):
    def __init__(self, n: int = 8, num_age_classes: int = 10, image_size: int = 56,
                with_polar: bool = False, with_zegar: bool = False, zegar_every: int = 2,
                with_strip: bool = False, multi_wycinek_k: int = 0,
                with_strip_valid_mask: bool = False, with_wedge: bool = False,
                with_wedge_bands: bool = False):
        self.n = n
        self.num_age_classes = num_age_classes
        self.image_size = image_size
        self.with_polar = with_polar
        self.with_zegar = with_zegar
        self.zegar_every = zegar_every   # every Nth sample carries a real target
        # 02.09, dendrochronology-strip experiment: mirrors OtolithDataset's
        # dual_branch_density="image_strip" key (non-square shape, on purpose — the
        # whole point is it's NOT the same grid as "image").
        self.with_strip = with_strip
        # 03.09, multi-wycinek experiment: >0 mirrors OtolithDataset's
        # multi_wycinek_k>1 path — image_strip becomes (K,3,14,28) instead of
        # (3,14,28). Takes precedence over with_strip when both are set.
        self.multi_wycinek_k = multi_wycinek_k
        # 08.09, background-activation penalty: mirrors OtolithDataset's
        # strip_mask_background_loss="strip_valid_mask" key. Fixed, non-trivial
        # pattern (one valid, one masked-out patch of the 2) so tests can tell a real
        # mask from an all-ones stand-in.
        self.with_strip_valid_mask = with_strip_valid_mask
        # 09.09, polar-wedge experiment: mirrors OtolithDataset's dual_branch_wedge
        # "image_wedge"/"wedge_polar_t"/"wedge_polar_theta"/"wedge_polar_valid" keys.
        # Non-square shape (h_p=2, w_p=3) and a non-trivial wedge_polar_valid pattern
        # (one masked-out patch of the 6) so tests can distinguish real values from an
        # all-ones/all-zeros stand-in.
        self.with_wedge = with_wedge
        # 09.09 follow-up, angular-resolution bands: mirrors OtolithDataset's
        # image_wedge_band{i}/wedge_band{i}_polar_{t,theta,valid} keys — 2 bands, deliberately
        # DIFFERENT shapes per band (band 0 smaller than band 1, like the real design where later
        # bands are wider), each with its own non-trivial (not all-ones) tissue-validity pattern.
        self.with_wedge_bands = with_wedge_bands

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict:
        age = (idx % (self.num_age_classes - 1)) + 1
        item: Dict = {
            "image": torch.randn(3, self.image_size, self.image_size),
            "age_ordinal": encode_age_ordinal(age, self.num_age_classes),
            "age": torch.tensor(age, dtype=torch.long),
            "image_id": f"img_{idx:03d}.png",
        }
        if self.with_polar:
            h_p = w_p = self.image_size // 14
            item["polar_grid"] = torch.rand(h_p, w_p)
            item["polar_valid"] = torch.ones(h_p, w_p, dtype=torch.bool)
        if self.with_zegar:
            h_p = w_p = self.image_size // 14
            # zegar_every<=0 is the sentinel for "never" — idx % N == 0 is always
            # True at idx=0 for any N>0, so that can't itself express "no sample".
            has_target = self.zegar_every > 0 and (idx % self.zegar_every == 0)
            heat = torch.zeros(h_p, w_p)
            if has_target:
                heat[0, 0] = 1.0
            item["zegar_heatmap"] = heat
            item["has_zegar_target"] = torch.tensor(has_target, dtype=torch.bool)
        if self.multi_wycinek_k > 0:
            item["image_strip"] = torch.randn(self.multi_wycinek_k, 3, 14, 28)
        elif self.with_strip:
            item["image_strip"] = torch.randn(3, 14, 28)   # deliberately non-square
        if self.with_strip_valid_mask:
            item["strip_valid_mask"] = torch.tensor([[1.0, 0.0]])   # (h_p=1, w_p=2)
        if self.with_wedge:
            item["image_wedge"] = torch.randn(3, 28, 42)   # deliberately non-square
            item["wedge_polar_t"] = torch.rand(2, 3)
            item["wedge_polar_theta"] = torch.rand(2, 3)
            item["wedge_polar_valid"] = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        if self.with_wedge_bands:
            item["image_wedge_band0"] = torch.randn(3, 14, 28)   # (h_p=1, w_p=2)
            item["wedge_band0_polar_t"] = torch.rand(1, 2)
            item["wedge_band0_polar_theta"] = torch.rand(1, 2)
            item["wedge_band0_polar_valid"] = torch.tensor([[1.0, 0.0]])
            item["image_wedge_band1"] = torch.randn(3, 28, 42)   # (h_p=2, w_p=3)
            item["wedge_band1_polar_t"] = torch.rand(2, 3)
            item["wedge_band1_polar_theta"] = torch.rand(2, 3)
            item["wedge_band1_polar_valid"] = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path, epochs: int = 2, freeze_epochs: int = 0,
              scheduler: str = "none"):
    from src.config import OtolithConfig
    cfg = OtolithConfig()
    cfg.model.num_age_classes = 10
    cfg.model.dropout = 0.0
    cfg.training.epochs = epochs
    cfg.training.lr = 1e-3
    cfg.training.weight_decay = 0.0
    cfg.training.freeze_backbone_epochs = freeze_epochs
    cfg.training.scheduler = scheduler
    cfg.training.device = "cpu"
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.training.log_dir = str(tmp_path / "logs")
    return cfg


def _make_loader(n: int = 8, batch_size: int = 4, with_polar: bool = False,
                 with_zegar: bool = False, with_strip: bool = False,
                 multi_wycinek_k: int = 0, with_strip_valid_mask: bool = False,
                 with_wedge: bool = False, with_wedge_bands: bool = False) -> DataLoader:
    ds = _SyntheticDataset(n=n, num_age_classes=10, image_size=56, with_polar=with_polar,
                           with_zegar=with_zegar, with_strip=with_strip,
                           multi_wycinek_k=multi_wycinek_k,
                           with_strip_valid_mask=with_strip_valid_mask, with_wedge=with_wedge,
                           with_wedge_bands=with_wedge_bands)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _make_model(cfg):
    from src.model import OtolithModel
    return OtolithModel(cfg, backbone=_MockDinoBackbone())


def _make_trainer(tmp_path: Path, epochs: int = 2, freeze_epochs: int = 0,
                  with_val: bool = True, scheduler: str = "none"):
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path, epochs=epochs, freeze_epochs=freeze_epochs,
                    scheduler=scheduler)
    model = _make_model(cfg)
    train_loader = _make_loader(n=8)
    val_loader = _make_loader(n=4) if with_val else None
    return Trainer(cfg, model, train_loader, val_loader)


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------

def test_resolve_device_cpu():
    from src.trainer import resolve_device
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_device():
    from src.trainer import resolve_device
    device = resolve_device("auto")
    assert isinstance(device, torch.device)


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

def test_train_one_epoch_returns_float(tmp_path):
    trainer = _make_trainer(tmp_path)
    loss = trainer.train_one_epoch()
    assert isinstance(loss, float)
    assert loss > 0
    assert not math.isnan(loss)


def test_train_one_epoch_updates_weights(tmp_path):
    trainer = _make_trainer(tmp_path)
    head_w_before = trainer.model.head[1].weight.data.clone()
    trainer.train_one_epoch()
    assert not torch.allclose(head_w_before, trainer.model.head[1].weight.data)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_validate_returns_two_floats(tmp_path):
    trainer = _make_trainer(tmp_path)
    val_loss, val_mae = trainer.validate()
    assert isinstance(val_loss, float)
    assert isinstance(val_mae, float)


def test_validate_loss_positive(tmp_path):
    trainer = _make_trainer(tmp_path)
    val_loss, _ = trainer.validate()
    assert val_loss > 0


def test_validate_mae_nonneg(tmp_path):
    trainer = _make_trainer(tmp_path)
    _, val_mae = trainer.validate()
    assert val_mae >= 0.0


def test_validate_no_val_loader_returns_nan(tmp_path):
    trainer = _make_trainer(tmp_path, with_val=False)
    val_loss, val_mae = trainer.validate()
    assert math.isnan(val_loss)
    assert math.isnan(val_mae)


def test_validate_does_not_update_weights(tmp_path):
    trainer = _make_trainer(tmp_path)
    w_before = trainer.model.head[1].weight.data.clone()
    trainer.validate()
    assert torch.allclose(w_before, trainer.model.head[1].weight.data)


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

def test_save_checkpoint_creates_file(tmp_path):
    trainer = _make_trainer(tmp_path)
    path = trainer.save_checkpoint(epoch=1, val_loss=0.5)
    assert path.exists()


def test_save_checkpoint_filename_contains_epoch_and_loss(tmp_path):
    trainer = _make_trainer(tmp_path)
    path = trainer.save_checkpoint(epoch=7, val_loss=0.1234)
    assert "007" in path.name
    assert "0.1234" in path.name


def test_load_checkpoint_round_trip(tmp_path):
    """Save weights → corrupt weights → reload → output must match original."""
    trainer = _make_trainer(tmp_path)
    images = torch.randn(2, 3, 56, 56)

    trainer.model.eval()
    with torch.no_grad():
        out_before = trainer.model(images)["coral_logits"].clone()

    ckpt_path = trainer.save_checkpoint(epoch=1, val_loss=0.5)

    # Corrupt all parameters
    for p in trainer.model.parameters():
        p.data.fill_(99.0)

    trainer.load_checkpoint(ckpt_path)

    trainer.model.eval()
    with torch.no_grad():
        out_after = trainer.model(images)["coral_logits"]

    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_load_checkpoint_returns_epoch(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.save_checkpoint(epoch=5, val_loss=0.3)
    ckpt_files = list((tmp_path / "checkpoints").glob("*.pt"))
    epoch = trainer.load_checkpoint(ckpt_files[0])
    assert epoch == 5


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def test_fit_creates_checkpoint_per_epoch(tmp_path):
    trainer = _make_trainer(tmp_path, epochs=3)
    trainer.cfg.training.keep_only_best = False   # test per-epoch mode explicitly
    trainer.fit()
    ckpts = list((tmp_path / "checkpoints").glob("checkpoint_epoch*.pt"))
    assert len(ckpts) == 3


def test_keep_only_best_prunes_epoch_checkpoints(tmp_path):
    """keep_only_best (default) leaves only best.pt — per-epoch checkpoints pruned."""
    trainer = _make_trainer(tmp_path, epochs=3)   # keep_only_best defaults to True
    trainer.fit()
    assert list(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt")) == []
    assert (trainer.checkpoint_dir / "best.pt").exists()


def test_fit_writes_log_file(tmp_path):
    trainer = _make_trainer(tmp_path, epochs=2)
    trainer.fit()
    log_file = tmp_path / "logs" / "train.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "epoch=" in content
    assert "Training complete" in content


def test_fit_log_contains_all_epochs(tmp_path):
    trainer = _make_trainer(tmp_path, epochs=3)
    trainer.fit()
    content = (tmp_path / "logs" / "train.log").read_text(encoding="utf-8")
    for ep in [1, 2, 3]:
        assert f"epoch={ep:3d}" in content


def test_fit_log_contains_lr(tmp_path):
    """Each epoch line must record the learning rate (for the report LR curve)."""
    trainer = _make_trainer(tmp_path, epochs=1)
    trainer.fit()
    content = (tmp_path / "logs" / "train.log").read_text(encoding="utf-8")
    assert "lr=" in content


def test_fit_with_cosine_scheduler(tmp_path):
    """fit() must not crash with cosine scheduler."""
    trainer = _make_trainer(tmp_path, epochs=2, scheduler="cosine")
    trainer.fit()


# ---------------------------------------------------------------------------
# Freeze / unfreeze in fit
# ---------------------------------------------------------------------------

def test_fit_backbone_unfrozen_after_warmup(tmp_path):
    """After fit() with freeze_epochs < total epochs, backbone must be unfrozen."""
    trainer = _make_trainer(tmp_path, epochs=3, freeze_epochs=2)
    trainer.fit()
    assert not trainer.model.backbone_is_frozen()


def test_fit_backbone_frozen_whole_training_if_epochs_equal_freeze(tmp_path):
    """If freeze_epochs == epochs, backbone stays frozen throughout."""
    trainer = _make_trainer(tmp_path, epochs=2, freeze_epochs=2)
    trainer.fit()
    assert trainer.model.backbone_is_frozen()


def test_fit_frozen_backbone_head_still_trains(tmp_path):
    """Even with frozen backbone, head weights must change after fit()."""
    trainer = _make_trainer(tmp_path, epochs=2, freeze_epochs=2)
    w_before = trainer.model.head[1].weight.data.clone()
    trainer.fit()
    assert not torch.allclose(w_before, trainer.model.head[1].weight.data)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class _ConstantValTrainer:
    """Trainer subclass whose validate() always returns (val_loss=1.0, val_mae=5.0).

    After epoch 1 the metric never improves, so early stopping fires after
    patience epochs.
    """
    pass  # defined per-test via local subclass to avoid import order issues


def test_early_stopping_triggers(tmp_path):
    """Training must stop before max_epochs when val_mae never improves."""
    from src.trainer import Trainer

    class ConstantValTrainer(Trainer):
        def validate(self):
            return 1.0, 5.0

    cfg = _make_cfg(tmp_path, epochs=10)
    cfg.training.early_stopping_patience = 2
    cfg.training.early_stopping_metric = "val_mae"
    cfg.training.early_stopping_min_delta = 0.001
    cfg.training.keep_only_best = False   # count per-epoch checkpoints as an epoch proxy

    model = _make_model(cfg)
    trainer = ConstantValTrainer(cfg, model, _make_loader(), _make_loader())
    trainer.fit()

    ckpt_files = list(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
    # epoch 1 improves inf→5.0; epochs 2+3 don't improve → stop after epoch 3
    assert len(ckpt_files) == 3


def test_early_stopping_saves_best_pt(tmp_path):
    """best.pt must exist after fit() completes."""
    from src.trainer import Trainer

    class ConstantValTrainer(Trainer):
        def validate(self):
            return 1.0, 5.0

    cfg = _make_cfg(tmp_path, epochs=5)
    cfg.training.early_stopping_patience = 2
    cfg.training.early_stopping_metric = "val_mae"
    cfg.training.early_stopping_min_delta = 0.001

    model = _make_model(cfg)
    trainer = ConstantValTrainer(cfg, model, _make_loader(), _make_loader())
    trainer.fit()

    assert (trainer.checkpoint_dir / "best.pt").exists()


def test_density_gate_delays_stop_until_density_alive(tmp_path):
    """16.07: with use_density_head, early-stopping/best.pt WAIT until density is alive.

    Constant val_mae would normally stop at e3 (patience 2). But while density_active=0 the
    gate holds (no patience, no stop); density wakes at e5, then patience(2) → stop ~e7.

    density_gate_max_wait_epochs is set explicitly large here to isolate THIS behaviour
    (waiting for a real, eventual density wake-up) from the separate 08.08 timeout safety
    valve (tested below in test_density_gate_gives_up_after_max_wait_epochs) — with a flat
    val_mae from epoch 1, the age metric "plateaus" immediately by construction, so the
    default timeout would otherwise fire (correctly, by design) before density's scripted
    wake-up at e5.
    """
    from src.trainer import Trainer

    class DeadThenAliveTrainer(Trainer):
        def validate(self):
            self._vc = getattr(self, "_vc", 0) + 1
            self.last_val_metrics = {"density_active": (0.0 if self._vc <= 4 else 2.0)}
            return 1.0, 5.0                       # constant val_mae

    cfg = _make_cfg(tmp_path, epochs=10)
    cfg.model.use_density_head = True
    cfg.training.early_stopping_patience = 2
    cfg.training.early_stopping_min_delta = 0.001
    cfg.training.min_epochs = 0
    cfg.training.min_density_active = 1.0
    cfg.training.density_gate_max_wait_epochs = 100   # disable the timeout for this test
    cfg.training.keep_only_best = False           # count per-epoch checkpoints as an epoch proxy

    model = _make_model(cfg)
    trainer = DeadThenAliveTrainer(cfg, model, _make_loader(), _make_loader())
    trainer.fit()

    ckpt_files = list(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
    assert len(ckpt_files) >= 6, f"gate should delay stop past naive e3, ran {len(ckpt_files)}"
    assert (trainer.checkpoint_dir / "best.pt").exists()


def test_density_gate_noop_without_density_head(tmp_path):
    """use_density_head=False → gate inert; identical to old early-stopping (stop at e3)."""
    from src.trainer import Trainer

    class ConstantValTrainer(Trainer):
        def validate(self):
            self.last_val_metrics = {}
            return 1.0, 5.0

    cfg = _make_cfg(tmp_path, epochs=10)
    cfg.model.use_density_head = False
    cfg.training.early_stopping_patience = 2
    cfg.training.early_stopping_min_delta = 0.001
    cfg.training.min_epochs = 0
    cfg.training.min_density_active = 1.0         # ignored — no density head
    cfg.training.keep_only_best = False

    model = _make_model(cfg)
    trainer = ConstantValTrainer(cfg, model, _make_loader(), _make_loader())
    trainer.fit()

    ckpt_files = list(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
    assert len(ckpt_files) == 3


def test_two_checkpoints_age_best_survives_gate_reset(tmp_path):
    """best_age.pt must keep the best RAW age epoch even after the density-maturity
    gate resets best_metric and best.pt moves to a worse (but density-mature) epoch —
    the two-checkpoint fix (20.07_trening_summary.md §5 pkt 2)."""
    from src.trainer import Trainer
    import torch as _torch

    class ScriptedTrainer(Trainer):
        def validate(self):
            self._vc = getattr(self, "_vc", 0) + 1
            # epochs 1-2: density dead, val_mae=2.0 (age-best).
            # epochs 3-4: density wakes (gate opens, best_metric resets), val_mae
            # degrades to 3.0 — worse than the pre-gate age-best.
            density_active = 0.0 if self._vc < 3 else 2.0
            val_mae = 2.0 if self._vc < 3 else 3.0
            self.last_val_metrics = {"density_active": density_active}
            return 1.0, val_mae

    cfg = _make_cfg(tmp_path, epochs=4)
    cfg.model.use_density_head = True
    cfg.training.early_stopping_patience = 0   # run all epochs
    cfg.training.min_epochs = 0
    cfg.training.min_density_active = 1.0

    model = _make_model(cfg)
    trainer = ScriptedTrainer(cfg, model, _make_loader(), _make_loader())
    trainer.fit()

    best_path = trainer.checkpoint_dir / "best.pt"
    best_age_path = trainer.checkpoint_dir / "best_age.pt"
    assert best_path.exists()
    assert best_age_path.exists()

    age_ckpt = _torch.load(best_age_path, map_location="cpu", weights_only=False)
    density_ckpt = _torch.load(best_path, map_location="cpu", weights_only=False)
    assert age_ckpt["epoch"] in (1, 2), "best_age.pt must come from the pre-gate age-best epoch"
    assert density_ckpt["epoch"] in (3, 4), "best.pt must come from a density-mature epoch"


def test_density_gate_gives_up_after_max_wait_epochs(tmp_path):
    """08.08 bug fix: if density NEVER matures, the gate must give up and fall back to
    normal early stopping instead of silently disabling it for the whole run.

    Real bug found on outputs/06.08_attention_first: density_active=0.0000 for all 50
    epochs → gate never opened → patience_counter (only incremented while gate_open)
    never moved → zero "Early stopping" log line → the run silently burned the full
    epoch ceiling (~26-28 wasted epochs) even though val_mae had long plateaued.

    Timeout is keyed off the ungated AGE metric's OWN patience, not a fixed epoch
    number (see config.py's density_gate_max_wait_epochs docstring for why). Here
    val_mae is flat from epoch 1, so best_age_metric only ever "improves" trivially at
    e1 (inf→5.0); with default max_wait=patience=2, age_patience_counter reaches 2 at
    e3 (min_epochs=3 is already satisfied by then) → gate times out and opens (with a
    reset) at e3, then 2 more non-improving epochs (e4, e5) → stop @e5.
    """
    from src.trainer import Trainer

    class ForeverDeadTrainer(Trainer):
        def validate(self):
            self.last_val_metrics = {"density_active": 0.0}   # never crosses min_density_active
            return 1.0, 5.0                                   # constant val_mae — never "improves"

    cfg = _make_cfg(tmp_path, epochs=20)
    cfg.model.use_density_head = True
    cfg.training.early_stopping_patience = 2
    cfg.training.early_stopping_min_delta = 0.001
    cfg.training.min_epochs = 3
    cfg.training.min_density_active = 1.0
    cfg.training.keep_only_best = False

    model = _make_model(cfg)
    trainer = ForeverDeadTrainer(cfg, model, _make_loader(), _make_loader())
    trainer.fit()

    ckpt_files = list(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
    assert len(ckpt_files) == 5, (
        f"gate should give up at e3 and stop by e5 (patience=2 after timeout), "
        f"ran {len(ckpt_files)} epochs instead"
    )
    assert (trainer.checkpoint_dir / "best.pt").exists()

    log_text = trainer.log_path.read_text(encoding="utf-8")
    assert "NIE dojrzała" in log_text, "timeout must be logged distinctly from real maturity"
    assert "Early stopping" in log_text


def test_density_gate_max_wait_epochs_override(tmp_path):
    """Explicit density_gate_max_wait_epochs overrides the default (= early_stopping_patience)
    used as the age-metric-plateau deadline for giving up on a density head that never matures."""
    from src.trainer import Trainer

    class ForeverDeadTrainer(Trainer):
        def validate(self):
            self.last_val_metrics = {"density_active": 0.0}
            return 1.0, 5.0

    cfg = _make_cfg(tmp_path, epochs=20)
    cfg.model.use_density_head = True
    cfg.training.early_stopping_patience = 2
    cfg.training.early_stopping_min_delta = 0.001
    cfg.training.min_epochs = 0
    cfg.training.min_density_active = 1.0
    cfg.training.density_gate_max_wait_epochs = 3   # wait longer than the default (2) before giving up
    cfg.training.keep_only_best = False

    model = _make_model(cfg)
    trainer = ForeverDeadTrainer(cfg, model, _make_loader(), _make_loader())
    trainer.fit()

    # age_patience_counter reaches 3 at e4 → gate times out/opens (reset) at e4, then
    # 2 more non-improving epochs (e5, e6) → stop @e6.
    ckpt_files = list(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
    assert len(ckpt_files) == 6, f"custom max_wait=3 should stop by e6, ran {len(ckpt_files)} epochs"


def test_no_best_age_checkpoint_without_density_head(tmp_path):
    """best_age.pt is only meaningful (and only created) when a density head exists —
    without one, best.pt is already the age-best checkpoint."""
    from src.trainer import Trainer

    class ConstantValTrainer(Trainer):
        def validate(self):
            self.last_val_metrics = {}
            return 1.0, 5.0

    cfg = _make_cfg(tmp_path, epochs=2)
    cfg.model.use_density_head = False
    model = _make_model(cfg)
    trainer = ConstantValTrainer(cfg, model, _make_loader(), _make_loader())
    trainer.fit()

    assert (trainer.checkpoint_dir / "best.pt").exists()
    assert not (trainer.checkpoint_dir / "best_age.pt").exists()


def test_trainer_supports_mil_head_type(tmp_path):
    """Trener musi działać dla head_type='mil' (bez CORAL)."""
    cfg = _make_cfg(tmp_path, epochs=1)
    cfg.model.head_type = "mil"
    model = _make_model(cfg)
    train_loader = _make_loader(n=8)
    val_loader   = _make_loader(n=4)

    from src.trainer import Trainer
    trainer = Trainer(cfg, model, train_loader, val_loader)
    trainer.fit()   # nie powinno rzucić wyjątku
    assert (trainer.checkpoint_dir / "best.pt").exists()


def test_trainer_supports_both_head_type(tmp_path):
    """Trener z head_type='both' liczy combined loss bez błędu."""
    cfg = _make_cfg(tmp_path, epochs=1)
    cfg.model.head_type = "both"
    model = _make_model(cfg)
    train_loader = _make_loader(n=8)
    val_loader   = _make_loader(n=4)

    from src.trainer import Trainer
    trainer = Trainer(cfg, model, train_loader, val_loader)
    trainer.fit()
    assert (trainer.checkpoint_dir / "best.pt").exists()


def test_early_stopping_disabled(tmp_path):
    """patience=0 must run all epochs without stopping."""
    from src.trainer import Trainer

    class ConstantValTrainer(Trainer):
        def validate(self):
            return 1.0, 5.0

    cfg = _make_cfg(tmp_path, epochs=3)
    cfg.training.early_stopping_patience = 0
    cfg.training.keep_only_best = False   # count per-epoch checkpoints as an epoch proxy

    model = _make_model(cfg)
    trainer = ConstantValTrainer(cfg, model, _make_loader(), _make_loader())
    trainer.fit()

    ckpt_files = list(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
    assert len(ckpt_files) == 3


def test_optimizer_has_discriminative_lr(tmp_path):
    """Backbone must sit in its own param group with a lower LR than the heads."""
    trainer = _make_trainer(tmp_path, epochs=1)
    groups = trainer.optimizer.param_groups
    assert len(groups) == 2
    lrs = sorted(g["lr"] for g in groups)
    assert lrs[0] < lrs[1]   # backbone_lr (lr * backbone_lr_mult) < head_lr


def test_optimizer_gives_density_head_its_own_lr_group(tmp_path):
    """22.07: with use_density_head=True, density_head params get a 3rd param group
    at lr * density_lr_mult — distinct from both backbone_lr and the CORAL/MIL head
    lr. Without a density head, behaviour is unchanged (2 groups, other test)."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path, epochs=1)
    cfg.model.use_density_head = True
    cfg.training.density_lr_mult = 2.0
    model = _make_model(cfg)
    trainer = Trainer(cfg, model, _make_loader(), _make_loader())
    groups = trainer.optimizer.param_groups
    assert len(groups) == 3
    lrs = sorted(g["lr"] for g in groups)
    expected = sorted([
        cfg.training.lr,
        cfg.training.lr * cfg.training.backbone_lr_mult,
        cfg.training.lr * cfg.training.density_lr_mult,
    ])
    assert lrs == pytest.approx(expected)


def test_loss_parts_includes_density_concentricity_when_enabled(tmp_path):
    """E9: with density_concentricity_weight>0 and a batch carrying polar_grid/
    polar_valid, _loss_parts must add a 'density_concentricity' component and fold
    it into 'total' — and train_one_epoch/validate must run end-to-end without
    the caller having to do anything special (loader-provided keys flow through)."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path, epochs=1)
    cfg.model.use_density_head = True
    cfg.model.density_concentricity_weight = 0.5
    cfg.model.density_concentricity_bins = 4
    model = _make_model(cfg)
    train_loader = _make_loader(n=8, with_polar=True)
    val_loader = _make_loader(n=4, with_polar=True)
    trainer = Trainer(cfg, model, train_loader, val_loader)

    # train_one_epoch/validate must not raise, and validate()'s diagnostics must
    # surface the new component.
    trainer.train_one_epoch()
    trainer.validate()
    assert "density_conc_loss" in trainer.last_val_metrics


def test_loss_parts_omits_density_concentricity_when_weight_zero(tmp_path):
    """Default weight (0.0): even with polar_grid present in the batch, the
    concentricity term must not appear — matches every other density_* weight's
    "0 = off" convention in this codebase."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path, epochs=1)
    cfg.model.use_density_head = True
    model = _make_model(cfg)
    trainer = Trainer(cfg, model, _make_loader(with_polar=True), _make_loader(with_polar=True))
    trainer.validate()
    assert "density_conc_loss" not in trainer.last_val_metrics


def test_loss_parts_includes_zegar_position_when_enabled_and_present(tmp_path):
    """26.08 Opcja A: with zegar_position_weight>0 and a batch where SOME samples
    carry a real zegar_heatmap target, _loss_parts must add a 'zegar_position'
    component and train_one_epoch/validate must run end-to-end without the caller
    doing anything special (loader-provided keys flow through, same as polar_grid)."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path, epochs=1)
    cfg.model.use_density_head = True
    cfg.model.zegar_position_weight = 0.5
    model = _make_model(cfg)
    train_loader = _make_loader(n=8, with_zegar=True)
    val_loader = _make_loader(n=4, with_zegar=True)
    trainer = Trainer(cfg, model, train_loader, val_loader)

    trainer.train_one_epoch()
    trainer.validate()
    assert "zegar_position_loss" in trainer.last_val_metrics


def test_loss_parts_omits_zegar_position_when_weight_zero(tmp_path):
    """Default weight (0.0): even with zegar_heatmap/has_zegar_target present in the
    batch, the term must not appear — matches every other density_*/zegar_* weight's
    "0 = off" convention in this codebase."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path, epochs=1)
    cfg.model.use_density_head = True
    model = _make_model(cfg)
    trainer = Trainer(cfg, model, _make_loader(with_zegar=True), _make_loader(with_zegar=True))
    trainer.validate()
    assert "zegar_position_loss" not in trainer.last_val_metrics


def test_loss_parts_omits_zegar_position_when_no_sample_has_target(tmp_path):
    """weight>0 but every has_zegar_target in the batch is False (the overwhelming
    common case — only ~0.2% of real epochs ever see a ZEGAR sample) — the term must
    still not appear, not silently compute a loss against an all-zero target."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path, epochs=1)
    cfg.model.use_density_head = True
    cfg.model.zegar_position_weight = 0.5
    model = _make_model(cfg)
    # zegar_every=0 is the sentinel for "never" -> has_zegar_target False for every sample.
    ds = _SyntheticDataset(n=4, num_age_classes=10, image_size=56, with_zegar=True, zegar_every=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    trainer = Trainer(cfg, model, loader, loader)
    trainer.validate()
    assert "zegar_position_loss" not in trainer.last_val_metrics


def test_zegar_position_weight_does_not_change_age_predictions(tmp_path):
    """Stop-gradient isolation, end-to-end through the trainer (not just the loss
    function in isolation, test_stage3_model.py's version): two trainers, identical
    seed/data/model init, differing ONLY in zegar_position_weight, must produce
    BIT-IDENTICAL CORAL logits after one training step — the established pattern this
    project uses to verify a density-head-only change can't touch the age head
    (e.g. the wide_window/smoothness checkpoint comparison, 19.08)."""
    from src.trainer import Trainer

    def _run(weight: float):
        torch.manual_seed(0)
        cfg = _make_cfg(tmp_path / f"w{weight}", epochs=1)
        cfg.model.use_density_head = True
        cfg.model.zegar_position_weight = weight
        model = _make_model(cfg)
        loader = _make_loader(n=8, with_zegar=True)
        trainer = Trainer(cfg, model, loader, loader)
        trainer.train_one_epoch()
        with torch.no_grad():
            probe = torch.randn(2, 3, 56, 56)
        return trainer.model(probe)["coral_logits"]

    logits_off = _run(0.0)
    logits_on = _run(0.5)
    assert torch.allclose(logits_off, logits_on, atol=1e-6), \
        "zegar_position_weight changed age (CORAL) predictions — stop-gradient isolation broken"


# ---------------------------------------------------------------------------
# Dendrochronology-strip experiment (02.09) — image_strip -> density_image wiring
# ---------------------------------------------------------------------------

def test_train_one_epoch_routes_image_strip_to_density_image(tmp_path):
    """batch["image_strip"] (only present when dual_branch_density=True) must reach
    forward() as density_image during training."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    model = _make_model(cfg)

    captured = []
    orig_forward = model.forward
    def _spy_forward(*args, **kwargs):
        captured.append(kwargs.get("density_image"))
        return orig_forward(*args, **kwargs)
    model.forward = _spy_forward

    loader = _make_loader(n=4, with_strip=True)
    trainer = Trainer(cfg, model, loader, None)
    trainer.train_one_epoch()

    assert len(captured) > 0
    assert all(t is not None for t in captured)
    assert all(t.shape[-2:] == (14, 28) for t in captured)


def test_validate_routes_image_strip_to_density_image(tmp_path):
    """Same wiring, validate() path."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    model = _make_model(cfg)

    captured = []
    orig_forward = model.forward
    def _spy_forward(*args, **kwargs):
        captured.append(kwargs.get("density_image"))
        return orig_forward(*args, **kwargs)
    model.forward = _spy_forward

    loader = _make_loader(n=4, with_strip=True)
    trainer = Trainer(cfg, model, loader, loader)
    trainer.validate()

    assert len(captured) > 0
    assert all(t is not None for t in captured)
    assert all(t.shape[-2:] == (14, 28) for t in captured)


def test_train_one_epoch_without_image_strip_passes_none(tmp_path):
    """Regression pin: a batch with no "image_strip" key (dual_branch_density=False,
    every existing config) must call forward() with density_image=None explicitly —
    confirms .get() absence is handled, not silently mis-keyed."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)

    captured = []
    orig_forward = model.forward
    def _spy_forward(*args, **kwargs):
        captured.append(kwargs.get("density_image"))
        return orig_forward(*args, **kwargs)
    model.forward = _spy_forward

    loader = _make_loader(n=4)   # no with_strip
    trainer = Trainer(cfg, model, loader, None)
    trainer.train_one_epoch()

    assert len(captured) > 0
    assert all(t is None for t in captured)


# ---------------------------------------------------------------------------
# Background-activation penalty (08.09) — strip_valid_mask -> density_count_loss wiring
# ---------------------------------------------------------------------------

def test_train_one_epoch_routes_strip_valid_mask_to_density_count_loss(tmp_path):
    """batch["strip_valid_mask"] (only present when
    cfg.data.strip_mask_background_loss=True) must reach density_count_loss as its
    valid_mask kwarg during training, flattened (B,Hp,Wp)->(B,N) like polar_grid."""
    import src.trainer as trainer_module
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    model = _make_model(cfg)

    captured = []
    orig = trainer_module.density_count_loss
    def _spy(*args, **kwargs):
        captured.append(kwargs.get("valid_mask"))
        return orig(*args, **kwargs)
    trainer_module.density_count_loss = _spy
    try:
        loader = _make_loader(n=4, with_strip=True, with_strip_valid_mask=True)
        trainer = Trainer(cfg, model, loader, None)
        trainer.train_one_epoch()
    finally:
        trainer_module.density_count_loss = orig

    assert len(captured) > 0
    assert all(t is not None for t in captured)
    assert all(t.shape == (4, 2) for t in captured)   # (B, Hp*Wp) = (4, 1*2)
    assert all(torch.equal(t, torch.tensor([[1.0, 0.0]] * 4)) for t in captured)


def test_validate_routes_strip_valid_mask_to_density_count_loss(tmp_path):
    """Same wiring, validate() path."""
    import src.trainer as trainer_module
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    model = _make_model(cfg)

    captured = []
    orig = trainer_module.density_count_loss
    def _spy(*args, **kwargs):
        captured.append(kwargs.get("valid_mask"))
        return orig(*args, **kwargs)
    trainer_module.density_count_loss = _spy
    try:
        loader = _make_loader(n=4, with_strip=True, with_strip_valid_mask=True)
        trainer = Trainer(cfg, model, loader, loader)
        trainer.validate()
    finally:
        trainer_module.density_count_loss = orig

    assert len(captured) > 0
    assert all(t is not None for t in captured)
    assert all(t.shape == (4, 2) for t in captured)


def test_train_one_epoch_without_strip_valid_mask_passes_none(tmp_path):
    """Regression pin: a batch with no "strip_valid_mask" key (the default for every
    existing config) must call density_count_loss with valid_mask=None explicitly —
    reproduces today's unmasked behaviour exactly."""
    import src.trainer as trainer_module
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    model = _make_model(cfg)

    captured = []
    orig = trainer_module.density_count_loss
    def _spy(*args, **kwargs):
        captured.append(kwargs.get("valid_mask"))
        return orig(*args, **kwargs)
    trainer_module.density_count_loss = _spy
    try:
        loader = _make_loader(n=4, with_strip=True)   # no with_strip_valid_mask
        trainer = Trainer(cfg, model, loader, None)
        trainer.train_one_epoch()
    finally:
        trainer_module.density_count_loss = orig

    assert len(captured) > 0
    assert all(t is None for t in captured)


# ---------------------------------------------------------------------------
# Polar-wedge experiment (09.09) — image_wedge/wedge_polar_* wiring
# ---------------------------------------------------------------------------

def test_train_one_epoch_routes_image_wedge_to_density_image(tmp_path):
    """batch["image_wedge"] (only present when dual_branch_wedge=True) must reach
    forward() as density_image during training, and its own polar coords
    (wedge_polar_t/theta, flattened (B,Hp,Wp)->(B,N)) as density_polar_t/theta — NOT
    the square image's own polar_t/theta (there are none here, with_polar=False)."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    model = _make_model(cfg)

    captured = []
    orig_forward = model.forward
    def _spy_forward(*args, **kwargs):
        captured.append({
            "density_image": kwargs.get("density_image"),
            "density_polar_t": kwargs.get("density_polar_t"),
            "density_polar_theta": kwargs.get("density_polar_theta"),
            "density_polar_valid": kwargs.get("density_polar_valid"),
        })
        return orig_forward(*args, **kwargs)
    model.forward = _spy_forward

    loader = _make_loader(n=4, with_wedge=True)
    trainer = Trainer(cfg, model, loader, None)
    trainer.train_one_epoch()

    assert len(captured) > 0
    for c in captured:
        assert c["density_image"] is not None and c["density_image"].shape[-2:] == (28, 42)
        assert c["density_polar_t"] is not None and c["density_polar_t"].shape == (4, 6)
        assert c["density_polar_theta"] is not None and c["density_polar_theta"].shape == (4, 6)
        # every wedge patch has a well-defined position by construction -> all-True,
        # regardless of wedge_polar_valid's own (tissue-fraction) values.
        assert c["density_polar_valid"] is not None
        assert c["density_polar_valid"].dtype == torch.bool
        assert bool(c["density_polar_valid"].all())


def test_validate_routes_image_wedge_to_density_image(tmp_path):
    """Same wiring, validate() path."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    model = _make_model(cfg)

    captured = []
    orig_forward = model.forward
    def _spy_forward(*args, **kwargs):
        captured.append({
            "density_image": kwargs.get("density_image"),
            "density_polar_t": kwargs.get("density_polar_t"),
            "density_polar_theta": kwargs.get("density_polar_theta"),
            "density_polar_valid": kwargs.get("density_polar_valid"),
        })
        return orig_forward(*args, **kwargs)
    model.forward = _spy_forward

    loader = _make_loader(n=4, with_wedge=True)
    trainer = Trainer(cfg, model, loader, loader)
    trainer.validate()

    assert len(captured) > 0
    for c in captured:
        assert c["density_image"] is not None and c["density_image"].shape[-2:] == (28, 42)
        assert c["density_polar_t"] is not None and c["density_polar_t"].shape == (4, 6)
        assert bool(c["density_polar_valid"].all())


def test_train_one_epoch_routes_wedge_polar_valid_to_density_count_loss(tmp_path):
    """batch["wedge_polar_valid"] (the wedge's own TISSUE-fraction validity, a
    different role than density_polar_valid above) must reach density_count_loss as
    its valid_mask kwarg, flattened (B,Hp,Wp)->(B,N) exactly like strip_valid_mask."""
    import src.trainer as trainer_module
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    model = _make_model(cfg)

    captured = []
    orig = trainer_module.density_count_loss
    def _spy(*args, **kwargs):
        captured.append(kwargs.get("valid_mask"))
        return orig(*args, **kwargs)
    trainer_module.density_count_loss = _spy
    try:
        loader = _make_loader(n=4, with_wedge=True)
        trainer = Trainer(cfg, model, loader, None)
        trainer.train_one_epoch()
    finally:
        trainer_module.density_count_loss = orig

    assert len(captured) > 0
    assert all(t is not None for t in captured)
    assert all(t.shape == (4, 6) for t in captured)
    expected_row = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert all(torch.equal(t, expected_row.expand(4, -1)) for t in captured)


def test_train_one_epoch_without_image_wedge_passes_none(tmp_path):
    """Regression pin: a batch with no "image_wedge" key (dual_branch_wedge=False,
    every existing config) must call forward() with density_image/density_polar_*
    =None explicitly."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)

    captured = []
    orig_forward = model.forward
    def _spy_forward(*args, **kwargs):
        captured.append((kwargs.get("density_image"), kwargs.get("density_polar_t")))
        return orig_forward(*args, **kwargs)
    model.forward = _spy_forward

    loader = _make_loader(n=4)   # no with_wedge
    trainer = Trainer(cfg, model, loader, None)
    trainer.train_one_epoch()

    assert len(captured) > 0
    assert all(img is None and pt is None for img, pt in captured)


# ---------------------------------------------------------------------------
# Angular-resolution bands (09.09 follow-up) — image_wedge_band{i}/wedge_band{i}_polar_* wiring
# ---------------------------------------------------------------------------

def _bands_cfg(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    cfg.model.density_head_type = "radial_attention"   # bands require a polar-aware head
    cfg.model.density_attn_num_heads = 4                # divides _MockDinoBackbone.embed_dim=64
    cfg.data.wedge_band_edges_t = [0.0, 0.5, 1.0]   # 2 bands, matches _SyntheticDataset's mock
    return cfg


def test_train_one_epoch_routes_density_image_bands_and_concatenated_polar(tmp_path):
    """batch["image_wedge_band{i}"] (only present when wedge bands are enabled) must reach
    forward() as a LIST via density_image_bands, and the per-band polar_t/theta (flattened
    (B,Hp,Wp)->(B,N) each) as density_bands_polar_t/theta — density_bands_polar_valid all-True
    regardless of the bands' own (tissue-fraction) wedge_band{i}_polar_valid values."""
    from src.trainer import Trainer
    cfg = _bands_cfg(tmp_path)
    model = _make_model(cfg)

    captured = []
    orig_forward = model.forward
    def _spy_forward(*args, **kwargs):
        captured.append({
            "bands": kwargs.get("density_image_bands"),
            "t": kwargs.get("density_bands_polar_t"),
            "theta": kwargs.get("density_bands_polar_theta"),
            "valid": kwargs.get("density_bands_polar_valid"),
        })
        return orig_forward(*args, **kwargs)
    model.forward = _spy_forward

    loader = _make_loader(n=4, with_wedge_bands=True)
    trainer = Trainer(cfg, model, loader, None)
    assert trainer.n_wedge_bands == 2
    trainer.train_one_epoch()

    assert len(captured) > 0
    for c in captured:
        assert c["bands"] is not None and len(c["bands"]) == 2
        assert c["bands"][0].shape[-2:] == (14, 28) and c["bands"][1].shape[-2:] == (28, 42)
        assert c["t"][0].shape == (4, 2) and c["t"][1].shape == (4, 6)
        assert c["theta"][0].shape == (4, 2) and c["theta"][1].shape == (4, 6)
        for v in c["valid"]:
            assert v.dtype == torch.bool and bool(v.all())


def test_validate_routes_density_image_bands(tmp_path):
    """Same wiring, validate() path."""
    from src.trainer import Trainer
    cfg = _bands_cfg(tmp_path)
    model = _make_model(cfg)

    captured = []
    orig_forward = model.forward
    def _spy_forward(*args, **kwargs):
        captured.append(kwargs.get("density_image_bands"))
        return orig_forward(*args, **kwargs)
    model.forward = _spy_forward

    loader = _make_loader(n=4, with_wedge_bands=True)
    trainer = Trainer(cfg, model, loader, loader)
    trainer.validate()

    assert len(captured) > 0
    assert all(bands is not None and len(bands) == 2 for bands in captured)


def test_train_one_epoch_routes_concatenated_wedge_bands_valid_to_density_count_loss(tmp_path):
    """The per-band TISSUE validity (wedge_band{i}_polar_valid) must reach density_count_loss as
    its valid_mask kwarg, concatenated across bands to match out["density"]'s own (B, sum(N_k))
    shape — a different concept from density_bands_polar_valid (always all-True) above."""
    import src.trainer as trainer_module
    from src.trainer import Trainer
    cfg = _bands_cfg(tmp_path)
    model = _make_model(cfg)

    captured = []
    orig = trainer_module.density_count_loss
    def _spy(*args, **kwargs):
        captured.append(kwargs.get("valid_mask"))
        return orig(*args, **kwargs)
    trainer_module.density_count_loss = _spy
    try:
        loader = _make_loader(n=4, with_wedge_bands=True)
        trainer = Trainer(cfg, model, loader, None)
        trainer.train_one_epoch()
    finally:
        trainer_module.density_count_loss = orig

    assert len(captured) > 0
    assert all(t is not None for t in captured)
    assert all(t.shape == (4, 8) for t in captured)   # (B, N0=2 + N1=6)
    expected_row = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    assert all(torch.equal(t, expected_row.expand(4, -1)) for t in captured)


def test_train_one_epoch_without_wedge_bands_passes_none(tmp_path):
    """Regression pin: a batch with no band keys (wedge_band_edges_t=None, every pre-09.09
    config) must call forward() with density_image_bands=None explicitly."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    model = _make_model(cfg)

    captured = []
    orig_forward = model.forward
    def _spy_forward(*args, **kwargs):
        captured.append(kwargs.get("density_image_bands"))
        return orig_forward(*args, **kwargs)
    model.forward = _spy_forward

    loader = _make_loader(n=4)   # no with_wedge_bands
    trainer = Trainer(cfg, model, loader, None)
    assert trainer.n_wedge_bands == 0
    trainer.train_one_epoch()

    assert len(captured) > 0
    assert all(bands is None for bands in captured)


# ---------------------------------------------------------------------------
# Multi-wycinek experiment (03.09) — candidate selection wiring
# ---------------------------------------------------------------------------

def test_select_multi_wycinek_noop_for_2d_density():
    """Regression pin: an ordinary (B,N) density (single-wycinek or no strip branch
    at all) must pass through _select_multi_wycinek completely unchanged."""
    from src.trainer import Trainer
    out = {"density": torch.rand(3, 16), "density_count": torch.rand(3)}
    ages = torch.tensor([2.0, 4.0, 6.0])
    result = Trainer._select_multi_wycinek(out, ages)
    assert result is out
    assert torch.equal(result["density"], out["density"])


def test_select_multi_wycinek_reduces_3d_to_2d_by_best_count_match():
    from src.trainer import Trainer
    density = torch.tensor([
        [[0.25, 0.25, 0.25, 0.25],   # count=1.0
         [0.5, 0.5, 0.5, 0.5]],      # count=2.0
    ])
    ages = torch.tensor([2.0])
    out = {"density": density, "density_count": density.sum(dim=-1)}
    result = Trainer._select_multi_wycinek(out, ages)
    assert result["density"].shape == (1, 4)
    assert torch.equal(result["density"][0], density[0, 1])   # count=2.0 matches age=2.0
    assert result["density_count"].shape == (1,)


def test_train_one_epoch_completes_with_multi_wycinek_density_image(tmp_path):
    """End-to-end smoke: multi_wycinek_k>1 batches must flow through forward() ->
    selection -> _combined_loss without shape errors, and produce a valid loss."""
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    model = _make_model(cfg)
    loader = _make_loader(n=4, multi_wycinek_k=3)
    trainer = Trainer(cfg, model, loader, None)
    loss = trainer.train_one_epoch()
    assert isinstance(loss, float)


def test_validate_completes_with_multi_wycinek_density_image(tmp_path):
    from src.trainer import Trainer
    cfg = _make_cfg(tmp_path)
    cfg.model.use_density_head = True
    model = _make_model(cfg)
    loader = _make_loader(n=4, multi_wycinek_k=3)
    trainer = Trainer(cfg, model, loader, loader)
    val_loss, val_mae = trainer.validate()
    assert isinstance(val_loss, float)
    assert isinstance(val_mae, float)


def test_fit_ema_selection_runs_and_saves_best(tmp_path):
    """early_stopping_ema>0 path (13.07): fit completes and writes best.pt via smoothed metric."""
    trainer = _make_trainer(tmp_path, epochs=3)
    trainer.cfg.training.early_stopping_ema = 0.5
    trainer.fit()
    assert (trainer.checkpoint_dir / "best.pt").exists()
    assert "Training complete" in trainer.log_path.read_text(encoding="utf-8")
