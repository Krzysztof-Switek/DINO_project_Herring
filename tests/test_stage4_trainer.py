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
    def __init__(self, n: int = 8, num_age_classes: int = 10, image_size: int = 56):
        self.n = n
        self.num_age_classes = num_age_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict:
        age = (idx % (self.num_age_classes - 1)) + 1
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "age_ordinal": encode_age_ordinal(age, self.num_age_classes),
            "age": torch.tensor(age, dtype=torch.long),
            "image_id": f"img_{idx:03d}.png",
        }


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


def _make_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    ds = _SyntheticDataset(n=n, num_age_classes=10, image_size=56)
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


def test_fit_ema_selection_runs_and_saves_best(tmp_path):
    """early_stopping_ema>0 path (13.07): fit completes and writes best.pt via smoothed metric."""
    trainer = _make_trainer(tmp_path, epochs=3)
    trainer.cfg.training.early_stopping_ema = 0.5
    trainer.fit()
    assert (trainer.checkpoint_dir / "best.pt").exists()
    assert "Training complete" in trainer.log_path.read_text(encoding="utf-8")
