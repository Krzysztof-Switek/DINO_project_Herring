"""Training loop, validation, checkpointing, and logging."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.config import OtolithConfig
from src.dataset import decode_age_ordinal
from src.model import OtolithModel, ordinal_loss
from src.utils import resolve_device  # re-exported for backwards compat


def _resolve_dir(cfg_path: str, root: Path) -> Path:
    """Return absolute directory: use as-is if absolute, else anchor to root."""
    p = Path(cfg_path)
    return p if p.is_absolute() else root / p


class Trainer:
    """Orchestrates training, validation, checkpointing, and logging.

    checkpoint_dir and log_dir from cfg are resolved relative to the project
    root unless they are already absolute paths (useful for tests with tmp_path).
    """

    def __init__(
        self,
        cfg: OtolithConfig,
        model: OtolithModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = resolve_device(cfg.training.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = self._build_scheduler()

        root = Path(__file__).resolve().parents[1]
        self.checkpoint_dir = _resolve_dir(cfg.training.checkpoint_dir, root)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log_dir = _resolve_dir(cfg.training.log_dir, root)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "train.log"

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _build_scheduler(self) -> Optional[object]:
        sched = self.cfg.training.scheduler
        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(self.cfg.training.epochs, 1)
            )
        if sched == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        return None

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> float:
        """One full pass over train_loader. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in self.train_loader:
            images  = batch["image"].to(self.device)
            targets = batch["age_ordinal"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = ordinal_loss(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            n += images.size(0)

        return total_loss / max(n, 1)

    def validate(self) -> tuple[float, float]:
        """Run validation. Returns (val_loss, val_mae).

        val_mae is mean absolute error between predicted and true integer age.
        Returns (nan, nan) when no val_loader is provided.
        """
        if self.val_loader is None:
            return float("nan"), float("nan")

        self.model.eval()
        total_loss = 0.0
        total_mae  = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images  = batch["image"].to(self.device)
                targets = batch["age_ordinal"].to(self.device)
                ages    = batch["age"].to(self.device)

                logits = self.model(images)
                loss = ordinal_loss(logits, targets)
                pred_ages = decode_age_ordinal(logits)

                total_loss += loss.item() * images.size(0)
                total_mae  += (pred_ages - ages).abs().float().sum().item()
                n += images.size(0)

        return total_loss / max(n, 1), total_mae / max(n, 1)

    # ------------------------------------------------------------------
    # Full fit loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Train for cfg.training.epochs epochs.

        Backbone is frozen for the first freeze_backbone_epochs epochs,
        then unfrozen. Checkpoint is saved after every epoch.
        """
        freeze_until = self.cfg.training.freeze_backbone_epochs
        if freeze_until > 0:
            self.model.freeze_backbone()
            self._log(f"Backbone frozen for first {freeze_until} epochs")

        for epoch in range(1, self.cfg.training.epochs + 1):
            if freeze_until > 0 and epoch == freeze_until + 1:
                self.model.unfreeze_backbone()
                self._log(f"Backbone unfrozen at epoch {epoch}")

            train_loss = self.train_one_epoch()
            val_loss, val_mae = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            self._log_epoch(epoch, train_loss, val_loss, val_mae)
            self.save_checkpoint(epoch, val_loss)

        self._log("Training complete")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        """Save model + optimizer state. Returns the checkpoint path."""
        fname = f"checkpoint_epoch{epoch:03d}_loss{val_loss:.4f}.pt"
        path = self.checkpoint_dir / fname
        try:
            cfg_dict = self.cfg.model_dump()
        except AttributeError:
            cfg_dict = {}
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "cfg": cfg_dict,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore model + optimizer from checkpoint. Returns saved epoch."""
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return int(ckpt["epoch"])

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _log_epoch(
        self, epoch: int, train_loss: float, val_loss: float, val_mae: float
    ) -> None:
        self._log(
            f"epoch={epoch:3d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_mae={val_mae:.3f}"
        )
