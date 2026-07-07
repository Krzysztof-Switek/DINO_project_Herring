"""Training loop, validation, checkpointing, and logging."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.config import OtolithConfig
from src.dataset import decode_age_ordinal
from src.model import OtolithModel, mil_count_loss, ordinal_loss
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

        # Loss weights for combined CORAL + MIL training
        self.coral_w    = cfg.model.coral_loss_weight
        self.count_w    = cfg.model.mil_count_weight
        self.sparsity_w = cfg.model.mil_sparsity_weight

        self.optimizer = self._build_optimizer()
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

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW with a lower LR for the pretrained backbone than for the heads.

        Fine-tuning a pretrained ViT works best when the backbone is updated more
        gently than the freshly-initialised heads (discriminative learning rate).
        ``backbone_lr_mult == 1.0`` reproduces the old uniform-LR behaviour.
        """
        lr = self.cfg.training.lr
        wd = self.cfg.training.weight_decay
        backbone_lr = lr * self.cfg.training.backbone_lr_mult

        backbone_ids = {id(p) for p in self.model.backbone.parameters()}
        backbone_params = [p for p in self.model.parameters() if id(p) in backbone_ids]
        head_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]

        groups = [
            {"params": head_params, "lr": lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]
        return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)

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

    def _combined_loss(self, out: dict, targets: torch.Tensor,
                       ages: torch.Tensor) -> torch.Tensor:
        """Combined CORAL + MIL loss based on which heads are active."""
        parts: list[torch.Tensor] = []
        if "coral_logits" in out:
            parts.append(self.coral_w * ordinal_loss(out["coral_logits"], targets))
        if "patch_probs" in out:
            parts.append(self.count_w * mil_count_loss(
                out["patch_probs"], ages, self.sparsity_w
            ))
        if not parts:
            raise RuntimeError("Model produced no recognised head outputs")
        return torch.stack(parts).sum()

    @staticmethod
    def _predict_age(out: dict) -> torch.Tensor:
        """Decode integer age from dict output.

        Prefers CORAL when available (matches existing val_mae semantics);
        falls back to rounded MIL count when only MIL is active.
        """
        if "coral_logits" in out:
            return decode_age_ordinal(out["coral_logits"])
        return out["patch_count"].round().long()

    def train_one_epoch(self) -> float:
        """One full pass over train_loader. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in self.train_loader:
            images  = batch["image"].to(self.device)
            targets = batch["age_ordinal"].to(self.device)
            ages    = batch["age"].to(self.device)

            metadata = batch.get("metadata")
            if metadata is not None:
                metadata = metadata.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(images, metadata=metadata)
            loss = self._combined_loss(out, targets, ages)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            n += images.size(0)

        return total_loss / max(n, 1)

    def validate(self) -> tuple[float, float]:
        """Run validation. Returns (val_loss, val_mae)."""
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

                metadata = batch.get("metadata")
                if metadata is not None:
                    metadata = metadata.to(self.device)

                out = self.model(images, metadata=metadata)
                loss = self._combined_loss(out, targets, ages)
                pred_ages = self._predict_age(out)

                total_loss += loss.item() * images.size(0)
                total_mae  += (pred_ages - ages).abs().float().sum().item()
                n += images.size(0)

        return total_loss / max(n, 1), total_mae / max(n, 1)

    # ------------------------------------------------------------------
    # Full fit loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Train for cfg.training.epochs epochs with optional early stopping.

        Backbone is frozen for the first freeze_backbone_epochs epochs,
        then unfrozen. Checkpoint is saved after every epoch. best.pt is
        updated whenever the monitored metric improves.
        """
        freeze_until = self.cfg.training.freeze_backbone_epochs
        if freeze_until > 0:
            self.model.freeze_backbone()
            self._log(f"Backbone frozen for first {freeze_until} epochs")

        patience = self.cfg.training.early_stopping_patience
        min_delta = self.cfg.training.early_stopping_min_delta
        metric_name = self.cfg.training.early_stopping_metric
        best_metric = float("inf")
        patience_counter = 0

        for epoch in range(1, self.cfg.training.epochs + 1):
            if freeze_until > 0 and epoch == freeze_until + 1:
                self.model.unfreeze_backbone()
                self._log(f"Backbone unfrozen at epoch {epoch}")

            train_loss = self.train_one_epoch()
            val_loss, val_mae = self.validate()

            # lr used during this epoch (before the scheduler advances it)
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            self._log_epoch(epoch, train_loss, val_loss, val_mae, current_lr)
            self.save_checkpoint(epoch, val_loss)

            current = val_mae if metric_name == "val_mae" else val_loss
            if current < best_metric - min_delta:
                best_metric = current
                patience_counter = 0
                self._save_best_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    self._log(
                        f"Early stopping — brak poprawy {metric_name} przez {patience} epok "
                        f"(best={best_metric:.4f})"
                    )
                    break

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

    def _save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
        import shutil as _shutil
        src = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}_loss{val_loss:.4f}.pt"
        if src.exists():
            _shutil.copy2(src, self.checkpoint_dir / "best.pt")

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
        self, epoch: int, train_loss: float, val_loss: float, val_mae: float,
        lr: float | None = None,
    ) -> None:
        line = (
            f"epoch={epoch:3d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_mae={val_mae:.3f}"
        )
        if lr is not None:
            line += f"  lr={lr:.2e}"
        self._log(line)
