"""Training loop, validation, checkpointing, and logging."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.config import OtolithConfig
from src.dataset import decode_age_ordinal
from src.model import OtolithModel, density_count_loss, mil_count_loss, ordinal_loss
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
        self.density_w      = getattr(cfg.model, "density_count_weight", 1.0)
        self.density_conc_w = getattr(cfg.model, "density_conc_weight", 1.0)
        self.density_tv_w   = getattr(cfg.model, "density_tv_weight", 0.0)
        self.last_val_metrics: dict = {}   # Section-B diagnostics from validate()

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        root = Path(__file__).resolve().parents[1]
        self.checkpoint_dir = _resolve_dir(cfg.training.checkpoint_dir, root)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log_dir = _resolve_dir(cfg.training.log_dir, root)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "train.log"
        # Fresh log per run — start empty so this run's log never concatenates a
        # previous run's epochs. A concatenated train.log (append mode + no cleanup)
        # inflated epoch counts in the summary/report; see
        # plans and summaries/11.07_pipeline_TO.DO.md Punkt 3.
        self.log_path.write_text("", encoding="utf-8")

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

    def _loss_parts(self, out: dict, targets: torch.Tensor,
                    ages: torch.Tensor) -> dict[str, torch.Tensor]:
        """Weighted CORAL / MIL components + their sum, keyed by name.

        Returned so the trainer can log the head losses separately (report
        Section B — "which head is learning"). ``total`` is what we backprop.
        """
        parts: dict[str, torch.Tensor] = {}
        if "coral_logits" in out:
            parts["coral"] = self.coral_w * ordinal_loss(out["coral_logits"], targets)
        if "patch_probs" in out:
            parts["mil"] = self.count_w * mil_count_loss(
                out["patch_probs"], ages, self.sparsity_w
            )
        if "density" in out:
            # Computed on the STOP-GRADIENT density output → updates only the density
            # head, never the backbone / CORAL / MIL (age head safe by construction).
            parts["density"] = self.density_w * density_count_loss(
                out["density"], ages, self.density_conc_w, self.density_tv_w
            )
        if not parts:
            raise RuntimeError("Model produced no recognised head outputs")
        parts["total"] = torch.stack(list(parts.values())).sum()
        return parts

    def _combined_loss(self, out: dict, targets: torch.Tensor,
                       ages: torch.Tensor) -> torch.Tensor:
        """Combined CORAL + MIL loss (the scalar we optimise)."""
        return self._loss_parts(out, targets, ages)["total"]

    @staticmethod
    def _predict_age(out: dict) -> torch.Tensor:
        """Decode integer age from dict output.

        Prefers CORAL when available (matches existing val_mae semantics);
        falls back to the MIL count when only MIL is active. With the top-k
        concentration loss the age is #active patches (prob>0.5), not the sum.
        """
        if "coral_logits" in out:
            return decode_age_ordinal(out["coral_logits"])
        return (out["patch_probs"] > 0.5).sum(dim=1).long()

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
        """Run validation. Returns (val_loss, val_mae).

        Also stashes report Section-B diagnostics on ``self.last_val_metrics``:
        the weighted CORAL and MIL component losses, and the MIL localisation
        metric (mean #active patches at prob>0.5 vs mean age — they should
        converge as the MIL head learns to fire ~age patches).
        """
        self.last_val_metrics = {}
        if self.val_loader is None:
            return float("nan"), float("nan")

        self.model.eval()
        total_loss = 0.0
        total_mae  = 0.0
        coral_sum = 0.0
        mil_sum   = 0.0
        active_sum = 0.0
        age_sum    = 0.0
        density_sum = 0.0
        density_active_sum = 0.0
        has_coral = has_mil = has_density = False
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
                parts = self._loss_parts(out, targets, ages)
                pred_ages = self._predict_age(out)

                bs = images.size(0)
                total_loss += parts["total"].item() * bs
                total_mae  += (pred_ages - ages).abs().float().sum().item()
                if "coral" in parts:
                    coral_sum += parts["coral"].item() * bs; has_coral = True
                if "mil" in parts:
                    mil_sum += parts["mil"].item() * bs; has_mil = True
                    active_sum += (out["patch_probs"] > 0.5).sum(dim=1).float().sum().item()
                if "density" in parts:
                    density_sum += parts["density"].item() * bs; has_density = True
                    density_active_sum += (out["density"] > 0.5).sum(dim=1).float().sum().item()
                age_sum += ages.float().sum().item()
                n += bs

        denom = max(n, 1)
        if has_coral:
            self.last_val_metrics["coral_loss"] = coral_sum / denom
        if has_mil:
            self.last_val_metrics["mil_loss"] = mil_sum / denom
            self.last_val_metrics["mil_active"] = active_sum / denom
            self.last_val_metrics["mean_age"] = age_sum / denom
        if has_density:
            self.last_val_metrics["density_loss"] = density_sum / denom
            self.last_val_metrics["density_active"] = density_active_sum / denom
            self.last_val_metrics["mean_age"] = age_sum / denom
        return total_loss / denom, total_mae / denom

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
        ema_alpha = getattr(self.cfg.training, "early_stopping_ema", 0.0)
        best_metric = float("inf")
        metric_ema: Optional[float] = None
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

            self._log_epoch(epoch, train_loss, val_loss, val_mae, current_lr,
                            getattr(self, "last_val_metrics", None))
            ckpt_path = self.save_checkpoint(epoch, val_loss)

            raw_metric = val_mae if metric_name == "val_mae" else val_loss
            # Smooth the noisy monitored metric (raw val_mae is chunky) so best.pt and
            # early stopping don't hinge on a single lucky epoch (13.07 diagnosis).
            # ema_alpha == 0 → use the raw metric (previous behaviour).
            if ema_alpha > 0.0 and raw_metric == raw_metric:   # skip NaN (no val loader)
                metric_ema = (raw_metric if metric_ema is None
                              else ema_alpha * raw_metric + (1.0 - ema_alpha) * metric_ema)
                current = metric_ema
            else:
                current = raw_metric
            improved = current < best_metric - min_delta
            if improved:
                best_metric = current
                patience_counter = 0
                self._save_best_checkpoint(epoch, val_loss)   # copies epoch ckpt → best.pt
            else:
                patience_counter += 1

            # Keep only best.pt — drop this epoch's checkpoint (best.pt already holds the
            # best model). Stops the run dir ballooning (each checkpoint ~265 MB).
            if getattr(self.cfg.training, "keep_only_best", True) and ckpt_path.name != "best.pt":
                try:
                    ckpt_path.unlink()
                except OSError:
                    pass

            if not improved and patience > 0 and patience_counter >= patience:
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
        lr: float | None = None, extra: dict | None = None,
    ) -> None:
        line = (
            f"epoch={epoch:3d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_mae={val_mae:.3f}"
        )
        if lr is not None:
            line += f"  lr={lr:.2e}"
        # Section-B diagnostics (only present with the relevant heads active).
        if extra:
            for key in ("coral_loss", "mil_loss", "mil_active",
                        "density_loss", "density_active", "mean_age"):
                if key in extra and extra[key] == extra[key]:   # skip NaN
                    line += f"  {key}={extra[key]:.4f}"
        self._log(line)
