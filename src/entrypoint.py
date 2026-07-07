"""Main entrypoint for OtolithDinoStandalone.

Usage:
    python -m src.entrypoint --mode info     # print resolved config and exit
    python -m src.entrypoint --mode train    # train
    python -m src.entrypoint --mode demo     # 1 epoch + full pipeline check
    python -m src.entrypoint --mode report   # rebuild HTML report from existing outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OtolithDinoStandalone — DINOv2-based otolith age prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["info", "train", "inference", "eval", "report", "demo"],
        help="Run mode",
    )
    return parser.parse_args(argv)


def print_config_summary(cfg) -> None:
    print("=" * 60)
    print(f"Project : {cfg.project.name} v{cfg.project.version}")
    print(f"Backbone: {cfg.model.backbone}")
    print(f"Target  : {cfg.model.target_type}  (classes={cfg.model.num_age_classes})")
    print(f"Metadata: {'yes' if cfg.model.use_metadata else 'no (image-only)'}")
    print(f"Device  : {cfg.training.device}")
    print(f"Epochs  : {cfg.training.epochs}  BS={cfg.training.batch_size}")
    print(f"Interp  : {cfg.interpretation.method}")
    print("=" * 60)


def _build_loaders(cfg):
    from torch.utils.data import DataLoader
    from src.dataset import OtolithDataset

    kw = dict(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(OtolithDataset(cfg, split="train"), shuffle=True,  **kw)
    val_loader   = DataLoader(OtolithDataset(cfg, split="val"),   shuffle=False, **kw)
    test_loader  = DataLoader(OtolithDataset(cfg, split="test"),  shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from src.config import load_config
    cfg = load_config(args.config)

    root    = PROJECT_ROOT
    out_dir = root / cfg.inference.output_dir

    # ------------------------------------------------------------------
    # info — print the resolved config and exit (no training)
    # ------------------------------------------------------------------
    if args.mode == "info":
        print_config_summary(cfg)
        return 0

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    if args.mode == "train":
        print_config_summary(cfg)
        from src.model import OtolithModel
        from src.trainer import Trainer

        train_loader, val_loader, _ = _build_loaders(cfg)
        trainer = Trainer(cfg, OtolithModel(cfg), train_loader, val_loader)
        trainer.fit()
        return 0

    # ------------------------------------------------------------------
    # demo — 1 epoch, then full pipeline: inference → heatmaps → candidates → report
    # ------------------------------------------------------------------
    if args.mode == "demo":
        cfg = cfg.model_copy(
            update={"training": cfg.training.model_copy(update={"epochs": 1})}
        )
        print_config_summary(cfg)

        from src.model import OtolithModel
        from src.trainer import Trainer
        from src.inference import load_model_from_checkpoint, run_inference
        from src.interpretation import run_interpretation
        from src.candidates import run_candidates
        from src.report import build_html_report, save_report

        print("[demo] Trening — 1 epoka...")
        train_loader, val_loader, test_loader = _build_loaders(cfg)
        model   = OtolithModel(cfg)
        trainer = Trainer(cfg, model, train_loader, val_loader)
        trainer.fit()

        ckpt_dir  = root / cfg.training.checkpoint_dir
        ckpts     = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
        if not ckpts:
            print("[demo] Błąd: brak checkpointu po treningu.")
            return 1
        ckpt_path = ckpts[-1]
        print(f"[demo] Checkpoint: {ckpt_path.name}")

        print("[demo] Inferencja na zbiorze testowym...")
        model = load_model_from_checkpoint(cfg, ckpt_path)
        run_inference(cfg, model, test_loader, out_dir)

        print("[demo] Heatmapy i nakładki...")
        run_interpretation(cfg, model, test_loader, out_dir)

        print("[demo] Detekcja kandydatów przyrostów...")
        run_candidates(cfg, model, test_loader, out_dir)

        print("[demo] Generowanie raportu HTML...")
        html = build_html_report(
            labels_csv        = root / cfg.data.labels_csv,
            log_path          = root / cfg.training.log_dir / "train.log",
            predictions_csv   = out_dir / "predictions.csv",
            heatmaps_dir      = out_dir / "heatmaps",
            overlays_dir      = out_dir / "overlays",
            cand_json_dir     = out_dir / "candidates",
            cand_overlays_dir = out_dir / "candidates_overlays",
        )
        report_path = out_dir / "report.html"
        save_report(html, report_path)

        print("\n[demo] Gotowe. Zapisane artefakty:")
        print(f"  checkpoint  : {ckpt_path}")
        print(f"  predykcje   : {out_dir / 'predictions.csv'}")
        print(f"  heatmapy    : {out_dir / 'heatmaps/'}")
        print(f"  nakładki    : {out_dir / 'overlays/'}")
        print(f"  kandydaci   : {out_dir / 'candidates/'}")
        print(f"  raport      : {report_path}")
        return 0

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    if args.mode == "inference":
        print_config_summary(cfg)
        print("[inference] Inference not yet implemented.")
        return 0

    # ------------------------------------------------------------------
    # eval
    # ------------------------------------------------------------------
    if args.mode == "eval":
        print_config_summary(cfg)
        print("[eval] Evaluation not yet implemented.")
        return 0

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    if args.mode == "report":
        from src.report import build_html_report, save_report

        html = build_html_report(
            labels_csv        = root / cfg.data.labels_csv,
            log_path          = root / cfg.training.log_dir / "train.log",
            predictions_csv   = out_dir / "predictions.csv",
            heatmaps_dir      = out_dir / "heatmaps",
            overlays_dir      = out_dir / "overlays",
            cand_json_dir     = out_dir / "candidates",
            cand_overlays_dir = out_dir / "candidates_overlays",
        )
        save_report(html, out_dir / "report.html")
        return 0

    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
