"""Main entrypoint for OtolithDinoStandalone.

Usage:
    python -m src.entrypoint --mode info     # print resolved config and exit
    python -m src.entrypoint --mode train    # train
    python -m src.entrypoint --mode demo     # unified quick-check → scripts/run_pipeline.py
    python -m src.entrypoint --mode report   # rebuild HTML report from existing outputs

`--mode demo` delegates to the single demo implementation in
scripts/run_pipeline.py (configs/config_demo.yaml), identical to `python main.py`
with MODE="demo".
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
    from src.utils import seed_worker, make_loader_generator

    kw = dict(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(OtolithDataset(cfg, split="train"), shuffle=True,
                              generator=make_loader_generator(cfg.project.seed), **kw)
    val_loader   = DataLoader(OtolithDataset(cfg, split="val"),   shuffle=False, **kw)
    test_loader  = DataLoader(OtolithDataset(cfg, split="test"),  shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def _test_loader(cfg):
    from torch.utils.data import DataLoader
    from src.dataset import OtolithDataset

    return DataLoader(
        OtolithDataset(cfg, split="test"),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )


def _resolve_checkpoint(root: Path, cfg) -> Path | None:
    """Return best.pt (preferred) or the latest epoch checkpoint, else None."""
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    best = ckpt_dir / "best.pt"
    if best.exists():
        return best
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
    return ckpts[-1] if ckpts else None


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from src.config import load_config
    cfg = load_config(args.config)

    # Before any backbone import so XFORMERS_DISABLED can take effect (true CLS attention).
    from src.utils import configure_attention
    configure_attention(cfg.interpretation.disable_fused_attention)

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
        from src.utils import seed_everything

        # Seed BEFORE model init + loader build so the run is reproducible.
        seed_everything(cfg.project.seed)
        train_loader, val_loader, _ = _build_loaders(cfg)
        trainer = Trainer(cfg, OtolithModel(cfg), train_loader, val_loader)
        trainer.fit()
        return 0

    # ------------------------------------------------------------------
    # demo — unified quick-check of the whole pipeline
    # ------------------------------------------------------------------
    # A single demo implementation lives in scripts/run_pipeline.py. Both this
    # mode and `python main.py` (MODE="demo") funnel into it with
    # configs/config_demo.yaml (1 epoch + demo-subsampled dataset), so they
    # always produce the same artefacts (outputs/demo/comparison_report.html).
    if args.mode == "demo":
        from scripts.run_pipeline import main as run_pipeline_main

        demo_cfg = root / "configs" / "config_demo.yaml"
        print(f"[demo] Delegacja do run_pipeline (config: {demo_cfg.name})")
        run_pipeline_main([
            "--base-config", str(demo_cfg),
            "--output-dir",  str(root / "outputs" / "demo"),
        ])
        return 0

    # ------------------------------------------------------------------
    # inference — predict on the test split using the trained checkpoint
    # ------------------------------------------------------------------
    if args.mode == "inference":
        print_config_summary(cfg)
        ckpt = _resolve_checkpoint(root, cfg)
        if ckpt is None:
            print(f"[inference] Brak checkpointu w {root / cfg.training.checkpoint_dir} "
                  f"— najpierw wytrenuj model (--mode train).")
            return 1

        from src.inference import load_model_from_checkpoint, run_inference
        print(f"[inference] Checkpoint: {ckpt}")
        model = load_model_from_checkpoint(cfg, ckpt)
        run_inference(cfg, model, _test_loader(cfg), out_dir)
        print(f"[inference] Zapisano: {out_dir / 'predictions.csv'}")
        return 0

    # ------------------------------------------------------------------
    # eval — inference on the test split + printed regression metrics
    # ------------------------------------------------------------------
    if args.mode == "eval":
        print_config_summary(cfg)
        ckpt = _resolve_checkpoint(root, cfg)
        if ckpt is None:
            print(f"[eval] Brak checkpointu w {root / cfg.training.checkpoint_dir} "
                  f"— najpierw wytrenuj model (--mode train).")
            return 1

        import pandas as pd
        from src.inference import load_model_from_checkpoint, run_inference
        from src.report_common import compute_metrics

        print(f"[eval] Checkpoint: {ckpt}")
        model = load_model_from_checkpoint(cfg, ckpt)
        run_inference(cfg, model, _test_loader(cfg), out_dir)

        preds = pd.read_csv(out_dir / "predictions.csv")
        labeled = preds[preds["target_age"].notna()]
        if labeled.empty:
            print("[eval] Zbiór testowy nie ma etykiet — metryki niedostępne.")
            return 0
        m = compute_metrics(labeled["target_age"].values, labeled["predicted_age"].values)
        print(f"[eval] Metryki na zbiorze testowym (n={len(labeled)}):")
        for key, val in m.items():
            print(f"  {key:8s} = {val:.4f}")
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
