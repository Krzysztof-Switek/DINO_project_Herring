"""Main entrypoint for OtolithDinoStandalone.

Usage:
    python -m src.entrypoint --config configs/config.yaml
    python -m src.entrypoint --config configs/config.yaml --mode train
    python -m src.entrypoint --config configs/config.yaml --mode inference
"""
from __future__ import annotations

import argparse
import json
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
        default="info",
        choices=["info", "train", "inference", "eval", "report"],
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


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # import here so import errors are surfaced clearly
    from src.config import load_config

    cfg = load_config(args.config)

    if args.mode == "info":
        print_config_summary(cfg)
        print("\nStage 1 scaffold OK — config loaded and validated.")
        return 0

    if args.mode == "train":
        print_config_summary(cfg)
        print("[train] Training loop not yet implemented (Stage 3+).")
        return 0

    if args.mode == "inference":
        print_config_summary(cfg)
        print("[inference] Inference not yet implemented (Stage 5+).")
        return 0

    if args.mode == "eval":
        print_config_summary(cfg)
        print("[eval] Evaluation not yet implemented.")
        return 0

    if args.mode == "report":
        from src.report import build_html_report, save_report
        root = Path(args.config).resolve().parent.parent
        html = build_html_report(
            labels_csv        = root / cfg.data.labels_csv,
            log_path          = root / cfg.training.log_dir / "train.log",
            predictions_csv   = root / cfg.inference.output_dir / "predictions.csv",
            heatmaps_dir      = root / cfg.inference.output_dir / "heatmaps",
            overlays_dir      = root / cfg.inference.output_dir / "overlays",
            cand_json_dir     = root / cfg.inference.output_dir / "candidates",
            cand_overlays_dir = root / cfg.inference.output_dir / "candidates_overlays",
        )
        out = root / cfg.inference.output_dir / "report.html"
        save_report(html, out)
        return 0

    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
