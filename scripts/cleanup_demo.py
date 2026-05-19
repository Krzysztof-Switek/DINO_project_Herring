"""Clean up demo-pipeline state so the next run starts from scratch.

Removes (each scope can be toggled off):
  - Output directory contents — predictions, heatmaps, overlays, masks,
    candidates, cards, pipeline_state.json, pipeline_summary.json,
    comparison_report.html, and the per-condition subfolders.
  - Model checkpoints under ``training.checkpoint_dir`` (resolved from the
    base config and the Embedded / NotEmbedded overrides).
  - Training logs under ``training.log_dir`` (same resolution).

Usage:
    # See what would be removed without touching disk
    python scripts/cleanup_demo.py --dry-run

    # Wipe everything for outputs/demo (interactive confirmation)
    python scripts/cleanup_demo.py

    # Keep checkpoints (e.g. when you plan to re-use --skip-train)
    python scripts/cleanup_demo.py --keep-checkpoints

    # Non-interactive (CI / scripted)
    python scripts/cleanup_demo.py --yes
"""
from __future__ import annotations

import argparse
import shutil
import stat
import sys
from pathlib import Path
from typing import Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config-path resolution
# ---------------------------------------------------------------------------

def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = PROJECT_ROOT / pp
    return pp


def _read_training_paths(config_path: Path) -> dict[str, Path]:
    """Extract ``training.checkpoint_dir`` and ``training.log_dir`` from one YAML."""
    if not config_path.exists():
        return {}
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    training = raw.get("training", {}) or {}
    out: dict[str, Path] = {}
    if "checkpoint_dir" in training:
        out["checkpoint_dir"] = _resolve(training["checkpoint_dir"])
    if "log_dir" in training:
        out["log_dir"] = _resolve(training["log_dir"])
    return out


def collect_checkpoint_dirs(configs: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    for c in configs:
        p = _read_training_paths(c).get("checkpoint_dir")
        if p is not None:
            seen.add(p.resolve())
    return sorted(seen)


def collect_log_dirs(configs: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    for c in configs:
        p = _read_training_paths(c).get("log_dir")
        if p is not None:
            seen.add(p.resolve())
    return sorted(seen)


# ---------------------------------------------------------------------------
# Target collection
# ---------------------------------------------------------------------------

def collect_targets(
    output_dir: Path,
    configs: Iterable[Path],
    *,
    clean_outputs: bool,
    clean_checkpoints: bool,
    clean_logs: bool,
) -> list[Path]:
    """Build the deduplicated list of paths to remove (existing only)."""
    targets: list[Path] = []
    if clean_outputs and output_dir.exists():
        for child in sorted(output_dir.iterdir()):
            targets.append(child)
    if clean_checkpoints:
        targets.extend(p for p in collect_checkpoint_dirs(configs) if p.exists())
    if clean_logs:
        targets.extend(p for p in collect_log_dirs(configs) if p.exists())
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in targets:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    # Drop nested paths whose ancestor is already on the list (would be removed
    # transitively, and re-deleting would raise FileNotFoundError).
    resolved = [p.resolve() for p in unique]
    keep: list[Path] = []
    for p, rp in zip(unique, resolved):
        if any(other != rp and other in rp.parents for other in resolved):
            continue
        keep.append(p)
    return keep


# ---------------------------------------------------------------------------
# Safe removal
# ---------------------------------------------------------------------------

def _on_rm_error(func, path, exc_info):
    """rmtree handler — strip read-only bit on Windows then retry once."""
    try:
        Path(path).chmod(stat.S_IWRITE)
        func(path)
    except OSError:
        raise


def remove_path(p: Path) -> None:
    """Remove a file, directory or symlink. Re-raises ``OSError`` on failure."""
    if p.is_symlink() or not p.is_dir():
        p.unlink()
        return
    shutil.rmtree(p, onerror=_on_rm_error)


# ---------------------------------------------------------------------------
# Sanity guard — refuse to delete obviously dangerous paths
# ---------------------------------------------------------------------------

def _is_safe_to_delete(p: Path) -> bool:
    """Reject paths that are filesystem roots, the project root, or its ancestors."""
    rp = p.resolve()
    # Drive root / filesystem root
    if str(rp) == str(rp.anchor):
        return False
    project = PROJECT_ROOT.resolve()
    # The project root itself — never delete it
    if rp == project:
        return False
    # An ancestor of the project root (e.g. parent dir, home dir) — never delete
    if rp in project.parents:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean demo-pipeline outputs, checkpoints and logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for examples.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/demo",
        help="Pipeline output directory to wipe (default: outputs/demo).",
    )
    parser.add_argument(
        "--base-config", default="configs/config.yaml",
        help="Base config providing default checkpoint_dir / log_dir.",
    )
    parser.add_argument(
        "--config-embedded", default="configs/config_embedded.yaml",
        help="Override config for the Embedded model.",
    )
    parser.add_argument(
        "--config-not-embedded", default="configs/config_not_embedded.yaml",
        help="Override config for the NotEmbedded model.",
    )
    parser.add_argument(
        "--extra-config", action="append", default=[],
        help="Additional config YAML to scan for paths. Can be passed multiple times.",
    )
    parser.add_argument(
        "--keep-outputs", action="store_true",
        help="Don't wipe the output directory (only checkpoints/logs).",
    )
    parser.add_argument(
        "--keep-checkpoints", action="store_true",
        help="Don't wipe model checkpoints (useful with --skip-train).",
    )
    parser.add_argument(
        "--keep-logs", action="store_true",
        help="Don't wipe training logs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be deleted without deleting anything.",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args(argv)

    output_dir = _resolve(args.output_dir)
    configs = [
        _resolve(args.base_config),
        _resolve(args.config_embedded),
        _resolve(args.config_not_embedded),
        *(_resolve(c) for c in args.extra_config),
    ]

    targets = collect_targets(
        output_dir, configs,
        clean_outputs=not args.keep_outputs,
        clean_checkpoints=not args.keep_checkpoints,
        clean_logs=not args.keep_logs,
    )

    # Safety check
    unsafe = [t for t in targets if not _is_safe_to_delete(t)]
    if unsafe:
        print("Odmowa: następujące ścieżki wyglądają niebezpiecznie i zostały pominięte:",
              file=sys.stderr)
        for t in unsafe:
            print(f"  - {t}", file=sys.stderr)
        targets = [t for t in targets if t not in unsafe]

    if not targets:
        print("Nic do posprzątania (wszystkie ścieżki nie istnieją lub zostały odfiltrowane).")
        return 0

    print(f"Cleanup demo — projekt: {PROJECT_ROOT}")
    print(f"Output dir:  {output_dir}")
    print("\nDo usunięcia:")
    for t in targets:
        kind = "dir " if t.is_dir() else "file"
        print(f"  [{kind}] {t}")

    if args.dry_run:
        print("\n[dry-run] Nic nie zostało usunięte.")
        return 0

    if not args.yes:
        try:
            reply = input("\nKontynuować? Wpisz 'tak' aby usunąć: ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("tak", "yes", "y"):
            print("Przerwane.")
            return 1

    print("\nUsuwam:")
    errors = 0
    for t in targets:
        try:
            remove_path(t)
            print(f"  OK     {t}")
        except OSError as e:
            errors += 1
            print(f"  BŁĄD   {t}  ({e})", file=sys.stderr)

    print(f"\nGotowe. Usunięto {len(targets) - errors}/{len(targets)} ścieżek.")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
