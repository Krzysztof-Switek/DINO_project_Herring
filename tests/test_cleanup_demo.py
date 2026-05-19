"""Tests for scripts/cleanup_demo.py — safe cleanup of demo pipeline state."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(path: Path, checkpoint_dir: Path, log_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "training": {
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
        }
    }), encoding="utf-8")


def _make_demo_output_dir(out: Path) -> None:
    """Create a realistic demo output layout."""
    (out / "emb_on_emb" / "heatmaps").mkdir(parents=True)
    (out / "emb_on_emb" / "overlays").mkdir(parents=True)
    (out / "cards" / "emb_on_emb" / "best").mkdir(parents=True)
    (out / "emb_on_emb" / "heatmaps" / "fish_heatmap.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (out / "pipeline_state.json").write_text('{"completed_steps": ["scan"]}')
    (out / "comparison_report.html").write_text("<html></html>")


# ---------------------------------------------------------------------------
# Path collection
# ---------------------------------------------------------------------------

def test_collect_checkpoint_dirs_dedup(tmp_path):
    from scripts.cleanup_demo import collect_checkpoint_dirs
    cfg1 = tmp_path / "c1.yaml"
    cfg2 = tmp_path / "c2.yaml"
    _make_config(cfg1, tmp_path / "checkpoints/embedded", tmp_path / "logs/embedded")
    _make_config(cfg2, tmp_path / "checkpoints/embedded", tmp_path / "logs/embedded")
    out = collect_checkpoint_dirs([cfg1, cfg2])
    # Same dir referenced twice → returned once
    assert len(out) == 1


def test_collect_log_dirs_distinct(tmp_path):
    from scripts.cleanup_demo import collect_log_dirs
    cfg1 = tmp_path / "c1.yaml"
    cfg2 = tmp_path / "c2.yaml"
    _make_config(cfg1, tmp_path / "ck/a", tmp_path / "log/a")
    _make_config(cfg2, tmp_path / "ck/b", tmp_path / "log/b")
    out = collect_log_dirs([cfg1, cfg2])
    assert len(out) == 2


def test_collect_handles_missing_config(tmp_path):
    from scripts.cleanup_demo import collect_checkpoint_dirs
    # Missing config files are silently ignored
    out = collect_checkpoint_dirs([tmp_path / "nope.yaml"])
    assert out == []


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def test_collect_targets_includes_outputs_children(tmp_path):
    from scripts.cleanup_demo import collect_targets
    out_dir = tmp_path / "outputs" / "demo"
    _make_demo_output_dir(out_dir)
    cfg = tmp_path / "c.yaml"
    _make_config(cfg, tmp_path / "checkpoints", tmp_path / "logs")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "logs").mkdir()

    targets = collect_targets(out_dir, [cfg],
                              clean_outputs=True,
                              clean_checkpoints=True,
                              clean_logs=True)
    names = {p.name for p in targets}
    # Output children
    assert "pipeline_state.json" in names
    assert "comparison_report.html" in names
    assert "emb_on_emb" in names
    assert "cards" in names
    # Checkpoints + logs
    assert any(p.name == "checkpoints" for p in targets)
    assert any(p.name == "logs" for p in targets)


def test_collect_targets_keep_flags(tmp_path):
    from scripts.cleanup_demo import collect_targets
    out_dir = tmp_path / "outputs" / "demo"
    _make_demo_output_dir(out_dir)
    cfg = tmp_path / "c.yaml"
    _make_config(cfg, tmp_path / "checkpoints", tmp_path / "logs")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "logs").mkdir()

    # Keep checkpoints only
    targets = collect_targets(out_dir, [cfg],
                              clean_outputs=True,
                              clean_checkpoints=False,
                              clean_logs=True)
    assert not any(p.name == "checkpoints" for p in targets)
    assert any(p.name == "logs" for p in targets)

    # Keep outputs only
    targets = collect_targets(out_dir, [cfg],
                              clean_outputs=False,
                              clean_checkpoints=True,
                              clean_logs=True)
    assert not any(p.parent == out_dir for p in targets)


def test_collect_targets_empty_when_nothing_exists(tmp_path):
    from scripts.cleanup_demo import collect_targets
    out_dir = tmp_path / "outputs" / "demo"   # does not exist
    cfg = tmp_path / "c.yaml"
    _make_config(cfg, tmp_path / "ck", tmp_path / "log")
    # checkpoint/log dirs also don't exist on disk
    targets = collect_targets(out_dir, [cfg],
                              clean_outputs=True,
                              clean_checkpoints=True,
                              clean_logs=True)
    assert targets == []


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

def test_remove_path_handles_file_and_dir(tmp_path):
    from scripts.cleanup_demo import remove_path
    f = tmp_path / "a.txt"
    f.write_text("hi")
    remove_path(f)
    assert not f.exists()

    d = tmp_path / "nested" / "deep"
    d.mkdir(parents=True)
    (d / "x.txt").write_text("y")
    remove_path(tmp_path / "nested")
    assert not (tmp_path / "nested").exists()


def test_is_safe_to_delete_rejects_project_root():
    from scripts.cleanup_demo import _is_safe_to_delete, PROJECT_ROOT
    assert _is_safe_to_delete(PROJECT_ROOT) is False
    assert _is_safe_to_delete(PROJECT_ROOT.parent) is False
    # Project drive root
    assert _is_safe_to_delete(Path(PROJECT_ROOT.anchor)) is False


def test_is_safe_to_delete_allows_demo_paths(tmp_path):
    from scripts.cleanup_demo import _is_safe_to_delete
    safe = tmp_path / "outputs" / "demo" / "emb_on_emb"
    safe.mkdir(parents=True)
    assert _is_safe_to_delete(safe) is True


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_cli_dry_run_does_not_delete(tmp_path, monkeypatch):
    """--dry-run lists targets but does not touch disk."""
    from scripts import cleanup_demo

    out_dir = tmp_path / "outputs" / "demo"
    _make_demo_output_dir(out_dir)
    cfg = tmp_path / "c.yaml"
    _make_config(cfg, tmp_path / "checkpoints", tmp_path / "logs")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "logs").mkdir()

    rc = cleanup_demo.main([
        "--output-dir", str(out_dir),
        "--base-config", str(cfg),
        "--config-embedded", str(tmp_path / "missing1.yaml"),
        "--config-not-embedded", str(tmp_path / "missing2.yaml"),
        "--dry-run",
        "--yes",
    ])
    assert rc == 0
    # Nothing was deleted
    assert (out_dir / "pipeline_state.json").exists()
    assert (tmp_path / "checkpoints").exists()
    assert (tmp_path / "logs").exists()


def test_cli_actually_deletes_with_yes(tmp_path):
    """Without --dry-run and with --yes, targets are removed."""
    from scripts import cleanup_demo

    out_dir = tmp_path / "outputs" / "demo"
    _make_demo_output_dir(out_dir)
    cfg = tmp_path / "c.yaml"
    _make_config(cfg, tmp_path / "checkpoints", tmp_path / "logs")
    (tmp_path / "checkpoints" / "best.pt").parent.mkdir(parents=True)
    (tmp_path / "checkpoints" / "best.pt").write_bytes(b"\x00")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "train.log").write_text("epoch=1")

    rc = cleanup_demo.main([
        "--output-dir", str(out_dir),
        "--base-config", str(cfg),
        "--config-embedded", str(tmp_path / "missing1.yaml"),
        "--config-not-embedded", str(tmp_path / "missing2.yaml"),
        "--yes",
    ])
    assert rc == 0
    # Output dir CONTENTS are gone but the directory itself can stay
    assert not (out_dir / "pipeline_state.json").exists()
    assert not (out_dir / "comparison_report.html").exists()
    assert not (out_dir / "emb_on_emb").exists()
    # Checkpoints and logs gone too
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "logs").exists()


def test_cli_keep_checkpoints(tmp_path):
    """--keep-checkpoints leaves the checkpoint dir intact."""
    from scripts import cleanup_demo

    out_dir = tmp_path / "outputs" / "demo"
    _make_demo_output_dir(out_dir)
    cfg = tmp_path / "c.yaml"
    _make_config(cfg, tmp_path / "ck", tmp_path / "lg")
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "best.pt").write_bytes(b"\x00")
    (tmp_path / "lg").mkdir()

    rc = cleanup_demo.main([
        "--output-dir", str(out_dir),
        "--base-config", str(cfg),
        "--config-embedded", str(tmp_path / "missing1.yaml"),
        "--config-not-embedded", str(tmp_path / "missing2.yaml"),
        "--keep-checkpoints",
        "--yes",
    ])
    assert rc == 0
    # Outputs cleaned
    assert not (out_dir / "pipeline_state.json").exists()
    # Checkpoint dir retained
    assert (tmp_path / "ck" / "best.pt").exists()
    # Logs gone
    assert not (tmp_path / "lg").exists()
