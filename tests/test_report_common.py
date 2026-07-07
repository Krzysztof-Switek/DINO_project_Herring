"""Tests for src/report_common.py and the report-generator consolidation."""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np


def test_comparison_reexports_shared_compute_metrics():
    """Both report generators must use the one shared compute_metrics."""
    from src.report_common import compute_metrics as shared
    from src.comparison_report import compute_metrics as reexported
    assert reexported is shared


def test_compute_metrics_values():
    from src.report_common import compute_metrics
    m = compute_metrics(np.array([1, 2, 3, 4, 5]), np.array([1, 3, 3, 5, 5]))
    assert abs(m["MAE"] - 0.4) < 1e-9
    assert m["RMSE"] >= m["MAE"]
    assert set(m) == {"MAE", "RMSE", "R2", "Acc1yr", "Acc2yr", "Bias"}


def test_fig_to_b64_is_png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.report_common import fig_to_b64
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    raw = base64.b64decode(fig_to_b64(fig))
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic bytes


def test_img_tag_and_png_to_b64(tmp_path):
    from src.report_common import img_tag, png_to_b64
    tag = img_tag("QQ==", style="width:50%;")
    assert 'src="data:image/png;base64,QQ=="' in tag
    assert "max-width:100%" in tag
    assert png_to_b64(tmp_path / "missing.png") is None


def test_report_build_html_smoke(tmp_path):
    """src/report.py must still assemble an HTML string after the refactor."""
    from src.report import build_html_report
    html = build_html_report(
        labels_csv=None, log_path=None, predictions_csv=None,
        heatmaps_dir=tmp_path, overlays_dir=tmp_path,
        cand_json_dir=tmp_path, cand_overlays_dir=tmp_path,
    )
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
