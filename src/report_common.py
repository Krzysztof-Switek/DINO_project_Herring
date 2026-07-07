"""Shared primitives for the HTML report generators.

Both report generators build self-contained HTML with base64-embedded PNGs:

  * ``src/report.py``            — single training-run report (report.html)
  * ``src/comparison_report.py`` — Embedded vs NotEmbedded report (comparison_report.html)

This module holds the pieces they have in common (figure/image → base64,
``<img>`` tag, regression metrics) so there is a single source of truth.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# base64 encoders
# ---------------------------------------------------------------------------

def fig_to_b64(fig: "plt.Figure", dpi: int = 110) -> str:
    """Render a matplotlib figure to a base64 PNG string and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def png_to_b64(path: str | Path) -> str | None:
    """Return the base64 of an existing PNG file, or None if it is missing."""
    path = Path(path)
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def pil_to_b64(path: str | Path, max_px: int = 300) -> str:
    """Load an image, downscale it to fit ``max_px``, return it as base64 PNG."""
    img = PILImage.open(path).convert("RGB")
    img.thumbnail((max_px, max_px), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def img_tag(b64: str, alt: str = "", style: str = "") -> str:
    """A responsive ``<img>`` tag for a base64 PNG (``max-width:100%`` always applied)."""
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;{style}">'


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Regression metrics for age prediction.

    Returns dict with keys: MAE, RMSE, R2, Acc1yr, Acc2yr, Bias.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    try:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except Exception:
        r2 = float("nan")
    acc1 = float(np.mean(np.abs(errors) <= 1.0))
    acc2 = float(np.mean(np.abs(errors) <= 2.0))
    bias = float(np.mean(errors))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Acc1yr": acc1, "Acc2yr": acc2, "Bias": bias}
