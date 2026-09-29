"""main_density_lab.py — the report states the gate outcome that the seed results imply."""
from __future__ import annotations

import json

import pandas as pd
import pytest

import main_density_lab as mdl


def _seed(root, cache, arm, seed, matured_epoch, final_ratio, final_active):
    d = root / cache / f"{arm}_seed{seed}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({
        "arm": arm, "seed": seed, "matured_epoch": matured_epoch,
        "matured": matured_epoch is not None}), encoding="utf-8")
    pd.DataFrame([
        {"epoch": 0, "zero_ratio": 5.0, "active": 0.0, "max_logit": 0.1, "median_logit": 0.0, "sum_p_over_age": 700},
        {"epoch": 30, "zero_ratio": final_ratio, "active": final_active, "max_logit": -1.0,
         "median_logit": -9.0, "sum_p_over_age": 1.0}]).to_csv(d / "metrics.csv", index=False)
    (d / "head.pt").write_bytes(b"")


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(mdl, "LAB_ROOT", tmp_path / "lab")
    monkeypatch.setattr(mdl, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mdl, "LOG", tmp_path / "log.txt")
    monkeypatch.setattr(mdl, "SEEDS", [0, 1, 2, 3, 4])
    monkeypatch.setattr(mdl, "GATE_SEEDS", 4)
    return tmp_path / "lab"


def test_report_gate_and_verdicts(lab):
    for s in range(5):
        _seed(lab, "raw", "A3", s, 8 + s if s < 4 else None, 0.3, 4.0)       # 4/5 → TAK
        _seed(lab, "raw", "A0", s, None, 0.97, 0.0)                          # 0/5 → NIE
        _seed(lab, "wedge_b_best_age", "A0", s, None, 0.98, 0.0)             # control collapses
    mdl.write_report()
    out = json.loads((lab / "wyniki.json").read_text(encoding="utf-8"))
    arms = {(a["cache"], a["arm"]): a for a in out["arms"]}
    assert arms[("raw", "A3")]["gate"] == "TAK" and arms[("raw", "A3")]["median_matured_epoch"] == 9.5
    assert arms[("raw", "A0")]["gate"] == "NIE" and arms[("raw", "A0")]["collapsed_seeds"] == 5
    v = " ".join(out["verdicts"])
    assert "odtwarza zjawisko" in v
    assert "A3 (4/5" in v and "(≥ 4/5 ziaren)" in v
    assert "sam brak dryfu nie wystarcza" in v
    md = (lab / "WYNIKI.md").read_text(encoding="utf-8")
    assert "| raw | A3 |" in md and "head.pt" in md


def test_gate_pending_while_seeds_missing(lab):
    for s in range(2):
        _seed(lab, "raw", "A4", s, 10, 0.3, 3.0)
    mdl.write_report()
    out = json.loads((lab / "wyniki.json").read_text(encoding="utf-8"))
    assert out["arms"][0]["gate"] == "w toku"
