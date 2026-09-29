"""The quarter-adjustment A/B configs must differ in exactly one field.

That is the entire experiment: `06.08_attention_first` vs `09.08_isolate_a` already showed
+10.1 pp exact, but those runs also differed in `density_concentricity_weight` and in their
whole trajectory, so the number was not attributable to the flag. If these two configs ever
drift apart in a second field, the replacement experiment inherits the same defect.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFF = PROJECT_ROOT / "configs" / "config_quarter_ab_off.yaml"
ON = PROJECT_ROOT / "configs" / "config_quarter_ab_on.yaml"
FLAG = "data.quarter_age_adjustment_enabled"


def _flat(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for key, value in (d or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, path + "."))
        else:
            out[path] = value
    return out


@pytest.fixture(scope="module")
def arms():
    return _flat(yaml.safe_load(OFF.read_text(encoding="utf-8"))), \
        _flat(yaml.safe_load(ON.read_text(encoding="utf-8")))


def test_the_arms_differ_in_exactly_one_field(arms):
    off, on = arms
    diff = {k: (off.get(k), on.get(k)) for k in set(off) | set(on) if off.get(k) != on.get(k)}
    assert diff == {FLAG: (False, True)}, diff


def test_e9_is_off_in_both_arms(arms):
    """E9 was the contaminating second variable in the original pair; it must be off here."""
    for cfg in arms:
        assert cfg.get("model.density_concentricity_weight") == 0.0


def test_density_branch_is_off_in_both_arms(arms):
    """Gradient-isolated, so switching it off cannot move the age result — only the clock.

    It also removes the maturity gate, which otherwise picks `best.pt` from the
    density-mature era rather than the age optimum (`src/trainer.py:631`).
    """
    for cfg in arms:
        assert cfg.get("model.use_density_head") is False
        assert cfg.get("training.min_density_active") == 0.0


def test_both_arms_emit_what_the_analysis_needs(arms):
    for cfg in arms:
        assert cfg.get("inference.dump_coral_logits") is True
        assert cfg.get("inference.aggregate_per_fish") is True


def test_both_arms_load_as_a_valid_config():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import OtolithConfig
    for path in (OFF, ON):
        cfg = OtolithConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))
        assert cfg.model.head_type == "both"          # same as the two runs being replaced
        assert cfg.data.quarter_age_adjustment_campaigns == ["BITS1q", "BITS2q"]
