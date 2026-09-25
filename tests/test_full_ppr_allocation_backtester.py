import numpy as np
import pandas as pd

from src.evaluation.full_ppr_allocation_backtester import _two_stage
from src.models.team_allocation.reconstruct import renormalize_shares


def test_two_stage_returns_bounded_predictions():
    x = pd.DataFrame({"f": np.linspace(0, 1, 80)})
    y = np.where(np.arange(80) % 3 == 0, 0.2, 0.0)
    pos = np.array(["RB"] * 80)
    pred = _two_stage(x.iloc[:60], x.iloc[60:], y[:60], pos[:60], pos[60:])
    assert len(pred) == 20
    assert np.isfinite(pred).all()
    assert (pred >= 0).all() and (pred <= 1).all()


def test_event_zero_fallback_does_not_invent_roster_shares():
    keys = pd.DataFrame({"team": ["A", "A", "B"], "season": [2024] * 3, "week": [1] * 3})
    out = renormalize_shares(np.zeros(3), keys, zero_fallback="zeros")
    assert np.all(out == 0)
