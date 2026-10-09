"""Panel actual repair: only clip-explained differences may be rescored."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "repair_oof_panel_actuals",
    Path(__file__).resolve().parents[1] / "scripts" / "repair_oof_panel_actuals.py")
repair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair)


def _panel():
    return pd.DataFrame({
        "player_id": list("abcd"), "season": 2025, "week": 1, "target_week": 2,
        "position": "WR", "predicted_points": [5.0, 6.0, 7.0, 8.0],
        "actual_points": [0.0, 10.0, 20.0, 30.0]})


def test_clipped_extremes_are_rescored_from_raw():
    raw = np.array([-1.5, 10.0, 20.0, 41.0])  # bottom and top were clipped
    fixed, summary = repair.repaired(_panel(), raw)
    assert list(fixed.actual_points) == list(raw)
    assert np.allclose(fixed.residual, fixed.predicted_points - raw)
    assert (summary["rows_repaired"], summary["repaired_high"], summary["repaired_low"]) == (2, 1, 1)


@pytest.mark.parametrize("raw", [
    [0.0, 12.0, 20.0, 30.0],   # interior row differs: not a clip
    [0.0, 10.0, 20.0, 25.0],   # top row's raw is BELOW the bound it sits at
])
def test_unexplained_differences_stop_the_repair(raw):
    with pytest.raises(ValueError, match="without being clipped"):
        repair.repaired(_panel(), np.array(raw))
