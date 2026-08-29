"""util_tier must be derived from a utilization index, not fantasy points.

get_utilization_tier's thresholds are a 0-100 utilization score (>=80 Elite,
>=70 Strong, >=60 Average, >=50 Below Average, else Low). In component mode --
production -- ensemble.py assigns the FANTASY POINT prediction to
`predicted_utilization`, the same value as predicted_points. Tiering off that
column graded weekly points against index thresholds, so every player came
back "Low": a realistic weekly projection is 3-15 points, below every cut.
"""
import pandas as pd
import pytest

from src.features.utilization_score import UtilizationScoreCalculator


@pytest.fixture
def calc():
    return UtilizationScoreCalculator(weights=None)


def test_thresholds_are_a_0_100_index(calc):
    assert calc.get_utilization_tier(85, "WR") == "Elite"
    assert calc.get_utilization_tier(75, "WR") == "Strong"
    assert calc.get_utilization_tier(65, "WR") == "Average"
    assert calc.get_utilization_tier(55, "WR") == "Below Average"
    assert calc.get_utilization_tier(45, "WR") == "Low"


def test_weekly_point_values_all_collapse_to_low(calc):
    """The bug, stated as a test: plausible weekly fantasy points are all
    under 50, so tiering points can only ever return 'Low'."""
    for weekly_points in (3.0, 8.5, 12.0, 15.0, 22.0, 30.0):
        assert calc.get_utilization_tier(weekly_points, "WR") == "Low"


def test_tier_uses_utilization_score_not_points():
    """Regression: the real case that exposed it. A player with utilization
    78.0 is Strong, even though his points prediction (47.7) would read Low."""
    from src.predict import NFLPredictor  # import guarded: no model load here
    calc = UtilizationScoreCalculator(weights=None)
    row = {"utilization_score": 78.0, "predicted_points": 47.7,
           "predicted_utilization": 47.7, "position": "WR"}
    assert calc.get_utilization_tier(row["utilization_score"], row["position"]) == "Strong"
    assert calc.get_utilization_tier(row["predicted_utilization"], row["position"]) == "Low"


def test_predict_module_no_longer_tiers_off_predicted_utilization():
    """Guards the wiring, not just the arithmetic."""
    import inspect
    from src import predict as m
    src = inspect.getsource(m)
    block = src[src.index("Utilization tier"):src.index("Utilization tier") + 1600]
    assert 'results["utilization_score"]' in block or 'row["utilization_score"]' in block
    assert 'row.get("predicted_utilization"' not in block
