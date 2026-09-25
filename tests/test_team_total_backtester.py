from src.evaluation.team_total_backtester import _choose_blend
import numpy as np


def test_choose_blend_uses_only_calibration_predictions():
    actual = np.array([10.0] * 40)
    ridge = np.array([10.0] * 40)
    xgb = np.array([20.0] * 40)
    assert _choose_blend(actual, ridge, xgb) == 1.0
