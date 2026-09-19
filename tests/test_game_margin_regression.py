"""Phase-2 margin/total regression invariants: MarketLineBaseline predicts
the line exactly, the pick/accuracy computation matches hand-computed
expectations, and pushes / no-pick predictions are excluded rather than
silently counted as wrong.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.game_margin_backtester import _regression_metrics
from src.models.game_outcome.baseline import MarketLineBaseline


def test_market_line_baseline_predicts_the_line_exactly():
    X = pd.DataFrame({"spread_line": [3.0, -7.5, 0.0, np.nan]})
    model = MarketLineBaseline("spread_line")
    preds = model.predict(X)
    assert np.array_equal(preds, X["spread_line"].to_numpy(), equal_nan=True)


def test_regression_metrics_mae_rmse_hand_computed():
    y_true = np.array([10.0, -5.0, 0.0, 20.0])
    y_pred = np.array([8.0, -5.0, 2.0, 15.0])
    line = np.zeros(4)
    m = _regression_metrics(y_true, y_pred, line, "ats_accuracy", 1)
    assert m["mae"] == pytest.approx(np.mean(np.abs(y_true - y_pred)))
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def test_pick_accuracy_excludes_pushes_and_no_picks():
    # Row 0: predicted above line, actual above line -> correct pick.
    # Row 1: predicted above line, actual below line -> wrong pick.
    # Row 2: predicted == line -> no pick, excluded regardless of actual.
    # Row 3: predicted above line, actual == line (push) -> excluded.
    y_true = np.array([5.0, -5.0, 3.0, 0.0])
    y_pred = np.array([2.0, 2.0, 2.0, 2.0])
    line = np.array([0.0, 0.0, 2.0, 0.0])
    m = _regression_metrics(y_true, y_pred, line, "ats_accuracy", 1)
    assert m["n_picks"] == 2
    assert m["ats_accuracy"] == pytest.approx(0.5)  # 1 correct, 1 wrong out of the 2 real picks


def test_market_line_baseline_never_makes_a_pick_against_itself():
    # predicted == line for every row -> n_picks is always 0 for this arm.
    y_true = np.array([5.0, -3.0, 10.0])
    line = np.array([1.0, -1.0, 2.0])
    y_pred = line.copy()
    m = _regression_metrics(y_true, y_pred, line, "ou_accuracy", 1)
    assert m["n_picks"] == 0
    assert m["ou_accuracy"] != m["ou_accuracy"]  # NaN when there are no picks to score
