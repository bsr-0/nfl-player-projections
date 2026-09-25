import numpy as np
import pandas as pd

from scripts.run_plan_a_improvements import _two_stage_position_prediction


def test_two_stage_rushing_returns_bounded_position_predictions():
    rng = np.random.default_rng(42)
    n_train = 80
    n_test = 8
    train = pd.DataFrame(rng.normal(size=(n_train, 3)), columns=["role", "accel", "team_rush"])
    test = pd.DataFrame(rng.normal(size=(n_test, 3)), columns=train.columns)
    positions_train = np.array(["RB"] * 40 + ["QB"] * 40)
    positions_test = np.array(["RB"] * 4 + ["QB"] * 4)
    y = np.r_[np.where(np.arange(40) % 2, 0.0, 0.2), np.where(np.arange(40) % 2, 0.0, 0.1)]

    pred = _two_stage_position_prediction(
        train, test, y, positions_train, positions_test, "ridge"
    )

    assert pred.shape == (n_test,)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()
    assert (pred <= 1).all()
