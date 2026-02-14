"""Tests for QB utilization->fantasy point conversion support."""

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.utilization_to_fp import UtilizationToFPConverter, train_utilization_to_fp_per_position


def test_qb_converter_can_fit_and_predict():
    n = 120
    rng = np.random.default_rng(42)
    util = rng.uniform(40, 90, size=n)
    df = pd.DataFrame(
        {
            "position": ["QB"] * n,
            "utilization_score": util,
            "completion_pct": rng.uniform(58, 72, size=n),
            "yards_per_attempt": rng.uniform(6.0, 9.0, size=n),
            "td_rate": rng.uniform(2.0, 8.0, size=n),
            "int_rate": rng.uniform(0.5, 3.5, size=n),
            "rushing_yards": rng.uniform(0, 80, size=n),
            "rushing_attempts": rng.uniform(0, 10, size=n),
            "fantasy_points": 0.22 * util + rng.normal(0, 2, size=n),
        }
    )
    conv = UtilizationToFPConverter("QB").fit(df, target_col="fantasy_points")
    assert conv.is_fitted
    preds = conv.predict(df["utilization_score"].values, efficiency_df=df)
    assert preds.shape[0] == n
    assert np.isfinite(preds).all()


def test_train_conversion_supports_explicit_qb_position():
    rng = np.random.default_rng(7)
    n = 80
    train_df = pd.DataFrame(
        {
            "position": ["QB"] * n + ["RB"] * n,
            "utilization_score": np.concatenate([rng.uniform(45, 90, size=n), rng.uniform(30, 85, size=n)]),
            "fantasy_points": np.concatenate([rng.uniform(10, 32, size=n), rng.uniform(5, 26, size=n)]),
        }
    )
    converters = train_utilization_to_fp_per_position(train_df, positions=["QB", "RB"])
    assert "QB" in converters
    assert "RB" in converters
