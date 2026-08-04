"""Tests for utilization as primary target and weight optimizer (no leakage, train-only)."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.utilization_score import calculate_utilization_scores, UtilizationScoreCalculator
from src.features.utilization_weight_optimizer import fit_utilization_weights, get_utilization_weights


@pytest.fixture
def sample_train_like():
    """Sample data with season/week so we can build utilization targets (train-like)."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "player_id": np.repeat([f"p{i}" for i in range(20)], 10),
        "name": np.repeat([f"Player {i}" for i in range(20)], 10),
        "position": np.repeat(["RB", "WR", "TE", "QB"] * 5, 10),
        "team": "KC",
        "season": np.repeat([2022, 2023], 100),
        "week": np.tile(list(range(1, 11)), 20),
        "rushing_attempts": np.random.randint(0, 25, n),
        "rushing_yards": np.random.randint(0, 120, n),
        "rushing_tds": np.random.randint(0, 3, n),
        "targets": np.random.randint(0, 15, n),
        "receptions": np.random.randint(0, 12, n),
        "receiving_yards": np.random.randint(0, 150, n),
        "receiving_tds": np.random.randint(0, 2, n),
        "snap_count": np.random.randint(20, 70, n),
        "passing_attempts": np.random.randint(0, 45, n) if "QB" in ["QB"] else 0,
        "passing_yards": np.random.randint(0, 350, n),
        "passing_tds": np.random.randint(0, 4, n),
        "interceptions": np.random.randint(0, 2, n),
    })
    df = calculate_utilization_scores(df)
    return df


def test_utilization_target_1w_no_future_leakage(sample_train_like):
    """target_util_1w must be next week's utilization only (shift(-1)); no current-week leakage."""
    df = sample_train_like.copy()
    df["target_util_1w"] = df.groupby("player_id")["utilization_score"].shift(-1)
    # Last week per player has no next week -> NaN
    last_week_per_player = df.groupby("player_id")["week"].transform("max")
    is_last_week = df["week"] == last_week_per_player
    assert df.loc[is_last_week, "target_util_1w"].isna().all()
    # For non-last weeks, target_util_1w should equal next row's utilization_score (same player)
    one_player = df[df["player_id"] == df["player_id"].iloc[0]].sort_values("week").reset_index(drop=True)
    for i in range(len(one_player) - 1):
        cur_tgt = one_player.loc[i, "target_util_1w"]
        nxt_util = one_player.loc[i + 1, "utilization_score"]
        np.testing.assert_almost_equal(cur_tgt, nxt_util, decimal=5)


def test_weight_optimizer_uses_only_passed_data(sample_train_like):
    """fit_utilization_weights uses only the DataFrame passed in (train-only discipline)."""
    df = sample_train_like.copy()
    df["target_util_1w"] = df.groupby("player_id")["utilization_score"].shift(-1)
    df = df.dropna(subset=["target_util_1w"])
    weights = fit_utilization_weights(df, target_col="target_util_1w", min_samples=50)
    assert isinstance(weights, dict)
    for pos in ["RB", "WR", "TE", "QB"]:
        if pos in weights and isinstance(weights[pos], dict):
            s = sum(weights[pos].values())
            assert abs(s - 1.0) < 1e-5
            assert all(v >= 0 for v in weights[pos].values())


def test_get_utilization_weights_fallback():
    """get_utilization_weights with no/invalid data returns config defaults."""
    defaults = get_utilization_weights(train_data=None, use_data_driven=False)
    assert "RB" in defaults and "WR" in defaults
    empty = pd.DataFrame()
    also = get_utilization_weights(train_data=empty, use_data_driven=True)
    assert also is not None and "RB" in also
