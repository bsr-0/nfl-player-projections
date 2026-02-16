"""Causality tests for forward targets and history-based season-long features."""

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train import _create_horizon_targets
from src.features.season_long_features import GamesPlayedProjector


def test_horizon_targets_are_forward_looking():
    df = pd.DataFrame(
        {
            "player_id": ["p1"] * 5,
            "season": [2024] * 5,
            "week": [1, 2, 3, 4, 5],
            "fantasy_points": [10.0, 20.0, 30.0, 40.0, 50.0],
            "utilization_score": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = _create_horizon_targets(df, n_weeks=[1, 4])

    np.testing.assert_allclose(
        out["target_1w"].values[:-1],
        np.array([20.0, 30.0, 40.0, 50.0]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        out["target_4w"].values[:-1],
        np.array([140.0, 120.0, 90.0, 50.0]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        out["target_util_4w"].values[:-1],
        np.array([3.5, 4.0, 4.5, 5.0]),
        atol=1e-8,
    )
    assert np.isnan(out["target_1w"].iloc[-1])


def test_horizon_targets_respect_season_boundary():
    df = pd.DataFrame(
        {
            "player_id": ["p1", "p1", "p1"],
            "season": [2024, 2024, 2025],
            "week": [17, 18, 1],
            "fantasy_points": [10.0, 20.0, 30.0],
            "utilization_score": [50.0, 60.0, 70.0],
        }
    )
    out = _create_horizon_targets(df, n_weeks=[1, 4])

    # 2024 week 18 should not point to 2025 week 1.
    assert np.isnan(out.loc[(out["season"] == 2024) & (out["week"] == 18), "target_1w"]).all()
    # 2024 week 17 only has one future game in same season.
    assert out.loc[(out["season"] == 2024) & (out["week"] == 17), "target_4w"].iloc[0] == 20.0


def test_games_played_projection_is_causal_to_future_seasons():
    projector = GamesPlayedProjector()

    base = pd.DataFrame(
        {
            "player_id": ["p1"] * 5,
            "position": ["RB"] * 5,
            "season": [2023] * 5,
            "week": [1, 2, 3, 4, 5],
        }
    )
    future = pd.DataFrame(
        {
            "player_id": ["p1"] * 4,
            "position": ["RB"] * 4,
            "season": [2024] * 4,
            "week": [1, 2, 3, 4],
        }
    )

    out_base = projector.project_games(base.copy())
    out_extended = projector.project_games(pd.concat([base, future], ignore_index=True))
    out_extended_2023 = out_extended[out_extended["season"] == 2023].reset_index(drop=True)

    np.testing.assert_allclose(
        out_base["historical_gpg"].values,
        out_extended_2023["historical_gpg"].values,
        atol=1e-8,
    )
