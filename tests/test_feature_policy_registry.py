import numpy as np
import pandas as pd
import pytest

from src.features.feature_policy_registry import FeaturePolicyRegistry
from src.features.feature_engineering import FeatureEngineer


def _minimal_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["p1", "p1", "p2", "p2"],
            "name": ["P1", "P1", "P2", "P2"],
            "position": ["RB", "RB", "WR", "WR"],
            "team": ["KC", "KC", "SF", "SF"],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 1, 2],
            "opponent": ["BUF", "BUF", "LAR", "LAR"],
            "home_away": ["home", "away", "home", "away"],
            "rushing_attempts": [10, 12, 1, 2],
            "rushing_yards": [45, 67, 4, 8],
            "rushing_tds": [0, 1, 0, 0],
            "targets": [4, np.nan, 7, np.nan],
            "receptions": [3, 4, 5, 6],
            "receiving_yards": [20, 30, 60, 68],
            "receiving_tds": [0, 0, 1, 0],
            "passing_attempts": [0, 0, 0, 0],
            "passing_completions": [0, 0, 0, 0],
            "passing_yards": [0, 0, 0, 0],
            "passing_tds": [0, 0, 0, 0],
            "interceptions": [0, 0, 0, 0],
            "snap_share": [0.6, np.nan, 0.8, np.nan],
            "utilization_score": [55.0, np.nan, 72.0, np.nan],
            "injury_score": [1.0, np.nan, 0.8, np.nan],
            "fantasy_points": [10.1, 14.2, 17.5, 15.0],
        }
    )


def test_policy_registry_sets_indicator_and_threshold_fail():
    df = pd.DataFrame(
        {
            "utilization_score": [np.nan, np.nan, 70.0, np.nan],
            "injury_score": [1.0, np.nan, 0.8, np.nan],
        }
    )
    registry = FeaturePolicyRegistry.from_config()

    with pytest.raises(ValueError):
        registry.apply(df, fail_on_threshold=True, context="policy_test")


def test_feature_engineering_policy_is_deterministic_with_partial_missing_data():
    engineer = FeatureEngineer(feature_mode="full")
    src = _minimal_df()

    first = engineer.create_features(src)
    second = engineer.create_features(src)

    pd.testing.assert_frame_equal(first.sort_index(axis=1), second.sort_index(axis=1), check_dtype=False)
    assert "utilization_score_imputed" in first.columns
    assert int(first["utilization_score_imputed"].sum()) == 2
    assert any(c.endswith("_imputed") for c in first.columns)
