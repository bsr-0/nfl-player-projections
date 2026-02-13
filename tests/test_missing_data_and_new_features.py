"""
Rigorous tests for missing data handling and new features (injury/rookie).

Ensures:
- No NaN or inf in model inputs after prepare_training_data / feature creation.
- New features (injury_score, is_injured, is_rookie) exist with safe defaults when missing.
- Pipelines are robust when team_stats / utilization / schedule data are absent.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_data_with_nulls():
    """Sample player data that can produce NaNs (e.g. from LEFT JOINs)."""
    return pd.DataFrame({
        "player_id": ["p1", "p1", "p2", "p2"],
        "name": ["P1", "P1", "P2", "P2"],
        "position": ["RB", "RB", "WR", "WR"],
        "team": ["KC", "KC", "SF", "SF"],
        "season": [2024, 2024, 2024, 2024],
        "week": [1, 2, 1, 2],
        "opponent": ["BAL", "LV", "ARI", "LAR"],
        "home_away": ["home", "away", "away", "home"],
        "rushing_attempts": [15, 18, 0, 0],
        "rushing_yards": [72, 85, 0, 0],
        "rushing_tds": [0, 1, 0, 0],
        "targets": [3, 4, 9, 8],
        "receptions": [2, 3, 6, 5],
        "receiving_yards": [18, 25, 78, 62],
        "receiving_tds": [0, 0, 1, 0],
        "passing_attempts": [0, 0, 0, 0],
        "passing_yards": [0, 0, 0, 0],
        "fantasy_points": [12.0, 18.5, 16.0, 14.2],
        "snap_count": [45, 48, 52, 50],
    })


def test_create_features_produces_no_nan_or_inf_in_numeric_columns(sample_data_with_nulls):
    """After create_features, no numeric column should contain NaN or inf."""
    eng = FeatureEngineer()
    result = eng.create_features(sample_data_with_nulls, include_target=False)
    numeric = result.select_dtypes(include=[np.number])
    for col in numeric.columns:
        assert not result[col].isna().any(), f"Column {col} has NaN"
        assert not np.isinf(result[col]).any(), f"Column {col} has inf"


def test_prepare_training_data_returns_clean_X_y(sample_data_with_nulls):
    """prepare_training_data must return X with no NaN and no inf."""
    eng = FeatureEngineer()
    df = eng.create_features(sample_data_with_nulls, include_target=True)
    X, y = eng.prepare_training_data(df, target_weeks=1)
    assert X is not None and len(X) > 0
    assert not X.isna().any().any(), "X contains NaN"
    assert not np.isinf(X.values).any(), "X contains inf"
    assert not y.isna().any(), "y contains NaN"


def test_injury_rookie_features_exist_with_defaults_when_missing(sample_data_with_nulls):
    """injury_score, is_injured, is_rookie must exist and have safe defaults when not provided."""
    eng = FeatureEngineer()
    # Sample has no injury_score, is_injured, games_count
    result = eng.create_features(sample_data_with_nulls, include_target=False)
    assert "injury_score" in result.columns
    assert "is_injured" in result.columns
    assert "is_rookie" in result.columns
    assert result["injury_score"].between(0, 1).all()
    assert result["is_injured"].isin([0, 1]).all()
    assert result["is_rookie"].isin([0, 1]).all()
    assert result["injury_score"].notna().all()
    assert result["is_injured"].notna().all()
    assert result["is_rookie"].notna().all()


def test_injury_rookie_features_preserve_valid_input():
    """When injury_score/is_injured are provided, they are clipped/filled but not overwritten arbitrarily."""
    eng = FeatureEngineer()
    df = pd.DataFrame({
        "player_id": ["p1", "p2"],
        "name": ["A", "B"],
        "position": ["RB", "WR"],
        "team": ["KC", "SF"],
        "season": [2024, 2024],
        "week": [1, 1],
        "opponent": ["BAL", "ARI"],
        "home_away": ["home", "away"],
        "rushing_attempts": [10, 0],
        "rushing_yards": [40, 0],
        "rushing_tds": [0, 0],
        "targets": [2, 8],
        "receptions": [1, 5],
        "receiving_yards": [10, 60],
        "receiving_tds": [0, 0],
        "passing_attempts": [0, 0],
        "passing_yards": [0, 0],
        "fantasy_points": [8.0, 14.0],
        "snap_count": [30, 45],
        "injury_score": [0.5, 1.0],
        "is_injured": [1, 0],
    })
    result = eng.create_features(df, include_target=False)
    assert result["injury_score"].tolist()[0] == 0.5
    assert result["is_injured"].tolist()[0] == 1


def test_impute_missing_removes_inf():
    """_impute_missing replaces inf with finite values."""
    eng = FeatureEngineer()
    df = pd.DataFrame({
        "player_id": ["p1"],
        "a": [1.0],
        "b": [np.inf],
        "c": [-np.inf],
    })
    out = eng._impute_missing(df)
    assert not np.isinf(out["b"]).any()
    assert not np.isinf(out["c"]).any()


def test_impute_missing_fills_nan():
    """_impute_missing fills NaN in numeric columns."""
    eng = FeatureEngineer()
    df = pd.DataFrame({
        "player_id": ["p1", "p2"],
        "x": [1.0, np.nan],
        "y": [np.nan, 2.0],
    })
    out = eng._impute_missing(df)
    assert out["x"].notna().all()
    assert out["y"].notna().all()


def test_refresh_matchup_features_handles_missing_with_defaults():
    """refresh_matchup_features fills team_sos, matchup_difficulty, opponent_rating when missing."""
    eng = FeatureEngineer()
    df = pd.DataFrame({
        "team": ["KC"],
        "season": [2025],
        "week": [1],
        "opponent": [""],
        "home_away": ["unknown"],
    })
    with patch.object(eng, "_add_schedule_features", side_effect=lambda x: x):
        with patch.object(eng, "_add_team_matchup_features", side_effect=lambda x: x):
            out = eng.refresh_matchup_features(df)
    for col in ("team_sos", "matchup_difficulty", "opponent_rating"):
        if col in out.columns:
            assert out[col].notna().all()
