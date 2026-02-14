"""
Tests for FastAPI /api/predictions endpoint: schedule_by_horizon and schedule_available.

Verifies that the API checks schedule availability for all three time horizons (1w, 4w, 18w)
and returns schedule_by_horizon with keys "1", "4", "18" and correct booleans.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def _minimal_predictions_df():
    """Minimal parquet-like DataFrame for predictions endpoint."""
    return pd.DataFrame({
        "season": [2025],
        "week": [18],
        "player_id": ["p1"],
        "name": ["A.Brown"],
        "position": ["WR"],
        "team": ["DET"],
        "projection_1w": [70.0],
        "projection_4w": [280.0],
        "projection_18w": [1000.0],
        "predicted_points": [70.0],
    })


@patch("api.main.load_predictions_parquet")
def test_predictions_returns_schedule_by_horizon_keys(mock_load_parquet):
    """Response must include schedule_by_horizon with keys '1', '4', '18'."""
    mock_load_parquet.return_value = _minimal_predictions_df()

    def fake_get_next_n_nfl_weeks(_today, n):
        if n == 1:
            return [(2025, 18)]
        if n == 4:
            return [(2025, 18), (2025, 19), (2025, 20), (2025, 21)]
        if n == 18:
            return [(2025, 18 + i) for i in range(18)]  # 2025 only
        return [(2025, 18)]

    mock_db = MagicMock()
    mock_db.has_schedule_for_season.return_value = True
    mock_db.get_schedule.return_value = pd.DataFrame({
        "home_team": ["DET"],
        "away_team": ["GB"],
    })

    with patch("src.utils.nfl_calendar.get_next_n_nfl_weeks", side_effect=fake_get_next_n_nfl_weeks):
        with patch("src.utils.database.DatabaseManager", return_value=mock_db):
            from api.main import get_predictions
            data = get_predictions(position=None, name=None, horizon="1")
    assert "schedule_by_horizon" in data
    sh = data["schedule_by_horizon"]
    assert list(sh.keys()) == ["1", "4", "18"]
    assert data["schedule_available"] is True


@patch("api.main.load_predictions_parquet")
def test_predictions_schedule_by_horizon_false_when_future_season_missing(mock_load_parquet):
    """When 18-week horizon spans 2026 and 2026 has no schedule, schedule_by_horizon['18'] is False."""
    mock_load_parquet.return_value = _minimal_predictions_df()

    def fake_get_next_n_nfl_weeks(_today, n):
        if n == 1:
            return [(2025, 18)]
        if n == 4:
            return [(2025, 18), (2025, 19), (2025, 20), (2025, 21)]
        if n == 18:
            # Include 2026 so schedule check fails for 18w
            return [(2025, 18 + i) for i in range(5)] + [(2026, 1)] + [(2026, 2)] * 12
        return [(2025, 18)]

    def has_schedule(season):
        return season == 2025  # 2026 not released

    mock_db = MagicMock()
    mock_db.has_schedule_for_season.side_effect = has_schedule
    mock_db.get_schedule.return_value = pd.DataFrame({
        "home_team": ["DET"],
        "away_team": ["GB"],
    })

    with patch("src.utils.nfl_calendar.get_next_n_nfl_weeks", side_effect=fake_get_next_n_nfl_weeks):
        with patch("src.utils.database.DatabaseManager", return_value=mock_db):
            from api.main import get_predictions
            data = get_predictions(position=None, name=None, horizon="all")
    sh = data["schedule_by_horizon"]
    assert sh["1"] is True
    assert sh["4"] is True
    assert sh["18"] is False
    assert data["schedule_available"] is False
    assert "schedule_note" in data
    assert len(data["rows"]) >= 1


@patch("api.main.load_predictions_parquet")
def test_predictions_empty_parquet_returns_schedule_by_horizon_defaults(mock_load_parquet):
    """When parquet is empty or None, response still has schedule_by_horizon with all True (default)."""
    mock_load_parquet.return_value = None

    from api.main import get_predictions
    data = get_predictions(position=None, name=None, horizon=None)
    assert data["rows"] == []
    assert data["qb_target"] in ("util", "fp")
    assert data["schedule_by_horizon"] == {"1": True, "4": True, "18": True}


@patch("api.main.load_qb_target_choice")
@patch("api.main.load_predictions_parquet")
def test_predictions_includes_qb_target(mock_load_parquet, mock_qb_target):
    """Predictions payload should include current QB dependent-variable choice."""
    mock_load_parquet.return_value = _minimal_predictions_df()
    mock_qb_target.return_value = "fp"
    from api.main import get_predictions
    data = get_predictions(position=None, name=None, horizon="1")
    assert data["qb_target"] == "fp"
