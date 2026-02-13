"""
Tests for data availability and train/test season selection.

Ensures:
- Train/test seasons are derived from actual DB data (player_weekly_stats), not schedule only.
- Test season always has data in DB (no 0 test records when DB is source of truth).
- When in-season, test season is current season; when not in-season, test = latest in DB.
"""

import pytest
from unittest.mock import MagicMock, patch


@patch("src.utils.nfl_calendar.current_season_has_weeks_played", return_value=False)
def test_get_train_test_seasons_uses_db_first(_mock_in_season):
    """When DB has seasons and not in-season, get_train_test_seasons uses latest in DB as test."""
    from src.utils.data_manager import DataManager

    with patch.object(DataManager, "get_available_seasons_from_db", return_value=[2020, 2021, 2022, 2023, 2024]):
        with patch.object(DataManager, "check_data_availability") as mock_check:
            dm = DataManager()
            train_seasons, test_season = dm.get_train_test_seasons()
            # DB said 2020-2024 only, not in-season -> test should be 2024, train 2020-2023
            assert test_season == 2024
            assert train_seasons == [2020, 2021, 2022, 2023]
            mock_check.assert_not_called()


@patch("src.utils.nfl_calendar.current_season_has_weeks_played", return_value=False)
def test_get_train_test_seasons_fallback_when_db_empty(_mock_in_season):
    """When DB has no seasons, fall back to check_data_availability (not in-season)."""
    from src.utils.data_manager import DataManager

    with patch.object(DataManager, "get_available_seasons_from_db", return_value=[]):
        with patch.object(
            DataManager,
            "check_data_availability",
            return_value={"available_seasons": [2022, 2023, 2024], "latest_season": 2024},
        ):
            dm = DataManager()
            train_seasons, test_season = dm.get_train_test_seasons()
            assert test_season == 2024
            assert 2024 not in train_seasons
            assert 2022 in train_seasons or 2023 in train_seasons


@patch("src.utils.nfl_calendar.get_current_nfl_season", return_value=2025)
@patch("src.utils.nfl_calendar.current_season_has_weeks_played", return_value=True)
def test_get_train_test_seasons_with_2025_in_db(_mock_in_season, _mock_current):
    """When in-season and DB has current season (2025), test season is forced to 2025."""
    from src.utils.data_manager import DataManager

    with patch.object(
        DataManager, "get_available_seasons_from_db", return_value=[2020, 2021, 2022, 2023, 2024, 2025]
    ):
        dm = DataManager()
        train_seasons, test_season = dm.get_train_test_seasons()
        assert test_season == 2025
        assert train_seasons == [2020, 2021, 2022, 2023, 2024]
        assert 2025 not in train_seasons


@patch("src.utils.nfl_calendar.get_current_nfl_season", return_value=2025)
@patch("src.utils.nfl_calendar.current_season_has_weeks_played", return_value=True)
def test_get_train_test_seasons_raises_when_in_season_but_current_not_in_db(_mock_in_season, _mock_current):
    """When in-season (e.g. 2025) but current season not in DB, get_train_test_seasons raises."""
    from src.utils.data_manager import DataManager

    with patch.object(
        DataManager, "get_available_seasons_from_db", return_value=[2020, 2021, 2022, 2023, 2024]
    ):
        dm = DataManager()
        with pytest.raises(ValueError) as exc_info:
            dm.get_train_test_seasons()
        assert "2025" in str(exc_info.value)
        assert "play-by-play" in str(exc_info.value).lower() or "auto_refresh" in str(exc_info.value).lower()
