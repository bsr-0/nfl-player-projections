"""
Tests for schedule loading (nfl-data-py first, scraper fallback) and team stats aggregation.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_manager import DataManager


def test_ensure_schedule_loaded_uses_nfl_data_py_first():
    """When DB has no schedule, ensure_schedule_loaded tries NFLDataLoader.load_schedules first."""
    dm = DataManager()
    dm.db = MagicMock()
    dm.db.has_schedule_for_season.return_value = False

    with patch.object(dm, "get_prediction_season", return_value=2025):
        with patch("src.data.nfl_data_loader.NFLDataLoader") as MockLoader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_schedules.return_value = pd.DataFrame({
                "season": [2025], "week": [1], "home_team": ["KC"], "away_team": ["BAL"]
            })
            MockLoader.return_value = mock_loader_instance

            result = dm.ensure_schedule_loaded(2025)

    assert result is True
    mock_loader_instance.load_schedules.assert_called_once_with([2025], store_in_db=True)


def test_ensure_schedule_loaded_fallback_to_scraper_when_nfl_empty():
    """When nfl-data-py returns empty, ensure_schedule_loaded falls back to scraper."""
    dm = DataManager()
    dm.db = MagicMock()
    dm.db.has_schedule_for_season.return_value = False

    with patch.object(dm, "get_prediction_season", return_value=2030):
        with patch("src.data.nfl_data_loader.NFLDataLoader") as MockLoader:
            mock_loader_instance = MagicMock()
            mock_loader_instance.load_schedules.return_value = pd.DataFrame()
            MockLoader.return_value = mock_loader_instance

        with patch("src.scrapers.schedule_scraper.import_schedule_to_db", return_value=5) as mock_import:
            result = dm.ensure_schedule_loaded(2030)

    assert result is True
    mock_import.assert_called_once_with(2030)


def test_ensure_schedule_loaded_returns_true_when_already_in_db():
    """When schedule already in DB, ensure_schedule_loaded returns True without loading."""
    dm = DataManager()
    dm.db = MagicMock()
    dm.db.has_schedule_for_season.return_value = True

    with patch.object(dm, "get_prediction_season", return_value=2025):
        with patch("src.data.nfl_data_loader.NFLDataLoader") as MockLoader:
            result = dm.ensure_schedule_loaded(2025)

    assert result is True
    MockLoader.assert_not_called()


# -----------------------------------------------------------------------------
# Team stats aggregation from player_weekly_stats
# -----------------------------------------------------------------------------

def test_aggregate_team_stats_from_players_schema():
    """aggregate_team_stats_from_players returns DataFrame with expected columns."""
    import tempfile
    import os
    from src.utils.database import DatabaseManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    try:
        db = DatabaseManager(db_path)
        db.insert_player({"player_id": "p1", "name": "P1", "position": "QB"})
        db.insert_player_weekly_stats({
            "player_id": "p1", "season": 2024, "week": 1, "team": "KC", "opponent": "BAL", "home_away": "home",
            "passing_attempts": 35, "rushing_attempts": 2, "passing_yards": 280, "rushing_yards": 10,
        })
        agg = db.aggregate_team_stats_from_players(season=2024)
        assert not agg.empty
        required = ["team", "season", "week", "pass_attempts", "rush_attempts", "passing_yards", "rushing_yards", "total_yards", "total_plays"]
        for col in required:
            assert col in agg.columns, f"missing {col}"
        assert agg.iloc[0]["team"] == "KC"
        assert agg.iloc[0]["pass_attempts"] == 35
        assert agg.iloc[0]["rush_attempts"] == 2
    finally:
        os.unlink(db_path)


def test_ensure_team_stats_from_players_backfills_and_skips_existing():
    """ensure_team_stats_from_players inserts missing rows and does not overwrite existing."""
    import tempfile
    import os
    from src.utils.database import DatabaseManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    try:
        db = DatabaseManager(db_path)
        db.insert_player({"player_id": "p1", "name": "P1", "position": "RB"})
        db.insert_player_weekly_stats({
            "player_id": "p1", "season": 2024, "week": 1, "team": "KC", "opponent": "LV", "home_away": "away",
            "passing_attempts": 0, "rushing_attempts": 18, "passing_yards": 0, "rushing_yards": 95,
        })
        n = db.ensure_team_stats_from_players(season=2024)
        assert n >= 1
        ts = db.get_team_stats(team="KC", season=2024)
        assert len(ts) >= 1
        assert ts.iloc[0]["rush_attempts"] == 18
        # Run again: should not duplicate (existing key skipped)
        n2 = db.ensure_team_stats_from_players(season=2024)
        assert n2 == 0
    finally:
        os.unlink(db_path)
