"""
Tests for eligible player filtering.

Ensures retired/inactive players are excluded from predictions
while active players are retained.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import DatabaseManager


# Attempt to import nfl_data_loader; skip tests that need it if nfl_data_py is missing.
try:
    from src.data.nfl_data_loader import (
        filter_to_eligible_players,
        get_eligible_seasons,
    )
    _HAS_NFL_DATA_LOADER = True
except ImportError:
    _HAS_NFL_DATA_LOADER = False

needs_nfl_data_loader = pytest.mark.skipif(
    not _HAS_NFL_DATA_LOADER,
    reason="nfl_data_py not installed",
)


@pytest.fixture
def db_with_players():
    """Create a temporary database with active and retired players."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    db = DatabaseManager(db_path)

    # Active player: has stats in 2024 and 2025
    db.insert_player({"player_id": "active_1", "name": "Patrick Mahomes", "position": "QB"})
    for season in (2023, 2024, 2025):
        for week in range(1, 4):
            db.insert_player_weekly_stats({
                "player_id": "active_1", "season": season, "week": week,
                "team": "KC", "fantasy_points": 20.0,
            })

    # Active player: has stats only in 2025
    db.insert_player({"player_id": "active_2", "name": "Bijan Robinson", "position": "RB"})
    for week in range(1, 4):
        db.insert_player_weekly_stats({
            "player_id": "active_2", "season": 2025, "week": week,
            "team": "ATL", "fantasy_points": 15.0,
        })

    # Retired player: last played in 2022 (e.g. Rob Gronkowski)
    db.insert_player({"player_id": "retired_1", "name": "R.Gronkowski", "position": "TE"})
    for season in (2020, 2021, 2022):
        for week in range(1, 4):
            db.insert_player_weekly_stats({
                "player_id": "retired_1", "season": season, "week": week,
                "team": "TB", "fantasy_points": 12.0,
            })

    # Retired player: last played in 2023
    db.insert_player({"player_id": "retired_2", "name": "Tom Brady", "position": "QB"})
    for week in range(1, 4):
        db.insert_player_weekly_stats({
            "player_id": "retired_2", "season": 2023, "week": week,
            "team": "TB", "fantasy_points": 18.0,
        })

    yield db, db_path

    os.unlink(db_path)


class TestGetEligiblePlayerIds:
    """Test DatabaseManager.get_eligible_player_ids."""

    def test_returns_players_in_specified_seasons(self, db_with_players):
        db, _ = db_with_players
        eligible = db.get_eligible_player_ids([2024, 2025])
        assert "active_1" in eligible
        assert "active_2" in eligible

    def test_excludes_retired_players(self, db_with_players):
        db, _ = db_with_players
        eligible = db.get_eligible_player_ids([2024, 2025])
        assert "retired_1" not in eligible  # Last played 2022

    def test_includes_recently_retired_if_in_window(self, db_with_players):
        db, _ = db_with_players
        # With a wider window that includes 2023, Brady should be included
        eligible = db.get_eligible_player_ids([2023, 2024, 2025])
        assert "retired_2" in eligible  # Brady last played 2023

    def test_excludes_recently_retired_outside_window(self, db_with_players):
        db, _ = db_with_players
        # With a narrow window of only 2024-2025, Brady should be excluded
        eligible = db.get_eligible_player_ids([2024, 2025])
        assert "retired_2" not in eligible  # Brady last played 2023

    def test_empty_seasons_returns_empty(self, db_with_players):
        db, _ = db_with_players
        eligible = db.get_eligible_player_ids([])
        assert eligible == []

    def test_future_season_returns_empty(self, db_with_players):
        db, _ = db_with_players
        eligible = db.get_eligible_player_ids([2030])
        assert eligible == []


@needs_nfl_data_loader
class TestFilterToEligiblePlayers:
    """Test the filter_to_eligible_players function."""

    def test_filters_retired_from_dataframe(self, db_with_players):
        db, _ = db_with_players
        eligible_ids = db.get_eligible_player_ids([2024, 2025])

        df = pd.DataFrame({
            "player_id": ["active_1", "active_2", "retired_1", "retired_2"],
            "name": ["Mahomes", "Robinson", "Gronkowski", "Brady"],
            "position": ["QB", "RB", "TE", "QB"],
        })

        filtered = filter_to_eligible_players(df, eligible_player_ids=eligible_ids)

        assert len(filtered) == 2
        assert "active_1" in filtered["player_id"].values
        assert "active_2" in filtered["player_id"].values
        assert "retired_1" not in filtered["player_id"].values
        assert "retired_2" not in filtered["player_id"].values

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["player_id", "name"])
        result = filter_to_eligible_players(df, eligible_player_ids=["x"])
        assert result.empty

    def test_no_player_id_column_returns_unchanged(self):
        df = pd.DataFrame({"name": ["Test"], "position": ["QB"]})
        result = filter_to_eligible_players(df, eligible_player_ids=["x"])
        assert len(result) == 1  # Unchanged, no player_id column

    def test_empty_eligible_list_returns_all(self):
        """Safety: if no eligible IDs (empty DB), don't filter everything out."""
        df = pd.DataFrame({
            "player_id": ["a", "b"],
            "name": ["Player A", "Player B"],
        })
        result = filter_to_eligible_players(df, eligible_player_ids=[])
        assert len(result) == 2


@needs_nfl_data_loader
class TestGetEligibleSeasons:
    """Test the get_eligible_seasons helper."""

    def test_returns_most_recent_seasons(self, db_with_players):
        db, db_path = db_with_players
        import src.data.nfl_data_loader as loader
        original_db_class = loader.DatabaseManager

        class PatchedDB(original_db_class):
            def __init__(self, *args, **kwargs):
                super().__init__(db_path)

        loader.DatabaseManager = PatchedDB
        try:
            seasons = get_eligible_seasons(lookback=2)
            # DB has seasons: 2020, 2021, 2022, 2023, 2024, 2025
            # Last 2 should be 2024, 2025
            assert seasons == [2024, 2025]
        finally:
            loader.DatabaseManager = original_db_class

    def test_lookback_of_3(self, db_with_players):
        db, db_path = db_with_players
        import src.data.nfl_data_loader as loader
        original_db_class = loader.DatabaseManager

        class PatchedDB(original_db_class):
            def __init__(self, *args, **kwargs):
                super().__init__(db_path)

        loader.DatabaseManager = PatchedDB
        try:
            seasons = get_eligible_seasons(lookback=3)
            assert seasons == [2023, 2024, 2025]
        finally:
            loader.DatabaseManager = original_db_class


class TestEligibleSeasonsSetting:
    """Test that the config setting exists and has sensible defaults."""

    def test_setting_exists(self):
        from config.settings import ELIGIBLE_SEASONS_LOOKBACK
        assert isinstance(ELIGIBLE_SEASONS_LOOKBACK, int)
        assert ELIGIBLE_SEASONS_LOOKBACK >= 1

    def test_default_lookback(self):
        from config.settings import ELIGIBLE_SEASONS_LOOKBACK
        assert ELIGIBLE_SEASONS_LOOKBACK == 2


# ---------------------------------------------------------------------------
# Roster-status-enhanced eligibility tests
# ---------------------------------------------------------------------------

@pytest.fixture
def db_with_roster_status():
    """Database with both game data and roster status entries."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    db = DatabaseManager(db_path)

    # Active player: game data in 2024 & 2025, roster status ACT
    db.insert_player({"player_id": "act_1", "name": "Josh Allen", "position": "QB"})
    for season in (2024, 2025):
        for week in range(1, 4):
            db.insert_player_weekly_stats({
                "player_id": "act_1", "season": season, "week": week,
                "team": "BUF", "fantasy_points": 22.0,
            })
        db.upsert_player_roster_status({
            "player_id": "act_1", "season": season,
            "status": "ACT", "team": "BUF", "position": "QB",
        })

    # Retired player with RET status but *has* recent game data (2025)
    # e.g. a player who played in 2025 then retired after the season
    db.insert_player({"player_id": "ret_recent", "name": "Derek Carr", "position": "QB"})
    for week in range(1, 4):
        db.insert_player_weekly_stats({
            "player_id": "ret_recent", "season": 2025, "week": week,
            "team": "NO", "fantasy_points": 14.0,
        })
    db.upsert_player_roster_status({
        "player_id": "ret_recent", "season": 2025,
        "status": "RET", "team": "NO", "position": "QB",
    })

    # Player with CUT status — still eligible (CUT != retired)
    db.insert_player({"player_id": "cut_1", "name": "Cut Player", "position": "WR"})
    for week in range(1, 3):
        db.insert_player_weekly_stats({
            "player_id": "cut_1", "season": 2025, "week": week,
            "team": "DAL", "fantasy_points": 8.0,
        })
    db.upsert_player_roster_status({
        "player_id": "cut_1", "season": 2025,
        "status": "CUT", "team": "DAL", "position": "WR",
    })

    # Player with no roster status entry but has game data (should still pass)
    db.insert_player({"player_id": "no_status", "name": "No Status", "position": "RB"})
    for week in range(1, 3):
        db.insert_player_weekly_stats({
            "player_id": "no_status", "season": 2025, "week": week,
            "team": "ATL", "fantasy_points": 10.0,
        })

    yield db, db_path

    os.unlink(db_path)


class TestRosterStatusDB:
    """Test roster status database operations."""

    def test_upsert_and_query(self, db_with_roster_status):
        db, _ = db_with_roster_status
        assert db.has_roster_status_data(2025)
        assert not db.has_roster_status_data(2030)

    def test_get_players_excludes_retired(self, db_with_roster_status):
        db, _ = db_with_roster_status
        eligible = db.get_players_with_roster_status([2025], exclude_statuses=["RET"])
        assert "act_1" in eligible
        assert "cut_1" in eligible
        assert "ret_recent" not in eligible  # RET excluded

    def test_get_players_without_exclusion(self, db_with_roster_status):
        db, _ = db_with_roster_status
        all_players = db.get_players_with_roster_status([2025])
        assert "act_1" in all_players
        assert "ret_recent" in all_players
        assert "cut_1" in all_players

    def test_empty_seasons(self, db_with_roster_status):
        db, _ = db_with_roster_status
        assert db.get_players_with_roster_status([]) == []


@needs_nfl_data_loader
class TestFilterWithRosterStatus:
    """Test that filter_to_eligible_players uses roster status when available."""

    def test_retired_player_excluded_despite_recent_game_data(self, db_with_roster_status):
        """A player with RET status should be excluded even with 2025 game data."""
        db, db_path = db_with_roster_status
        import src.data.nfl_data_loader as loader
        original_db_class = loader.DatabaseManager

        class PatchedDB(original_db_class):
            def __init__(self, *args, **kwargs):
                super().__init__(db_path)

        loader.DatabaseManager = PatchedDB
        try:
            df = pd.DataFrame({
                "player_id": ["act_1", "ret_recent", "cut_1", "no_status"],
                "name": ["Allen", "Carr", "Cut", "NoStatus"],
            })
            filtered = filter_to_eligible_players(df)

            assert "act_1" in filtered["player_id"].values
            assert "cut_1" in filtered["player_id"].values
            assert "no_status" in filtered["player_id"].values
            # ret_recent has game data in 2025 but RET roster status
            assert "ret_recent" not in filtered["player_id"].values
        finally:
            loader.DatabaseManager = original_db_class

    def test_no_roster_status_falls_back_to_game_data(self, db_with_players):
        """When no roster status data exists, fall back to game-data-only filtering."""
        db, db_path = db_with_players
        import src.data.nfl_data_loader as loader
        original_db_class = loader.DatabaseManager

        class PatchedDB(original_db_class):
            def __init__(self, *args, **kwargs):
                super().__init__(db_path)

        loader.DatabaseManager = PatchedDB
        try:
            df = pd.DataFrame({
                "player_id": ["active_1", "active_2", "retired_1"],
                "name": ["Mahomes", "Robinson", "Gronkowski"],
            })
            filtered = filter_to_eligible_players(df)

            # No roster status data → only game-data check applies
            assert "active_1" in filtered["player_id"].values
            assert "active_2" in filtered["player_id"].values
            assert "retired_1" not in filtered["player_id"].values  # No game data in 2024/2025
        finally:
            loader.DatabaseManager = original_db_class
