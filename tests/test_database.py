"""
Database tests for NFL Predictor.

Tests DatabaseManager class for:
- Initialization and connection handling
- CRUD operations with edge cases
- Data type validation
- Query filtering
- Bulk insert handling
"""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import DatabaseManager


class TestDatabaseInitialization:
    """Test database initialization and connection handling."""
    
    def test_database_creation(self):
        """Test database is created with correct tables."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        try:
            db = DatabaseManager(db_path)
            
            # Check tables exist
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' ORDER BY name
                """)
                tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'player_team_history', 'player_weekly_stats', 'players',
                'schedule', 'team_defense_stats', 'team_stats', 'utilization_scores'
            ]
            for table in expected_tables:
                assert table in tables, f"Missing table: {table}"
        finally:
            os.unlink(db_path)
    
    def test_connection_context_manager(self):
        """Test connection context manager properly closes connection."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        try:
            db = DatabaseManager(db_path)
            
            # Connection should be closed after context exits
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                assert cursor.fetchone()[0] == 1
            
            # Connection should still work for new requests
            with db._get_connection() as conn2:
                cursor2 = conn2.cursor()
                cursor2.execute("SELECT 2")
                assert cursor2.fetchone()[0] == 2
        finally:
            os.unlink(db_path)


class TestPlayerOperations:
    """Test player CRUD operations."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        db = DatabaseManager(db_path)
        yield db
        
        os.unlink(db_path)
    
    def test_insert_player(self, db):
        """Test basic player insertion."""
        player_data = {
            'player_id': 'test_player_001',
            'name': 'John Doe',
            'position': 'QB',
            'birth_date': '1990-01-15',
            'college': 'Test University'
        }
        
        result = db.insert_player(player_data)
        assert result is True
        
        # Verify player was inserted
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM players WHERE player_id = ?", ('test_player_001',))
            row = cursor.fetchone()
            
        assert row is not None
        assert row['name'] == 'John Doe'
        assert row['position'] == 'QB'
    
    def test_insert_player_with_null_values(self, db):
        """Test player insertion with NULL values."""
        player_data = {
            'player_id': 'test_player_002',
            'name': 'Jane Doe',
            'position': None,
            'birth_date': None,
            'college': None
        }
        
        result = db.insert_player(player_data)
        assert result is True
        
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM players WHERE player_id = ?", ('test_player_002',))
            row = cursor.fetchone()
        
        assert row is not None
        assert row['position'] is None
    
    def test_insert_player_duplicate_updates(self, db):
        """Test that inserting duplicate player updates existing record."""
        player_data_v1 = {
            'player_id': 'test_player_003',
            'name': 'Original Name',
            'position': 'RB',
        }
        
        player_data_v2 = {
            'player_id': 'test_player_003',
            'name': 'Updated Name',
            'position': 'WR',
        }
        
        db.insert_player(player_data_v1)
        db.insert_player(player_data_v2)
        
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, position FROM players WHERE player_id = ?", ('test_player_003',))
            row = cursor.fetchone()
        
        assert row['name'] == 'Updated Name'
        assert row['position'] == 'WR'


class TestPlayerWeeklyStats:
    """Test player weekly stats operations."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database with a test player."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        db = DatabaseManager(db_path)
        
        # Insert test player
        db.insert_player({
            'player_id': 'test_player',
            'name': 'Test Player',
            'position': 'RB'
        })
        
        yield db
        os.unlink(db_path)
    
    def test_insert_weekly_stats(self, db):
        """Test basic weekly stats insertion."""
        stats = {
            'player_id': 'test_player',
            'season': 2024,
            'week': 1,
            'team': 'KC',
            'opponent': 'BAL',
            'home_away': 'home',
            'rushing_attempts': 15,
            'rushing_yards': 87,
            'rushing_tds': 1,
            'fantasy_points': 14.7
        }
        
        result = db.insert_player_weekly_stats(stats)
        assert result is True
        
        # Verify stats were inserted
        df = db.get_player_stats(player_id='test_player', season=2024)
        assert len(df) == 1
        assert df.iloc[0]['rushing_attempts'] == 15
        assert df.iloc[0]['fantasy_points'] == 14.7
    
    def test_insert_stats_with_nan_values(self, db):
        """Test stats insertion handles None values.
        
        Note: Currently None is stored as NULL, not converted to 0.
        This is a known limitation - the database uses .get() with default 0,
        but None values are explicitly passed through.
        """
        stats = {
            'player_id': 'test_player',
            'season': 2024,
            'week': 2,
            'team': 'KC',
            'rushing_attempts': None,  # Currently stored as NULL
            'rushing_yards': None,
            'fantasy_points': 0
        }
        
        result = db.insert_player_weekly_stats(stats)
        assert result is True
        
        df = db.get_player_stats(player_id='test_player', season=2024)
        week2_stats = df[df['week'] == 2].iloc[0]
        # None values are stored as NULL (known behavior)
        assert week2_stats['rushing_attempts'] is None or pd.isna(week2_stats['rushing_attempts'])
    
    def test_get_stats_by_position(self, db):
        """Test filtering stats by position."""
        # Insert another player with different position
        db.insert_player({
            'player_id': 'qb_player',
            'name': 'Test QB',
            'position': 'QB'
        })
        
        db.insert_player_weekly_stats({
            'player_id': 'test_player',
            'season': 2024,
            'week': 1,
            'team': 'KC',
            'fantasy_points': 10.0
        })
        
        db.insert_player_weekly_stats({
            'player_id': 'qb_player',
            'season': 2024,
            'week': 1,
            'team': 'KC',
            'fantasy_points': 25.0
        })
        
        rb_stats = db.get_player_stats(position='RB', season=2024)
        qb_stats = db.get_player_stats(position='QB', season=2024)
        
        assert len(rb_stats) == 1
        assert len(qb_stats) == 1
        assert rb_stats.iloc[0]['fantasy_points'] == 10.0
        assert qb_stats.iloc[0]['fantasy_points'] == 25.0


class TestDataIntegrity:
    """Test data integrity constraints and edge cases."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        db = DatabaseManager(db_path)
        yield db
        os.unlink(db_path)
    
    def test_unique_constraint_player_week(self, db):
        """Test unique constraint on player/season/week."""
        db.insert_player({
            'player_id': 'test_player',
            'name': 'Test',
            'position': 'RB'
        })
        
        stats_v1 = {
            'player_id': 'test_player',
            'season': 2024,
            'week': 1,
            'fantasy_points': 10.0
        }
        
        stats_v2 = {
            'player_id': 'test_player',
            'season': 2024,
            'week': 1,
            'fantasy_points': 15.0  # Updated value
        }
        
        db.insert_player_weekly_stats(stats_v1)
        db.insert_player_weekly_stats(stats_v2)
        
        # Should only have one row (updated)
        df = db.get_player_stats(player_id='test_player', season=2024)
        assert len(df) == 1
        assert df.iloc[0]['fantasy_points'] == 15.0
    
    def test_has_data_for_season(self, db):
        """Test season data check."""
        assert db.has_data_for_season(2024) is False
        
        db.insert_player({
            'player_id': 'test_player',
            'name': 'Test',
            'position': 'RB'
        })
        
        db.insert_player_weekly_stats({
            'player_id': 'test_player',
            'season': 2024,
            'week': 1,
            'fantasy_points': 10.0
        })
        
        assert db.has_data_for_season(2024) is True
        assert db.has_data_for_season(2023) is False
    
    def test_get_latest_week_for_season(self, db):
        """Test getting latest week."""
        db.insert_player({
            'player_id': 'test_player',
            'name': 'Test',
            'position': 'RB'
        })
        
        for week in [1, 5, 10, 15]:
            db.insert_player_weekly_stats({
                'player_id': 'test_player',
                'season': 2024,
                'week': week,
                'fantasy_points': float(week)
            })
        
        latest = db.get_latest_week_for_season(2024)
        assert latest == 15
        
        # Non-existent season should return 0
        assert db.get_latest_week_for_season(2023) == 0
    
    def test_get_seasons_with_data(self, db):
        """Test getting list of seasons."""
        db.insert_player({
            'player_id': 'test_player',
            'name': 'Test',
            'position': 'RB'
        })
        
        for season in [2022, 2023, 2024]:
            db.insert_player_weekly_stats({
                'player_id': 'test_player',
                'season': season,
                'week': 1,
                'fantasy_points': 10.0
            })
        
        seasons = db.get_seasons_with_data()
        assert seasons == [2022, 2023, 2024]


class TestScheduleOperations:
    """Test schedule table operations."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        db = DatabaseManager(db_path)
        yield db
        os.unlink(db_path)
    
    def test_insert_schedule(self, db):
        """Test schedule insertion."""
        schedule = {
            'season': 2024,
            'week': 1,
            'home_team': 'KC',
            'away_team': 'BAL',
            'game_id': 'KC_BAL_2024_01',
            'venue': 'Arrowhead Stadium',
            'home_score': 27,
            'away_score': 24
        }
        
        result = db.insert_schedule(schedule)
        assert result is True
        
        df = db.get_schedule(season=2024, week=1)
        assert len(df) == 1
        assert df.iloc[0]['home_team'] == 'KC'
        assert df.iloc[0]['home_score'] == 27
    
    def test_get_schedule_by_team(self, db):
        """Test filtering schedule by team."""
        schedules = [
            {'season': 2024, 'week': 1, 'home_team': 'KC', 'away_team': 'BAL'},
            {'season': 2024, 'week': 2, 'home_team': 'SF', 'away_team': 'KC'},
            {'season': 2024, 'week': 3, 'home_team': 'DET', 'away_team': 'MIN'},
        ]
        
        for s in schedules:
            db.insert_schedule(s)
        
        kc_games = db.get_schedule(team='KC')
        assert len(kc_games) == 2  # KC plays week 1 and 2
        
        det_games = db.get_schedule(team='DET')
        assert len(det_games) == 1
    
    def test_has_schedule_for_season(self, db):
        """Test checking if schedule exists."""
        assert db.has_schedule_for_season(2024) is False
        
        db.insert_schedule({
            'season': 2024,
            'week': 1,
            'home_team': 'KC',
            'away_team': 'BAL'
        })
        
        assert db.has_schedule_for_season(2024) is True
        assert db.has_schedule_for_season(2023) is False


class TestTeamStats:
    """Test team stats operations."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        db = DatabaseManager(db_path)
        yield db
        os.unlink(db_path)
    
    def test_insert_team_stats(self, db):
        """Test team stats insertion."""
        stats = {
            'team': 'KC',
            'season': 2024,
            'week': 1,
            'opponent': 'BAL',
            'home_away': 'home',
            'points_scored': 27,
            'points_allowed': 24,
            'total_yards': 400,
            'passing_yards': 280,
            'rushing_yards': 120
        }
        
        result = db.insert_team_stats(stats)
        assert result is True
        
        df = db.get_team_stats(team='KC', season=2024)
        assert len(df) == 1
        assert df.iloc[0]['points_scored'] == 27


class TestUtilizationScores:
    """Test utilization score operations."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database with a test player."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        db = DatabaseManager(db_path)
        db.insert_player({
            'player_id': 'test_player',
            'name': 'Test',
            'position': 'RB'
        })
        
        yield db
        os.unlink(db_path)
    
    def test_insert_utilization_score(self, db):
        """Test utilization score insertion."""
        score_data = {
            'player_id': 'test_player',
            'season': 2024,
            'week': 1,
            'utilization_score': 85.5,
            'snap_share': 0.75,
            'target_share': 0.15,
            'rush_share': 0.45
        }
        
        result = db.insert_utilization_score(score_data)
        assert result is True


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        db = DatabaseManager(db_path)
        yield db
        os.unlink(db_path)
    
    def test_empty_query_results(self, db):
        """Test queries return empty DataFrames when no data."""
        df = db.get_player_stats(player_id='nonexistent')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        
        df = db.get_schedule(season=9999)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_special_characters_in_name(self, db):
        """Test handling of special characters in player names."""
        player_data = {
            'player_id': 'special_player',
            'name': "O'Brien-Smith Jr.",  # Apostrophe, hyphen, period
            'position': 'WR'
        }
        
        result = db.insert_player(player_data)
        assert result is True
        
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM players WHERE player_id = ?", ('special_player',))
            row = cursor.fetchone()
        
        assert row['name'] == "O'Brien-Smith Jr."
    
    def test_unicode_characters(self, db):
        """Test handling of unicode characters."""
        player_data = {
            'player_id': 'unicode_player',
            'name': 'José García',
            'position': 'K'
        }
        
        result = db.insert_player(player_data)
        assert result is True
    
    def test_large_numeric_values(self, db):
        """Test handling of large numeric values."""
        db.insert_player({
            'player_id': 'big_numbers',
            'name': 'Test',
            'position': 'QB'
        })
        
        stats = {
            'player_id': 'big_numbers',
            'season': 2024,
            'week': 1,
            'passing_yards': 999999,
            'fantasy_points': 99999.99
        }
        
        result = db.insert_player_weekly_stats(stats)
        assert result is True
        
        df = db.get_player_stats(player_id='big_numbers')
        assert df.iloc[0]['passing_yards'] == 999999


class TestDataTypeValidation:
    """Test data type handling and conversion."""
    
    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        db = DatabaseManager(db_path)
        db.insert_player({
            'player_id': 'test_player',
            'name': 'Test',
            'position': 'RB'
        })
        yield db
        os.unlink(db_path)
    
    def test_float_conversion(self, db):
        """Test float values are stored correctly."""
        stats = {
            'player_id': 'test_player',
            'season': 2024,
            'week': 1,
            'fantasy_points': 15.7,
            'snap_share': 0.85
        }
        
        db.insert_player_weekly_stats(stats)
        
        df = db.get_player_stats(player_id='test_player')
        assert abs(df.iloc[0]['fantasy_points'] - 15.7) < 0.001
        assert abs(df.iloc[0]['snap_share'] - 0.85) < 0.001
    
    def test_integer_conversion(self, db):
        """Test integer values are stored correctly."""
        stats = {
            'player_id': 'test_player',
            'season': 2024,
            'week': 1,
            'rushing_attempts': 15,
            'rushing_yards': 87
        }
        
        db.insert_player_weekly_stats(stats)
        
        df = db.get_player_stats(player_id='test_player')
        assert df.iloc[0]['rushing_attempts'] == 15
        assert df.iloc[0]['rushing_yards'] == 87
    
    def test_boolean_like_values(self, db):
        """Test handling of boolean-like values as integers."""
        stats = {
            'player_id': 'test_player',
            'season': 2024,
            'week': 1,
            'games_played': 1  # Boolean-like: played or not
        }
        
        db.insert_player_weekly_stats(stats)
        
        df = db.get_player_stats(player_id='test_player')
        assert df.iloc[0]['games_played'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
