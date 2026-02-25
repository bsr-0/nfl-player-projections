"""Database management for NFL data storage."""
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from contextlib import contextmanager

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DB_PATH


class DatabaseManager:
    """Manages SQLite database for NFL statistics."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Player info table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    position TEXT,
                    birth_date TEXT,
                    college TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Player weekly stats
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_weekly_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    team TEXT,
                    opponent TEXT,
                    home_away TEXT,
                    games_played INTEGER DEFAULT 1,
                    passing_attempts INTEGER DEFAULT 0,
                    passing_completions INTEGER DEFAULT 0,
                    passing_yards INTEGER DEFAULT 0,
                    passing_tds INTEGER DEFAULT 0,
                    interceptions INTEGER DEFAULT 0,
                    rushing_attempts INTEGER DEFAULT 0,
                    rush_inside_10 INTEGER DEFAULT 0,
                    rush_inside_5 INTEGER DEFAULT 0,
                    rushing_yards INTEGER DEFAULT 0,
                    rushing_tds INTEGER DEFAULT 0,
                    targets INTEGER DEFAULT 0,
                    targets_15_plus INTEGER DEFAULT 0,
                    receptions INTEGER DEFAULT 0,
                    air_yards REAL DEFAULT 0,
                    receiving_yards INTEGER DEFAULT 0,
                    receiving_tds INTEGER DEFAULT 0,
                    fumbles INTEGER DEFAULT 0,
                    fumbles_lost INTEGER DEFAULT 0,
                    two_point_conversions INTEGER DEFAULT 0,
                    snap_count INTEGER DEFAULT 0,
                    snap_share REAL DEFAULT 0,
                    team_snaps INTEGER DEFAULT 0,
                    pass_plays INTEGER DEFAULT 0,
                    rush_plays INTEGER DEFAULT 0,
                    recv_targets INTEGER DEFAULT 0,
                    pass_epa REAL DEFAULT 0,
                    rush_epa REAL DEFAULT 0,
                    recv_epa REAL DEFAULT 0,
                    pass_wpa REAL DEFAULT 0,
                    rush_wpa REAL DEFAULT 0,
                    recv_wpa REAL DEFAULT 0,
                    pass_success_rate REAL DEFAULT 0,
                    rush_success_rate REAL DEFAULT 0,
                    recv_success_rate REAL DEFAULT 0,
                    neutral_targets INTEGER DEFAULT 0,
                    neutral_rushes INTEGER DEFAULT 0,
                    third_down_targets INTEGER DEFAULT 0,
                    short_yardage_rushes INTEGER DEFAULT 0,
                    redzone_targets INTEGER DEFAULT 0,
                    goal_line_touches INTEGER DEFAULT 0,
                    two_minute_targets INTEGER DEFAULT 0,
                    high_leverage_touches INTEGER DEFAULT 0,
                    fantasy_points REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season, week),
                    FOREIGN KEY (player_id) REFERENCES players(player_id)
                )
            """)

            # Backward-compatible migration for existing DBs: add missing columns.
            try:
                cursor.execute("PRAGMA table_info(player_weekly_stats)")
                existing_cols = {row[1] for row in cursor.fetchall()}
                add_cols = {
                    "rush_inside_10": "INTEGER DEFAULT 0",
                    "rush_inside_5": "INTEGER DEFAULT 0",
                    "targets_15_plus": "INTEGER DEFAULT 0",
                    "air_yards": "REAL DEFAULT 0",
                    "pass_plays": "INTEGER DEFAULT 0",
                    "rush_plays": "INTEGER DEFAULT 0",
                    "recv_targets": "INTEGER DEFAULT 0",
                    "pass_epa": "REAL DEFAULT 0",
                    "rush_epa": "REAL DEFAULT 0",
                    "recv_epa": "REAL DEFAULT 0",
                    "pass_wpa": "REAL DEFAULT 0",
                    "rush_wpa": "REAL DEFAULT 0",
                    "recv_wpa": "REAL DEFAULT 0",
                    "pass_success_rate": "REAL DEFAULT 0",
                    "rush_success_rate": "REAL DEFAULT 0",
                    "recv_success_rate": "REAL DEFAULT 0",
                    "neutral_targets": "INTEGER DEFAULT 0",
                    "neutral_rushes": "INTEGER DEFAULT 0",
                    "third_down_targets": "INTEGER DEFAULT 0",
                    "short_yardage_rushes": "INTEGER DEFAULT 0",
                    "redzone_targets": "INTEGER DEFAULT 0",
                    "goal_line_touches": "INTEGER DEFAULT 0",
                    "two_minute_targets": "INTEGER DEFAULT 0",
                    "high_leverage_touches": "INTEGER DEFAULT 0",
                }
                for col, ddl in add_cols.items():
                    if col not in existing_cols:
                        cursor.execute(f"ALTER TABLE player_weekly_stats ADD COLUMN {col} {ddl}")
            except Exception:
                pass
            
            # Team stats
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS team_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    opponent TEXT,
                    home_away TEXT,
                    points_scored INTEGER DEFAULT 0,
                    points_allowed INTEGER DEFAULT 0,
                    total_yards INTEGER DEFAULT 0,
                    passing_yards INTEGER DEFAULT 0,
                    rushing_yards INTEGER DEFAULT 0,
                    turnovers INTEGER DEFAULT 0,
                    time_of_possession REAL DEFAULT 0,
                    total_plays INTEGER DEFAULT 0,
                    pass_attempts INTEGER DEFAULT 0,
                    rush_attempts INTEGER DEFAULT 0,
                    redzone_attempts INTEGER DEFAULT 0,
                    redzone_scores INTEGER DEFAULT 0,
                    third_down_conv REAL DEFAULT 0,
                    sacks_allowed INTEGER DEFAULT 0,
                    neutral_pass_plays INTEGER DEFAULT 0,
                    neutral_run_plays INTEGER DEFAULT 0,
                    neutral_pass_rate REAL DEFAULT 0,
                    neutral_pass_rate_lg REAL DEFAULT 0,
                    neutral_pass_rate_oe REAL DEFAULT 0,
                    drive_count INTEGER DEFAULT 0,
                    drive_success_rate REAL DEFAULT 0,
                    avg_drive_epa REAL DEFAULT 0,
                    points_per_drive REAL DEFAULT 0,
                    pace_sec_per_play REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team, season, week)
                )
            """)

            # Backward-compatible migration for existing DBs: add missing team_stats columns.
            try:
                cursor.execute("PRAGMA table_info(team_stats)")
                existing_team_cols = {row[1] for row in cursor.fetchall()}
                add_team_cols = {
                    "neutral_pass_plays": "INTEGER DEFAULT 0",
                    "neutral_run_plays": "INTEGER DEFAULT 0",
                    "neutral_pass_rate": "REAL DEFAULT 0",
                    "neutral_pass_rate_lg": "REAL DEFAULT 0",
                    "neutral_pass_rate_oe": "REAL DEFAULT 0",
                    "drive_count": "INTEGER DEFAULT 0",
                    "drive_success_rate": "REAL DEFAULT 0",
                    "avg_drive_epa": "REAL DEFAULT 0",
                    "points_per_drive": "REAL DEFAULT 0",
                    "pace_sec_per_play": "REAL DEFAULT 0",
                }
                for col, ddl in add_team_cols.items():
                    if col not in existing_team_cols:
                        cursor.execute(f"ALTER TABLE team_stats ADD COLUMN {col} {ddl}")
            except Exception:
                pass
            
            # NFL Schedule
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schedule (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    game_id TEXT,
                    game_time TEXT,
                    venue TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(season, week, home_team, away_team)
                )
            """)
            
            # Team defense stats (for opponent analysis)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS team_defense_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    points_allowed INTEGER DEFAULT 0,
                    yards_allowed INTEGER DEFAULT 0,
                    passing_yards_allowed INTEGER DEFAULT 0,
                    rushing_yards_allowed INTEGER DEFAULT 0,
                    sacks INTEGER DEFAULT 0,
                    interceptions INTEGER DEFAULT 0,
                    fumbles_recovered INTEGER DEFAULT 0,
                    fantasy_points_allowed_qb REAL DEFAULT 0,
                    fantasy_points_allowed_rb REAL DEFAULT 0,
                    fantasy_points_allowed_wr REAL DEFAULT 0,
                    fantasy_points_allowed_te REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team, season, week)
                )
            """)
            
            # Player team history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_team_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    team TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    start_week INTEGER DEFAULT 1,
                    end_week INTEGER DEFAULT 18,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, team, season),
                    FOREIGN KEY (player_id) REFERENCES players(player_id)
                )
            """)
            
            # Utilization scores
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS utilization_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    utilization_score REAL DEFAULT 0,
                    snap_share REAL DEFAULT 0,
                    target_share REAL DEFAULT 0,
                    rush_share REAL DEFAULT 0,
                    redzone_share REAL DEFAULT 0,
                    air_yards_share REAL DEFAULT 0,
                    weighted_opportunity REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season, week),
                    FOREIGN KEY (player_id) REFERENCES players(player_id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def has_data_for_season(self, season: int, position: str = None) -> bool:
        """Check if data exists for a given season."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if position:
                cursor.execute("""
                    SELECT COUNT(*) FROM player_weekly_stats pws
                    JOIN players p ON pws.player_id = p.player_id
                    WHERE pws.season = ? AND p.position = ?
                """, (season, position))
            else:
                cursor.execute(
                    "SELECT COUNT(*) FROM player_weekly_stats WHERE season = ?",
                    (season,)
                )
            count = cursor.fetchone()[0]
            return count > 0
    
    def get_latest_week_for_season(self, season: int) -> int:
        """Get the latest week of data for a season."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(week) FROM player_weekly_stats WHERE season = ?",
                (season,)
            )
            result = cursor.fetchone()[0]
            return result if result else 0
    
    def get_seasons_with_data(self) -> List[int]:
        """Get list of seasons that have data in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT season FROM player_weekly_stats ORDER BY season")
            return [row[0] for row in cursor.fetchall()]
    
    def insert_player(self, player_data: Dict[str, Any]) -> bool:
        """Insert or update player info."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO players 
                (player_id, name, position, birth_date, college, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                player_data.get("player_id"),
                player_data.get("name"),
                player_data.get("position"),
                player_data.get("birth_date"),
                player_data.get("college"),
            ))
            conn.commit()
            return True
    
    def insert_player_weekly_stats(self, stats: Dict[str, Any]) -> bool:
        """Insert or update player weekly stats."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO player_weekly_stats 
                (player_id, season, week, team, opponent, home_away, games_played,
                 passing_attempts, passing_completions, passing_yards, passing_tds,
                 interceptions, rushing_attempts, rushing_yards, rushing_tds,
                 targets, receptions, receiving_yards, receiving_tds,
                 fumbles, fumbles_lost, two_point_conversions,
                 snap_count, snap_share, team_snaps, fantasy_points,
                 rush_inside_10, rush_inside_5, targets_15_plus, air_yards,
                 pass_plays, rush_plays, recv_targets,
                 pass_epa, rush_epa, recv_epa,
                 pass_wpa, rush_wpa, recv_wpa,
                 pass_success_rate, rush_success_rate, recv_success_rate,
                 neutral_targets, neutral_rushes, third_down_targets, short_yardage_rushes,
                 redzone_targets, goal_line_touches, two_minute_targets, high_leverage_touches)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats.get("player_id"),
                stats.get("season"),
                stats.get("week"),
                stats.get("team"),
                stats.get("opponent"),
                stats.get("home_away"),
                stats.get("games_played", 1),
                stats.get("passing_attempts", 0),
                stats.get("passing_completions", 0),
                stats.get("passing_yards", 0),
                stats.get("passing_tds", 0),
                stats.get("interceptions", 0),
                stats.get("rushing_attempts", 0),
                stats.get("rushing_yards", 0),
                stats.get("rushing_tds", 0),
                stats.get("targets", 0),
                stats.get("receptions", 0),
                stats.get("receiving_yards", 0),
                stats.get("receiving_tds", 0),
                stats.get("fumbles", 0),
                stats.get("fumbles_lost", 0),
                stats.get("two_point_conversions", 0),
                stats.get("snap_count", 0),
                stats.get("snap_share", 0),
                stats.get("team_snaps", 0),
                stats.get("fantasy_points", 0),
                stats.get("rush_inside_10", 0),
                stats.get("rush_inside_5", 0),
                stats.get("targets_15_plus", 0),
                stats.get("air_yards", 0.0),
                stats.get("pass_plays", 0),
                stats.get("rush_plays", 0),
                stats.get("recv_targets", 0),
                stats.get("pass_epa", 0.0),
                stats.get("rush_epa", 0.0),
                stats.get("recv_epa", 0.0),
                stats.get("pass_wpa", 0.0),
                stats.get("rush_wpa", 0.0),
                stats.get("recv_wpa", 0.0),
                stats.get("pass_success_rate", 0.0),
                stats.get("rush_success_rate", 0.0),
                stats.get("recv_success_rate", 0.0),
                stats.get("neutral_targets", 0),
                stats.get("neutral_rushes", 0),
                stats.get("third_down_targets", 0),
                stats.get("short_yardage_rushes", 0),
                stats.get("redzone_targets", 0),
                stats.get("goal_line_touches", 0),
                stats.get("two_minute_targets", 0),
                stats.get("high_leverage_touches", 0),
            ))
            conn.commit()
            return True
    
    def insert_team_stats(self, stats: Dict[str, Any]) -> bool:
        """Insert or update team stats."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO team_stats 
                (team, season, week, opponent, home_away, points_scored, points_allowed,
                 total_yards, passing_yards, rushing_yards, turnovers, time_of_possession,
                 total_plays, pass_attempts, rush_attempts, redzone_attempts, redzone_scores,
                 third_down_conv, sacks_allowed,
                 neutral_pass_plays, neutral_run_plays, neutral_pass_rate,
                 neutral_pass_rate_lg, neutral_pass_rate_oe, drive_count,
                 drive_success_rate, avg_drive_epa, points_per_drive, pace_sec_per_play)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team, season, week) DO UPDATE SET
                    opponent = COALESCE(excluded.opponent, team_stats.opponent),
                    home_away = COALESCE(excluded.home_away, team_stats.home_away),
                    points_scored = COALESCE(excluded.points_scored, team_stats.points_scored),
                    points_allowed = COALESCE(excluded.points_allowed, team_stats.points_allowed),
                    total_yards = COALESCE(excluded.total_yards, team_stats.total_yards),
                    passing_yards = COALESCE(excluded.passing_yards, team_stats.passing_yards),
                    rushing_yards = COALESCE(excluded.rushing_yards, team_stats.rushing_yards),
                    turnovers = COALESCE(excluded.turnovers, team_stats.turnovers),
                    time_of_possession = COALESCE(excluded.time_of_possession, team_stats.time_of_possession),
                    total_plays = COALESCE(excluded.total_plays, team_stats.total_plays),
                    pass_attempts = COALESCE(excluded.pass_attempts, team_stats.pass_attempts),
                    rush_attempts = COALESCE(excluded.rush_attempts, team_stats.rush_attempts),
                    redzone_attempts = COALESCE(excluded.redzone_attempts, team_stats.redzone_attempts),
                    redzone_scores = COALESCE(excluded.redzone_scores, team_stats.redzone_scores),
                    third_down_conv = COALESCE(excluded.third_down_conv, team_stats.third_down_conv),
                    sacks_allowed = COALESCE(excluded.sacks_allowed, team_stats.sacks_allowed),
                    neutral_pass_plays = COALESCE(excluded.neutral_pass_plays, team_stats.neutral_pass_plays),
                    neutral_run_plays = COALESCE(excluded.neutral_run_plays, team_stats.neutral_run_plays),
                    neutral_pass_rate = COALESCE(excluded.neutral_pass_rate, team_stats.neutral_pass_rate),
                    neutral_pass_rate_lg = COALESCE(excluded.neutral_pass_rate_lg, team_stats.neutral_pass_rate_lg),
                    neutral_pass_rate_oe = COALESCE(excluded.neutral_pass_rate_oe, team_stats.neutral_pass_rate_oe),
                    drive_count = COALESCE(excluded.drive_count, team_stats.drive_count),
                    drive_success_rate = COALESCE(excluded.drive_success_rate, team_stats.drive_success_rate),
                    avg_drive_epa = COALESCE(excluded.avg_drive_epa, team_stats.avg_drive_epa),
                    points_per_drive = COALESCE(excluded.points_per_drive, team_stats.points_per_drive),
                    pace_sec_per_play = COALESCE(excluded.pace_sec_per_play, team_stats.pace_sec_per_play)
            """, (
                stats.get("team"),
                stats.get("season"),
                stats.get("week"),
                stats.get("opponent"),
                stats.get("home_away"),
                stats.get("points_scored", 0),
                stats.get("points_allowed", 0),
                stats.get("total_yards", 0),
                stats.get("passing_yards", 0),
                stats.get("rushing_yards", 0),
                stats.get("turnovers", 0),
                stats.get("time_of_possession", 0),
                stats.get("total_plays", 0),
                stats.get("pass_attempts", 0),
                stats.get("rush_attempts", 0),
                stats.get("redzone_attempts", 0),
                stats.get("redzone_scores", 0),
                stats.get("third_down_conv", 0),
                stats.get("sacks_allowed", 0),
                stats.get("neutral_pass_plays", 0),
                stats.get("neutral_run_plays", 0),
                stats.get("neutral_pass_rate", 0.0),
                stats.get("neutral_pass_rate_lg", 0.0),
                stats.get("neutral_pass_rate_oe", 0.0),
                stats.get("drive_count", 0),
                stats.get("drive_success_rate", 0.0),
                stats.get("avg_drive_epa", 0.0),
                stats.get("points_per_drive", 0.0),
                stats.get("pace_sec_per_play", 0.0),
            ))
            conn.commit()
            return True
    
    def insert_utilization_score(self, score_data: Dict[str, Any]) -> bool:
        """Insert or update utilization score."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO utilization_scores 
                (player_id, season, week, utilization_score, snap_share, target_share,
                 rush_share, redzone_share, air_yards_share, weighted_opportunity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                score_data.get("player_id"),
                score_data.get("season"),
                score_data.get("week"),
                score_data.get("utilization_score", 0),
                score_data.get("snap_share", 0),
                score_data.get("target_share", 0),
                score_data.get("rush_share", 0),
                score_data.get("redzone_share", 0),
                score_data.get("air_yards_share", 0),
                score_data.get("weighted_opportunity", 0),
            ))
            conn.commit()
            return True
    
    def insert_schedule(self, schedule_data: Dict[str, Any]) -> bool:
        """Insert or update schedule entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO schedule 
                (season, week, home_team, away_team, game_id, game_time, venue, home_score, away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                schedule_data.get("season"),
                schedule_data.get("week"),
                schedule_data.get("home_team"),
                schedule_data.get("away_team"),
                schedule_data.get("game_id"),
                schedule_data.get("game_time"),
                schedule_data.get("venue"),
                schedule_data.get("home_score"),
                schedule_data.get("away_score"),
            ))
            conn.commit()
            return True
    
    def get_schedule(self, season: int = None, week: int = None, 
                     team: str = None, include_scores: bool = False) -> pd.DataFrame:
        """Get schedule with optional filters.

        By default, final score columns are removed to prevent leakage when
        schedules are used as model features.
        """
        query = "SELECT * FROM schedule WHERE 1=1"
        params = []
        
        if season:
            query += " AND season = ?"
            params.append(season)
        if week:
            query += " AND week = ?"
            params.append(week)
        if team:
            query += " AND (home_team = ? OR away_team = ?)"
            params.extend([team, team])
        
        query += " ORDER BY season, week"
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        if not include_scores:
            try:
                from src.utils.leakage import sanitize_schedule_df
                df = sanitize_schedule_df(df)
            except Exception:
                pass
        return df
    
    def import_schedule_from_csv(self, csv_path: str) -> int:
        """
        Import schedule from CSV file.
        
        Expected columns: season, week, home_team, away_team
        Optional columns: game_id, game_time, venue
        
        Returns number of games imported.
        """
        df = pd.read_csv(csv_path)
        required_cols = ['season', 'week', 'home_team', 'away_team']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        count = 0
        for _, row in df.iterrows():
            self.insert_schedule(row.to_dict())
            count += 1
        
        return count
    
    def has_schedule_for_season(self, season: int) -> bool:
        """Check if schedule exists for a season."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM schedule WHERE season = ?", (season,))
            return cursor.fetchone()[0] > 0
    
    def get_player_stats(self, player_id: str = None, season: int = None, 
                         position: str = None) -> pd.DataFrame:
        """Get player weekly stats with optional filters."""
        query = """
            SELECT pws.*, p.name, p.position 
            FROM player_weekly_stats pws
            JOIN players p ON pws.player_id = p.player_id
            WHERE 1=1
        """
        params = []
        
        if player_id:
            query += " AND pws.player_id = ?"
            params.append(player_id)
        if season:
            query += " AND pws.season = ?"
            params.append(season)
        if position:
            query += " AND p.position = ?"
            params.append(position)
        
        query += " ORDER BY pws.season, pws.week"
        
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_team_stats(self, team: str = None, season: int = None) -> pd.DataFrame:
        """Get team stats with optional filters."""
        query = "SELECT * FROM team_stats WHERE 1=1"
        params = []
        
        if team:
            query += " AND team = ?"
            params.append(team)
        if season:
            query += " AND season = ?"
            params.append(season)
        
        query += " ORDER BY season, week"
        
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def aggregate_team_stats_from_players(self, season: int = None) -> pd.DataFrame:
        """
        Aggregate team-level stats from player_weekly_stats (no new DBs).
        Produces one row per (team, season, week) with pass/rush attempts and yards,
        total_yards, total_plays; opponent/home_away from any player on that team.
        Use when team_stats is empty so training has team tendency features.
        """
        query = """
            SELECT
                team,
                season,
                week,
                MAX(opponent) AS opponent,
                MAX(home_away) AS home_away,
                COALESCE(SUM(passing_attempts), 0) AS pass_attempts,
                COALESCE(SUM(rushing_attempts), 0) AS rush_attempts,
                COALESCE(SUM(passing_yards), 0) AS passing_yards,
                COALESCE(SUM(rushing_yards), 0) AS rushing_yards,
                COALESCE(SUM(passing_yards), 0) + COALESCE(SUM(rushing_yards), 0) AS total_yards,
                COALESCE(SUM(passing_attempts), 0) + COALESCE(SUM(rushing_attempts), 0) AS total_plays,
                0 AS points_scored,
                0 AS points_allowed,
                0 AS turnovers,
                0.0 AS time_of_possession,
                0 AS redzone_attempts,
                0 AS redzone_scores,
                0.0 AS third_down_conv,
                0 AS sacks_allowed,
                0 AS neutral_pass_plays,
                0 AS neutral_run_plays,
                0.0 AS neutral_pass_rate,
                0.0 AS neutral_pass_rate_lg,
                0.0 AS neutral_pass_rate_oe,
                0 AS drive_count,
                0.0 AS drive_success_rate,
                0.0 AS avg_drive_epa,
                0.0 AS points_per_drive,
                0.0 AS pace_sec_per_play
            FROM player_weekly_stats
            WHERE team IS NOT NULL AND team != ''
        """
        params = []
        if season is not None:
            query += " AND season = ?"
            params.append(season)
        query += " GROUP BY team, season, week ORDER BY season, week"
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def ensure_team_stats_from_players(self, season: int = None) -> int:
        """
        Backfill team_stats from player_weekly_stats for rows that don't exist.
        Skips (team, season, week) that already have team_stats so scraper data is preserved.
        Returns number of rows inserted.
        """
        agg = self.aggregate_team_stats_from_players(season=season)
        if agg.empty:
            return 0
        existing = self.get_team_stats()
        if not existing.empty:
            existing_keys = set(
                zip(existing["team"].astype(str), existing["season"].astype(int), existing["week"].astype(int))
            )
        else:
            existing_keys = set()
        count = 0
        for _, row in agg.iterrows():
            key = (str(row["team"]), int(row["season"]), int(row["week"]))
            if key in existing_keys:
                continue
            self.insert_team_stats({
                "team": row["team"],
                "season": int(row["season"]),
                "week": int(row["week"]),
                "opponent": row.get("opponent") or "",
                "home_away": row.get("home_away") or "",
                "points_scored": int(row.get("points_scored", 0)),
                "points_allowed": int(row.get("points_allowed", 0)),
                "total_yards": int(row.get("total_yards", 0)),
                "passing_yards": int(row.get("passing_yards", 0)),
                "rushing_yards": int(row.get("rushing_yards", 0)),
                "turnovers": int(row.get("turnovers", 0)),
                "time_of_possession": float(row.get("time_of_possession", 0)),
                "total_plays": int(row.get("total_plays", 0)),
                "pass_attempts": int(row.get("pass_attempts", 0)),
                "rush_attempts": int(row.get("rush_attempts", 0)),
                "redzone_attempts": int(row.get("redzone_attempts", 0)),
                "redzone_scores": int(row.get("redzone_scores", 0)),
                "third_down_conv": float(row.get("third_down_conv", 0)),
                "sacks_allowed": int(row.get("sacks_allowed", 0)),
                "neutral_pass_plays": int(row.get("neutral_pass_plays", 0)),
                "neutral_run_plays": int(row.get("neutral_run_plays", 0)),
                "neutral_pass_rate": float(row.get("neutral_pass_rate", 0.0)),
                "neutral_pass_rate_lg": float(row.get("neutral_pass_rate_lg", 0.0)),
                "neutral_pass_rate_oe": float(row.get("neutral_pass_rate_oe", 0.0)),
                "drive_count": int(row.get("drive_count", 0)),
                "drive_success_rate": float(row.get("drive_success_rate", 0.0)),
                "avg_drive_epa": float(row.get("avg_drive_epa", 0.0)),
                "points_per_drive": float(row.get("points_per_drive", 0.0)),
                "pace_sec_per_play": float(row.get("pace_sec_per_play", 0.0)),
            })
            count += 1
            existing_keys.add(key)
        return count

    def aggregate_team_defense_from_players(self, season: int = None) -> pd.DataFrame:
        """
        Aggregate team_defense_stats from player_weekly_stats: for each (opponent, season, week),
        sum fantasy_points by position (QB/RB/WR/TE) to get points allowed per position.
        """
        query = """
            SELECT
                pws.opponent AS team,
                pws.season,
                pws.week,
                0 AS points_allowed,
                0 AS yards_allowed,
                0 AS passing_yards_allowed,
                0 AS rushing_yards_allowed,
                0 AS sacks,
                0 AS interceptions,
                0 AS fumbles_recovered,
                SUM(CASE WHEN p.position = 'QB' THEN pws.fantasy_points ELSE 0 END) AS fantasy_points_allowed_qb,
                SUM(CASE WHEN p.position = 'RB' THEN pws.fantasy_points ELSE 0 END) AS fantasy_points_allowed_rb,
                SUM(CASE WHEN p.position = 'WR' THEN pws.fantasy_points ELSE 0 END) AS fantasy_points_allowed_wr,
                SUM(CASE WHEN p.position = 'TE' THEN pws.fantasy_points ELSE 0 END) AS fantasy_points_allowed_te
            FROM player_weekly_stats pws
            JOIN players p ON pws.player_id = p.player_id
            WHERE pws.opponent IS NOT NULL AND pws.opponent != ''
        """
        params = []
        if season is not None:
            query += " AND pws.season = ?"
            params.append(season)
        query += " GROUP BY pws.opponent, pws.season, pws.week ORDER BY pws.season, pws.week"
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def ensure_team_defense_stats(self, season: int = None) -> int:
        """Populate team_defense_stats from player_weekly_stats. Returns number of rows inserted."""
        agg = self.aggregate_team_defense_from_players(season=season)
        if agg.empty:
            return 0
        existing = set()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT team, season, week FROM team_defense_stats")
            existing = set((r[0], r[1], r[2]) for r in cur.fetchall())
        count = 0
        with self._get_connection() as conn:
            cur = conn.cursor()
            for _, row in agg.iterrows():
                key = (str(row["team"]), int(row["season"]), int(row["week"]))
                if key in existing:
                    continue
                try:
                    cur.execute("""
                        INSERT OR REPLACE INTO team_defense_stats
                        (team, season, week, points_allowed, yards_allowed, passing_yards_allowed,
                         rushing_yards_allowed, sacks, interceptions, fumbles_recovered,
                         fantasy_points_allowed_qb, fantasy_points_allowed_rb,
                         fantasy_points_allowed_wr, fantasy_points_allowed_te)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row["team"], int(row["season"]), int(row["week"]),
                        int(row.get("points_allowed", 0)), int(row.get("yards_allowed", 0)),
                        int(row.get("passing_yards_allowed", 0)), int(row.get("rushing_yards_allowed", 0)),
                        int(row.get("sacks", 0)), int(row.get("interceptions", 0)),
                        int(row.get("fumbles_recovered", 0)),
                        float(row.get("fantasy_points_allowed_qb", 0)),
                        float(row.get("fantasy_points_allowed_rb", 0)),
                        float(row.get("fantasy_points_allowed_wr", 0)),
                        float(row.get("fantasy_points_allowed_te", 0)),
                    ))
                    count += 1
                    existing.add(key)
                except Exception:
                    pass
            conn.commit()
        return count
    
    def get_player_with_team_history(self, player_id: str) -> pd.DataFrame:
        """Get player stats joined with their team stats for each week."""
        query = """
            SELECT pws.*, p.name, p.position,
                   ts.points_scored as team_points, ts.total_yards as team_yards,
                   ts.pass_attempts as team_pass_attempts, ts.rush_attempts as team_rush_attempts,
                   ts.redzone_attempts as team_redzone_attempts
            FROM player_weekly_stats pws
            JOIN players p ON pws.player_id = p.player_id
            LEFT JOIN team_stats ts ON pws.team = ts.team 
                AND pws.season = ts.season AND pws.week = ts.week
            WHERE pws.player_id = ?
            ORDER BY pws.season, pws.week
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[player_id])
    
    def get_all_players_for_training(self, position: str = None,
                                      min_games: int = 4) -> pd.DataFrame:
        """Get all player data suitable for model training.

        NOTE: Opponent defense stats are joined from the PRIOR week to prevent
        data leakage. Same-week opponent stats encode the target (the player's
        own fantasy points contribute to the opponent's "points allowed").
        """
        query = """
            SELECT pws.*, p.name, p.position,
                   us.utilization_score, us.snap_share as util_snap_share,
                   us.target_share as util_target_share, us.rush_share as util_rush_share,
                   us.redzone_share as util_redzone_share,
                   ts.points_scored as team_points, ts.total_yards as team_yards,
                   ts.pass_attempts as team_pass_attempts, ts.rush_attempts as team_rush_attempts,
                   ts.redzone_attempts as team_redzone_attempts, ts.total_plays as team_plays,
                   ts.neutral_pass_plays as team_neutral_pass_plays,
                   ts.neutral_run_plays as team_neutral_run_plays,
                   ts.neutral_pass_rate as team_neutral_pass_rate,
                   ts.neutral_pass_rate_lg as team_neutral_pass_rate_lg,
                   ts.neutral_pass_rate_oe as team_neutral_pass_rate_oe,
                   ts.drive_count as team_drive_count,
                   ts.drive_success_rate as team_drive_success_rate,
                   ts.avg_drive_epa as team_avg_drive_epa,
                   ts.points_per_drive as team_points_per_drive,
                   ts.pace_sec_per_play as team_pace_sec_per_play,
                   tds.fantasy_points_allowed_qb, tds.fantasy_points_allowed_rb,
                   tds.fantasy_points_allowed_wr, tds.fantasy_points_allowed_te,
                   tds.week as opp_defense_week
            FROM player_weekly_stats pws
            JOIN players p ON pws.player_id = p.player_id
            LEFT JOIN utilization_scores us ON pws.player_id = us.player_id
                AND pws.season = us.season AND pws.week = us.week
            LEFT JOIN team_stats ts ON pws.team = ts.team
                AND pws.season = ts.season AND pws.week = ts.week
            LEFT JOIN team_defense_stats tds ON pws.opponent = tds.team
                AND tds.season = pws.season AND tds.week = pws.week - 1
        """
        params = []
        
        if position:
            query += " WHERE p.position = ?"
            params.append(position)
        
        query += " ORDER BY pws.player_id, pws.season, pws.week"
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        # Defensive stats must be strictly prior-week; block any same-week leakage.
        if "opp_defense_week" in df.columns and "week" in df.columns:
            bad = df["opp_defense_week"].notna() & (df["opp_defense_week"] >= df["week"])
            if bad.any():
                sample = df.loc[bad, ["player_id", "season", "week", "opponent", "opp_defense_week"]].head(3)
                raise ValueError(f"Leakage detected: opponent defense stats not shifted. Sample: {sample.to_dict(orient='records')}")

        # Filter to players with minimum games
        if min_games > 0:
            game_counts = df.groupby('player_id').size()
            valid_players = game_counts[game_counts >= min_games].index
            df = df[df['player_id'].isin(valid_players)]
        
        return df
    
    def get_eligible_player_ids(self, eligible_seasons: List[int]) -> List[str]:
        """Return player_ids that have game data in at least one of the given seasons.

        Players without any weekly stats in these seasons are considered inactive
        (retired, unsigned, etc.) and excluded from predictions.
        """
        if not eligible_seasons:
            return []
        placeholders = ",".join("?" for _ in eligible_seasons)
        query = f"""
            SELECT DISTINCT player_id
            FROM player_weekly_stats
            WHERE season IN ({placeholders})
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, eligible_seasons)
            return [row[0] for row in cursor.fetchall()]

    def bulk_insert_dataframe(self, df: pd.DataFrame, table_name: str) -> int:
        """Bulk insert a DataFrame into a table."""
        with self._get_connection() as conn:
            rows_inserted = df.to_sql(table_name, conn, if_exists='append', index=False)
            return rows_inserted or len(df)
