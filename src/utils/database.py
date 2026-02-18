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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(team, season, week)
                )
            """)
            
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
                 rush_inside_10, rush_inside_5, targets_15_plus, air_yards)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ))
            conn.commit()
            return True
    
    def insert_team_stats(self, stats: Dict[str, Any]) -> bool:
        """Insert or update team stats."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO team_stats 
                (team, season, week, opponent, home_away, points_scored, points_allowed,
                 total_yards, passing_yards, rushing_yards, turnovers, time_of_possession,
                 total_plays, pass_attempts, rush_attempts, redzone_attempts, redzone_scores,
                 third_down_conv, sacks_allowed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                     team: str = None) -> pd.DataFrame:
        """Get schedule with optional filters."""
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
            return pd.read_sql_query(query, conn, params=params)
    
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
                0 AS sacks_allowed
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
                   tds.fantasy_points_allowed_qb, tds.fantasy_points_allowed_rb,
                   tds.fantasy_points_allowed_wr, tds.fantasy_points_allowed_te
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
        
        # Filter to players with minimum games
        if min_games > 0:
            game_counts = df.groupby('player_id').size()
            valid_players = game_counts[game_counts >= min_games].index
            df = df[df['player_id'].isin(valid_players)]
        
        return df
    
    def bulk_insert_dataframe(self, df: pd.DataFrame, table_name: str) -> int:
        """Bulk insert a DataFrame into a table."""
        with self._get_connection() as conn:
            rows_inserted = df.to_sql(table_name, conn, if_exists='append', index=False)
            return rows_inserted or len(df)
