"""Database management for NFL data storage."""
import re
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from contextlib import contextmanager

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALID_DDL_TYPE = re.compile(r"^[A-Z]+ *(DEFAULT *[\w.]+)?$", re.IGNORECASE)


def _validate_identifier(name: str) -> str:
    """Validate a SQL identifier (column/table name) against injection."""
    if not _VALID_IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def _validate_ddl(ddl: str) -> str:
    """Validate a DDL type expression (e.g. 'REAL DEFAULT 0')."""
    if not _VALID_DDL_TYPE.match(ddl):
        raise ValueError(f"Invalid DDL type expression: {ddl!r}")
    return ddl

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
                        cursor.execute(
                            f"ALTER TABLE player_weekly_stats ADD COLUMN "
                            f"{_validate_identifier(col)} {_validate_ddl(ddl)}"
                        )
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
                        cursor.execute(
                            f"ALTER TABLE team_stats ADD COLUMN "
                            f"{_validate_identifier(col)} {_validate_ddl(ddl)}"
                        )
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
                    spread_line REAL,
                    total_line REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(season, week, home_team, away_team)
                )
            """)

            # Idempotent migration for pre-existing schedule tables.  SQLite has
            # no "IF NOT EXISTS" on ALTER TABLE, so catch the duplicate-column
            # error rather than gate on schema introspection.  Landed 2026-04-21
            # as part of Phase 1 Step C (Vegas cache); see
            # docs/PHASE_1_VEGAS_FINDINGS.md.
            for col_sql in (
                "ALTER TABLE schedule ADD COLUMN spread_line REAL",
                "ALTER TABLE schedule ADD COLUMN total_line REAL",
            ):
                try:
                    cursor.execute(col_sql)
                except Exception:
                    pass

            # Injury reports — point-in-time pre-game status per (player, week).
            # Populated by scripts/backfill_injuries.py from nfl_data_py's
            # import_injuries().  See docs/PHASE_3_INJURY_FINDINGS.md.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_injuries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    team TEXT,
                    full_name TEXT,
                    position TEXT,
                    report_status TEXT,
                    practice_status TEXT,
                    report_primary_injury TEXT,
                    date_modified TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season, week)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_player_injuries_season_week "
                "ON player_injuries(season, week)"
            )

            # Weekly roster snapshots — point-in-time "was this player
            # on the 53-man active roster for this (season, week)".
            # Populated by scripts/backfill_weekly_rosters.py from the
            # nflverse weekly_rosters parquet.  Used by the paper-trade
            # harness and retrospective symmetric-replay measurement
            # as the pre-lock active-roster filter described in
            # docs/INACTIVE_PICK_GAP_DIAGNOSIS.md.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weekly_rosters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    team TEXT,
                    position TEXT,
                    status TEXT,
                    status_description_abbr TEXT,
                    full_name TEXT,
                    game_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season, week)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_weekly_rosters_season_week "
                "ON weekly_rosters(season, week)"
            )

            # Forward paper-trade entries — one row per locked week.
            # Populated by scripts/paper_trade_lock.py per the protocol
            # in docs/PAPER_TRADE_PROTOCOL_20260422.md (re-council Step 5).
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_trade_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    lock_timestamp TEXT NOT NULL,
                    lock_git_sha TEXT NOT NULL,
                    model_config_json TEXT NOT NULL,
                    model_lineup_json TEXT NOT NULL,
                    opponent_lineup_json TEXT NOT NULL,
                    opponent_method TEXT NOT NULL,
                    notional_entry_usd REAL NOT NULL,
                    notional_payout_multiplier REAL NOT NULL,
                    model_actual REAL,
                    opponent_actual REAL,
                    won INTEGER,
                    score_timestamp TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(season, week)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_paper_trade_season_week "
                "ON paper_trade_entries(season, week)"
            )

            # Historical ADP (FantasyPros ECR via dynastyprocess/data mirror).
            # Populated by scripts/backfill_adp.py. One row per
            # (scrape_date, page_type, player) — we store multiple weekly
            # snapshots per season so the draft simulator can pick the
            # closest-to-draft-day scrape.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adp_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    season INTEGER NOT NULL,
                    scrape_date TEXT NOT NULL,
                    fp_player_id TEXT,
                    player_name TEXT NOT NULL,
                    mergename TEXT,
                    position TEXT,
                    team TEXT,
                    ecr REAL NOT NULL,
                    sd REAL,
                    best REAL,
                    worst REAL,
                    ecr_type TEXT,
                    page_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(season, scrape_date, page_type, fp_player_id, player_name)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_adp_season_date "
                "ON adp_history(season, scrape_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_adp_mergename "
                "ON adp_history(mergename)"
            )

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
            
            # Draft picks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS draft_picks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT,
                    player_name TEXT NOT NULL,
                    position TEXT,
                    college TEXT,
                    draft_season INTEGER NOT NULL,
                    draft_round INTEGER NOT NULL,
                    draft_pick INTEGER NOT NULL,
                    draft_team TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_name, draft_season)
                )
            """)

            # NFL Combine data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS combine_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_name TEXT NOT NULL,
                    position TEXT,
                    season INTEGER NOT NULL,
                    height REAL,
                    weight REAL,
                    forty REAL,
                    bench INTEGER,
                    vertical REAL,
                    broad_jump INTEGER,
                    cone REAL,
                    shuttle REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_name, season)
                )
            """)

            # Rosters (years_exp, birth_date, college, etc.)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rosters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    player_name TEXT,
                    position TEXT,
                    team TEXT,
                    season INTEGER NOT NULL,
                    years_exp INTEGER DEFAULT 0,
                    birth_date TEXT,
                    height TEXT,
                    weight INTEGER,
                    college TEXT,
                    jersey_number INTEGER,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season)
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

            # Game odds — one row per (event, bookmaker, market, snapshot time).
            # Markets: h2h (moneylines), spreads, totals.
            # home_point / away_point = spread or total line; home_price / away_price = American odds.
            # fetched_at = UTC ISO-8601 datetime of the API snapshot.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_odds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    game_id TEXT,
                    season INTEGER,
                    week INTEGER,
                    commence_time TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    bookmaker TEXT NOT NULL,
                    market TEXT NOT NULL,
                    home_price REAL,
                    away_price REAL,
                    home_point REAL,
                    away_point REAL,
                    fetched_at TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_id, bookmaker, market, fetched_at)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_game_odds_season_week "
                "ON game_odds(season, week)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_game_odds_event "
                "ON game_odds(event_id)"
            )

            # Player prop odds — one row per (event, bookmaker, market, player, over/under, snapshot).
            # market examples: player_pass_yds, player_rush_tds, player_receptions, etc.
            # description: "Over" or "Under".
            # point: the line value (e.g., 274.5 for passing yards).
            # price: American odds for that side.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_props_odds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    game_id TEXT,
                    season INTEGER,
                    week INTEGER,
                    commence_time TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    bookmaker TEXT NOT NULL,
                    market TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    description TEXT,
                    price REAL NOT NULL,
                    point REAL,
                    fetched_at TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_id, bookmaker, market, player_name, description, fetched_at)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_player_props_event "
                "ON player_props_odds(event_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_player_props_player "
                "ON player_props_odds(player_name, market)"
            )

            # Game weather — one row per game (home stadium at kickoff).
            # Dome/retractable-roof games: is_dome=1, weather fields NULL.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_weather (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    season INTEGER NOT NULL,
                    week INTEGER NOT NULL,
                    game_date TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    stadium TEXT,
                    is_dome INTEGER DEFAULT 0,
                    kickoff_time TEXT,
                    temp_f REAL,
                    wind_mph REAL,
                    precip_mm REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(season, week, home_team, away_team)
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_game_weather_season_week "
                "ON game_weather(season, week)"
            )

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
                (season, week, home_team, away_team, game_id, game_time, venue,
                 home_score, away_score, spread_line, total_line)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                schedule_data.get("spread_line"),
                schedule_data.get("total_line"),
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
    
    # ------------------------------------------------------------------
    # Draft picks
    # ------------------------------------------------------------------

    def bulk_insert_draft_picks(self, df: pd.DataFrame) -> int:
        """Bulk insert draft picks from a DataFrame (e.g. nfl.import_draft_picks())."""
        col_map = {
            "pfr_player_name": "player_name",
            "round": "draft_round",
            "pick": "draft_pick",
            "season": "draft_season",
        }
        renamed = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        keep = [c for c in ["player_id", "player_name", "position", "college",
                            "draft_season", "draft_round", "draft_pick", "draft_team",
                            "team"] if c in renamed.columns]
        out = renamed[keep].copy()
        if "draft_team" not in out.columns and "team" in out.columns:
            out["draft_team"] = out["team"]
        out = out.drop(columns=["team"], errors="ignore")
        with self._get_connection() as conn:
            out.to_sql("draft_picks", conn, if_exists="replace", index=False)
            conn.commit()
        return len(out)

    def get_draft_picks(self, season: int = None, position: str = None) -> pd.DataFrame:
        """Get draft picks with optional filters.

        Reads from ``draft_picks_v2`` (nfl-data-py source, GSIS IDs) which
        has data.  The legacy ``draft_picks`` table is empty on most installs.
        """
        query = "SELECT player_id, position, draft_season, draft_round, draft_pick, draft_team FROM draft_picks_v2 WHERE player_id LIKE '00-%'"
        params: list = []
        if season:
            query += " AND draft_season = ?"
            params.append(season)
        if position:
            query += " AND position = ?"
            params.append(position)
        query += " ORDER BY draft_season, draft_round, draft_pick"
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ------------------------------------------------------------------
    # Combine data
    # ------------------------------------------------------------------

    def bulk_insert_combine_data(self, df: pd.DataFrame) -> int:
        """Bulk insert combine data from a DataFrame (e.g. nfl.import_combine_data())."""
        col_map = {
            "player_name": "player_name",
            "pos": "position",
            "ht": "height",
            "wt": "weight",
            "forty": "forty",
            "bench": "bench",
            "vertical": "vertical",
            "broad_jump": "broad_jump",
            "cone": "cone",
            "shuttle": "shuttle",
        }
        renamed = df.rename(columns={k: v for k, v in col_map.items()
                                     if k in df.columns and v not in df.columns})
        keep = [c for c in ["player_name", "position", "season",
                            "height", "weight", "forty", "bench",
                            "vertical", "broad_jump", "cone", "shuttle"]
                if c in renamed.columns]
        out = renamed[keep].copy()
        with self._get_connection() as conn:
            out.to_sql("combine_data", conn, if_exists="replace", index=False)
            conn.commit()
        return len(out)

    def get_combine_data(self, season: int = None, position: str = None) -> pd.DataFrame:
        """Get combine data with optional filters."""
        query = "SELECT * FROM combine_data WHERE 1=1"
        params: list = []
        if season:
            query += " AND season = ?"
            params.append(season)
        if position:
            query += " AND position = ?"
            params.append(position)
        query += " ORDER BY season, position"
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ------------------------------------------------------------------
    # NGS (Next Gen Stats)
    # ------------------------------------------------------------------

    def get_ngs_data(self, seasons: list = None) -> pd.DataFrame:
        """Load NGS passing/rushing/receiving data, merged into one DataFrame.

        Returns columns prefixed with ``ngs_`` keyed on
        (player_id, season, week).  Only regular-season weekly rows
        (week > 0) are returned.
        """
        frames = []
        table_cols = {
            "ngs_passing": [
                "player_gsis_id", "season", "week",
                "avg_time_to_throw", "completion_percentage_above_expectation",
                "aggressiveness", "avg_air_yards_to_sticks",
            ],
            "ngs_rushing": [
                "player_gsis_id", "season", "week",
                "rush_yards_over_expected_per_att", "efficiency",
                "percent_attempts_gte_eight_defenders",
            ],
            "ngs_receiving": [
                "player_gsis_id", "season", "week",
                "avg_separation", "avg_cushion",
                "avg_intended_air_yards", "catch_percentage",
                "avg_yac_above_expectation",
            ],
        }
        with self._get_connection() as conn:
            for table, cols in table_cols.items():
                try:
                    safe_cols = ", ".join(cols)
                    where = "WHERE week > 0"
                    params: list = []
                    if seasons:
                        placeholders = ", ".join("?" for _ in seasons)
                        where += f" AND season IN ({placeholders})"
                        params.extend(seasons)
                    q = f"SELECT {safe_cols} FROM {table} {where}"
                    tdf = pd.read_sql_query(q, conn, params=params)
                    # Prefix non-key columns with ngs_
                    rename = {c: f"ngs_{c}" for c in tdf.columns
                              if c not in ("player_gsis_id", "season", "week")}
                    tdf = tdf.rename(columns=rename)
                    tdf = tdf.rename(columns={"player_gsis_id": "player_id"})
                    frames.append(tdf)
                except Exception as e:
                    print(f"  WARNING: Could not load {table}: {e}")
        if not frames:
            return pd.DataFrame()
        # Outer-merge so we keep passing-only, rushing-only, etc.
        result = frames[0]
        for f in frames[1:]:
            result = result.merge(f, on=["player_id", "season", "week"], how="outer")
        return result

    # ------------------------------------------------------------------
    # Rosters
    # ------------------------------------------------------------------

    def bulk_insert_rosters(self, df: pd.DataFrame) -> int:
        """Bulk insert roster data from a DataFrame (e.g. nfl.import_rosters())."""
        col_map = {
            "full_name": "player_name",
            "gsis_id": "player_id",
            "years_exp": "years_exp",
        }
        renamed = df.rename(columns={k: v for k, v in col_map.items()
                                     if k in df.columns and v not in df.columns})
        keep = [c for c in ["player_id", "player_name", "position", "team",
                            "season", "years_exp", "birth_date", "height",
                            "weight", "college", "jersey_number", "status"]
                if c in renamed.columns]
        out = renamed[keep].copy()
        out = out.dropna(subset=["player_id"])
        with self._get_connection() as conn:
            out.to_sql("rosters", conn, if_exists="replace", index=False)
            conn.commit()
        return len(out)

    def get_rosters(self, season: int = None, position: str = None) -> pd.DataFrame:
        """Get roster data with optional filters."""
        query = "SELECT * FROM rosters WHERE 1=1"
        params: list = []
        if season:
            query += " AND season = ?"
            params.append(season)
        if position:
            query += " AND position = ?"
            params.append(position)
        query += " ORDER BY season, position, player_name"
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_player_years_exp(self, player_id: str, season: int) -> Optional[int]:
        """Get years of experience for a player in a given season."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT years_exp FROM rosters WHERE player_id = ? AND season = ?",
                (player_id, season)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def backfill_players_from_rosters(self) -> int:
        """Backfill college and birth_date in the players table from rosters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE players SET
                    college = COALESCE(players.college, (
                        SELECT r.college FROM rosters r
                        WHERE r.player_id = players.player_id
                        AND r.college IS NOT NULL AND r.college != ''
                        LIMIT 1
                    )),
                    birth_date = COALESCE(players.birth_date, (
                        SELECT r.birth_date FROM rosters r
                        WHERE r.player_id = players.player_id
                        AND r.birth_date IS NOT NULL AND r.birth_date != ''
                        LIMIT 1
                    )),
                    updated_at = CURRENT_TIMESTAMP
                WHERE players.college IS NULL OR players.birth_date IS NULL
            """)
            count = cursor.rowcount
            conn.commit()
            return count

    def get_authoritative_player_positions(self) -> Dict[str, str]:
        """Return authoritative skill-position labels keyed by ``player_id``.

        Priority order:
        1. ``weekly_rosters_v2`` latest (season, week) snapshot
        2. ``weekly_rosters`` latest (season, week) snapshot
        3. ``rosters`` latest season snapshot
        """
        def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            return row is not None

        def _latest_rows(
            conn: sqlite3.Connection,
            table_name: str,
            *,
            season_col: str = "season",
            week_col: Optional[str] = "week",
        ) -> List[Tuple[str, str]]:
            if not _table_exists(conn, table_name):
                return []
            order_by = f"{season_col} DESC"
            if week_col is not None:
                order_by += f", {week_col} DESC"
            query = f"""
                SELECT player_id, position
                FROM (
                    SELECT player_id,
                           position,
                           ROW_NUMBER() OVER (
                               PARTITION BY player_id
                               ORDER BY {order_by}
                           ) AS rn
                    FROM {table_name}
                    WHERE position IN ('QB','RB','WR','TE')
                      AND player_id IS NOT NULL
                      AND TRIM(player_id) != ''
                )
                WHERE rn = 1
            """
            return conn.execute(query).fetchall()

        pos_map: Dict[str, str] = {}
        with self._get_connection() as conn:
            for table_name, week_col in [
                ("weekly_rosters_v2", "week"),
                ("weekly_rosters", "week"),
                ("rosters", None),
            ]:
                for player_id, position in _latest_rows(
                    conn,
                    table_name,
                    week_col=week_col,
                ):
                    pos_map.setdefault(player_id, position)
        return pos_map

    def reconcile_player_positions_from_rosters(self) -> int:
        """Update ``players.position`` from authoritative roster snapshots."""
        pos_map = self.get_authoritative_player_positions()
        if not pos_map:
            return 0

        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT player_id, position FROM players"
            ).fetchall()
            updates = [
                (authoritative_pos, player_id)
                for player_id, current_pos in rows
                if (authoritative_pos := pos_map.get(player_id))
                and authoritative_pos != current_pos
            ]
            if not updates:
                return 0
            conn.executemany(
                """
                UPDATE players
                   SET position = ?,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE player_id = ?
                """,
                updates,
            )
            conn.commit()
            return len(updates)

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
                NULL AS points_scored,
                NULL AS points_allowed,
                NULL AS turnovers,
                NULL AS time_of_possession,
                NULL AS redzone_attempts,
                NULL AS redzone_scores,
                NULL AS third_down_conv,
                NULL AS sacks_allowed,
                NULL AS neutral_pass_plays,
                NULL AS neutral_run_plays,
                NULL AS neutral_pass_rate,
                NULL AS neutral_pass_rate_lg,
                NULL AS neutral_pass_rate_oe,
                NULL AS drive_count,
                NULL AS drive_success_rate,
                NULL AS avg_drive_epa,
                NULL AS points_per_drive,
                NULL AS pace_sec_per_play
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
        Skips (team, season, week) keys that already have team_stats so that
        PBP-derived or scraper-sourced values are never overwritten by the
        less-accurate player-aggregated approximations.
        Fields that cannot be computed from player stats (redzone, drive metrics,
        neutral pass rate, etc.) are passed as None so the ON CONFLICT … COALESCE
        logic in insert_team_stats preserves any existing values.
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

        def _safe_int(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return int(val)

        def _safe_float(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return float(val)

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
                "points_scored": _safe_int(row.get("points_scored")),
                "points_allowed": _safe_int(row.get("points_allowed")),
                "total_yards": _safe_int(row.get("total_yards")),
                "passing_yards": _safe_int(row.get("passing_yards")),
                "rushing_yards": _safe_int(row.get("rushing_yards")),
                "turnovers": _safe_int(row.get("turnovers")),
                "time_of_possession": _safe_float(row.get("time_of_possession")),
                "total_plays": _safe_int(row.get("total_plays")),
                "pass_attempts": _safe_int(row.get("pass_attempts")),
                "rush_attempts": _safe_int(row.get("rush_attempts")),
                "redzone_attempts": _safe_int(row.get("redzone_attempts")),
                "redzone_scores": _safe_int(row.get("redzone_scores")),
                "third_down_conv": _safe_float(row.get("third_down_conv")),
                "sacks_allowed": _safe_int(row.get("sacks_allowed")),
                "neutral_pass_plays": _safe_int(row.get("neutral_pass_plays")),
                "neutral_run_plays": _safe_int(row.get("neutral_run_plays")),
                "neutral_pass_rate": _safe_float(row.get("neutral_pass_rate")),
                "neutral_pass_rate_lg": _safe_float(row.get("neutral_pass_rate_lg")),
                "neutral_pass_rate_oe": _safe_float(row.get("neutral_pass_rate_oe")),
                "drive_count": _safe_int(row.get("drive_count")),
                "drive_success_rate": _safe_float(row.get("drive_success_rate")),
                "avg_drive_epa": _safe_float(row.get("avg_drive_epa")),
                "points_per_drive": _safe_float(row.get("points_per_drive")),
                "pace_sec_per_play": _safe_float(row.get("pace_sec_per_play")),
            })
            count += 1
            existing_keys.add(key)
        return count

    def nullify_zero_redzone_team_stats(self) -> int:
        """Set redzone_attempts/redzone_scores to NULL where they are 0.

        This allows a subsequent insert (e.g. from PBP data) to fill them in
        via the COALESCE logic in insert_team_stats. Only touches rows that
        are clearly unpopulated (both columns are 0).
        """
        query = """
            UPDATE team_stats
            SET redzone_attempts = NULL, redzone_scores = NULL
            WHERE redzone_attempts = 0 AND redzone_scores = 0
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            count = cursor.rowcount
            conn.commit()
        return count

    def aggregate_team_defense_from_players(self, season: int = None) -> pd.DataFrame:
        """
        Aggregate team_defense_stats from player_weekly_stats and team_stats.
        Computes fantasy points allowed by position plus real counting stats
        (yards, sacks, interceptions) from team_stats cross-referenced by opponent.
        """
        # Fantasy points allowed by position
        fp_query = """
            SELECT
                pws.opponent AS team,
                pws.season,
                pws.week,
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
            fp_query += " AND pws.season = ?"
            params.append(season)
        fp_query += " GROUP BY pws.opponent, pws.season, pws.week ORDER BY pws.season, pws.week"

        with self._get_connection() as conn:
            fp_df = pd.read_sql_query(fp_query, conn, params=params)

        if fp_df.empty:
            return fp_df

        # Yards allowed = opponent's offensive yards from team_stats
        ts_query = """
            SELECT team AS offense_team, season, week, opponent,
                   COALESCE(points_scored, 0) AS opp_points,
                   COALESCE(total_yards, 0) AS opp_yards,
                   COALESCE(passing_yards, 0) AS opp_pass_yards,
                   COALESCE(rushing_yards, 0) AS opp_rush_yards,
                   COALESCE(sacks_allowed, 0) AS opp_sacks_taken
            FROM team_stats
            WHERE opponent IS NOT NULL AND opponent != ''
        """
        ts_params = []
        if season is not None:
            ts_query += " AND season = ?"
            ts_params.append(season)

        with self._get_connection() as conn:
            ts_df = pd.read_sql_query(ts_query, conn, params=ts_params)

        if not ts_df.empty:
            # Defense team = opponent column in team_stats
            ts_def = ts_df.rename(columns={
                'opponent': 'team',
                'opp_points': 'points_allowed',
                'opp_yards': 'yards_allowed',
                'opp_pass_yards': 'passing_yards_allowed',
                'opp_rush_yards': 'rushing_yards_allowed',
                'opp_sacks_taken': 'sacks',
            })
            fp_df = fp_df.merge(
                ts_def[['team', 'season', 'week', 'points_allowed', 'yards_allowed',
                        'passing_yards_allowed', 'rushing_yards_allowed', 'sacks']],
                on=['team', 'season', 'week'], how='left'
            )
            for col in ['points_allowed', 'yards_allowed', 'passing_yards_allowed', 'rushing_yards_allowed', 'sacks']:
                fp_df[col] = fp_df[col].fillna(0).astype(int)
        else:
            for col in ['points_allowed', 'yards_allowed', 'passing_yards_allowed', 'rushing_yards_allowed', 'sacks']:
                fp_df[col] = 0

        # Interceptions: sum of QB interceptions thrown against this defense
        int_query = """
            SELECT opponent AS team, season, week,
                   SUM(interceptions) AS interceptions
            FROM player_weekly_stats
            WHERE opponent IS NOT NULL AND opponent != ''
        """
        int_params = []
        if season is not None:
            int_query += " AND season = ?"
            int_params.append(season)
        int_query += " GROUP BY opponent, season, week"

        with self._get_connection() as conn:
            int_df = pd.read_sql_query(int_query, conn, params=int_params)

        if not int_df.empty:
            fp_df = fp_df.merge(int_df, on=['team', 'season', 'week'], how='left', suffixes=('', '_from_players'))
            if 'interceptions_from_players' in fp_df.columns:
                fp_df['interceptions'] = fp_df['interceptions_from_players'].fillna(0).astype(int)
                fp_df = fp_df.drop(columns=['interceptions_from_players'])
            elif 'interceptions' not in fp_df.columns:
                fp_df['interceptions'] = 0
        else:
            if 'interceptions' not in fp_df.columns:
                fp_df['interceptions'] = 0

        fp_df['fumbles_recovered'] = 0  # Not reliably derivable from player stats

        return fp_df

    def ensure_team_defense_stats(self, season: int = None) -> int:
        """Populate team_defense_stats from player_weekly_stats. Returns number of rows upserted."""
        agg = self.aggregate_team_defense_from_players(season=season)
        if agg.empty:
            return 0
        count = 0
        with self._get_connection() as conn:
            cur = conn.cursor()
            for _, row in agg.iterrows():
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
                except Exception:
                    pass
            conn.commit()
        return count
    
    def populate_player_team_history(self) -> int:
        """Derive player_team_history from player_weekly_stats.

        Groups by (player_id, team, season) to find the first and last week
        each player was on each team per season. Uses INSERT OR REPLACE to
        be idempotent.
        """
        query = """
            SELECT player_id, team, season,
                   MIN(week) AS start_week, MAX(week) AS end_week
            FROM player_weekly_stats
            WHERE player_id IS NOT NULL AND team IS NOT NULL AND team != ''
            GROUP BY player_id, team, season
        """
        with self._get_connection() as conn:
            rows = pd.read_sql_query(query, conn)
        if rows.empty:
            return 0
        count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for _, row in rows.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO player_team_history
                        (player_id, team, season, start_week, end_week)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        str(row["player_id"]),
                        str(row["team"]),
                        int(row["season"]),
                        int(row["start_week"]),
                        int(row["end_week"]),
                    ))
                    count += 1
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
