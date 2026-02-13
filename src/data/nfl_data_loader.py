"""
NFL Data Loader using nfl-data-py package

This module provides a clean interface to load NFL data from the nflverse
data repository via the nfl-data-py package.

Data sources:
- Weekly player stats (fantasy-relevant)
- Seasonal aggregates
- Snap counts
- Schedules
- Rosters

Advantages over scraping:
- Reliable, maintained data source
- Fast (parquet files)
- Comprehensive (all positions, all stats)
- Historical data back to 1999
"""

import os
import ssl
import certifi
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Fix SSL certificate issue before importing nfl_data_py
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import nfl_data_py as nfl

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import POSITIONS, SCORING
from src.utils.database import DatabaseManager
from src.utils.helpers import calculate_fantasy_points
from src.utils.nfl_calendar import get_current_nfl_season, get_current_nfl_week, current_season_has_weeks_played


def _to_scalar_int(x, default: int = 0) -> int:
    """Coerce to int; handle Series from duplicate columns (take first value)."""
    if hasattr(x, "iloc"):
        x = x.iloc[0] if len(x) else default
    if pd.isna(x):
        return default
    return int(x)


def _to_scalar_float(x, default: float = 0.0) -> float:
    """Coerce to float; handle Series from duplicate columns (take first value)."""
    if hasattr(x, "iloc"):
        x = x.iloc[0] if len(x) else default
    if pd.isna(x):
        return default
    return float(x)


def _to_scalar_str(x, default: str = "") -> str:
    """Coerce to str; handle Series from duplicate columns (take first value)."""
    if hasattr(x, "iloc"):
        x = x.iloc[0] if len(x) else default
    if pd.isna(x):
        return default
    return str(x)


class NFLDataLoader:
    """
    Load NFL data from nfl-data-py package and store in database.
    
    This replaces the Pro Football Reference scraper with a more
    reliable data source.
    """
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def load_weekly_data(self, seasons: List[int], 
                         store_in_db: bool = True,
                         use_pbp_fallback: bool = True) -> pd.DataFrame:
        """
        Load weekly player statistics. Falls back to PBP aggregation when weekly is empty.
        
        Args:
            seasons: List of seasons to load (e.g., [2020, 2021, 2022])
            store_in_db: Whether to store data in the database
            use_pbp_fallback: If True, use play-by-play aggregation when weekly data is missing
            
        Returns:
            DataFrame with weekly player stats
        """
        print(f"Loading weekly data for seasons: {seasons}")
        
        current_season = get_current_nfl_season()
        in_season_current = current_season_has_weeks_played()
        
        all_dfs = []
        for season in seasons:
            df = pd.DataFrame()
            # Current season with at least one week played: try PBP first (weekly often empty for in-progress season)
            if use_pbp_fallback and season == current_season and in_season_current:
                try:
                    from src.data.pbp_stats_aggregator import get_weekly_stats_from_pbp
                    pbp_df = get_weekly_stats_from_pbp(season)
                    if not pbp_df.empty:
                        df = self._standardize_pbp_columns(pbp_df.copy())
                        print(f"  Current season {season}: loaded from PBP ({len(df)} records)")
                        # Optionally merge with weekly if available (prefer weekly for same player/week)
                        try:
                            weekly_df = nfl.import_weekly_data([season])
                            if not weekly_df.empty and len(weekly_df) >= 10:
                                weekly_df = self._standardize_weekly_columns(weekly_df)
                                key_cols = ["player_id", "season", "week"]
                                if all(c in df.columns and c in weekly_df.columns for c in key_cols):
                                    combined = pd.concat([df, weekly_df], ignore_index=True)
                                    df = combined.drop_duplicates(subset=key_cols, keep="first")
                                    print(f"  Merged weekly data for {season}: {len(df)} records")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"  PBP for current season {season}: {e}")
            
            # If we don't have current-season PBP data yet, try weekly
            if df.empty:
                try:
                    df = nfl.import_weekly_data([season])
                except Exception as e:
                    print(f"  Weekly data for {season}: {e}")
                    df = pd.DataFrame()
            
            # PBP fallback when weekly is empty or (current season) has fewer weeks than current NFL week
            need_pbp = df.empty or len(df) < 50
            if not need_pbp and use_pbp_fallback and "week" in df.columns:
                cur = get_current_nfl_week()
                if season == cur.get("season") and cur.get("week_num", 0) > 0:
                    max_week = int(df["week"].max()) if len(df) else 0
                    if max_week < cur["week_num"]:
                        need_pbp = True
            if need_pbp and use_pbp_fallback:
                try:
                    from src.data.pbp_stats_aggregator import get_weekly_stats_from_pbp
                    pbp_df = get_weekly_stats_from_pbp(season)
                    if not pbp_df.empty:
                        pbp_df = self._standardize_pbp_columns(pbp_df)
                        if not df.empty:
                            # Merge: prefer weekly when both have same player_id, season, week
                            key_cols = ["player_id", "season", "week"]
                            if all(c in df.columns and c in pbp_df.columns for c in key_cols):
                                combined = pd.concat([df, pbp_df], ignore_index=True)
                                df = combined.drop_duplicates(subset=key_cols, keep="first")
                            else:
                                df = pbp_df
                        else:
                            df = pbp_df
                        print(f"  Using PBP for {season}: {len(df)} records")
                except Exception as e2:
                    print(f"  PBP fallback for {season}: {e2}")
            if df.empty:
                continue
            
            if not df.empty:
                df = self._standardize_weekly_columns(df)
                df = df[df['position'].isin(POSITIONS)]
                df['fantasy_points'] = df.apply(
                    lambda row: self._calculate_fantasy_points(row), axis=1
                )
                all_dfs.append(df)
        
        if not all_dfs:
            return pd.DataFrame()
        
        # Ensure unique column names so concat does not raise InvalidIndexError
        def _unique_columns(d: pd.DataFrame) -> pd.DataFrame:
            if d.columns.duplicated().any():
                return d.loc[:, ~d.columns.duplicated()]
            return d
        all_dfs = [_unique_columns(d) for d in all_dfs]
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"  Loaded {len(df)} records total")
        print(f"  After filtering: {len(df)} records for {POSITIONS}")
        
        if store_in_db:
            self._store_weekly_data(df)
        
        return df
    
    def _standardize_pbp_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align PBP-aggregated columns with our schema (same names as weekly)."""
        if 'completions' in df.columns and 'passing_completions' not in df.columns:
            df = df.rename(columns={'completions': 'passing_completions'})
        for col in ['opponent', 'home_away', 'fumbles_lost']:
            if col not in df.columns:
                df[col] = "" if col == "opponent" else ("unknown" if col == "home_away" else 0)
        return df
    
    def _standardize_weekly_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match our database schema."""
        column_mapping = {
            'player_id': 'player_id',
            'player_name': 'name',
            'player_display_name': 'display_name',
            'position': 'position',
            'recent_team': 'team',
            'season': 'season',
            'week': 'week',
            'opponent_team': 'opponent',
            'completions': 'passing_completions',
            'attempts': 'passing_attempts',
            'passing_yards': 'passing_yards',
            'passing_tds': 'passing_tds',
            'interceptions': 'interceptions',
            'carries': 'rushing_attempts',
            'rushing_yards': 'rushing_yards',
            'rushing_tds': 'rushing_tds',
            'targets': 'targets',
            'receptions': 'receptions',
            'receiving_yards': 'receiving_yards',
            'receiving_tds': 'receiving_tds',
            'sack_fumbles_lost': 'fumbles_lost',
            'rushing_fumbles_lost': 'rushing_fumbles_lost',
            'receiving_fumbles_lost': 'receiving_fumbles_lost',
        }
        
        # Rename columns that exist (skip if target name already present to avoid duplicate columns)
        rename_dict = {
            k: v for k, v in column_mapping.items()
            if k in df.columns and (k == v or v not in df.columns)
        }
        df = df.rename(columns=rename_dict)
        
        # Use display_name if name is missing
        if 'name' not in df.columns and 'display_name' in df.columns:
            df['name'] = df['display_name']
        
        # Combine fumbles lost
        fumble_cols = ['fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost']
        existing_fumble_cols = [c for c in fumble_cols if c in df.columns]
        if existing_fumble_cols:
            df['fumbles_lost'] = df[existing_fumble_cols].fillna(0).sum(axis=1)
        else:
            df['fumbles_lost'] = 0
        
        # Fill missing numeric columns with 0
        numeric_cols = [
            'passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds',
            'interceptions', 'rushing_attempts', 'rushing_yards', 'rushing_tds',
            'targets', 'receptions', 'receiving_yards', 'receiving_tds', 'fumbles_lost'
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)
        
        # Determine home/away
        if 'home_team' in df.columns:
            df['home_away'] = np.where(df['team'] == df['home_team'], 'home', 'away')
        else:
            df['home_away'] = 'unknown'
        
        return df
    
    def _calculate_fantasy_points(self, row: pd.Series) -> float:
        """Calculate PPR fantasy points for a player."""
        points = 0.0
        
        # Passing
        points += row.get('passing_yards', 0) * SCORING.get('passing_yards', 0.04)
        points += row.get('passing_tds', 0) * SCORING.get('passing_tds', 4)
        points += row.get('interceptions', 0) * SCORING.get('interceptions', -2)
        
        # Rushing
        points += row.get('rushing_yards', 0) * SCORING.get('rushing_yards', 0.1)
        points += row.get('rushing_tds', 0) * SCORING.get('rushing_tds', 6)
        
        # Receiving (PPR)
        points += row.get('receptions', 0) * SCORING.get('receptions', 1)
        points += row.get('receiving_yards', 0) * SCORING.get('receiving_yards', 0.1)
        points += row.get('receiving_tds', 0) * SCORING.get('receiving_tds', 6)
        
        # Fumbles
        points += row.get('fumbles_lost', 0) * SCORING.get('fumbles_lost', -2)
        
        return round(points, 1)
    
    def _store_weekly_data(self, df: pd.DataFrame):
        """Store weekly data in the database."""
        print("  Storing in database...")
        # Avoid duplicate column names (e.g. from merge/concat) so row values are scalars
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]

        players_inserted = 0
        stats_inserted = 0

        for _, row in df.iterrows():
            # Insert player (coerce to scalar in case of duplicate columns)
            player_data = {
                'player_id': _to_scalar_str(row['player_id'], ''),
                'name': _to_scalar_str(row['name'], ''),
                'position': _to_scalar_str(row['position'], ''),
            }
            self.db.insert_player(player_data)
            players_inserted += 1

            # Insert weekly stats (coerce to scalar; duplicate columns can make row values Series)
            stats_data = {
                'player_id': _to_scalar_str(row['player_id'], ''),
                'season': _to_scalar_int(row['season'], 0),
                'week': _to_scalar_int(row['week'], 0),
                'team': _to_scalar_str(row['team'], ''),
                'opponent': _to_scalar_str(row.get('opponent', ''), ''),
                'home_away': _to_scalar_str(row.get('home_away', 'unknown'), 'unknown'),
                'games_played': 1,
                'passing_attempts': _to_scalar_int(row.get('passing_attempts', 0), 0),
                'passing_completions': _to_scalar_int(row.get('passing_completions', 0), 0),
                'passing_yards': _to_scalar_int(row.get('passing_yards', 0), 0),
                'passing_tds': _to_scalar_int(row.get('passing_tds', 0), 0),
                'interceptions': _to_scalar_int(row.get('interceptions', 0), 0),
                'rushing_attempts': _to_scalar_int(row.get('rushing_attempts', 0), 0),
                'rush_inside_10': _to_scalar_int(row.get('rush_inside_10', 0), 0),
                'rush_inside_5': _to_scalar_int(row.get('rush_inside_5', 0), 0),
                'rushing_yards': _to_scalar_int(row.get('rushing_yards', 0), 0),
                'rushing_tds': _to_scalar_int(row.get('rushing_tds', 0), 0),
                'targets': _to_scalar_int(row.get('targets', 0), 0),
                'targets_15_plus': _to_scalar_int(row.get('targets_15_plus', 0), 0),
                'receptions': _to_scalar_int(row.get('receptions', 0), 0),
                'air_yards': _to_scalar_float(row.get('air_yards', 0.0), 0.0),
                'receiving_yards': _to_scalar_int(row.get('receiving_yards', 0), 0),
                'receiving_tds': _to_scalar_int(row.get('receiving_tds', 0), 0),
                'fumbles_lost': _to_scalar_int(row.get('fumbles_lost', 0), 0),
                'fantasy_points': _to_scalar_float(row.get('fantasy_points', 0), 0.0),
            }
            self.db.insert_player_weekly_stats(stats_data)
            stats_inserted += 1
        
        print(f"  Stored {players_inserted} players, {stats_inserted} weekly records")

    def store_weekly_dataframe(self, df: pd.DataFrame) -> None:
        """
        Standardize, filter to POSITIONS, compute fantasy_points if missing, then store in DB.

        Shared path for PBP-only or auto_refresh: ingest a weekly DataFrame into player_weekly_stats
        without duplicating insert logic. Accepts both weekly-style and PBP-style column names.
        """
        if df.empty:
            return
        if not all(c in df.columns for c in ["player_id", "name", "position", "team", "season", "week"]):
            raise ValueError("DataFrame must have player_id, name, position, team, season, week")
        df = df.copy()
        df = self._standardize_pbp_columns(df)
        df = self._standardize_weekly_columns(df)
        df = df[df["position"].isin(POSITIONS)]
        if "fantasy_points" not in df.columns or df["fantasy_points"].isna().any():
            df["fantasy_points"] = df.apply(self._calculate_fantasy_points, axis=1)
        self._store_weekly_data(df)

    def load_snap_counts(self, seasons: List[int], 
                         store_in_db: bool = True) -> pd.DataFrame:
        """Load snap count data."""
        print(f"Loading snap counts for seasons: {seasons}")
        
        try:
            df = nfl.import_snap_counts(seasons)
        except Exception as e:
            print(f"Error loading snap counts: {e}")
            return pd.DataFrame()
        
        print(f"  Loaded {len(df)} records")
        
        # Note: When using PBP path (get_weekly_stats_from_pbp), snap_count/team_snaps come from pbp_stats_aggregator.merge_with_snaps.
        
        return df
    
    def load_schedules(self, seasons: List[int],
                       store_in_db: bool = True) -> pd.DataFrame:
        """
        Load NFL schedules.
        
        Args:
            seasons: List of seasons to load
            store_in_db: Whether to store in database
            
        Returns:
            DataFrame with schedule data
        """
        print(f"Loading schedules for seasons: {seasons}")
        
        try:
            df = nfl.import_schedules(seasons)
        except Exception as e:
            print(f"Error loading schedules: {e}")
            return pd.DataFrame()
        
        print(f"  Loaded {len(df)} games")
        
        if store_in_db:
            self._store_schedules(df)
        
        return df
    
    def _store_schedules(self, df: pd.DataFrame):
        """Store schedule data in database. Maps game_type to week 19-22 for playoffs (SB=22)."""
        print("  Storing schedules in database...")
        # nflverse game_type: REG, WC, DIV, CON, SB -> align with nfl_calendar week_num 19-22
        _GAME_TYPE_WEEK = {"WC": 19, "DIV": 20, "CON": 21, "SB": 22}
        count = 0
        for _, row in df.iterrows():
            week = _to_scalar_int(row.get("week"), 0)
            if "game_type" in row.index and row.get("game_type") is not None:
                gt = str(row.get("game_type", "")).strip().upper()
                if gt in _GAME_TYPE_WEEK:
                    week = _GAME_TYPE_WEEK[gt]
            schedule_data = {
                "season": _to_scalar_int(row["season"], 0),
                "week": week,
                "home_team": _to_scalar_str(row["home_team"], ""),
                "away_team": _to_scalar_str(row["away_team"], ""),
                "game_id": _to_scalar_str(row.get("game_id", ""), ""),
                "game_time": str(_to_scalar_str(row.get("gameday", ""), "")),
                "venue": _to_scalar_str(row.get("stadium", ""), ""),
                "home_score": row.get("home_score"),
                "away_score": row.get("away_score"),
            }
            self.db.insert_schedule(schedule_data)
            count += 1
        print(f"  Stored {count} games")
    
    def get_available_seasons(self) -> List[int]:
        """Get list of seasons to consider (from config AVAILABLE_SEASONS_START_YEAR through current + 1)."""
        from config.settings import AVAILABLE_SEASONS_START_YEAR
        current = get_current_nfl_season()
        return list(range(AVAILABLE_SEASONS_START_YEAR, current + 2))
    
    def check_schedule_availability(self, season: int) -> bool:
        """Check if schedule is available for a given season."""
        try:
            df = nfl.import_schedules([season])
            return len(df) > 0
        except Exception:
            return False
    
    def get_latest_available_season(self) -> int:
        """Get the most recent season with weekly or PBP data available."""
        current = get_current_nfl_season()
        for year in range(current + 1, 2015, -1):
            try:
                df = nfl.import_weekly_data([year])
                if len(df) > 0:
                    return year
            except Exception:
                pass
        return current  # Fallback to current NFL season
    
    def load_upcoming_schedule(self, season: int = None) -> pd.DataFrame:
        """
        Load upcoming season schedule.
        
        For seasons not yet in nfl-data-py, this will return empty.
        Defaults to current NFL season when not specified.
        """
        if season is None:
            season = get_current_nfl_season()
        print(f"Loading {season} schedule...")
        
        try:
            df = nfl.import_schedules([season])
            if len(df) > 0:
                print(f"  Found {len(df)} games for {season}")
                return df
            else:
                print(f"  No schedule available for {season} yet")
                return pd.DataFrame()
        except Exception as e:
            print(f"  Schedule not available: {e}")
            return pd.DataFrame()
    
    def load_rosters(self, seasons: List[int]) -> pd.DataFrame:
        """
        Load NFL rosters with current team assignments.
        
        This is critical for tracking:
        - Offseason trades and signings
        - Current team assignments
        - Depth chart positions
        - Contract status
        
        Args:
            seasons: List of seasons to load rosters for
            
        Returns:
            DataFrame with roster data including player-team assignments
        """
        print(f"Loading rosters for seasons: {seasons}")
        
        try:
            df = nfl.import_rosters(seasons)
            if len(df) > 0:
                print(f"  Loaded {len(df)} roster entries")
                return df
            else:
                print(f"  No roster data available")
                return pd.DataFrame()
        except Exception as e:
            print(f"  Error loading rosters: {e}")
            return pd.DataFrame()
    
    def sync_player_teams(self, season: int = None) -> pd.DataFrame:
        """
        Sync player-team assignments for the current/upcoming season.
        
        This should be run before each season to capture:
        - Free agent signings
        - Trades
        - Roster cuts
        
        Args:
            season: Season to sync (defaults to current year)
            
        Returns:
            DataFrame with updated player-team mappings
        """
        from datetime import datetime
        season = season or datetime.now().year
        
        print(f"Syncing player-team assignments for {season}...")
        
        roster_df = self.load_rosters([season])
        
        if roster_df.empty:
            print("  No roster data available for sync")
            return pd.DataFrame()
        
        # Extract key fields for team assignment
        team_assignments = roster_df[['player_id', 'player_name', 'position', 'team', 
                                       'depth_chart_position', 'status', 'jersey_number']].copy()
        team_assignments = team_assignments.rename(columns={'player_name': 'name'})
        
        # Filter to fantasy-relevant positions
        team_assignments = team_assignments[team_assignments['position'].isin(POSITIONS)]
        
        print(f"  Synced {len(team_assignments)} player-team assignments")
        
        return team_assignments
    
    def get_roster_changes(self, current_season: int = None, 
                           previous_season: int = None) -> pd.DataFrame:
        """
        Identify players who changed teams between seasons.
        
        Args:
            current_season: Current season year
            previous_season: Previous season year
            
        Returns:
            DataFrame with roster changes (trades, signings, etc.)
        """
        from datetime import datetime
        current_season = current_season or datetime.now().year
        previous_season = previous_season or current_season - 1
        
        print(f"Finding roster changes from {previous_season} to {current_season}...")
        
        # Load rosters for both seasons
        current_roster = self.load_rosters([current_season])
        previous_roster = self.load_rosters([previous_season])
        
        if current_roster.empty or previous_roster.empty:
            print("  Cannot compare rosters - missing data")
            return pd.DataFrame()
        
        # Get last team for each player in previous season
        prev_teams = previous_roster.groupby('player_id').agg({
            'player_name': 'first',
            'team': 'last',
            'position': 'first'
        }).reset_index()
        prev_teams = prev_teams.rename(columns={'team': 'previous_team', 'player_name': 'name'})
        
        # Get current team for each player
        curr_teams = current_roster.groupby('player_id').agg({
            'player_name': 'first',
            'team': 'last',
            'position': 'first',
            'depth_chart_position': 'first'
        }).reset_index()
        curr_teams = curr_teams.rename(columns={'team': 'current_team', 'player_name': 'name'})
        
        # Merge to find changes
        merged = prev_teams.merge(curr_teams[['player_id', 'current_team', 'depth_chart_position']], 
                                   on='player_id', how='outer')
        
        # Identify changes
        changes = merged[merged['previous_team'] != merged['current_team']].copy()
        changes = changes[changes['position'].isin(POSITIONS)]
        
        # Categorize change type
        def categorize_change(row):
            if pd.isna(row['previous_team']):
                return 'New Player'
            elif pd.isna(row['current_team']):
                return 'Left League'
            else:
                return 'Team Change'
        
        changes['change_type'] = changes.apply(categorize_change, axis=1)
        changes = changes[changes['change_type'] == 'Team Change']  # Focus on team changes
        
        print(f"  Found {len(changes)} roster changes")
        
        return changes[['name', 'position', 'previous_team', 'current_team', 'depth_chart_position', 'change_type']]
    
    def load_depth_charts(self, season: int = None) -> pd.DataFrame:
        """
        Load depth chart data for utilization modeling.
        
        Depth chart position is critical for predicting:
        - Snap share expectations
        - Target share expectations
        - Red zone opportunity share
        
        Args:
            season: Season to load depth charts for
            
        Returns:
            DataFrame with depth chart positions
        """
        from datetime import datetime
        season = season or datetime.now().year
        
        print(f"Loading depth charts for {season}...")
        
        roster_df = self.load_rosters([season])
        
        if roster_df.empty:
            return pd.DataFrame()
        
        # Extract depth chart info
        depth_df = roster_df[['player_id', 'player_name', 'position', 'team', 
                              'depth_chart_position', 'status']].copy()
        depth_df = depth_df.rename(columns={'player_name': 'name'})
        
        # Filter to fantasy positions
        depth_df = depth_df[depth_df['position'].isin(POSITIONS)]
        
        # Calculate depth chart rank within position/team
        depth_df['depth_rank'] = depth_df.groupby(['team', 'position']).cumcount() + 1
        
        # Create utilization expectation based on depth
        def expected_utilization(row):
            pos = row['position']
            rank = row['depth_rank']
            
            if pos == 'QB':
                return 1.0 if rank == 1 else 0.05
            elif pos == 'RB':
                if rank == 1: return 0.55
                elif rank == 2: return 0.30
                elif rank == 3: return 0.10
                else: return 0.05
            elif pos == 'WR':
                if rank == 1: return 0.30
                elif rank == 2: return 0.25
                elif rank == 3: return 0.20
                elif rank == 4: return 0.15
                else: return 0.05
            elif pos == 'TE':
                if rank == 1: return 0.70
                elif rank == 2: return 0.20
                else: return 0.10
            return 0.0
        
        depth_df['expected_utilization'] = depth_df.apply(expected_utilization, axis=1)
        
        print(f"  Loaded depth charts for {len(depth_df)} players")
        
        return depth_df


def load_all_historical_data(seasons: List[int] = None):
    """
    Load all historical data from nfl-data-py.
    
    This is the main entry point for populating the database.
    """
    loader = NFLDataLoader()
    
    if seasons is None:
        from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
        seasons = list(range(MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON + 1))
    
    print("="*60)
    print("Loading NFL Data from nfl-data-py")
    print("="*60)
    
    # Load weekly player data
    weekly_df = loader.load_weekly_data(seasons)
    
    # Load schedules
    schedule_df = loader.load_schedules(seasons)
    
    # Check what we have
    db = DatabaseManager()
    db_seasons = db.get_seasons_with_data()
    
    print("\n" + "="*60)
    print("Data Loading Complete")
    print("="*60)
    print(f"Seasons in database: {db_seasons}")
    
    for s in db_seasons:
        latest_week = db.get_latest_week_for_season(s)
        print(f"  {s}: through week {latest_week}")
    
    return weekly_df


if __name__ == "__main__":
    import argparse
    from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
    parser = argparse.ArgumentParser(description='Load NFL data from nfl-data-py (PBP fallback when weekly missing)')
    parser.add_argument('--seasons', type=str, default=None,
                        help=f'Seasons to load (e.g. "{MIN_HISTORICAL_YEAR}-{CURRENT_NFL_SEASON}" or "2022,2023,2024"). Default: {MIN_HISTORICAL_YEAR} through current NFL season ({CURRENT_NFL_SEASON}).')
    
    args = parser.parse_args()
    
    if args.seasons is None:
        seasons = None
    elif '-' in args.seasons:
        start, end = args.seasons.split('-')
        seasons = list(range(int(start), int(end) + 1))
    else:
        seasons = [int(s.strip()) for s in args.seasons.split(',')]
    
    load_all_historical_data(seasons)
