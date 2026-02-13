"""
Play-by-Play Stats Aggregator

Derives weekly player stats from nfl-data-py play-by-play data.
This is used when weekly stats are not directly available (e.g., current season).
"""
import os
import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from pathlib import Path
from datetime import datetime


class PBPStatsAggregator:
    """Aggregate play-by-play data into weekly player stats."""
    
    def __init__(self):
        self.pbp_data = None
        self.snap_data = None
    
    def load_data(self, season: int) -> "PBPStatsAggregator":
        """Load PBP (required) and snap count (optional) data for a season. PBP-only path when snaps unavailable."""
        print(f"Loading play-by-play data for {season}...")
        self.pbp_data = nfl.import_pbp_data([season])
        print(f"  Loaded {len(self.pbp_data)} plays")
        
        self.snap_data = None
        try:
            snap_df = nfl.import_snap_counts([season])
            if snap_df is not None and not snap_df.empty:
                self.snap_data = snap_df
                print(f"  Loaded {len(self.snap_data)} snap records")
            else:
                print(f"  Snap counts for {season}: not available (using position inference)")
        except Exception as e:
            print(f"  Snap counts for {season}: not available ({e}); using position inference")
        
        return self
    
    def aggregate_passing_stats(self) -> pd.DataFrame:
        """Aggregate passing stats by player/week."""
        pbp = self.pbp_data
        
        # Filter to pass plays
        pass_plays = pbp[pbp['play_type'] == 'pass'].copy()
        
        # Aggregate by passer
        passing = pass_plays.groupby(['season', 'week', 'passer_player_id', 'passer_player_name', 'posteam']).agg({
            'passing_yards': 'sum',
            'pass_touchdown': 'sum',
            'interception': 'sum',
            'complete_pass': 'sum',
            'play_id': 'count'  # attempts
        }).reset_index()
        
        passing = passing.rename(columns={
            'passer_player_id': 'player_id',
            'passer_player_name': 'name',
            'posteam': 'team',
            'play_id': 'passing_attempts',
            'complete_pass': 'completions',
            'pass_touchdown': 'passing_tds',
            'interception': 'interceptions'
        })
        
        passing['position'] = 'QB'
        return passing
    
    def aggregate_rushing_stats(self) -> pd.DataFrame:
        """Aggregate rushing stats by player/week."""
        pbp = self.pbp_data
        
        # Filter to rush plays
        rush_plays = pbp[pbp['play_type'] == 'run'].copy()

        # Goal-line / high-value rushing opportunities
        # nflverse convention: yardline_100 is distance to opponent endzone.
        # inside 10 => yardline_100 <= 10, inside 5 => yardline_100 <= 5.
        if 'yardline_100' in rush_plays.columns:
            rush_plays['is_rush_inside_10'] = (rush_plays['yardline_100'] <= 10).astype(int)
            rush_plays['is_rush_inside_5'] = (rush_plays['yardline_100'] <= 5).astype(int)
        else:
            rush_plays['is_rush_inside_10'] = 0
            rush_plays['is_rush_inside_5'] = 0
        
        # Aggregate by rusher
        rushing = rush_plays.groupby(['season', 'week', 'rusher_player_id', 'rusher_player_name', 'posteam']).agg({
            'rushing_yards': 'sum',
            'rush_touchdown': 'sum',
            'play_id': 'count',  # attempts
            'is_rush_inside_10': 'sum',
            'is_rush_inside_5': 'sum',
        }).reset_index()
        
        rushing = rushing.rename(columns={
            'rusher_player_id': 'player_id',
            'rusher_player_name': 'name',
            'posteam': 'team',
            'play_id': 'rushing_attempts',
            'rush_touchdown': 'rushing_tds',
            'is_rush_inside_10': 'rush_inside_10',
            'is_rush_inside_5': 'rush_inside_5',
        })
        
        return rushing
    
    def aggregate_receiving_stats(self) -> pd.DataFrame:
        """Aggregate receiving stats by player/week."""
        pbp = self.pbp_data
        
        # Filter to pass plays with a receiver
        rec_plays = pbp[(pbp['play_type'] == 'pass') & (pbp['receiver_player_id'].notna())].copy()

        # Air yards + deep target proxies (15+ air yards)
        if 'air_yards' in rec_plays.columns:
            rec_plays['air_yards_filled'] = rec_plays['air_yards'].fillna(0)
            rec_plays['is_target_15_plus'] = (rec_plays['air_yards_filled'] >= 15).astype(int)
        else:
            rec_plays['air_yards_filled'] = 0.0
            rec_plays['is_target_15_plus'] = 0
        
        # Aggregate by receiver
        receiving = rec_plays.groupby(['season', 'week', 'receiver_player_id', 'receiver_player_name', 'posteam']).agg({
            'receiving_yards': 'sum',
            'complete_pass': 'sum',  # receptions
            'pass_touchdown': 'sum',
            'play_id': 'count',  # targets
            'air_yards_filled': 'sum',
            'is_target_15_plus': 'sum',
        }).reset_index()
        
        receiving = receiving.rename(columns={
            'receiver_player_id': 'player_id',
            'receiver_player_name': 'name',
            'posteam': 'team',
            'play_id': 'targets',
            'complete_pass': 'receptions',
            'pass_touchdown': 'receiving_tds',
            'air_yards_filled': 'air_yards',
            'is_target_15_plus': 'targets_15_plus',
        })
        
        return receiving
    
    def merge_with_snaps(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Merge stats with snap count data; set snap_count (offense_snaps) and team_snaps for utilization_score/DB."""
        if self.snap_data is None:
            return stats_df
        
        snap_pos = self.snap_data[['season', 'week', 'player', 'team', 'position', 'offense_snaps', 'offense_pct']].copy()
        snap_pos = snap_pos.rename(columns={'player': 'name'})
        snap_pos['snap_count'] = snap_pos['offense_snaps'].fillna(0).astype(int)
        team_snaps = (
            snap_pos.groupby(['season', 'week', 'team'], as_index=False)['offense_snaps']
            .sum()
            .rename(columns={'offense_snaps': 'team_snaps'})
        )
        team_snaps['team_snaps'] = team_snaps['team_snaps'].fillna(0).astype(int)
        snap_pos = snap_pos.merge(team_snaps, on=['season', 'week', 'team'], how='left')
        snap_pos['team_snaps'] = snap_pos['team_snaps'].fillna(0).astype(int)
        merged = stats_df.merge(
            snap_pos[['season', 'week', 'name', 'team', 'position', 'snap_count', 'team_snaps', 'offense_pct']],
            on=['season', 'week', 'name', 'team'],
            how='left'
        )
        if 'snap_count' not in merged.columns:
            merged['snap_count'] = 0
        if 'team_snaps' not in merged.columns:
            merged['team_snaps'] = 0
        merged['snap_count'] = merged['snap_count'].fillna(0).astype(int)
        merged['team_snaps'] = merged['team_snaps'].fillna(0).astype(int)
        merged['snap_share'] = np.where(
            merged['team_snaps'] > 0,
            merged['snap_count'] / merged['team_snaps'],
            0.0
        )
        return merged
    
    def calculate_fantasy_points(self, df: pd.DataFrame, ppr: float = 1.0) -> pd.DataFrame:
        """Calculate fantasy points (PPR scoring)."""
        df = df.copy()
        
        # Initialize
        df['fantasy_points'] = 0.0
        
        # Passing: 0.04 per yard, 4 per TD, -2 per INT
        if 'passing_yards' in df.columns:
            df['fantasy_points'] += df['passing_yards'].fillna(0) * 0.04
        if 'passing_tds' in df.columns:
            df['fantasy_points'] += df['passing_tds'].fillna(0) * 4
        if 'interceptions' in df.columns:
            df['fantasy_points'] -= df['interceptions'].fillna(0) * 2
        
        # Rushing: 0.1 per yard, 6 per TD
        if 'rushing_yards' in df.columns:
            df['fantasy_points'] += df['rushing_yards'].fillna(0) * 0.1
        if 'rushing_tds' in df.columns:
            df['fantasy_points'] += df['rushing_tds'].fillna(0) * 6
        
        # Receiving: 0.1 per yard, 6 per TD, PPR per reception
        if 'receiving_yards' in df.columns:
            df['fantasy_points'] += df['receiving_yards'].fillna(0) * 0.1
        if 'receiving_tds' in df.columns:
            df['fantasy_points'] += df['receiving_tds'].fillna(0) * 6
        if 'receptions' in df.columns:
            df['fantasy_points'] += df['receptions'].fillna(0) * ppr
        
        df['fantasy_points'] = df['fantasy_points'].round(1)
        
        return df
    
    def aggregate_all_stats(self, season: int) -> pd.DataFrame:
        """
        Aggregate all player stats from PBP data for a season.
        
        Returns DataFrame with weekly player stats similar to nfl.import_weekly_data()
        """
        self.load_data(season)
        if self.pbp_data is None or self.pbp_data.empty:
            return pd.DataFrame()
        
        # Get passing stats
        passing = self.aggregate_passing_stats()
        
        # Get rushing stats
        rushing = self.aggregate_rushing_stats()
        
        # Get receiving stats
        receiving = self.aggregate_receiving_stats()
        
        # Merge rushing and receiving for skill players
        skill = rushing.merge(
            receiving,
            on=['season', 'week', 'player_id', 'name', 'team'],
            how='outer'
        )
        
        # Combine with passing (QBs)
        all_stats = pd.concat([passing, skill], ignore_index=True)
        
        # Fill NaN with 0 for numeric columns
        numeric_cols = ['passing_yards', 'passing_tds', 'interceptions', 'completions', 'passing_attempts',
                        'rushing_yards', 'rushing_tds', 'rushing_attempts',
                        'receiving_yards', 'receiving_tds', 'receptions', 'targets']
        for col in numeric_cols:
            if col in all_stats.columns:
                all_stats[col] = all_stats[col].fillna(0)
        
        # Merge with snap data for positions
        all_stats = self.merge_with_snaps(all_stats)
        
        # Infer position if not from snaps
        if 'position' not in all_stats.columns or all_stats['position'].isna().any():
            all_stats['position'] = all_stats.apply(self._infer_position, axis=1)
        
        # Calculate fantasy points
        all_stats = self.calculate_fantasy_points(all_stats)
        
        # Filter to fantasy-relevant positions
        all_stats = all_stats[all_stats['position'].isin(['QB', 'RB', 'WR', 'TE', 'FB', 'HB'])]
        all_stats.loc[all_stats['position'].isin(['FB', 'HB']), 'position'] = 'RB'
        
        return all_stats
    
    def _infer_position(self, row) -> str:
        """Infer position from stats if not available."""
        if pd.notna(row.get('position')):
            return row['position']
        
        # If has passing attempts, likely QB
        if row.get('passing_attempts', 0) > 0:
            return 'QB'
        
        # If has rushing attempts but few/no targets, likely RB
        if row.get('rushing_attempts', 0) > row.get('targets', 0):
            return 'RB'
        
        # Default to WR for receivers
        return 'WR'


def _ensure_store_weekly_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has all columns required by NFLDataLoader._store_weekly_data.

    Adds defaults for opponent, home_away, fumbles_lost; renames completions -> passing_completions;
    ensures all stat columns exist so PBP-only output is valid for DB insert.
    """
    if df.empty:
        return df
    df = df.copy()
    if "completions" in df.columns and "passing_completions" not in df.columns:
        df["passing_completions"] = df["completions"]
    for col, default in [("opponent", ""), ("home_away", "unknown"), ("fumbles_lost", 0)]:
        if col not in df.columns:
            df[col] = default
    if "games_played" not in df.columns:
        df["games_played"] = 1
    stat_cols = [
        "passing_attempts", "passing_completions", "passing_yards", "passing_tds",
        "interceptions", "rushing_attempts", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
    ]
    for col in stat_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    # High-value touch source columns (optional; default to 0 when unavailable)
    for col, default in [
        ("rush_inside_10", 0),
        ("rush_inside_5", 0),
        ("targets_15_plus", 0),
        ("air_yards", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    return df


def get_weekly_stats_from_pbp(season: int) -> pd.DataFrame:
    """
    Return weekly player stats from PBP data in same shape as nfl.import_weekly_data.

    Used by nfl_data_loader and auto_refresh when weekly data is missing or incomplete.
    Column names and defaults match NFLDataLoader._store_weekly_data. Works for current
    season (current/next) when import_weekly_data is empty; snap counts are optional.
    """
    aggregator = PBPStatsAggregator()
    stats = aggregator.aggregate_all_stats(season)
    if stats.empty:
        return stats
    return _ensure_store_weekly_schema(stats)


def load_current_season_stats_from_pbp() -> pd.DataFrame:
    """
    Load current NFL season stats from PBP data (convenience wrapper).
    
    Returns DataFrame compatible with nfl_data_loader schema.
    """
    from src.utils.nfl_calendar import get_current_nfl_season
    season = get_current_nfl_season()
    stats = get_weekly_stats_from_pbp(season)
    if not stats.empty:
        print(f"\n{season} Season Stats Summary (from PBP):")
        print(f"  Total player-weeks: {len(stats)}")
        print(f"  Weeks: {sorted(stats['week'].unique())}")
        print(f"  Positions: {stats['position'].value_counts().to_dict()}")
    return stats


if __name__ == "__main__":
    stats = load_current_season_stats_from_pbp()
    
    # Show top players from latest week
    latest_week = stats['week'].max()
    latest = stats[stats['week'] == latest_week].sort_values('fantasy_points', ascending=False)
    
    print(f"\nTop 10 Players Week {latest_week}:")
    print(latest[['name', 'team', 'position', 'fantasy_points']].head(10).to_string())
