"""
Real-Time Data Integration
Fetches current season data and caches it efficiently.
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class RealTimeDataFetcher:
    """
    Fetches and caches real-time NFL data.
    Updates every 30 minutes during season, daily in offseason.
    """
    
    def __init__(self, cache_dir: str = "../data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = Path("../data/nfl_data.db")
    
    def get_current_season(self) -> int:
        """Determine current NFL season year."""
        today = datetime.now()
        # NFL season runs Sep-Feb, so if Jan-Feb use previous year
        if today.month <= 2:
            return today.year - 1
        return today.year
    
    def get_current_week(self) -> int:
        """Determine current NFL week (1-19)."""
        today = datetime.now()
        season = self.get_current_season()
        
        # Season typically starts first Thursday after Labor Day (approx Sep 4)
        season_start = datetime(season, 9, 4)
        
        # Add days until Thursday
        days_until_thursday = (3 - season_start.weekday()) % 7
        season_start = season_start + timedelta(days=days_until_thursday)
        
        # Calculate weeks elapsed
        if today < season_start:
            return 0  # Preseason
        
        weeks_elapsed = (today - season_start).days // 7 + 1
        return min(weeks_elapsed, 19)
    
    def should_refresh_cache(self, cache_file: Path) -> bool:
        """Check if cache should be refreshed."""
        if not cache_file.exists():
            return True
        
        # Get file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        
        # During season (Sep-Jan): refresh every 30 min
        # Offseason: refresh daily
        today = datetime.now()
        if 9 <= today.month <= 12 or today.month <= 1:
            return file_age > timedelta(minutes=30)
        else:
            return file_age > timedelta(hours=24)
    
    def fetch_current_season_pbp(self) -> pd.DataFrame:
        """
        Fetch play-by-play data for current season.
        Cached to avoid excessive API calls.
        """
        season = self.get_current_season()
        current_week = self.get_current_week()
        
        cache_file = self.cache_dir / f"pbp_{season}_week{current_week}.parquet"
        
        if not self.should_refresh_cache(cache_file):
            print(f"📦 Loading cached PBP data (Week {current_week})")
            return pd.read_parquet(cache_file)
        
        print(f"🔄 Fetching fresh PBP data for {season} (through Week {current_week})...")
        
        try:
            # Fetch current season play-by-play
            pbp = nfl.import_pbp_data([season])
            
            # Filter to completed weeks only
            if current_week > 0:
                pbp = pbp[pbp['week'] <= current_week]
            
            # Cache it
            pbp.to_parquet(cache_file, index=False)
            print(f"✅ Cached {len(pbp):,} plays")
            
            return pbp
            
        except Exception as e:
            print(f"⚠️  Error fetching PBP: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['season', 'week', 'player_id', 'play_type'])
    
    def fetch_weekly_stats(self) -> pd.DataFrame:
        """
        Fetch aggregated weekly stats for current season.
        More efficient than aggregating from PBP.
        """
        season = self.get_current_season()
        current_week = self.get_current_week()
        
        cache_file = self.cache_dir / f"weekly_{season}_week{current_week}.parquet"
        
        if not self.should_refresh_cache(cache_file):
            print(f"📦 Loading cached weekly stats")
            return pd.read_parquet(cache_file)
        
        print(f"🔄 Fetching weekly stats for {season}...")
        
        try:
            # Fetch weekly data
            weekly = nfl.import_weekly_data([season])
            
            # Filter to completed weeks
            if current_week > 0:
                weekly = weekly[weekly['week'] <= current_week]
            
            # Cache it
            weekly.to_parquet(cache_file, index=False)
            print(f"✅ Cached {len(weekly):,} player-weeks")
            
            return weekly
            
        except Exception as e:
            print(f"⚠️  Error fetching weekly stats: {e}")
            return pd.DataFrame()
    
    def fetch_roster_data(self) -> pd.DataFrame:
        """Fetch current roster/player info."""
        cache_file = self.cache_dir / "rosters_current.parquet"
        
        if not self.should_refresh_cache(cache_file):
            return pd.read_parquet(cache_file)
        
        print("🔄 Fetching roster data...")
        
        try:
            season = self.get_current_season()
            rosters = nfl.import_rosters([season])
            rosters.to_parquet(cache_file, index=False)
            return rosters
        except Exception as e:
            print(f"⚠️  Error fetching rosters: {e}")
            return pd.DataFrame()
    
    def aggregate_to_player_stats(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate PBP to player-week stats.
        Calculate utilization metrics.
        """
        if pbp.empty:
            return pd.DataFrame()
        
        # Passing stats
        passing = pbp[pbp['pass_attempt'] == 1].groupby([
            'season', 'week', 'passer_player_id', 'passer_player_name', 'posteam'
        ]).agg({
            'pass_attempt': 'sum',
            'complete_pass': 'sum',
            'passing_yards': 'sum',
            'pass_touchdown': 'sum',
            'interception': 'sum',
        }).reset_index()
        
        passing.columns = ['season', 'week', 'player_id', 'player_name', 'team',
                          'attempts', 'completions', 'yards', 'tds', 'ints']
        passing['position'] = 'QB'
        
        # Rushing stats
        rushing = pbp[pbp['rush_attempt'] == 1].groupby([
            'season', 'week', 'rusher_player_id', 'rusher_player_name', 'posteam'
        ]).agg({
            'rush_attempt': 'sum',
            'rushing_yards': 'sum',
            'rush_touchdown': 'sum',
        }).reset_index()
        
        rushing.columns = ['season', 'week', 'player_id', 'player_name', 'team',
                          'carries', 'yards', 'tds']
        
        # Receiving stats
        receiving = pbp[pbp['pass_attempt'] == 1].groupby([
            'season', 'week', 'receiver_player_id', 'receiver_player_name', 'posteam'
        ]).agg({
            'pass_attempt': 'sum',  # targets
            'complete_pass': 'sum',  # receptions
            'receiving_yards': 'sum',
            'pass_touchdown': 'sum',
        }).reset_index()
        
        receiving.columns = ['season', 'week', 'player_id', 'player_name', 'team',
                            'targets', 'receptions', 'yards', 'tds']
        
        return passing, rushing, receiving
    
    def update_database(self):
        """
        Update local database with fresh real-time data.
        Merges with historical data.
        """
        print("\n" + "="*60)
        print("REAL-TIME DATA UPDATE")
        print("="*60)
        
        # Fetch fresh data
        weekly = self.fetch_weekly_stats()
        
        if weekly.empty:
            print("⚠️  No new data to update")
            return
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        # Get current season/week
        season = self.get_current_season()
        week = self.get_current_week()
        
        print(f"\n📅 Current: {season} Season, Week {week}")
        
        # Delete existing data for current season (will re-insert fresh data)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM player_weekly_stats 
            WHERE season = ? AND week <= ?
        """, (season, week))
        
        deleted = cursor.rowcount
        print(f"🗑️  Removed {deleted} old records for re-insertion")
        
        # Prepare data for insertion
        weekly_clean = weekly[[
            'player_id', 'player_name', 'season', 'week', 'recent_team',
            'position', 'completions', 'attempts', 'passing_yards', 'passing_tds',
            'interceptions', 'carries', 'rushing_yards', 'rushing_tds',
            'targets', 'receptions', 'receiving_yards', 'receiving_tds'
        ]].copy()
        
        weekly_clean.columns = [
            'player_id', 'player_name', 'season', 'week', 'team', 'position',
            'completions', 'pass_attempts', 'passing_yards', 'passing_tds',
            'interceptions', 'carries', 'rushing_yards', 'rushing_tds',
            'targets', 'receptions', 'receiving_yards', 'receiving_tds'
        ]
        
        # Insert fresh data
        weekly_clean.to_sql('player_weekly_stats', conn, if_exists='append', index=False)
        
        print(f"✅ Inserted {len(weekly_clean):,} fresh records")
        print(f"📊 Positions: {weekly_clean['position'].value_counts().to_dict()}")
        
        conn.commit()
        conn.close()
        
        print("\n✅ Database update complete!")
        
        return weekly_clean


# ============================================================================
# SCHEDULED EXECUTION
# ============================================================================

def run_scheduled_update():
    """Run this on a schedule (cron/GitHub Actions)."""
    fetcher = RealTimeDataFetcher()
    
    # Check if it's during season
    week = fetcher.get_current_week()
    
    if week > 0 and week <= 18:
        print(f"🏈 In-season (Week {week}): Fetching real-time data")
        fetcher.update_database()
    else:
        print(f"🏖️  Offseason: Skipping update")


if __name__ == "__main__":
    fetcher = RealTimeDataFetcher()
    
    # Manual update
    fetcher.update_database()
