"""
NFL Data Auto-Refresh System

Automatically checks for and loads the latest NFL data from nfl-data-py.
Designed to keep the database up-to-date as new seasons and weeks become available.

Usage:
    python3 src/data/auto_refresh.py           # Check and load any new data
    python3 src/data/auto_refresh.py --force   # Force reload all data
"""

import os
import ssl
import certifi
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path

# Fix SSL certificate issue
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import nfl_data_py as nfl

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.database import DatabaseManager
from src.data.nfl_data_loader import NFLDataLoader
from src.utils.nfl_calendar import (
    get_current_nfl_season as get_current_nfl_season_calendar,
    get_current_nfl_week,
    current_season_has_weeks_played,
)


class NFLDataRefresher:
    """
    Automatically refresh NFL data from nfl-data-py.
    
    Checks for new seasons and weeks, and loads any missing data.
    """
    
    def __init__(self):
        self.db = DatabaseManager()
        self.loader = NFLDataLoader()
        self.current_year = datetime.now().year
        self.current_month = datetime.now().month
    
    def get_current_nfl_season(self) -> int:
        """Current NFL season (shared nfl_calendar logic)."""
        return get_current_nfl_season_calendar()
    
    def check_data_availability(self) -> Dict:
        """
        Check what data is available in nfl-data-py vs what we have locally.
        
        Returns dict with:
        - available_seasons: Seasons with weekly data in nfl-data-py
        - local_seasons: Seasons we have in our database
        - missing_seasons: Seasons we need to load
        - schedule_seasons: Seasons with schedule data available
        """
        current_season = self.get_current_nfl_season()
        
        # Check what's available remotely (weekly data; current season may be empty)
        available_seasons = []
        from config.settings import MIN_HISTORICAL_YEAR
        for year in range(MIN_HISTORICAL_YEAR, current_season + 2):  # Check up to next year
            try:
                df = nfl.import_weekly_data([year])
                if len(df) > 0:
                    available_seasons.append(year)
            except Exception:
                pass
        
        # In-season: current season is loadable from PBP even if weekly is empty
        in_season = current_season_has_weeks_played()
        if in_season and current_season not in available_seasons:
            available_seasons.append(current_season)
            available_seasons.sort()
        
        # Check what we have locally
        local_seasons = self.db.get_seasons_with_data()
        
        # Find missing (include current season when in-season so we load from PBP)
        missing_seasons = [s for s in available_seasons if s not in local_seasons]
        if current_season not in local_seasons and current_season not in missing_seasons:
            missing_seasons.append(current_season)
            missing_seasons.sort()
        
        # Check schedule availability
        schedule_seasons = []
        for year in range(current_season, current_season + 2):
            try:
                sched = nfl.import_schedules([year])
                if len(sched) > 0:
                    schedule_seasons.append(year)
            except Exception:
                pass
        
        return {
            'current_season': current_season,
            'available_seasons': available_seasons,
            'local_seasons': local_seasons,
            'missing_seasons': missing_seasons,
            'schedule_seasons': schedule_seasons,
        }
    
    def check_for_new_weeks(self, season: int) -> Tuple[int, int]:
        """
        Check if there are new weeks available for a season.
        
        Returns (local_max_week, remote_max_week)
        """
        # Get local max week
        local_max = self.db.get_latest_week_for_season(season)
        
        # Get remote max week
        try:
            df = nfl.import_weekly_data([season])
            remote_max = df['week'].max() if len(df) > 0 else 0
        except Exception:
            remote_max = 0
        
        return local_max, remote_max
    
    def refresh(self, force: bool = False) -> Dict:
        """
        Refresh data from nfl-data-py.
        
        Args:
            force: If True, reload all data even if we have it
            
        Returns:
            Dict with refresh results
        """
        print("="*60)
        print("NFL Data Auto-Refresh")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*60)
        
        status = self.check_data_availability()
        
        print(f"\nCurrent NFL Season: {status['current_season']}")
        print(f"Available in nfl-data-py: {status['available_seasons']}")
        print(f"In local database: {status['local_seasons']}")
        print(f"Missing seasons: {status['missing_seasons']}")
        print(f"Schedules available: {status['schedule_seasons']}")
        
        results = {
            'seasons_loaded': [],
            'weeks_updated': [],
            'schedules_loaded': [],
            'team_stats_backfilled': 0,
            'errors': []
        }
        
        # Load missing seasons (loader uses PBP fallback when weekly is empty/incomplete)
        if status['missing_seasons'] or force:
            seasons_to_load = status['available_seasons'] if force else status['missing_seasons']
            
            if seasons_to_load:
                print(f"\nLoading seasons: {seasons_to_load}")
                try:
                    self.loader.load_weekly_data(seasons_to_load, store_in_db=True, use_pbp_fallback=True)
                    results['seasons_loaded'] = seasons_to_load
                except Exception as e:
                    results['errors'].append(f"Error loading weekly data: {e}")
        
        # Check for new weeks in current season; use PBP if weekly has fewer weeks than current NFL week
        current = status['current_season']
        current_week_info = get_current_nfl_week()
        current_week_num = current_week_info.get("week_num", 0)
        if current in status['local_seasons']:
            local_week, remote_week = self.check_for_new_weeks(current)
            # If we're in-season and local is behind current NFL week, try PBP to fill gaps
            if current_week_num > 0 and local_week < current_week_num:
                try:
                    from src.data.pbp_stats_aggregator import get_weekly_stats_from_pbp
                    pbp_df = get_weekly_stats_from_pbp(current)
                    if not pbp_df.empty and pbp_df["week"].max() > local_week:
                        self.loader.load_weekly_data([current], store_in_db=True, use_pbp_fallback=True)
                        new_local = self.db.get_latest_week_for_season(current)
                        results['weeks_updated'].append({
                            'season': current,
                            'from_week': local_week,
                            'to_week': new_local
                        })
                except Exception as e:
                    pass  # Non-fatal; weekly refresh below may still help
            if remote_week > local_week:
                print(f"\nNew weeks available for {current}: {local_week} -> {remote_week}")
                try:
                    self.loader.load_weekly_data([current], store_in_db=True, use_pbp_fallback=True)
                    results['weeks_updated'].append({
                        'season': current,
                        'from_week': local_week,
                        'to_week': remote_week
                    })
                except Exception as e:
                    results['errors'].append(f"Error updating weeks: {e}")
        
        # Load schedules for current and upcoming seasons (needed for matchup/Super Bowl week 22)
        for season in status['schedule_seasons']:
            need_schedule = (
                season > max(status['local_seasons'] or [0])
                or not self.db.has_schedule_for_season(season)
            )
            if need_schedule:
                print(f"\nLoading {season} schedule...")
                try:
                    self.loader.load_schedules([season], store_in_db=True)
                    results['schedules_loaded'].append(season)
                except Exception as e:
                    results['errors'].append(f"Error loading schedule: {e}")
        
        # Backfill team_stats from player_weekly_stats when missing (no scrapers required)
        try:
            n_backfill = self.db.ensure_team_stats_from_players(season=None)
            if n_backfill > 0:
                results['team_stats_backfilled'] = n_backfill
        except Exception as e:
            results['errors'].append(f"Team stats backfill: {e}")
        
        # Summary
        print("\n" + "="*60)
        print("Refresh Complete")
        print("="*60)
        
        if results['seasons_loaded']:
            print(f"✅ Loaded seasons: {results['seasons_loaded']}")
        if results['weeks_updated']:
            for update in results['weeks_updated']:
                print(f"✅ Updated {update['season']}: week {update['from_week']} -> {update['to_week']}")
        if results['schedules_loaded']:
            print(f"✅ Loaded schedules: {results['schedules_loaded']}")
        if results.get('team_stats_backfilled', 0) > 0:
            print(f"✅ Team stats backfilled: {results['team_stats_backfilled']} rows")
        if results['errors']:
            for error in results['errors']:
                print(f"❌ {error}")
        
        if not any([results['seasons_loaded'], results['weeks_updated'],
                    results['schedules_loaded'], results.get('team_stats_backfilled', 0), results['errors']]):
            print("✅ Database is up-to-date")
        
        return results
    
    def get_status(self) -> Dict:
        """Get current data status for display."""
        status = self.check_data_availability()
        
        # Add week info for each season
        season_details = {}
        for season in status['local_seasons']:
            max_week = self.db.get_latest_week_for_season(season)
            season_details[season] = {'max_week': max_week}
        
        status['season_details'] = season_details
        return status


def auto_refresh(force: bool = False) -> Dict:
    """Convenience function to run auto-refresh."""
    refresher = NFLDataRefresher()
    return refresher.refresh(force=force)


def get_data_status() -> Dict:
    """Get current data status."""
    refresher = NFLDataRefresher()
    return refresher.get_status()


def schedule_auto_refresh(interval_hours: int = 24):
    """
    Schedule automatic data refresh at regular intervals.
    
    This can be run as a background process or cron job.
    
    Args:
        interval_hours: Hours between refresh checks (default: 24)
    """
    import time as time_module
    
    print(f"Starting scheduled auto-refresh (every {interval_hours} hours)")
    print("Press Ctrl+C to stop")
    
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running auto-refresh...")
            auto_refresh()
            
            print(f"Next refresh in {interval_hours} hours")
            time_module.sleep(interval_hours * 3600)
            
        except KeyboardInterrupt:
            print("\nStopping scheduled refresh")
            break
        except Exception as e:
            print(f"Error during refresh: {e}")
            print(f"Retrying in 1 hour...")
            time_module.sleep(3600)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-refresh NFL data')
    parser.add_argument('--force', action='store_true', 
                        help='Force reload all data')
    parser.add_argument('--status', action='store_true',
                        help='Just show status, do not refresh')
    parser.add_argument('--schedule', action='store_true',
                        help='Run scheduled refresh (every 24 hours)')
    parser.add_argument('--interval', type=int, default=24,
                        help='Hours between scheduled refreshes (default: 24)')
    
    args = parser.parse_args()
    
    if args.status:
        status = get_data_status()
        print("Current Data Status:")
        print(f"  Current NFL Season: {status['current_season']}")
        print(f"  Local seasons: {status['local_seasons']}")
        print(f"  Season details:")
        for season, details in status.get('season_details', {}).items():
            print(f"    {season}: through week {details['max_week']}")
    elif args.schedule:
        schedule_auto_refresh(interval_hours=args.interval)
    else:
        auto_refresh(force=args.force)
