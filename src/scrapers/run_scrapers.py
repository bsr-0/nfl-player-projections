"""Main script to load NFL data from nfl-data-py and populate the database."""
import argparse
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.nfl_data_loader import NFLDataLoader
from src.utils.database import DatabaseManager
from config.settings import SEASONS_TO_SCRAPE, CURRENT_NFL_SEASON


def run_all_scrapers(seasons: List[int] = None, refresh_only: bool = False,
                     force_rescrape: bool = False):
    """
    Load data from nfl-data-py and store in database.

    Args:
        seasons: List of seasons to load
        refresh_only: Only get latest week's data (current season)
        force_rescrape: Force re-loading even if data exists
    """
    seasons = seasons or SEASONS_TO_SCRAPE
    if refresh_only:
        seasons = [CURRENT_NFL_SEASON]

    db = DatabaseManager()

    print("=" * 60)
    print("NFL Data Loading Pipeline (nfl-data-py)")
    print("=" * 60)

    # Check which seasons need loading
    if not force_rescrape and not refresh_only:
        existing_seasons = db.get_seasons_with_data()
        seasons_to_load = []
        for season in seasons:
            if season not in existing_seasons:
                seasons_to_load.append(season)
                print(f"  Season {season}: No data found, will load")
            else:
                latest_week = db.get_latest_week_for_season(season)
                # Current season might need updates
                from datetime import datetime
                if season == datetime.now().year and latest_week < 18:
                    seasons_to_load.append(season)
                    print(f"  Season {season}: Partial data (week {latest_week}), will update")
                else:
                    print(f"  Season {season}: Data exists (through week {latest_week}), skipping")

        if not seasons_to_load:
            print("\nAll requested seasons already have data. Use --force to re-load.")
            print("=" * 60)
            return

        seasons = seasons_to_load

    loader = NFLDataLoader()

    # 1. Load weekly player data (with PBP fallback for in-season completeness)
    print("\n[1/3] Loading weekly player data...")
    weekly_df = loader.load_weekly_data(seasons, store_in_db=True, use_pbp_fallback=True)
    if weekly_df is not None:
        print(f"  Loaded {len(weekly_df)} player stat records")

    # 2. Load schedules
    print("\n[2/3] Loading schedules...")
    schedule_df = loader.load_schedules(seasons, store_in_db=True)
    if schedule_df is not None:
        print(f"  Loaded {len(schedule_df)} schedule records")

    # 3. Load rosters + snap counts (optional, non-fatal)
    print("\n[3/3] Loading rosters and snap counts...")
    try:
        roster_df = loader.load_rosters(seasons)
        print(f"  Loaded {len(roster_df) if roster_df is not None else 0} roster records")
    except Exception as e:
        print(f"  Warning: Could not load rosters: {e}")
    try:
        snap_df = loader.load_snap_counts(seasons)
        print(f"  Loaded {len(snap_df) if snap_df is not None else 0} snap count records")
    except Exception as e:
        print(f"  Warning: Could not load snap counts: {e}")

    print("\n" + "=" * 60)
    print("Data loading complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run NFL data loader (nfl-data-py)")
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Seasons to load (e.g., '2020-2024' or '2023,2024')",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Only refresh with latest data (current season)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-loading even if data already exists",
    )

    args = parser.parse_args()

    # Parse seasons argument
    seasons = None
    if args.seasons:
        if "-" in args.seasons:
            start, end = args.seasons.split("-")
            seasons = list(range(int(start), int(end) + 1))
        else:
            seasons = [int(s.strip()) for s in args.seasons.split(",")]

    run_all_scrapers(seasons=seasons, refresh_only=args.refresh, force_rescrape=args.force)


if __name__ == "__main__":
    main()
