"""Main script to run all scrapers and populate database."""
import argparse
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.scrapers.player_scraper import PlayerStatsScraper, FantasyProssScraper
from src.scrapers.team_scraper import TeamStatsScraper, SnapCountScraper
from src.utils.database import DatabaseManager
from src.utils.helpers import calculate_fantasy_points, generate_player_id
from config.settings import SEASONS_TO_SCRAPE, POSITIONS


def run_all_scrapers(seasons: List[int] = None, refresh_only: bool = False, 
                      force_rescrape: bool = False):
    """
    Run all scrapers and store data in database.
    
    Args:
        seasons: List of seasons to scrape
        refresh_only: Only get latest week's data
        force_rescrape: Force re-scraping even if data exists
    """
    seasons = seasons or SEASONS_TO_SCRAPE
    db = DatabaseManager()
    
    print("=" * 60)
    print("NFL Data Scraping Pipeline")
    print("=" * 60)
    
    # Check which seasons need scraping
    if not force_rescrape and not refresh_only:
        existing_seasons = db.get_seasons_with_data()
        seasons_to_scrape = []
        for season in seasons:
            if season not in existing_seasons:
                seasons_to_scrape.append(season)
                print(f"  Season {season}: No data found, will scrape")
            else:
                latest_week = db.get_latest_week_for_season(season)
                # Current season might need updates
                from datetime import datetime
                if season == datetime.now().year and latest_week < 18:
                    seasons_to_scrape.append(season)
                    print(f"  Season {season}: Partial data (week {latest_week}), will update")
                else:
                    print(f"  Season {season}: Data exists (through week {latest_week}), skipping")
        
        if not seasons_to_scrape:
            print("\nAll requested seasons already have data. Use --force to re-scrape.")
            print("=" * 60)
            return
        
        seasons = seasons_to_scrape
    
    # 1. Scrape player stats
    print("\n[1/4] Scraping player statistics...")
    player_scraper = PlayerStatsScraper()
    
    if refresh_only:
        player_df = player_scraper.get_latest_data()
    else:
        player_df = player_scraper.scrape(seasons=seasons, positions=POSITIONS)
    
    if not player_df.empty:
        print(f"  Scraped {len(player_df)} player stat records")
        _store_player_data(db, player_df)
    
    # 2. Scrape team stats
    print("\n[2/4] Scraping team statistics...")
    team_scraper = TeamStatsScraper()
    
    if refresh_only:
        team_df = team_scraper.get_latest_data()
    else:
        team_df = team_scraper.scrape(seasons=seasons)
    
    if not team_df.empty:
        print(f"  Scraped {len(team_df)} team stat records")
        _store_team_data(db, team_df)
    
    # 3. Scrape weekly team stats
    print("\n[3/4] Scraping weekly team statistics...")
    for season in seasons:
        weekly_df = team_scraper.scrape_weekly_team_stats(season)
        if not weekly_df.empty:
            print(f"  Scraped {len(weekly_df)} weekly team records for {season}")
            _store_weekly_team_data(db, weekly_df)
    
    # 4. Scrape snap counts
    print("\n[4/4] Scraping snap count data...")
    snap_scraper = SnapCountScraper()
    
    if refresh_only:
        snap_df = snap_scraper.get_latest_data()
    else:
        snap_df = snap_scraper.scrape(seasons=seasons)
    
    if not snap_df.empty:
        print(f"  Scraped {len(snap_df)} snap count records")
        _store_snap_data(db, snap_df)
    
    print("\n" + "=" * 60)
    print("Scraping complete!")
    print("=" * 60)


def _store_player_data(db: DatabaseManager, df):
    """Store player data in database."""
    # Insert player info
    players_inserted = 0
    stats_inserted = 0
    
    for _, row in df.iterrows():
        player_data = {
            "player_id": row.get("player_id"),
            "name": row.get("name"),
            "position": row.get("position"),
            "birth_date": row.get("birth_date"),
            "college": row.get("college"),
        }
        
        if player_data["player_id"]:
            db.insert_player(player_data)
            players_inserted += 1
        
        # Insert weekly stats if week is present
        if "week" in row and row.get("week"):
            stats_data = row.to_dict()
            db.insert_player_weekly_stats(stats_data)
            stats_inserted += 1
    
    print(f"  Stored {players_inserted} players, {stats_inserted} weekly stat records")


def _store_team_data(db: DatabaseManager, df):
    """Store team data in database."""
    records_inserted = 0
    
    for _, row in df.iterrows():
        stats_data = row.to_dict()
        # Add week=0 for season aggregates
        if "week" not in stats_data:
            stats_data["week"] = 0
        db.insert_team_stats(stats_data)
        records_inserted += 1
    
    print(f"  Stored {records_inserted} team stat records")


def _store_weekly_team_data(db: DatabaseManager, df):
    """Store weekly team data in database."""
    records_inserted = 0
    
    for _, row in df.iterrows():
        stats_data = row.to_dict()
        db.insert_team_stats(stats_data)
        records_inserted += 1
    
    print(f"  Stored {records_inserted} weekly team records")


def _store_snap_data(db: DatabaseManager, df):
    """Store snap count data - merge with player stats."""
    # This would update existing player records with snap data
    print(f"  Processed {len(df)} snap count records")


def main():
    parser = argparse.ArgumentParser(description="Run NFL data scrapers")
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Seasons to scrape (e.g., '2020-2024' or '2023,2024')"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Only refresh with latest data (current season)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-scraping even if data already exists"
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
