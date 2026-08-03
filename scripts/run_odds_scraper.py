#!/usr/bin/env python3
"""CLI for the NFL Odds API scraper.

Usage examples:

  # Check remaining API quota (costs 1 request):
  python scripts/run_odds_scraper.py --quota

  # Fetch current/upcoming game odds + player props:
  python scripts/run_odds_scraper.py --current

  # Fetch current game odds only (no props):
  python scripts/run_odds_scraper.py --current --no-props

  # Scrape historical odds for a single season:
  python scripts/run_odds_scraper.py --historical --year 2023

  # Scrape historical odds for a range of seasons:
  python scripts/run_odds_scraper.py --historical --year 2018 --end-year 2025

  # Dry run (prints dates, no API calls):
  python scripts/run_odds_scraper.py --historical --year 2023 --dry-run

  # Game lines only (no player props, cheaper on API credits):
  python scripts/run_odds_scraper.py --historical --year 2024 --markets spreads,totals
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CURRENT_NFL_SEASON, MIN_HISTORICAL_YEAR
from src.scrapers.odds_scraper import NFLOddsScraper, GAME_MARKETS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape NFL odds data from The Odds API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--current",
        action="store_true",
        help="Fetch odds for upcoming NFL games (current/next week)",
    )
    mode.add_argument(
        "--historical",
        action="store_true",
        help="Fetch historical odds snapshots (requires paid API tier)",
    )
    mode.add_argument(
        "--quota",
        action="store_true",
        help="Check remaining API quota without pulling odds data",
    )

    # Historical-mode options
    parser.add_argument(
        "--year",
        type=int,
        default=CURRENT_NFL_SEASON,
        help=f"NFL season year to scrape (default: {CURRENT_NFL_SEASON})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End season for a range scrape (inclusive). Omit for single year.",
    )
    parser.add_argument(
        "--markets",
        type=str,
        default=GAME_MARKETS,
        help=f"Comma-separated market keys (default: '{GAME_MARKETS}')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without making API calls",
    )

    # Current-mode options
    parser.add_argument(
        "--no-props",
        action="store_true",
        help="Skip player props when running --current (saves API credits)",
    )

    args = parser.parse_args()

    scraper = NFLOddsScraper()

    # ------------------------------------------------------------------
    if args.quota:
        print("Checking API quota...")
        status = scraper.check_quota()
        print(f"  Requests used:      {status['requests_used']}")
        print(f"  Requests remaining: {status['requests_remaining']}")
        return

    # ------------------------------------------------------------------
    if args.current:
        print("=" * 60)
        print("Fetching current NFL game odds")
        print("=" * 60)
        totals = scraper.scrape_current(include_props=not args.no_props)
        print()
        print(f"Done.  Game-odds rows stored : {totals['game_odds']}")
        print(f"       Player-prop rows stored: {totals['player_props']}")
        return

    # ------------------------------------------------------------------
    if args.historical:
        start_year = args.year
        end_year = args.end_year or start_year

        if start_year < MIN_HISTORICAL_YEAR:
            print(
                f"Warning: {start_year} is before MIN_HISTORICAL_YEAR "
                f"({MIN_HISTORICAL_YEAR}). The Odds API historical data may not "
                "exist that far back."
            )

        print("=" * 60)
        if start_year == end_year:
            print(f"Fetching historical odds — season {start_year}")
        else:
            print(f"Fetching historical odds — seasons {start_year}–{end_year}")
        print(f"Markets : {args.markets}")
        if args.dry_run:
            print("(DRY RUN — no API calls will be made)")
        print("=" * 60)

        if start_year == end_year:
            result = scraper.scrape_historical_season(
                start_year, markets=args.markets, dry_run=args.dry_run
            )
        else:
            result = scraper.scrape_historical_range(
                start_year, end_year, markets=args.markets, dry_run=args.dry_run
            )

        print()
        print(f"Done.  Dates fetched : {result['dates_fetched']}")
        print(f"       Rows stored   : {result['game_odds']}")
        return


if __name__ == "__main__":
    main()
