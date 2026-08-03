#!/usr/bin/env python3
"""Master data pull script — fetches all available NFL data sources.

Runs each scraper in sequence and reports totals.  Safe to re-run; all
scrapers skip rows that already exist.

Sources pulled:
  1. Historical player props      (The Odds API — costs ~100 credits/event)
  2. Season win totals            (The Odds API — cheap, season-level futures)
  3. Game-day weather             (Open-Meteo — free, no key)

Usage:
    # Full pull: props 2024-2025 + win totals 2020-2025 + weather 2006-2025
    python scripts/pull_all_data.py

    # Props only for specific seasons
    python scripts/pull_all_data.py --props-only --seasons 2024,2025

    # Weather only
    python scripts/pull_all_data.py --weather-only

    # Check credits first
    python scripts/pull_all_data.py --quota
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CURRENT_NFL_SEASON, MIN_HISTORICAL_YEAR


def parse_seasons(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",")]


def run_props(seasons: list[int], credit_floor: int) -> None:
    from src.scrapers.odds_scraper import NFLOddsScraper
    scraper = NFLOddsScraper()
    print(f"\n{'='*60}")
    print(f"Player props — seasons {seasons}")
    print(f"{'='*60}")
    quota = scraper.check_quota()
    remaining = quota['requests_remaining']
    print(f"Credits remaining: {remaining:,}" if remaining is not None else "Credits remaining: unknown (cached)")
    result = scraper.scrape_historical_props(
        seasons=seasons, credit_floor=credit_floor
    )
    print(f"  → {result['events_fetched']} events, {result['props_rows']:,} rows")


def run_win_totals(seasons: list[int]) -> None:
    from src.scrapers.odds_scraper import NFLOddsScraper
    scraper = NFLOddsScraper()
    print(f"\n{'='*60}")
    print(f"Win totals — seasons {seasons}")
    print(f"{'='*60}")
    result = scraper.scrape_win_totals(seasons=seasons)
    print(f"  → {result['win_total_rows']} rows")


def run_weather(seasons: list[int]) -> None:
    from src.scrapers.weather_scraper import WeatherScraper
    scraper = WeatherScraper()
    print(f"\n{'='*60}")
    print(f"Weather — seasons {seasons}")
    print(f"{'='*60}")
    total = scraper.scrape_seasons(seasons)
    print(f"  → {total:,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull all NFL data sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--props-only", action="store_true")
    parser.add_argument("--weather-only", action="store_true")
    parser.add_argument("--win-totals-only", action="store_true")
    parser.add_argument("--quota", action="store_true", help="Check API credits and exit")
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Seasons to pull (e.g. '2024,2025' or '2020-2025'). "
             "Defaults differ per source.",
    )
    parser.add_argument(
        "--credit-floor",
        type=int,
        default=5000,
        help="Stop props scrape when remaining credits fall below this (default: 5000)",
    )

    args = parser.parse_args()

    if args.quota:
        from src.scrapers.odds_scraper import NFLOddsScraper
        q = NFLOddsScraper().check_quota()
        used = q['requests_used']
        rem = q['requests_remaining']
        print(f"Credits used:      {used:,}" if used is not None else "Credits used: unknown")
        print(f"Credits remaining: {rem:,}" if rem is not None else "Credits remaining: unknown")
        return

    explicit_seasons = parse_seasons(args.seasons) if args.seasons else None

    if args.weather_only:
        run_weather(explicit_seasons or list(range(MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON + 1)))
        return

    if args.win_totals_only:
        run_win_totals(explicit_seasons or list(range(2020, CURRENT_NFL_SEASON + 1)))
        return

    if args.props_only:
        run_props(explicit_seasons or [2024, 2025], args.credit_floor)
        return

    # Full pull
    prop_seasons = explicit_seasons or [2024, 2025]
    win_seasons = explicit_seasons or list(range(2020, CURRENT_NFL_SEASON + 1))
    weather_seasons = explicit_seasons or list(range(MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON + 1))

    run_props(prop_seasons, args.credit_floor)
    run_win_totals(win_seasons)
    run_weather(weather_seasons)

    print(f"\n{'='*60}")
    print("All data sources pulled.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
