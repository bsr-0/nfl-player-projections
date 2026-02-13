#!/usr/bin/env python3
"""
Verify that the latest season in the database is used as the test dataset.

Run from project root:
  python scripts/verify_latest_season_test.py           # auto-refreshes if current season missing
  python scripts/verify_latest_season_test.py --no-refresh   # fail fast if current season missing

When in-season and current season is not in DB, by default runs auto_refresh to load
current season from play-by-play, then re-checks.
"""

import argparse
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify latest season is test dataset")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Do not run auto_refresh when current season is missing; fail immediately",
    )
    args = parser.parse_args()

    from src.utils.nfl_calendar import get_current_nfl_season, current_season_has_weeks_played
    from src.utils.data_manager import DataManager

    print("=" * 60)
    print("Verify: Latest season is test dataset")
    print("=" * 60)

    dm = DataManager()
    available = dm.get_available_seasons_from_db()
    current_season = get_current_nfl_season()
    in_season = current_season_has_weeks_played()

    # When in-season and current season missing, optionally run auto_refresh to load from PBP
    if not args.no_refresh and in_season and available and current_season not in available:
        print(f"  Current season {current_season} not in DB. Running auto_refresh to load from PBP...")
        try:
            from src.data.auto_refresh import auto_refresh
            auto_refresh(force=False)
            available = dm.get_available_seasons_from_db()
            print(f"  After refresh, seasons in DB: {sorted(available)}")
        except Exception as e:
            print(f"  Auto-refresh failed: {e}")
            print("  Run manually: python -m src.data.auto_refresh")
            return 1

    if not available:
        print("No seasons in database. Run data load / auto_refresh first.")
        print("  python -m src.data.auto_refresh")
        return 1

    latest_in_db = max(available)
    print(f"  Seasons in DB: {sorted(available)}")
    print(f"  Latest in DB: {latest_in_db}")
    print(f"  Current NFL season: {current_season}")
    print(f"  In-season (week >= 1): {in_season}")

    train_seasons, test_season = dm.get_train_test_seasons()
    print(f"  Train seasons: {train_seasons}")
    print(f"  Test season:  {test_season}")

    # Assert: test must be the latest available season
    ok = True
    if test_season != latest_in_db:
        print(f"\n  FAIL: test_season ({test_season}) != latest in DB ({latest_in_db})")
        ok = False
    else:
        print(f"\n  OK: test_season ({test_season}) == latest in DB ({latest_in_db})")

    if in_season:
        if test_season != current_season:
            print(f"  FAIL: in-season but test_season ({test_season}) != current_season ({current_season})")
            ok = False
        else:
            print(f"  OK: in-season and test_season == current_season ({current_season})")
        if current_season not in available:
            print(f"  WARN: current season {current_season} not in DB (run: python -m src.data.auto_refresh)")

    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
