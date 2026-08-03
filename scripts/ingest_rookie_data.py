#!/usr/bin/env python3
"""
Ingest rookie-related data (draft picks, combine, rosters) into the SQLite DB.

Fetches from nflverse via nfl_data_py and persists to the draft_picks,
combine_data, and rosters tables so downstream feature engineering can
read locally instead of hitting the API every run.

Usage:
    python scripts/ingest_rookie_data.py                # Ingest all three datasets
    python scripts/ingest_rookie_data.py --only draft   # Just draft picks
    python scripts/ingest_rookie_data.py --only combine # Just combine data
    python scripts/ingest_rookie_data.py --only rosters # Just rosters
    python scripts/ingest_rookie_data.py --seasons 2020 2021 2022 2023 2024 2025
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.database import DatabaseManager


def ingest_draft_picks(db: DatabaseManager, seasons: list[int]) -> int:
    """Fetch and persist draft picks."""
    import nfl_data_py as nfl

    print(f"Fetching draft picks for {len(seasons)} seasons...")
    df = nfl.import_draft_picks(seasons)
    if df.empty:
        print("  No draft pick data returned.")
        return 0
    count = db.bulk_insert_draft_picks(df)
    print(f"  Inserted {count} draft pick records.")
    return count


def ingest_combine_data(db: DatabaseManager, seasons: list[int]) -> int:
    """Fetch and persist combine data."""
    import nfl_data_py as nfl

    print("Fetching combine data...")
    df = nfl.import_combine_data()
    if df.empty:
        print("  No combine data returned.")
        return 0
    # Filter to requested seasons if the column exists
    if "season" in df.columns:
        df = df[df["season"].isin(seasons)]
    count = db.bulk_insert_combine_data(df)
    print(f"  Inserted {count} combine records.")
    return count


def ingest_rosters(db: DatabaseManager, seasons: list[int]) -> int:
    """Fetch and persist roster data."""
    import nfl_data_py as nfl

    print(f"Fetching rosters for {len(seasons)} seasons...")
    df = nfl.import_rosters(seasons)
    if df.empty:
        print("  No roster data returned.")
        return 0
    count = db.bulk_insert_rosters(df)
    print(f"  Inserted {count} roster records.")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Ingest rookie data (draft picks, combine, rosters) into SQLite DB",
    )
    parser.add_argument(
        "--only",
        choices=["draft", "combine", "rosters"],
        default=None,
        help="Ingest only one dataset instead of all three",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Seasons to ingest (default: 2000-current)",
    )
    args = parser.parse_args()

    seasons = args.seasons or list(range(2000, 2027))

    db = DatabaseManager()
    total = 0

    targets = [args.only] if args.only else ["draft", "combine", "rosters"]

    for target in targets:
        if target == "draft":
            total += ingest_draft_picks(db, seasons)
        elif target == "combine":
            total += ingest_combine_data(db, seasons)
        elif target == "rosters":
            total += ingest_rosters(db, seasons)

    # Backfill college / birth_date into players table from rosters
    if "rosters" in targets:
        updated = db.backfill_players_from_rosters()
        print(f"Backfilled {updated} player records (college/birth_date).")

    print(f"\nDone. {total} total records ingested.")


if __name__ == "__main__":
    main()
