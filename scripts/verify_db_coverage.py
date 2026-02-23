#!/usr/bin/env python3
"""Verify DB has data for requested seasons and positions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import SEASONS_TO_SCRAPE, POSITIONS
from src.utils.database import DatabaseManager
from src.utils.nfl_calendar import get_current_nfl_season, get_current_nfl_week


def _parse_seasons(arg: str | None) -> List[int] | None:
    if not arg:
        return None
    if "-" in arg:
        start, end = arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s.strip()) for s in arg.split(",") if s.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify database coverage for seasons/positions")
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Seasons to verify (e.g., '2020-2024' or '2023,2024'); defaults to config range",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if missing seasons or positions",
    )
    args = parser.parse_args()

    seasons = _parse_seasons(args.seasons) or list(SEASONS_TO_SCRAPE)
    expected_positions = list(POSITIONS)

    db = DatabaseManager()
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    query = """
        SELECT pws.season as season,
               p.position as position,
               COUNT(*) as rows,
               COUNT(DISTINCT pws.player_id) as players,
               COUNT(DISTINCT pws.week) as weeks
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        GROUP BY pws.season, p.position
        ORDER BY pws.season, p.position
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    print("=" * 70)
    print("DB COVERAGE CHECK")
    print("=" * 70)

    missing_seasons = [s for s in seasons if s not in df["season"].unique()]
    if missing_seasons:
        print(f"Missing seasons in DB: {missing_seasons}")

    missing_positions = []
    for season in seasons:
        for pos in expected_positions:
            if not ((df["season"] == season) & (df["position"] == pos)).any():
                missing_positions.append((season, pos))

    if missing_positions:
        print("Missing season/position pairs:")
        for season, pos in missing_positions:
            print(f"  - {season}: {pos}")

    # Week coverage summary
    current_season = get_current_nfl_season()
    cur_week_info = get_current_nfl_week()
    current_week = int(cur_week_info.get("week_num", 0)) if cur_week_info else 0

    print("\nWeek coverage summary:")
    for season in seasons:
        season_rows = df[df["season"] == season]
        if season_rows.empty:
            continue
        max_weeks = int(season_rows["weeks"].max()) if not season_rows.empty else 0
        status = "OK"
        if season < current_season and max_weeks < 18:
            status = "INCOMPLETE"
        if season == current_season and current_week > 0 and max_weeks < current_week:
            status = "BEHIND"
        print(f"  {season}: weeks={max_weeks} ({status})")

    if not df.empty:
        print("\nPosition coverage (rows/players per season):")
        for season in seasons:
            season_rows = df[df["season"] == season]
            if season_rows.empty:
                continue
            print(f"  {season}:")
            for pos in expected_positions:
                row = season_rows[season_rows["position"] == pos]
                if row.empty:
                    print(f"    {pos}: MISSING")
                else:
                    rows = int(row["rows"].iloc[0])
                    players = int(row["players"].iloc[0])
                    weeks = int(row["weeks"].iloc[0])
                    print(f"    {pos}: rows={rows}, players={players}, weeks={weeks}")

    ok = not missing_seasons and not missing_positions
    if ok:
        print("\n✅ DB coverage looks good.")
    else:
        print("\n⚠️ DB coverage has gaps.")

    return 0 if ok or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
