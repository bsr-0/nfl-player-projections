"""One-time backfill of high-value touch source columns from play-by-play.

Backfills the following columns in player_weekly_stats:
- rush_inside_10
- rush_inside_5
- targets_15_plus
- air_yards

This script re-aggregates nflverse play-by-play via nfl_data_py and updates existing
rows in SQLite without wiping the DB.

Usage:
  python -m src.data.pbp_backfill_high_value_touches --seasons 2018-2024
  python -m src.data.pbp_backfill_high_value_touches --use-db-seasons

Notes:
- This relies on nfl_data_py.import_pbp_data, which can be slow.
- For seasons already present in your DB, this will update only the new columns.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.data.pbp_stats_aggregator import get_weekly_stats_from_pbp
from src.utils.database import DatabaseManager


def _parse_seasons_arg(seasons: Optional[str]) -> Optional[List[int]]:
    if not seasons:
        return None
    seasons = seasons.strip()
    if not seasons:
        return None
    if "-" in seasons:
        a, b = seasons.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in seasons.split(",") if x.strip()]


def _coerce_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _coerce_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)


def backfill_high_value_touch_columns(
    seasons: Iterable[int],
    dry_run: bool = False,
) -> pd.DataFrame:
    """Backfill high-value touch columns for the specified seasons.

    Returns a summary DataFrame (per season): rows_updated, rows_seen.
    """

    db = DatabaseManager()

    summaries = []
    for season in seasons:
        print(f"\n=== Backfill season {season} ===")
        weekly = get_weekly_stats_from_pbp(int(season))
        if weekly is None or weekly.empty:
            print("  No PBP weekly data returned; skipping")
            summaries.append({"season": int(season), "rows_seen": 0, "rows_updated": 0})
            continue

        needed = ["player_id", "season", "week", "rush_inside_10", "rush_inside_5", "targets_15_plus", "air_yards"]
        for c in needed:
            if c not in weekly.columns:
                weekly[c] = 0

        upd = weekly[needed].copy()
        upd["season"] = _coerce_int(upd["season"])
        upd["week"] = _coerce_int(upd["week"])
        upd["rush_inside_10"] = _coerce_int(upd["rush_inside_10"])
        upd["rush_inside_5"] = _coerce_int(upd["rush_inside_5"])
        upd["targets_15_plus"] = _coerce_int(upd["targets_15_plus"])
        upd["air_yards"] = _coerce_float(upd["air_yards"])
        upd["player_id"] = upd["player_id"].astype(str)

        # De-duplicate just in case (should not be necessary, but protects UPDATE loop)
        upd = (
            upd.groupby(["player_id", "season", "week"], as_index=False)
            .agg(
                {
                    "rush_inside_10": "sum",
                    "rush_inside_5": "sum",
                    "targets_15_plus": "sum",
                    "air_yards": "sum",
                }
            )
        )

        print(f"  Rows in PBP aggregation: {len(upd)}")

        if dry_run:
            print("  Dry run: not writing to DB")
            summaries.append({"season": int(season), "rows_seen": len(upd), "rows_updated": 0})
            continue

        rows_updated = 0
        with db._get_connection() as conn:
            cur = conn.cursor()
            # Update only new columns; do not touch other fields.
            for _, r in upd.iterrows():
                cur.execute(
                    """
                    UPDATE player_weekly_stats
                    SET rush_inside_10 = ?,
                        rush_inside_5 = ?,
                        targets_15_plus = ?,
                        air_yards = ?
                    WHERE player_id = ? AND season = ? AND week = ?
                    """,
                    (
                        int(r["rush_inside_10"]),
                        int(r["rush_inside_5"]),
                        int(r["targets_15_plus"]),
                        float(r["air_yards"]),
                        str(r["player_id"]),
                        int(r["season"]),
                        int(r["week"]),
                    ),
                )
                # rowcount = 1 when a row existed and was updated
                if cur.rowcount:
                    rows_updated += int(cur.rowcount)
            conn.commit()

        print(f"  Rows updated in DB: {rows_updated}")
        summaries.append({"season": int(season), "rows_seen": len(upd), "rows_updated": int(rows_updated)})

    return pd.DataFrame(summaries)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill high-value touch columns from PBP")
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Seasons to backfill. Format: '2018-2024' or '2022,2023,2024'",
    )
    parser.add_argument(
        "--use-db-seasons",
        action="store_true",
        help="Use seasons detected from the DB (player_weekly_stats)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute but do not write to DB",
    )

    args = parser.parse_args()

    db = DatabaseManager()

    seasons = _parse_seasons_arg(args.seasons)
    if args.use_db_seasons:
        try:
            seasons = db.get_seasons_with_data()
        except Exception:
            seasons = seasons

    if not seasons:
        raise ValueError("No seasons provided. Use --seasons or --use-db-seasons")

    seasons = sorted(set(int(s) for s in seasons))
    print(f"Backfilling seasons: {seasons}")

    summary = backfill_high_value_touch_columns(seasons, dry_run=bool(args.dry_run))
    if not summary.empty:
        print("\nSummary:")
        print(summary.to_string(index=False))
        print("\nNote: If rows_updated is low, it likely means those seasons/weeks weren't loaded in DB yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
