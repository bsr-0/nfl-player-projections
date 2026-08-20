#!/usr/bin/env python3
"""Backfill `player_weekly_stats.home_away` from the schedule.

The weekly ingest never wrote this column: it was populated on 100% of
`inferred_snap_verified_zero` rows (the panel build sets it) and 0% of
`nflverse_stats` rows, in every season except 2025, which a re-ingest
happened to fill. `is_dome` is derived from `home_away`, so it read ~3%
across all training seasons and 35.7% in the 2025 test fold -- while
`game_weather` says 32-35% in every season. That single discontinuity
accounted for most of the 2025 fold anomaly, and `is_dome` ranks 4th of
RB's 64 features (GAPS.md 2026-08-20).

`opponent` is already 100% populated, so (season, week, team, opponent)
resolves against the schedule for 123,183 of 123,254 rows.

Usage:
    python scripts/backfill_home_away.py            # dry-run
    python scripts/backfill_home_away.py --write
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH

# Both orientations of every scheduled game, so a player row joins on the
# team it played for regardless of which side that was.
SIDES_SQL = """
    SELECT season, week, home_team AS team, away_team AS opponent, 'home' AS ha FROM schedule
    UNION ALL
    SELECT season, week, away_team AS team, home_team AS opponent, 'away' AS ha FROM schedule
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    sides = pd.read_sql(SIDES_SQL, conn)
    pws = pd.read_sql(
        "SELECT id, season, week, team, opponent, home_away FROM player_weekly_stats", conn)

    merged = pws.merge(sides, on=["season", "week", "team", "opponent"], how="left")
    known = merged.home_away.isin(["home", "away"])
    resolved = merged.ha.notna()

    print(f"player_weekly_stats: {len(merged)} rows")
    print(f"  already set:  {int(known.sum())}")
    print(f"  resolvable:   {int(resolved.sum())}")
    print(f"  unresolvable: {int((~resolved).sum())} (left as-is)")

    conflicts = merged[known & resolved & (merged.home_away != merged.ha)]
    print(f"  conflicts with existing values: {len(conflicts)} (existing wins)")
    if len(conflicts):
        print(conflicts.head(5)[["season", "week", "team", "opponent", "home_away", "ha"]]
              .to_string(index=False))

    to_write = merged[~known & resolved]
    print(f"\n  to write: {len(to_write)}")
    by_season = to_write.groupby("season").size()
    print("  by season: " + ", ".join(f"{s}:{n}" for s, n in by_season.items()))

    if not args.write:
        print("\nDRY RUN -- nothing written. Re-run with --write to apply.")
        conn.close()
        return

    backup = DB_PATH.parent / f"nfl_data.db.bak-homeaway-{datetime.now():%Y%m%d%H%M%S}"
    shutil.copy2(DB_PATH, backup)
    print(f"\nBacked up database to {backup}")
    conn.executemany(
        "UPDATE player_weekly_stats SET home_away = ? WHERE id = ?",
        list(zip(to_write.ha, to_write.id)),
    )
    conn.commit()

    after = pd.read_sql(
        "SELECT season, SUM(CASE WHEN home_away IN ('home','away') THEN 1 ELSE 0 END) ok, "
        "COUNT(*) n FROM player_weekly_stats GROUP BY season ORDER BY season", conn)
    after["pct"] = (100 * after.ok / after.n).round(1)
    print(f"\nWrote {len(to_write)} values. Coverage now:")
    print(after.tail(10).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
