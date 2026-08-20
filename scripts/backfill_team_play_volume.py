#!/usr/bin/env python3
"""Backfill `team_stats.total_plays` / `pass_attempts` / `rush_attempts`.

These are NULL for ~558 of ~600 rows in each of 2020-2024, but populated
for 2006-2019 and 2025. `team_plays_roll3_mean` therefore reads ~52 through
2019, ~3 for 2021-2024, and 46.2 in 2025 -- a train/test discontinuity
landing on the most heavily recency-weighted training seasons, and part of
the 2025 fold anomaly (GAPS.md 2026-08-20).

The values are reconstructible exactly from `player_weekly_stats`, because
that is how the surviving ones were computed in the first place: summing
player passing_attempts + rushing_attempts per (season, week, team)
reproduces all 8,249 non-null rows with correlation 1.0000 and zero mean
error, in all 20 seasons. This is a restoration, not an estimate.

Only updates rows that are NULL or zero; never overwrites a real value,
and never inserts new team-weeks.

Usage:
    python scripts/backfill_team_play_volume.py            # dry-run
    python scripts/backfill_team_play_volume.py --write
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

COLUMNS = ["total_plays", "pass_attempts", "rush_attempts"]

RECON_SQL = """
    SELECT season, week, team,
           SUM(passing_attempts) + SUM(rushing_attempts) AS total_plays,
           SUM(passing_attempts) AS pass_attempts,
           SUM(rushing_attempts) AS rush_attempts
    FROM player_weekly_stats
    GROUP BY season, week, team
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    recon = pd.read_sql(RECON_SQL, conn)
    ts = pd.read_sql(
        f"SELECT id, season, week, team, {', '.join(COLUMNS)} FROM team_stats", conn)

    m = ts.merge(recon, on=["season", "week", "team"], how="left", suffixes=("", "_recon"))

    # Agreement check on rows that already have values -- if the formula ever
    # stops matching, this backfill is no longer a restoration and should not run.
    have = m.total_plays.notna() & (m.total_plays > 0) & m.total_plays_recon.notna()
    if have.any():
        err = (m.loc[have, "total_plays_recon"] - m.loc[have, "total_plays"]).abs()
        print(f"agreement on {int(have.sum())} existing rows: "
              f"max|err|={err.max():.0f}, mismatches={int((err > 0).sum())}")
        if err.max() > 0:
            print("!! reconstruction no longer matches stored values -- refusing to write")
            conn.close()
            sys.exit(2)

    blank = (m.total_plays.isna() | (m.total_plays == 0)) & m.total_plays_recon.notna()
    print(f"\nteam_stats: {len(m)} rows, {int(blank.sum())} fillable")
    fill = m[blank]
    print("  by season: " + ", ".join(
        f"{s}:{n}" for s, n in fill.groupby("season").size().items()))
    unresolved = (m.total_plays.isna() | (m.total_plays == 0)) & m.total_plays_recon.isna()
    print(f"  unresolvable (no player rows for that team-week): {int(unresolved.sum())}")

    if not args.write:
        print("\nDRY RUN -- nothing written. Re-run with --write to apply.")
        conn.close()
        return

    backup = DB_PATH.parent / f"nfl_data.db.bak-teamplays-{datetime.now():%Y%m%d%H%M%S}"
    shutil.copy2(DB_PATH, backup)
    print(f"\nBacked up database to {backup}")
    conn.executemany(
        f"UPDATE team_stats SET {', '.join(f'{c} = ?' for c in COLUMNS)} WHERE id = ?",
        list(zip(*[fill[f"{c}_recon"] for c in COLUMNS], fill.id)),
    )
    conn.commit()

    after = pd.read_sql(
        "SELECT season, COUNT(*) n, SUM(CASE WHEN total_plays>0 THEN 1 ELSE 0 END) ok, "
        "ROUND(AVG(total_plays),1) avg_plays FROM team_stats GROUP BY season "
        "ORDER BY season", conn)
    print(f"\nWrote {len(fill)} rows. Coverage now:")
    print(after.tail(9).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
