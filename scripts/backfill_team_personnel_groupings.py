"""
Backfill team_personnel_stats (11/12/21/13/other offensive personnel usage
%, per team-week) from PBP's offense_personnel column via
get_personnel_groupings_from_pbp.

GAPS.md §3.3/§11.1.E: team-level personnel grouping (11 vs. 12 vs. 21
personnel) was previously entirely unmodeled. The column exists in PBP
starting 2016 (verified directly against nfl_data_py -- not ~2013 as
originally guessed in GAPS.md, and it's 100% non-null on real pass/run
plays from 2016 on, not partially missing).

Usage:
    python scripts/backfill_team_personnel_groupings.py            # only missing seasons
    python scripts/backfill_team_personnel_groupings.py --force     # re-fetch everything
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import DatabaseManager

PERSONNEL_COVERAGE_START = 2016


def main():
    force = "--force" in sys.argv
    db = DatabaseManager()

    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(db.db_path if hasattr(db, "db_path") else "data/nfl_data.db")
    all_seasons = [
        int(s) for s in pd.read_sql(
            "SELECT DISTINCT season FROM player_weekly_stats", conn
        )["season"].tolist()
    ]
    seasons = [s for s in all_seasons if s >= PERSONNEL_COVERAGE_START]
    skipped = sorted(set(all_seasons) - set(seasons))
    if skipped:
        print(f"Skipping seasons before PBP offense_personnel coverage starts ({PERSONNEL_COVERAGE_START}): {skipped}")

    n = db.ensure_team_personnel_stats(seasons=seasons, force_refresh=force)
    print(f"Upserted {n} team_personnel_stats rows.")

    coverage = pd.read_sql(
        "SELECT season, COUNT(*) n, "
        "ROUND(AVG(pct_11), 3) avg_11, ROUND(AVG(pct_12), 3) avg_12, "
        "ROUND(AVG(pct_21), 3) avg_21, ROUND(AVG(n_plays), 1) avg_n_plays "
        "FROM team_personnel_stats GROUP BY season ORDER BY season",
        conn,
    )
    print(coverage.to_string())
    conn.close()


if __name__ == "__main__":
    main()
