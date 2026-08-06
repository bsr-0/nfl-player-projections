"""
Backfill pbp_pass_participation (per-player pass-play participation %,
per player-week) from PBP's offense_players column via
get_pass_play_participation_from_pbp.

GAPS.md §11.1.C/D follow-up: true route participation and YPRR aren't
derivable from any accessible data source (no source distinguishes a
receiver running a route from one staying in to block on a given pass
play). This is a real, PBP-derived proxy instead -- fraction of a team's
actual dropback pass plays a player was on the field for. The column
exists in PBP starting 2016 (same coverage start as offense_personnel,
used for team_personnel_stats), 100% non-null on real pass plays from
2016 on.

Usage:
    python scripts/backfill_pbp_pass_participation.py            # only missing seasons
    python scripts/backfill_pbp_pass_participation.py --force     # re-fetch everything
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import DatabaseManager

PARTICIPATION_COVERAGE_START = 2016


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
    seasons = [s for s in all_seasons if s >= PARTICIPATION_COVERAGE_START]
    skipped = sorted(set(all_seasons) - set(seasons))
    if skipped:
        print(f"Skipping seasons before PBP offense_players coverage starts ({PARTICIPATION_COVERAGE_START}): {skipped}")

    n = db.ensure_pbp_pass_participation(seasons=seasons, force_refresh=force)
    print(f"Upserted {n} pbp_pass_participation rows.")

    coverage = pd.read_sql(
        "SELECT season, COUNT(*) n, COUNT(DISTINCT player_id) n_players, "
        "ROUND(AVG(pass_play_participation_pct), 3) avg_pct, "
        "ROUND(AVG(team_pass_plays), 1) avg_team_pass_plays "
        "FROM pbp_pass_participation GROUP BY season ORDER BY season",
        conn,
    )
    print(coverage.to_string())
    conn.close()


if __name__ == "__main__":
    main()
