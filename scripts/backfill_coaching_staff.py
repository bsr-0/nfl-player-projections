"""
Backfill team_coaching_staff (head coach identity per team-week) for all
seasons in player_weekly_stats, from nflverse's games.csv
(home_coach/away_coach) via nfl_data_py.import_schedules.

GAPS.md §4.G / §11.1.G: coaching-staff identity was previously entirely
unavailable, so the fully-built CoachingChangeDetector in
src/features/advanced_analytics.py had never had real data to work with
and always fell back to a scheme-pass-rate proxy. This wires the real
signal in.

Usage:
    python scripts/backfill_coaching_staff.py            # only missing seasons
    python scripts/backfill_coaching_staff.py --force     # re-fetch everything
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import DatabaseManager


def main():
    force = "--force" in sys.argv
    db = DatabaseManager()
    n = db.ensure_team_coaching_staff(force_refresh=force)
    print(f"Upserted {n} team_coaching_staff rows.")

    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(db.db_path if hasattr(db, "db_path") else "data/nfl_data.db")
    coverage = pd.read_sql(
        "SELECT season, COUNT(*) n, SUM(CASE WHEN head_coach IS NULL OR head_coach = '' "
        "THEN 1 ELSE 0 END) null_n FROM team_coaching_staff GROUP BY season ORDER BY season",
        conn,
    )
    print(coverage.to_string())
    conn.close()


if __name__ == "__main__":
    main()
