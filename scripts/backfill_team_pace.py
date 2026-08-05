"""
One-time backfill: populate team_stats.total_plays / pace_sec_per_play where
missing (found while wiring the GAPS.md §4.D tempo feature —
team_plays_roll3_mean/team_pace_sec_per_play_roll3_mean would otherwise be
mostly zero for large chunks of the training window).

Root cause: these two columns were populated by a PBP-based aggregation path
(pbp_stats_aggregator.get_team_stats_from_pbp) inconsistently across seasons:
- 2018-2019: total_plays populated, but pace_sec_per_play 100% zero
- 2020-2024: both ~93% zero (2025 current-season pipeline populates both
  correctly)
All other team_stats columns (drive_count, neutral_pass_rate, etc.) are
already fully populated for every season — only these two are affected.

Safe to re-run: UPDATE only touches rows currently zero on either column.
"""
import sqlite3
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.pbp_stats_aggregator import get_team_stats_from_pbp

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nfl_data.db")
SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for season in SEASONS:
        df = get_team_stats_from_pbp(season)
        if df.empty or "total_plays" not in df.columns:
            print(f"{season}: no PBP team stats returned, skipping")
            continue

        updates = [
            (
                int(row["total_plays"]) if pd.notna(row["total_plays"]) else 0,
                float(row["pace_sec_per_play"]) if pd.notna(row.get("pace_sec_per_play")) else 0.0,
                row["team"],
                season,
                int(row["week"]),
            )
            for _, row in df.iterrows()
        ]

        cur.executemany(
            """
            UPDATE team_stats
            SET total_plays = ?, pace_sec_per_play = ?
            WHERE team = ? AND season = ? AND week = ?
              AND (total_plays = 0 OR pace_sec_per_play = 0)
            """,
            updates,
        )
        print(f"{season}: patched {cur.rowcount} rows ({len(updates)} PBP team-weeks available)")

    conn.commit()

    remaining = pd.read_sql(
        "SELECT season, "
        "SUM(CASE WHEN total_plays = 0 THEN 1 ELSE 0 END) plays_zero_n, "
        "SUM(CASE WHEN pace_sec_per_play = 0 THEN 1 ELSE 0 END) pace_zero_n, "
        "COUNT(*) n "
        "FROM team_stats WHERE season BETWEEN 2018 AND 2024 GROUP BY season",
        conn,
    )
    print(remaining)
    conn.close()


if __name__ == "__main__":
    main()
