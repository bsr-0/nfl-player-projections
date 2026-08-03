#!/usr/bin/env python3
"""
Backfill snap_count and team_snaps into player_weekly_stats from the
snap_counts table.  The two tables use different player ID systems
(GSIS vs PFR), so we match on name + team + season + week.

Usage:
    python scripts/backfill_snap_counts_to_pws.py
    python scripts/backfill_snap_counts_to_pws.py --dry-run
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "nfl_data.db"

# Positions we care about in player_weekly_stats
SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}


def _normalize_name(full_name: str) -> str:
    """Convert 'Josh Allen' -> 'J.Allen' to match pws format."""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    first_initial = parts[0][0]
    last = parts[-1]
    return f"{first_initial}.{last}"


def _team_alias(team: str) -> str:
    """Normalize team codes to match pws conventions."""
    aliases = {
        "OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA",
        "JAX": "JAC",
    }
    return aliases.get(team, team)


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Step 1: Compute team total offense snaps per game from snap_counts
    print("Computing team total snaps per game...")
    cur.execute("""
        SELECT season, week, team, MAX(offense_snaps) as team_snaps
        FROM snap_counts
        WHERE offense_snaps > 0 AND position IN ('QB','RB','WR','TE','FB','T','G','C','OL')
        GROUP BY season, week, team
    """)
    team_snaps_map = {}
    for row in cur.fetchall():
        team = _team_alias(row["team"])
        key = (row["season"], row["week"], team)
        team_snaps_map[key] = max(team_snaps_map.get(key, 0), row["team_snaps"])

    print(f"  {len(team_snaps_map)} team-game entries")

    # Step 2: Get player snap counts (skill positions only)
    print("Loading player snap counts...")
    cur.execute("""
        SELECT season, week, player, team, position, offense_snaps
        FROM snap_counts
        WHERE offense_snaps > 0 AND position IN ('QB','RB','WR','TE','FB')
    """)
    snap_rows = cur.fetchall()
    print(f"  {len(snap_rows)} skill-position snap entries")

    # Step 3: Build lookup from normalized name + team + season + week
    snap_lookup = {}
    for row in snap_rows:
        norm = _normalize_name(row["player"])
        team = _team_alias(row["team"])
        key = (norm, team, row["season"], row["week"])
        # Keep the higher snap count if duplicates
        if key not in snap_lookup or row["offense_snaps"] > snap_lookup[key]:
            snap_lookup[key] = row["offense_snaps"]

    print(f"  {len(snap_lookup)} unique player-game snap entries")

    # Step 4: Load pws rows that need updating (2018-2025, snap_count = 0)
    print("Loading player_weekly_stats rows to update...")
    cur.execute("""
        SELECT p.name, pws.player_id, pws.team, pws.season, pws.week
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season >= 2018 AND pws.snap_count = 0
    """)
    pws_rows = cur.fetchall()
    print(f"  {len(pws_rows)} pws rows to check")

    # Step 5: Match and update
    matched = 0
    updates = []
    for row in pws_rows:
        name = row["name"]
        team = _team_alias(row["team"])
        season = row["season"]
        week = row["week"]
        key = (name, team, season, week)

        if key in snap_lookup:
            offense_snaps = snap_lookup[key]
            ts_key = (season, week, team)
            team_total = team_snaps_map.get(ts_key, 0)
            updates.append((offense_snaps, team_total, row["player_id"], season, week))
            matched += 1

    print(f"  Matched {matched} of {len(pws_rows)} rows ({matched/max(len(pws_rows),1)*100:.1f}%)")

    if dry_run:
        print("\n--dry-run: no writes.")
        return 0

    # Batch update
    print("Writing updates...")
    cur.executemany("""
        UPDATE player_weekly_stats
        SET snap_count = ?, team_snaps = ?
        WHERE player_id = ? AND season = ? AND week = ?
    """, updates)
    conn.commit()
    print(f"  Updated {cur.rowcount} rows")

    # Verification
    cur.execute("""
        SELECT season,
               SUM(CASE WHEN snap_count > 0 THEN 1 ELSE 0 END) as nonzero,
               COUNT(*) as total
        FROM player_weekly_stats
        WHERE season >= 2018
        GROUP BY season ORDER BY season
    """)
    print("\nVerification:")
    for row in cur.fetchall():
        pct = row["nonzero"] / row["total"] * 100
        print(f"  {row['season']}: {row['nonzero']}/{row['total']} ({pct:.0f}%) have snap counts")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
