"""
One-time backfill: collapse the JAC/LAR team codes introduced by the (now
fixed) entity_resolver.TEAM_CODE_ALIASES direction bug back onto the
canonical JAX/LA codes used everywhere else (schedule table, historical
rows, GAPS.md §4/§9 audit).

Root cause: entity_resolver.py's TEAM_CODE_ALIASES used to map JAX->JAC and
LA->LAR. Any row ingested/refreshed through nfl_data_loader.py's
resolver.build_keys() after that alias was added picked up the new codes,
while historical rows loaded before then kept JAX/LA — leaving team_stats,
team_defense_stats, and player_weekly_stats with BOTH codes for the same
team split across different weeks/seasons (first observed: 2025 season).
The alias direction has been flipped; this script cleans up the rows
written before the fix.

team_stats has a UNIQUE(team, season, week) constraint and, for 2025, holds a
genuine duplicate under both codes for every (team, week): a complete row
under the old JAX/LA code and a sparse duplicate (all derived stat columns
NULL, e.g. points_scored/pace_sec_per_play) under the new JAC/LAR code,
written later by a run that picked up the not-yet-fixed alias. Verified
100% consistent (39/39 old-code rows fully populated, 38/38 new-code rows
null on points_scored/pace_sec_per_play) before writing this script — the
sparse duplicates are simply deleted rather than merged. player_weekly_stats
and team_defense_stats have no such collision (verified: zero duplicate
player-weeks in pws; team_defense_stats 2025 only exists under the new code,
no old-code row to collide with), so those get a plain rename.

Safe to re-run: every UPDATE/DELETE only touches rows currently on the old
code / matching the verified-sparse pattern.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nfl_data.db")

# old (wrong) code -> canonical code
_RENAME = {"JAC": "JAX", "LAR": "LA"}

_RENAME_TABLES_COLS = {
    "player_weekly_stats": ["team", "opponent"],
    "team_defense_stats": ["team"],
}


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # team_stats: delete the sparse new-code duplicate where a complete
    # old-code row already exists for the same (season, week).
    for old, new in _RENAME.items():
        cur.execute(
            """
            DELETE FROM team_stats
            WHERE team = ?
              AND points_scored IS NULL
              AND EXISTS (
                  SELECT 1 FROM team_stats t2
                  WHERE t2.team = ? AND t2.season = team_stats.season
                    AND t2.week = team_stats.week
                    AND t2.points_scored IS NOT NULL
              )
            """,
            (old, new),
        )
        print(f"team_stats: deleted {cur.rowcount} sparse {old} duplicate rows "
              f"(complete {new} row already present)")

    for table, cols in _RENAME_TABLES_COLS.items():
        for col in cols:
            for old, new in _RENAME.items():
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (old,))
                n = cur.fetchone()[0]
                if n == 0:
                    continue
                cur.execute(f"UPDATE {table} SET {col} = ? WHERE {col} = ?", (new, old))
                print(f"{table}.{col}: {old} -> {new} ({n} rows)")

    conn.commit()

    # Verify no old codes remain, and no leftover JAC/LAR team_stats rows
    remaining = []
    for table, cols in {**_RENAME_TABLES_COLS, "team_stats": ["team"]}.items():
        for col in cols:
            cur.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} IN ('JAC', 'LAR')"
            )
            n = cur.fetchone()[0]
            if n:
                remaining.append((table, col, n))

    if remaining:
        print("WARNING: old codes still present after backfill:")
        for table, col, n in remaining:
            print(f"  {table}.{col}: {n} rows")
    else:
        print("Verified: no JAC/LAR codes remain in player_weekly_stats, "
              "team_stats, or team_defense_stats.")

    conn.close()


if __name__ == "__main__":
    main()
