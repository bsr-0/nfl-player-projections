"""
One-time backfill: populate player_weekly_stats.opponent for LAR/JAC and LA/JAX rows in 2025.

Root cause: nflverse schedule uses 'LA'/'JAX'; canonical player_weekly_stats
convention (see entity_resolver.TEAM_CODE_ALIASES) is also 'LA'/'JAX' as of
the 2026-08 team-code normalization fix, so this backfill no longer needs to
translate codes — it just fills opponent from the schedule directly.

Safe to re-run: UPDATE only touches rows where opponent IS NULL OR opponent = ''.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nfl_data.db")

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT home_team, away_team, week, season
        FROM schedule
        WHERE season = 2025
    """)
    schedule_rows = cur.fetchall()

    if not schedule_rows:
        print("No 2025 schedule rows found — nothing to do.")
        conn.close()
        return

    updates = []
    for home, away, week, season in schedule_rows:
        # home team's opponent is away; away team's opponent is home
        updates.append((away, home, season, week))
        updates.append((home, away, season, week))

    cur.executemany("""
        UPDATE player_weekly_stats
        SET opponent = ?
        WHERE team = ? AND season = ? AND week = ?
          AND (opponent IS NULL OR opponent = '')
    """, updates)
    conn.commit()
    patched = cur.rowcount
    print(f"Patched {patched} rows")

    # Verify
    cur.execute("""
        SELECT team, COUNT(*) FROM player_weekly_stats
        WHERE season = 2025 AND (opponent IS NULL OR opponent = '')
        GROUP BY team
    """)
    remaining = cur.fetchall()
    if remaining:
        print(f"WARNING: {sum(r[1] for r in remaining)} rows still empty after backfill:")
        for row in remaining:
            print(f"  {row[0]}: {row[1]} rows")
    else:
        print("Verification passed: zero empty-opponent rows in 2025.")

    conn.close()


if __name__ == "__main__":
    main()
