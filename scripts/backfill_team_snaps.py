#!/usr/bin/env python
"""Recompute player_weekly_stats.team_snaps / snap_share from snap_counts.

The 2026-08-07 "team_snaps inflation" fix corrected the ingest, but rows
written before it kept the bad values -- 2025 sits at avg 640 / max 2112
against 51-55 / 90-100 for every other season, because team_snaps had been
summed across players (and across both teams in the game) instead of being
the team's offensive play count. snap_share = snap_count / team_snaps is
therefore ~12x too small for that season, and snap_share_pct_roll3_mean and
snap_share_accel are causal features for RB/WR/TE.

Calls `compute_team_snaps` from the ingest module rather than re-deriving
the formula: a second copy is exactly how a fixed formula and a stale table
drift apart again.

Only team_snaps and snap_share are touched. Backs the DB up first, and
refuses to write if the recomputed values fail a sanity band.

Usage:
    python scripts/backfill_team_snaps.py --seasons 2025 --dry-run
    python scripts/backfill_team_snaps.py --seasons 2025
"""
import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data.pbp_stats_aggregator import compute_team_snaps

# A team runs roughly 40-90 offensive plays. Anything outside this is not a
# play count and must not be written.
MIN_PLAUSIBLE, MAX_PLAUSIBLE = 20, 120


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from config.settings import DB_PATH
    db_path = Path(DB_PATH)

    conn = sqlite3.connect(str(db_path))
    placeholders = ",".join("?" * len(args.seasons))
    snaps = pd.read_sql(
        f"""SELECT season, week, team, offense_snaps, offense_pct
            FROM snap_counts WHERE season IN ({placeholders})""",
        conn, params=args.seasons)
    if snaps.empty:
        print(f"No snap_counts rows for {args.seasons} — nothing to do.")
        return

    team_snaps = compute_team_snaps(snaps)
    print(f"Recomputed {len(team_snaps)} team-weeks from {len(snaps):,} snap rows")

    band = team_snaps[(team_snaps.team_snaps < MIN_PLAUSIBLE)
                      | (team_snaps.team_snaps > MAX_PLAUSIBLE)]
    print(f"  team_snaps: min {team_snaps.team_snaps.min()}, "
          f"median {team_snaps.team_snaps.median():.0f}, max {team_snaps.team_snaps.max()}")
    if len(band):
        print(f"  {len(band)} team-week(s) outside [{MIN_PLAUSIBLE},{MAX_PLAUSIBLE}]:")
        print(band.head(10).to_string(index=False))
        print("REFUSING to write — recomputed values are not plausible play counts.")
        return

    # Fixing the denominator alone is only valid if the numerator is sound.
    # 2025 also has snap_count doubled for ~10% of rows (an ingest-side
    # double-count, exactly 2x the source offense_snaps), which would come
    # out as snap_share = 2.0. Refuse rather than write a different wrong
    # number over the current one.
    pws_chk = pd.read_sql(
        f"""SELECT season, week, team, snap_count FROM player_weekly_stats
            WHERE season IN ({placeholders})""", conn, params=args.seasons)
    chk = pws_chk.merge(team_snaps, on=["season", "week", "team"], how="left")
    share = (chk.snap_count / chk.team_snaps).where(chk.team_snaps > 0)
    over = int((share > 1.0).sum())
    if over:
        print(f"\n  {over} row(s) would get snap_share > 1.0 (max {share.max():.2f}).")
        print("  That means snap_count itself is wrong, not just team_snaps.")
        print("REFUSING to write — recompute snap_count from source first "
              "(full re-ingest), or this just moves the error.")
        conn.close()
        return

    before = pd.read_sql(
        f"""SELECT season, ROUND(AVG(team_snaps),1) avg_team_snaps,
                   MAX(team_snaps) max_team_snaps, ROUND(AVG(snap_share),4) avg_snap_share,
                   COUNT(*) rows
            FROM player_weekly_stats WHERE season IN ({placeholders}) GROUP BY season""",
        conn, params=args.seasons)
    print("\nBEFORE:"); print(before.to_string(index=False))

    if args.dry_run:
        # Show what the new values would be without touching anything.
        pws = pd.read_sql(
            f"""SELECT season, week, team, snap_count FROM player_weekly_stats
                WHERE season IN ({placeholders})""", conn, params=args.seasons)
        merged = pws.merge(team_snaps, on=["season", "week", "team"], how="left")
        matched = merged.team_snaps.notna()
        print(f"\nDRY RUN — would update {int(matched.sum()):,} of {len(merged):,} rows "
              f"({100*matched.mean():.1f}% matched a team-week)")
        new_share = (merged.snap_count / merged.team_snaps).where(merged.team_snaps > 0)
        print(f"  new team_snaps : avg {merged.team_snaps.mean():.1f}, max {merged.team_snaps.max():.0f}")
        print(f"  new snap_share : avg {new_share.mean():.4f}, max {new_share.max():.4f}")
        conn.close()
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = db_path.with_suffix(f".db.bak-teamsnaps-{stamp}")
    conn.close()
    shutil.copy2(db_path, backup)
    print(f"\nBacked up DB -> {backup}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TEMP TABLE _ts (season INT, week INT, team TEXT, team_snaps INT)")
    cur.executemany("INSERT INTO _ts VALUES (?,?,?,?)",
                    team_snaps[["season", "week", "team", "team_snaps"]].values.tolist())
    cur.execute(f"""
        UPDATE player_weekly_stats AS p
           SET team_snaps = (SELECT t.team_snaps FROM _ts t
                          WHERE t.season=p.season AND t.week=p.week AND t.team=p.team),
            snap_share = CASE
                WHEN (SELECT t.team_snaps FROM _ts t
                      WHERE t.season=p.season AND t.week=p.week AND t.team=p.team) > 0
                THEN CAST(p.snap_count AS REAL) / (SELECT t.team_snaps FROM _ts t
                      WHERE t.season=p.season AND t.week=p.week AND t.team=p.team)
                ELSE 0.0 END
        WHERE p.season IN ({placeholders})
          AND EXISTS (SELECT 1 FROM _ts t
                      WHERE t.season=p.season AND t.week=p.week AND t.team=p.team)""",
        args.seasons)
    print(f"Updated {cur.rowcount:,} rows")
    conn.commit()

    after = pd.read_sql(
        f"""SELECT season, ROUND(AVG(team_snaps),1) avg_team_snaps,
                   MAX(team_snaps) max_team_snaps, ROUND(AVG(snap_share),4) avg_snap_share,
                   COUNT(*) rows
            FROM player_weekly_stats WHERE season IN ({placeholders}) GROUP BY season""",
        conn, params=args.seasons)
    print("\nAFTER:"); print(after.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
