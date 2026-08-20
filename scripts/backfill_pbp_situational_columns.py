#!/usr/bin/env python3
"""Backfill the situational `player_weekly_stats` columns that only 2025 has.

The 2025 re-ingest went through `store_weekly_dataframe` (the PBP path),
which derives a much wider column set than the historical
`import_weekly_data` path ever wrote. Six columns are populated in 2025 and
NO other season:

    redzone_targets, neutral_targets, third_down_targets,
    goal_line_touches, two_minute_targets, high_leverage_touches

and four more start only in 2020 (rush_inside_10, rush_inside_5,
targets_15_plus, air_yards). Every one is a train/test discontinuity of the
same kind as `home_away` and `team_stats.total_plays`, which together
accounted for ~0.43 MAE on the 2025 fold (GAPS.md 2026-08-20).

Definitions are not re-implemented here: this calls
`PBPStatsAggregator.aggregate_all_stats`, the exact code that produced the
2025 values. Validated against 2025 before writing -- 99.9-100% exact
agreement, correlation 0.9988-1.0000 on all 12 columns.

A column is backfilled for a season only when that season's stored total is
zero, i.e. it was never populated. Seasons where the column already has
data are left untouched rather than overwritten.

Usage:
    python scripts/backfill_pbp_situational_columns.py                  # dry-run, all seasons
    python scripts/backfill_pbp_situational_columns.py --seasons 2018 2019
    python scripts/backfill_pbp_situational_columns.py --write
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH, MIN_HISTORICAL_YEAR

TARGET_COLUMNS = [
    "redzone_targets", "neutral_targets", "third_down_targets",
    "goal_line_touches", "two_minute_targets", "high_leverage_touches",
    "rush_inside_10", "rush_inside_5", "targets_15_plus", "air_yards",
    "neutral_rushes", "short_yardage_rushes",
]
# Newest season is the reference implementation; never rewrite it.
REFERENCE_SEASON = 2025


def reconstruct(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl
    from src.data.pbp_stats_aggregator import PBPStatsAggregator

    pbp = nfl.import_pbp_data([season])
    if pbp is None or pbp.empty:
        return pd.DataFrame()
    stats = PBPStatsAggregator().aggregate_all_stats(pbp_df=pbp, include_advanced=True)
    if stats.empty:
        return stats
    keep = ["player_id", "season", "week"] + [c for c in TARGET_COLUMNS if c in stats.columns]
    out = stats[keep].copy()
    for c in keep[3:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    # A player can appear once per (id, week) after the aggregator's own
    # collapse, but guard anyway -- a duplicate here would double-count.
    return out.groupby(["player_id", "season", "week"], as_index=False).sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--seasons", nargs="+", type=int)
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    seasons = args.seasons or [
        s for (s,) in conn.execute(
            "SELECT DISTINCT season FROM player_weekly_stats WHERE season >= ? ORDER BY season",
            (MIN_HISTORICAL_YEAR,)).fetchall()
        if s != REFERENCE_SEASON
    ]
    print(f"Seasons to process: {seasons[0]}-{seasons[-1]} ({len(seasons)})")

    totals = pd.read_sql(
        "SELECT season, " + ", ".join(f"SUM({c}) AS {c}" for c in TARGET_COLUMNS) +
        " FROM player_weekly_stats GROUP BY season", conn).set_index("season")

    if args.write:
        backup = DB_PATH.parent / f"nfl_data.db.bak-pbpcols-{datetime.now():%Y%m%d%H%M%S}"
        shutil.copy2(DB_PATH, backup)
        print(f"Backed up database to {backup}\n")

    grand = 0
    for season in seasons:
        empty_cols = [c for c in TARGET_COLUMNS
                      if season in totals.index and not totals.loc[season, c]]
        if not empty_cols:
            print(f"{season}: already populated, skipped")
            continue
        try:
            recon = reconstruct(season)
        except Exception as e:
            print(f"{season}: FAILED to reconstruct ({type(e).__name__}: {e})")
            continue
        if recon.empty:
            print(f"{season}: no PBP rows returned, skipped")
            continue

        cols = [c for c in empty_cols if c in recon.columns]
        nonzero = {c: int((recon[c] != 0).sum()) for c in cols}
        print(f"{season}: {len(recon)} PBP player-weeks; filling {len(cols)} cols; "
              f"nonzero rows " + ", ".join(f"{c}={n}" for c, n in nonzero.items() if n))

        if not args.write:
            continue

        cur = conn.cursor()
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS _recon
            (player_id TEXT, season INT, week INT, %s)
        """ % ", ".join(f"{c} REAL" for c in cols))
        cur.execute("DELETE FROM _recon")
        cur.executemany(
            f"INSERT INTO _recon VALUES ({','.join('?' * (3 + len(cols)))})",
            recon[["player_id", "season", "week"] + cols].itertuples(index=False, name=None),
        )
        set_clause = ", ".join(
            f"{c} = COALESCE((SELECT r.{c} FROM _recon r WHERE r.player_id = player_weekly_stats.player_id "
            f"AND r.season = player_weekly_stats.season AND r.week = player_weekly_stats.week), {c})"
            for c in cols
        )
        cur.execute(f"UPDATE player_weekly_stats SET {set_clause} WHERE season = ?", (season,))
        grand += cur.rowcount
        conn.commit()
        cur.execute("DROP TABLE _recon")

    if not args.write:
        print("\nDRY RUN -- nothing written. Re-run with --write to apply.")
        conn.close()
        return

    print(f"\nUpdated {grand} player-week rows. Coverage now:")
    after = pd.read_sql(
        "SELECT season, " + ", ".join(f"SUM({c}) AS {c}" for c in
                                      ["redzone_targets", "neutral_targets", "high_leverage_touches",
                                       "air_yards", "rush_inside_10"]) +
        " FROM player_weekly_stats GROUP BY season ORDER BY season", conn)
    print(after.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
