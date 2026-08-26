#!/usr/bin/env python3
"""Backfill `player_weekly_stats.fumbles_lost` and correct `fantasy_points`.

2025 was ingested through the PBP fallback (`import_weekly_data` 404s for it,
so `nfl_data_loader` fell back to `get_weekly_stats_from_pbp`). That path does
not produce fumbles, and `_ensure_store_weekly_schema` defaults the column to
0 -- so all 6,764 rows read as "no fumbles lost" against ~240/season in every
other year, and `fantasy_points` was computed from that, overstating the
target by 2 points per lost fumble.

Source is the authoritative nflverse weekly release rather than a PBP
derivation. Validated against 2024, where the stored values are known good:
the release agrees on 6,692 of 6,695 rows (stored 241 vs source 242). A
straight PBP derivation was tried first and reached only 234/241 -- it counts
special-teams and aborted-snap fumbles that the fantasy definition excludes.

    fumbles_lost = sack_fumbles_lost + rushing_fumbles_lost
                   + receiving_fumbles_lost

Usage:
    python scripts/backfill_fumbles_lost.py --seasons 2025            # dry-run
    python scripts/backfill_fumbles_lost.py --seasons 2025 --write
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DB_PATH  # noqa: E402

RELEASE = ("https://github.com/nflverse/nflverse-data/releases/download/"
           "stats_player/stats_player_week_{season}.parquet")
FUMBLE_PARTS = ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
POINTS_PER_FUMBLE_LOST = 2.0


def load_source(season: int) -> pd.DataFrame:
    """Player-week fumbles lost for `season`, all season types.

    POST rows are kept: our week numbering runs straight through the playoffs
    and player_weekly_stats carries those rows, so filtering to REG here would
    leave playoff-week fumbles unfixed.
    """
    df = pd.read_parquet(RELEASE.format(season=season))
    missing = [c for c in FUMBLE_PARTS if c not in df.columns]
    if missing:
        raise SystemExit(f"{season}: release is missing {missing}")
    id_col = "player_id" if "player_id" in df.columns else "gsis_id"
    df["fumbles_lost"] = sum(df[c].fillna(0) for c in FUMBLE_PARTS)
    out = (df[[id_col, "week", "fumbles_lost"]]
           .rename(columns={id_col: "player_id"})
           .groupby(["player_id", "week"], as_index=False)["fumbles_lost"].sum())
    out["week"] = out["week"].astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--write", action="store_true",
                    help="apply changes (default is a dry run)")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    try:
        for season in args.seasons:
            src = load_source(season)
            cur = pd.read_sql(
                "SELECT player_id, week, fumbles_lost, fantasy_points "
                "FROM player_weekly_stats WHERE season = ?", conn, params=[season])
            m = cur.merge(src, on=["player_id", "week"], how="left",
                          suffixes=("_db", "_src"))
            m["fumbles_lost_src"] = m["fumbles_lost_src"].fillna(0)
            m["delta"] = m["fumbles_lost_src"] - m["fumbles_lost_db"]
            changed = m[m.delta != 0]

            print(f"\n=== {season} ===")
            print(f"  rows in DB              : {len(m)}")
            print(f"  stored fumbles_lost     : {m.fumbles_lost_db.sum():.0f}")
            print(f"  source fumbles_lost     : {m.fumbles_lost_src.sum():.0f}")
            print(f"  rows changing           : {len(changed)}")
            print(f"  fantasy_points to remove: "
                  f"{changed.delta.sum() * POINTS_PER_FUMBLE_LOST:.0f}")
            if changed.empty:
                continue
            if not args.write:
                print("  (dry run -- pass --write to apply)")
                continue

            # fantasy_points was computed with fumbles_lost = 0, so the penalty
            # for the newly-known fumbles has never been applied. Adjust by the
            # DELTA rather than recomputing from components, which would silently
            # re-derive every other scoring term too.
            rows = [(int(r.fumbles_lost_src),
                     -POINTS_PER_FUMBLE_LOST * float(r.delta),
                     r.player_id, int(r.week), season)
                    for r in changed.itertuples()]
            conn.executemany(
                "UPDATE player_weekly_stats "
                "SET fumbles_lost = ?, fantasy_points = fantasy_points + ? "
                "WHERE player_id = ? AND week = ? AND season = ?", rows)
            conn.commit()
            print(f"  updated {len(rows)} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    if "--write" in sys.argv:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup = Path(str(DB_PATH) + f".bak-fumbles-{stamp}")
        shutil.copy2(DB_PATH, backup)
        print(f"backup: {backup}")
    main()
