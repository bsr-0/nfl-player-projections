#!/usr/bin/env python
"""Extend snap_counts coverage backwards using nflverse's PFR feed.

Our snap_counts table starts at 2018; nflverse carries the same game-level
PFR data from 2013. The 2012 release file exists but is EMPTY (0 rows), so
2013 is the real floor -- nflreadr's docs say "starting with the 2012
season", which overstates it by a year. PFR's own player pages do show 2012
(e.g. Luke McCown 2012-2015), so recovering 2012 would need direct scraping,
not this path.

Purpose is target/population construction, NOT another availability
multiplier: a snap row with offense_snaps == 0 is evidence of legitimate
non-participation, which lets a rostered player-week with no stat line be
classified rather than synthesised. The feed emits ~13,600 zero-offensive-snap
rows per season, so that signal is real and dense.

Usage:
    python scripts/backfill_snap_counts_history.py --seasons 2013 2014 2015 2016 2017
    python scripts/backfill_snap_counts_history.py --seasons 2013 --dry-run
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

COLUMNS = ["game_id", "pfr_game_id", "season", "game_type", "week", "player",
           "pfr_player_id", "position", "team", "opponent", "offense_snaps",
           "offense_pct", "defense_snaps", "defense_pct", "st_snaps", "st_pct"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import nfl_data_py as nfl
    from config.settings import DB_PATH
    conn = sqlite3.connect(str(DB_PATH))

    existing = pd.read_sql(
        "SELECT season, COUNT(*) n FROM snap_counts GROUP BY season", conn)
    have = set(existing.season.tolist())
    print("existing coverage:", sorted(have))

    frames = []
    for season in args.seasons:
        if season in have:
            print(f"  {season}: already present ({int(existing[existing.season==season].n.iloc[0]):,} rows) — skipping")
            continue
        try:
            d = nfl.import_snap_counts([season])
        except Exception as e:
            print(f"  {season}: FAILED — {type(e).__name__}: {e}")
            continue
        if d.empty:
            print(f"  {season}: upstream file is EMPTY — no data to load")
            continue
        missing = [c for c in COLUMNS if c not in d.columns]
        if missing:
            print(f"  {season}: missing columns {missing} — skipping")
            continue
        d = d[COLUMNS]
        zero = int((d.offense_snaps == 0).sum())
        print(f"  {season}: {len(d):,} rows, {zero:,} with offense_snaps == 0")
        frames.append(d)

    if not frames:
        print("\nNothing to load.")
        return
    new = pd.concat(frames, ignore_index=True)

    if args.dry_run:
        print(f"\nDRY RUN — would insert {len(new):,} rows for "
              f"{sorted(new.season.unique().tolist())}")
        return

    before = pd.read_sql("SELECT COUNT(*) n FROM snap_counts", conn).n.iloc[0]
    new.to_sql("snap_counts", conn, if_exists="append", index=False)
    after = pd.read_sql("SELECT COUNT(*) n FROM snap_counts", conn).n.iloc[0]
    print(f"\ninserted {after - before:,} rows ({before:,} -> {after:,})")

    dupes = pd.read_sql(
        """SELECT COUNT(*) n FROM (SELECT season, week, player, team, COUNT(*) c
           FROM snap_counts GROUP BY season, week, player, team HAVING c > 1)""",
        conn).n.iloc[0]
    print(f"duplicate player-weeks after load: {dupes}")
    print(pd.read_sql(
        """SELECT season, COUNT(*) rows, SUM(offense_snaps=0) off_zero
           FROM snap_counts WHERE game_type='REG' GROUP BY season ORDER BY season""",
        conn).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
