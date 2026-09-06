#!/usr/bin/env python
"""Extend depth_charts and NGS coverage backwards to their true upstream floor.

Both tables start later than nflverse actually carries them, for the same
reason: scripts/backfill_all_data.py hardcodes `SEASONS = 2018+` (and
`range(2024, 2026)` for depth charts). Nothing about the data required
those floors.

    depth_charts   had 2020-2024  ->  upstream carries 2013+
    ngs_*          had 2018-2025  ->  upstream carries 2016+

Both are pure appends in the existing schema -- unlike 2025's depth charts,
which needed a schema bridge (scripts/backfill_depth_charts.py).

APPEND, never replace. `backfill_all_data.py::_save_df` defaults to
`if_exists="replace"`, which drops the whole table; running it would undo
this. That is the footgun this script deliberately does not reuse.

Usage:
    python scripts/backfill_historical_coverage.py --dry-run
    python scripts/backfill_historical_coverage.py --datasets depth_charts
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

DEPTH_CHART_SEASONS = range(2013, 2020)
NGS_SEASONS = range(2016, 2018)
NGS_STAT_TYPES = ("passing", "rushing", "receiving")

DEPTH_CHART_COLUMNS = ["season", "club_code", "week", "depth_team", "last_name",
                       "first_name", "football_name", "position",
                       "jersey_number", "gsis_id", "depth_position", "full_name"]


def prepare_depth_chart_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Reduce a raw weekly depth-chart pull to the stored schema.

    Postseason is already encoded as continuing week numbers (2013: 18=WC ..
    21=SB), matching the stored 2020-2024 convention, so weeks pass through
    untouched. Rows without a week are the Super Bowl bye (game_type SBBYE)
    and are unusable.

    The feed separates some listings only by `formation`/`game_type` -- a
    player in both the base and a sub-package. Our table carries neither
    column, so those rows arrive identical and are collapsed. Rows differing
    in `depth_team` are deliberately NOT collapsed: those are genuine
    conflicting listings (610 of them across 2013-2019), resolved by MIN in
    _load_depth_chart_asof_table. Dropping them here would silently pick a
    winner at load time instead, in a different place from where that policy
    is documented.
    """
    d = d.dropna(subset=["week", "gsis_id"])
    d = d[[c for c in DEPTH_CHART_COLUMNS if c in d.columns]]
    return d.drop_duplicates()


def _existing_seasons(conn: sqlite3.Connection, table: str) -> set:
    df = pd.read_sql(f"SELECT DISTINCT season FROM {table} WHERE season IS NOT NULL", conn)
    return {int(s) for s in df.season}


def backfill_depth_charts(conn, dry_run: bool) -> None:
    import nfl_data_py as nfl

    have = _existing_seasons(conn, "depth_charts")
    print(f"depth_charts: have {sorted(have)}")

    frames = []
    for season in DEPTH_CHART_SEASONS:
        if season in have:
            print(f"  {season}: already present — skipping")
            continue
        d = nfl.import_depth_charts([season])
        if d.empty:
            print(f"  {season}: upstream empty")
            continue
        if "depth_team" not in d.columns:
            # 2025+ ships the daily ESPN-style feed instead; that needs the
            # dedicated bridge, not a straight append.
            print(f"  {season}: new-schema feed — use backfill_depth_charts.py")
            continue
        # Postseason is already encoded as continuing week numbers (2013:
        # 18=WC .. 21=SB), matching the stored 2020-2024 convention, so weeks
        # pass through untouched. The only unusable rows are the Super Bowl
        # bye (game_type SBBYE), which carries no week at all.
        before = len(d)
        d = prepare_depth_chart_frame(d)
        print(f"  {season}: {len(d):,} rows (dropped {before - len(d):,} "
              f"without week/gsis_id or exactly duplicated)")
        frames.append(d)

    if not frames:
        print("  nothing to load")
        return
    new = pd.concat(frames, ignore_index=True)
    if dry_run:
        print(f"  DRY RUN — would append {len(new):,} rows")
        return
    new.to_sql("depth_charts", conn, if_exists="append", index=False)
    print(f"  appended {len(new):,} rows")


def backfill_ngs(conn, dry_run: bool) -> None:
    import nfl_data_py as nfl

    for stat_type in NGS_STAT_TYPES:
        table = f"ngs_{stat_type}"
        have = _existing_seasons(conn, table)
        wanted = [s for s in NGS_SEASONS if s not in have]
        if not wanted:
            print(f"{table}: {sorted(have)} — nothing missing")
            continue

        d = nfl.import_ngs_data(stat_type=stat_type, years=list(wanted))
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        missing = [c for c in cols if c not in d.columns]
        if missing:
            print(f"{table}: upstream missing {missing} — skipping")
            continue
        d = d[cols]
        print(f"{table}: {len(d):,} rows for {list(wanted)}")
        if dry_run:
            continue
        d.to_sql(table, conn, if_exists="append", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["depth_charts", "ngs"],
                    choices=["depth_charts", "ngs"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from config.settings import DB_PATH
    conn = sqlite3.connect(str(DB_PATH))

    if "depth_charts" in args.datasets:
        backfill_depth_charts(conn, args.dry_run)
    if "ngs" in args.datasets:
        backfill_ngs(conn, args.dry_run)

    if not args.dry_run:
        conn.commit()
        print("\ncoverage now:")
        for table in ["depth_charts", "ngs_passing", "ngs_rushing", "ngs_receiving"]:
            df = pd.read_sql(
                f"SELECT MIN(season) lo, MAX(season) hi, COUNT(*) rows FROM {table}", conn)
            print(f"  {table:16s} {int(df.lo[0])}-{int(df.hi[0])}  {int(df.rows[0]):,} rows")
    conn.close()


if __name__ == "__main__":
    main()
