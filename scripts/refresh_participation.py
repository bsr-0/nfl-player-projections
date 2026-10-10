#!/usr/bin/env python3
"""Load current-season personnel groupings and pass-play participation every week.

The served model reads `team_pct_11/12/13/21_personnel_roll3_mean`
(`team_personnel_stats`) and `pbp_pass_play_participation_pct_roll3_mean`
(`pbp_pass_participation`), both derived from nflverse's participation file.
For 2026 that file is not published yet (HTTP 404), so both tables have no 2026
rows. Nothing here can fill them before nflverse publishes, but the existing
paths would also fail afterwards:

  * `ensure_team_personnel_stats()` (run by every live predict) fetches only
    seasons with no rows at all, so the first week it finds would be the last;
  * `get_*_from_pbp(season)` caches each season in data/raw on first success
    and returns that cache from then on;
  * nothing in the live path loads `pbp_pass_participation`.

This re-derives the season from upstream without the cache and upserts every
team-week (UNIQUE keys on both tables, so re-running replaces, never duplicates).
Only weeks already in `player_weekly_stats` are written, and a team-week with
fewer than MIN_PLAYS plays (a partly published game) is left out. Until the file
exists it prints "not published" and writes nothing.

    python scripts/refresh_participation.py --seasons 2026            # dry run
    python scripts/refresh_participation.py --seasons 2026 --write
    python scripts/refresh_participation.py --validate 2025           # re-derive a stored season
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import DB_PATH

MIN_PLAYS = {"team_personnel_stats": 20, "pbp_pass_participation": 10}
COLUMNS = {
    "team_personnel_stats": ["team", "season", "week", "pct_11", "pct_12", "pct_21", "pct_13",
                             "pct_other", "n_plays"],
    "pbp_pass_participation": ["player_id", "team", "season", "week", "team_pass_plays",
                               "player_pass_plays", "pass_play_participation_pct"],
}
KEYS = {"team_personnel_stats": ["team", "season", "week"],
        "pbp_pass_participation": ["player_id", "season", "week"]}
PLAYS = {"team_personnel_stats": "n_plays", "pbp_pass_participation": "team_pass_plays"}


def derive(table: str, season: int) -> pd.DataFrame:
    """Fresh upstream derivation, bypassing the per-season cache."""
    from src.data.pbp_stats_aggregator import (get_pass_play_participation_from_pbp,
                                               get_personnel_groupings_from_pbp)
    fn = get_personnel_groupings_from_pbp if table == "team_personnel_stats" \
        else get_pass_play_participation_from_pbp
    df = fn(season, use_cache=False)
    return df[COLUMNS[table]].copy() if not df.empty else pd.DataFrame(columns=COLUMNS[table])


def rows_to_write(con: sqlite3.Connection, table: str, derived: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict]:
    """Played weeks only, and only team-weeks with enough plays to be a whole game."""
    last = con.execute("SELECT MAX(week) FROM player_weekly_stats WHERE season = ?", (season,)).fetchone()[0] or 0
    d = derived[derived.season == season]
    played = d[(d.week >= 1) & (d.week <= last)]
    full = played[played[PLAYS[table]] >= MIN_PLAYS[table]]
    return full.reset_index(drop=True), {"derived": len(d), "after_week_filter": len(played),
                                         "left_out_short": len(played) - len(full), "last_week": int(last)}


def compare(stored: pd.DataFrame, derived: pd.DataFrame, table: str) -> dict:
    """How well a re-derivation reproduces stored rows (shared keys, value columns)."""
    key = KEYS[table]
    j = stored.merge(derived, on=key, how="outer", suffixes=("_s", "_d"), indicator=True)
    both = j[j._merge == "both"]
    vals = [c for c in COLUMNS[table] if c not in key and c != "team"]
    differing = {c: int((~np.isclose(both[c + "_s"].astype(float), both[c + "_d"].astype(float),
                                      rtol=1e-6, atol=1e-9)).sum()) for c in vals}
    return {"stored": len(stored), "derived": len(derived), "shared": len(both),
            "only_stored": int((j._merge == "left_only").sum()), "only_derived": int((j._merge == "right_only").sum()),
            "differing_cells": {c: n for c, n in differing.items() if n}}


def upsert(con: sqlite3.Connection, table: str, rows: pd.DataFrame) -> int:
    cols = COLUMNS[table]
    before = con.total_changes
    con.executemany(f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [tuple(r) for r in rows[cols].itertuples(index=False, name=None)])
    return con.total_changes - before


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=[])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--validate", type=int, metavar="SEASON",
                    help="re-derive a stored season and compare (downloads its PBP and participation)")
    a = ap.parse_args()
    if not a.seasons and a.validate is None:
        ap.error("give --seasons or --validate")

    if a.validate is not None:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
            for table in COLUMNS:
                stored = pd.read_sql(f"SELECT {', '.join(COLUMNS[table])} FROM {table} WHERE season = ?",
                                     con, params=[a.validate])
                print(f"validate {table} {a.validate}: {compare(stored, derive(table, a.validate), table)}")
        return

    plan = {}
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
        for season in a.seasons:
            for table in COLUMNS:
                derived = derive(table, season)
                if derived.empty:
                    print(f"{table} {season}: not published upstream (no participation data); nothing to load")
                    continue
                rows, info = rows_to_write(con, table, derived, season)
                stored = con.execute(f"SELECT COUNT(*) FROM {table} WHERE season = ?", (season,)).fetchone()[0]
                weeks = sorted(rows.week.astype(int).unique().tolist())
                print(f"{table} {season}: {info}, stored {stored}, to upsert {len(rows)}"
                      + (f", weeks {weeks}" if weeks else ""))
                if len(rows):
                    plan[(table, season)] = rows
    if not a.write or not plan:
        print("dry run: nothing written" if not a.write else "nothing to write")
        return
    backup = DB_PATH.parent / "backups" / f"nfl_data_pre_participation_{datetime.now():%Y%m%d_%H%M%S}.db"
    backup.parent.mkdir(exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as con, sqlite3.connect(str(backup)) as dst:
        con.backup(dst)
        print(f"backup: {backup}")
        for (table, season), rows in plan.items():
            print(f"  {table} {season}: upserted {upsert(con, table, rows)} rows")
        con.commit()


if __name__ == "__main__":
    main()
