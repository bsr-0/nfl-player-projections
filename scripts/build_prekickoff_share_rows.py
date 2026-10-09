#!/usr/bin/env python3
"""Build Plan A's share rows for a week before it is played, and audit them for leakage.

Plan A forecasts from `team_week_player_shares` rows, which exist only for
games already played. Before kickoff, this builds the same rows for a frozen
player list (scripts/build_prekickoff_population.py): the history strictly
before (season, week) from the audited panel, plus one stub row per listed
player at the target week with unknown (zero) volumes. Every feature is a
lag, so it is computed exactly as in the stored table.

    build   rows for one week's frozen list.
    audit   leakage gate: rebuild past weeks with their *played* players as
            the list, from pre-kickoff data only, and require every feature
            column to equal the stored table's row. Any difference fails.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import DB_PATH, POSITIONS
from scripts.build_team_week_player_shares import (
    build_shares, build_shares_from, drop_unplayed_team_weeks, load_volumes)
from src.evaluation.team_reconstruction_candidates import _team_feature_columns
from src.models.team_allocation.features import feature_columns, load_share_rows

KEY = ["player_id", "season", "week", "team", "position"]


def history(con: sqlite3.Connection, season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel rows and volumes strictly before (season, week), back to the prior season."""
    pop = pd.read_sql(
        "SELECT player_id, season, week, team, position FROM canonical_player_weeks "
        f"WHERE position IN ({','.join('?' * len(POSITIONS))}) "
        "AND (season = ? OR (season = ? AND week < ?))",
        con, params=[*POSITIONS, season - 1, season, week])
    vol = load_volumes(con, season - 1, season)
    vol = vol[(vol.season < season) | (vol.week < week)]
    pop, _ = drop_unplayed_team_weeks(con, pop, vol)
    if week > 1 and (season, week - 1) not in set(zip(vol.season, vol.week)):
        raise ValueError(f"{season} week {week - 1} has no stats yet: run the data refresh "
                         f"(src.data.auto_refresh) before forecasting week {week}")
    return pop, vol


def prekickoff_rows(season: int, week: int, listed: pd.DataFrame,
                    con: sqlite3.Connection | None = None) -> pd.DataFrame:
    """Share rows for `listed` (player_id, team, position) at (season, week), before kickoff."""
    if listed.player_id.duplicated().any():
        raise ValueError("player listed twice")
    own = con is None
    con = con or sqlite3.connect(str(DB_PATH))
    try:
        pop, vol = history(con, season, week)
    finally:
        if own:
            con.close()
    stubs = listed[["player_id", "team", "position"]].assign(season=season, week=week)[KEY]
    rows = build_shares_from(pd.concat([pop, stubs], ignore_index=True), vol)
    out = rows[(rows.season == season) & (rows.week == week)].reset_index(drop=True)
    if len(out) != len(stubs):
        raise ValueError(f"expected {len(stubs)} stub rows, built {len(out)}")
    return out


def audit(weeks: list[tuple[int, int]]) -> dict:
    report = {}
    with sqlite3.connect(str(DB_PATH)) as con:
        for season, week in weeks:
            stored = load_share_rows(seasons=[season], con=con)
            reference = "stored team_week_player_shares"
            if stored.empty:
                # Season not yet written to the table: rebuild it in memory with
                # the table's own write path (full panel, nothing truncated).
                stored = build_shares(con, season - 1, season)
                stored = stored[stored.season == season]
                reference = "build_shares over the full panel, in memory"
            stored = stored[stored.week == week].reset_index(drop=True)
            if stored.empty:
                raise ValueError(f"no reference rows for {season} week {week}")
            built = prekickoff_rows(season, week, stored[["player_id", "team", "position"]], con)
            # Player features plus every column a team-total arm may read.
            cols = feature_columns(stored, include_full_ppr=True)
            cols += [c for c in _team_feature_columns(stored, "receiving_yards") if c not in cols]
            if set(cols) - set(built.columns):
                raise SystemExit(f"{season} w{week}: builder lacks {sorted(set(cols) - set(built.columns))}")
            a = stored.set_index("player_id").sort_index()
            b = built.set_index("player_id").sort_index()
            if not a.index.equals(b.index) or not (a.team == b.team).all():
                raise SystemExit(f"{season} w{week}: keys differ")
            bad = {}
            for c in cols:
                x, y = a[c].to_numpy(float), b[c].to_numpy(float)
                same = (x == y) | (np.isnan(x) & np.isnan(y))
                if not same.all():
                    bad[c] = int((~same).sum())
            report[f"{season}-w{week:02d}"] = {"rows": len(a), "features": len(cols), "reference": reference, "differing": bad}
            print(json.dumps({f"{season}-w{week:02d}": report[f"{season}-w{week:02d}"]}), flush=True)
            if bad:
                raise SystemExit(f"LEAKAGE AUDIT FAILED {season} w{week}: {bad}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--population", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    au = sub.add_parser("audit")
    au.add_argument("--weeks", nargs="+", required=True, help="season:week, e.g. 2025:6")
    a = ap.parse_args()
    if a.cmd == "audit":
        audit([tuple(int(x) for x in w.split(":")) for w in a.weeks])
        print("leakage audit passed")
        return
    listed = pd.read_csv(a.population, dtype={"player_id": str})
    (season,), (week,) = listed.season.unique(), listed.week.unique()
    rows = prekickoff_rows(int(season), int(week), listed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(a.output, index=False)
    print(f"wrote {len(rows)} rows to {a.output}")


if __name__ == "__main__":
    main()
