#!/usr/bin/env python3
"""Fill the PBP-derived team_stats columns that are NULL for played weeks.

For 2026, weeks 2-4 had all 18 of these columns NULL (points, turnovers,
time of possession, red-zone, third-down, sacks allowed, neutral pass rate,
drives, pace): the team-level PBP cache was written for week 1 and never
refreshed, because only the player cache was checked for staleness (fixed in
`get_pbp_advanced_stats`). The served model reads two of them
(`team_neutral_pass_rate_oe_roll3_mean`, `team_pace_sec_per_play_roll3_mean`),
which on live rows quietly averaged whatever weeks were filled.

A targeted UPDATE of NULL cells only, never `insert_team_stats` (its upsert
binds 0 for any column the frame lacks; see scripts/backfill_team_pbp_stats.py),
and only for weeks already in `player_weekly_stats`. Points come from the local
schedule, as in the loader. `--validate` re-derives a stored season and compares.

    python scripts/repair_team_pbp_columns.py --seasons 2026            # dry run
    python scripts/repair_team_pbp_columns.py --seasons 2026 --write
    python scripts/repair_team_pbp_columns.py --validate 2025
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

COLS = ["points_scored", "points_allowed", "turnovers", "time_of_possession", "redzone_attempts",
        "redzone_scores", "third_down_conv", "sacks_allowed", "neutral_pass_plays", "neutral_run_plays",
        "neutral_pass_rate", "neutral_pass_rate_lg", "neutral_pass_rate_oe", "drive_count",
        "drive_success_rate", "avg_drive_epa", "points_per_drive", "pace_sec_per_play"]
KEY = ["team", "season", "week"]


def derive(season: int) -> pd.DataFrame:
    """Team-week PBP stats with schedule points, the same frame the loader stores."""
    from src.data.nfl_data_loader import NFLDataLoader, _enrich_team_stats_from_schedule
    from src.data.pbp_stats_aggregator import get_team_stats_from_pbp
    team = get_team_stats_from_pbp(season, use_cache=True)
    if team.empty:
        return pd.DataFrame(columns=KEY + COLS)
    sched = NFLDataLoader()._local_schedule_df(season)
    if not sched.empty:
        team = _enrich_team_stats_from_schedule(team, sched)
    return team[[c for c in KEY + COLS if c in team.columns]]


def plan_updates(stored: pd.DataFrame, derived: pd.DataFrame, last_week: int) -> pd.DataFrame:
    """Long frame (team, season, week, column, value): stored NULL cells the derivation fills."""
    d = derived[(derived.week >= 1) & (derived.week <= last_week)].drop_duplicates(KEY)
    j = stored.merge(d, on=KEY, suffixes=("_old", "_new"))
    parts = []
    for c in [c for c in COLS if c + "_new" in j.columns]:
        take = j[c + "_old"].isna() & j[c + "_new"].notna()
        if take.any():
            parts.append(j.loc[take, KEY].assign(column=c, value=j.loc[take, c + "_new"].astype(float)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY + ["column", "value"])


def compare(stored: pd.DataFrame, derived: pd.DataFrame) -> dict:
    j = stored.merge(derived.drop_duplicates(KEY), on=KEY, suffixes=("_s", "_d"))
    out = {}
    for c in [c for c in COLS if c + "_d" in j.columns]:
        a, b = j[c + "_s"].astype(float), j[c + "_d"].astype(float)
        out[c] = int((~np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)).sum())
    return {"rows_compared": len(j), "differing": {c: n for c, n in out.items() if n}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=[])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--validate", type=int, metavar="SEASON")
    a = ap.parse_args()
    if not a.seasons and a.validate is None:
        ap.error("give --seasons or --validate")
    sel = f"SELECT {', '.join(KEY + COLS)} FROM team_stats WHERE season = ? AND week >= 1"
    if a.validate is not None:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
            stored = pd.read_sql(sel, con, params=[a.validate])
        print(f"validate {a.validate}: {compare(stored, derive(a.validate))}")
        return

    plans = []
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
        for season in a.seasons:
            last = con.execute("SELECT MAX(week) FROM player_weekly_stats WHERE season = ?",
                               (season,)).fetchone()[0] or 0
            stored = pd.read_sql(sel, con, params=[season])
            plan = plan_updates(stored, derive(season), last)
            null_before = stored[stored.week <= last][COLS].isna().sum().sum()
            print(f"{season} weeks 1-{last}: {len(stored)} stored team-weeks, {null_before} NULL cells, "
                  f"{len(plan)} to fill, weeks {sorted(plan.week.unique().tolist())}")
            plans.append(plan)
    plan = pd.concat(plans, ignore_index=True)
    if not a.write or plan.empty:
        print("dry run: nothing written" if not a.write else "nothing to write")
        return
    backup = DB_PATH.parent / "backups" / f"nfl_data_pre_team_pbp_{datetime.now():%Y%m%d_%H%M%S}.db"
    backup.parent.mkdir(exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as con, sqlite3.connect(str(backup)) as dst:
        con.backup(dst)
        print(f"backup: {backup}")
        before = con.total_changes
        for r in plan.itertuples(index=False):
            con.execute(f"UPDATE team_stats SET {r.column} = ? WHERE team = ? AND season = ? AND week = ? "
                        f"AND {r.column} IS NULL", (float(r.value), r.team, int(r.season), int(r.week)))
        con.commit()
        print(f"updated {con.total_changes - before} cells")


if __name__ == "__main__":
    main()
