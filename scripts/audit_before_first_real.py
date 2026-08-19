#!/usr/bin/env python
"""Audit whether pre-debut synthetic weeks represent an eligible player.

`before_first_real` weeks -- synthetic weeks falling before a player's first
real appearance of the season -- are 46.5% of synthetic weeks and 51% of
manufactured fantasy points in the 75-100%-synthetic bucket
(data/experiments/synthetic_row_weeks.csv). Those weeks assert "here is what
this player would have scored", which is only a coherent counterfactual if
the player was actually part of the eligible population that week.

This is a DATA AUDIT, not a modelling experiment: it classifies each such
week against roster/depth-chart/snap evidence and quantifies the
consequences. It changes nothing and fits no model.

Evidence, in order of authority:
  weekly_rosters  status per player-week, complete 2023-2025. Primary axis.
  depth_charts    depth_team (1 = starter), 2023-2024 ONLY -- absent for
                  2025, and 2024 uses a wider full-roster format than
                  earlier years (GAPS.md). Corroborating only.
  snap_counts     offensive snaps, joined by name. Used to confirm
                  non-participation, not to establish role.

A QB on the active roster is NOT automatically a plausible participant --
the backup is active every week and takes no snaps -- so ACT is split by
depth-chart rank where that exists and left explicitly ambiguous where it
doesn't, rather than silently meaning different things across seasons.

Usage:
    python scripts/audit_before_first_real.py
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# weekly_rosters.status -> what it means for eligibility that week.
STATUS_MEANING = {
    "ACT": "active roster",
    "INA": "declared inactive",
    "DEV": "practice squad",
    "RES": "reserve (IR/PUP/NFI)",
    "CUT": "waived",
    "RET": "retired",
    "EXE": "exempt",
    "E01": "exempt",
}

# Categories, ordered from most to least representable.
CAT_STARTER = "on roster, listed starter"
CAT_ACTIVE_BACKUP = "on roster, listed backup"
CAT_ACTIVE_UNKNOWN = "on roster, role unknown (no depth chart)"
CAT_INACTIVE = "on roster, declared inactive"
CAT_PRACTICE_SQUAD = "practice squad"
CAT_RESERVE = "reserve / waived / retired"
CAT_NOT_ROSTERED = "not on a roster that week"
CAT_PRE_ARRIVAL = "not yet in league / pre-acquisition"

# Only a listed starter is unambiguously a plausible participant. Everything
# else is either ineligible or a counterfactual the data cannot support.
REPRESENTABLE = {CAT_STARTER}


def classify(row) -> str:
    if pd.isna(row["status"]):
        return CAT_PRE_ARRIVAL if row["before_first_roster_week"] else CAT_NOT_ROSTERED
    s = row["status"]
    if s in ("RES", "CUT", "RET", "EXE", "E01"):
        return CAT_RESERVE
    if s == "DEV":
        return CAT_PRACTICE_SQUAD
    if s == "INA":
        return CAT_INACTIVE
    if s == "ACT":
        if pd.isna(row["depth_team"]):
            return CAT_ACTIVE_UNKNOWN
        return CAT_STARTER if int(row["depth_team"]) == 1 else CAT_ACTIVE_BACKUP
    return CAT_ACTIVE_UNKNOWN


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weeks-csv", type=Path,
                    default=Path("data/experiments/synthetic_row_weeks.csv"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/experiments/before_first_real_audit.csv"))
    args = ap.parse_args()

    from config.settings import DB_PATH
    conn = sqlite3.connect(str(DB_PATH))

    w = pd.read_csv(args.weeks_csv)
    seasons = sorted(w.season.unique().tolist())
    lo, hi = min(seasons), max(seasons)

    bfr = w[(~w.is_real) & (w.position_class == "before_first_real")].copy()
    bfr = bfr[["season", "player", "week", "prediction"]]

    rosters = pd.read_sql(
        f"""SELECT player_id, season, week, team, status, full_name
            FROM weekly_rosters
            WHERE game_type='REG' AND season BETWEEN {lo} AND {hi}""", conn)
    rosters = rosters.drop_duplicates(["player_id", "season", "week"])

    # First week the player appears on ANY roster that season -- distinguishes
    # "signed later" from "cut mid-season and off the roster this week".
    first_roster = (rosters.groupby(["player_id", "season"])["week"].min()
                    .rename("first_roster_week").reset_index())

    depth = pd.read_sql(
        f"""SELECT gsis_id AS player_id, CAST(season AS INT) season,
                   CAST(week AS INT) week, MIN(CAST(depth_team AS INT)) depth_team
            FROM depth_charts
            WHERE position='QB' AND season BETWEEN {lo} AND {hi}
              AND gsis_id IS NOT NULL
            GROUP BY gsis_id, season, week""", conn)

    a = (bfr
         .merge(rosters, left_on=["player", "season", "week"],
                right_on=["player_id", "season", "week"], how="left")
         .merge(first_roster, left_on=["player", "season"],
                right_on=["player_id", "season"], how="left", suffixes=("", "_fr"))
         .merge(depth, left_on=["player", "season", "week"],
                right_on=["player_id", "season", "week"], how="left",
                suffixes=("", "_d")))

    a["before_first_roster_week"] = (
        a.first_roster_week.isna() | (a.week < a.first_roster_week))
    a["status_meaning"] = a.status.map(STATUS_MEANING).fillna("no roster row")
    a["category"] = a.apply(classify, axis=1)
    a["representable"] = a.category.isin(REPRESENTABLE)
    a["depth_chart_available"] = a.season.isin(
        depth.season.unique().tolist())

    keep = ["season", "player", "full_name", "week", "prediction", "team",
            "status", "status_meaning", "depth_team", "depth_chart_available",
            "first_roster_week", "category", "representable"]
    a = a[keep].sort_values(["season", "player", "week"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    a.to_csv(args.output, index=False)
    print(f"Wrote {len(a)} classified before_first_real weeks -> {args.output}")

    print("\n" + "=" * 78)
    print("STEP 1 — CLASSIFICATION OF EVERY before_first_real WEEK")
    print("=" * 78)
    t = a.groupby("category").agg(
        weeks=("week", "size"),
        points=("prediction", "sum"),
        QBs=("player", "nunique"))
    t["weeks_%"] = 100 * t.weeks / t.weeks.sum()
    t["points_%"] = 100 * t.points / t.points.sum()
    print(t.sort_values("points", ascending=False).round(1).to_string())
    print(f"\n  representable (listed starter only): "
          f"{100*a.representable.mean():.1f}% of weeks, "
          f"{100*a[a.representable].prediction.sum()/a.prediction.sum():.1f}% of points")
    print(f"  depth chart unavailable (2025)     : "
          f"{100*(~a.depth_chart_available).mean():.1f}% of weeks — role cannot be established")

    conn.close()


if __name__ == "__main__":
    main()
