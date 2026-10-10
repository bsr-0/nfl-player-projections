#!/usr/bin/env python3
"""Stop a forward-test forecast from being written on empty live inputs.

Plan A's team-total model and the served model read team-level situational
volume (neutral-situation targets and rushes, third-down and red-zone targets,
high-leverage touches), built from PBP columns of `player_weekly_stats`. When
the weekly load stores those columns as zero, every team reads 0 and Plan A
silently forecasts a run-heavy league (GAPS.md, 2026-10-09: team passing yards
205 against a 229 norm, 503 of 661 players moved by 0.39 on average when the
columns were repaired). Nothing downstream notices, so this does.

For every loaded week of the season it computes the share of teams whose summed
column is exactly zero and fails if that exceeds a limit set well above the
worst week of 2020-2025 and well below the broken state (every team):

    column               worst week 2020-25   limit
    neutral_targets            3.8%             25%
    neutral_rushes             8.3%             25%
    third_down_targets         0.0%             25%
    high_leverage_touches      0.0%             25%
    redzone_targets           18.8%             50%

Missing 2026 weekly PFR / NGS / snap-count / participation rows for the newest week and a draft
table without official ids are reported as warnings only: those lag upstream
for a day or two and degrade a forecast less than they are worth missing a
deadline for.

    python scripts/check_live_inputs.py [--season 2026] [--db PATH]

Exit 0 = pass (warnings allowed), 1 = fail. Run by forward_week_job.sh after the
refresh steps and before scripts/run_forward_week.py.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

LIMITS = {  # max share of teams with a zero weekly sum
    "neutral_targets": 0.25, "neutral_rushes": 0.25, "third_down_targets": 0.25,
    "high_leverage_touches": 0.25, "redzone_targets": 0.50,
}
WEEKLY_TABLES = ["weekly_pfr", "snap_counts", "ngs_passing", "ngs_receiving", "ngs_rushing",
                 # nflverse participation; empty for 2026 until it is published
                 "team_personnel_stats", "pbp_pass_participation"]


def zero_share_by_week(con: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Rows = loaded weeks, columns = LIMITS; the share of teams whose weekly sum is zero."""
    sums = ",".join(f"COALESCE(SUM({c}), 0) AS {c}" for c in LIMITS)
    t = pd.read_sql(
        f"SELECT week, team, {sums} FROM player_weekly_stats "
        "WHERE season = ? AND week >= 1 AND team IS NOT NULL GROUP BY week, team",
        con, params=[season])
    if t.empty:
        return pd.DataFrame(columns=list(LIMITS))
    return (t[list(LIMITS)] == 0).groupby(t.week).mean()


def check(con: sqlite3.Connection, season: int) -> dict:
    """{'failures': [...], 'warnings': [...], 'weeks': [...], 'zero_share': DataFrame}."""
    failures, warnings = [], []
    z = zero_share_by_week(con, season)
    if z.empty:
        return {"failures": [f"no {season} rows in player_weekly_stats"], "warnings": [],
                "weeks": [], "zero_share": z}
    for week, row in z.iterrows():
        over = [f"{col} {row[col]:.0%} (limit {limit:.0%})" for col, limit in LIMITS.items() if row[col] > limit]
        if over:
            failures.append(f"{season} week {int(week)}: teams with a zero weekly sum over the limit: "
                            + ", ".join(over))
    newest = int(z.index.max())
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for table in WEEKLY_TABLES:
        if table not in have:
            warnings.append(f"table {table} does not exist")
        elif not con.execute(f"SELECT 1 FROM {table} WHERE season = ? AND week = ? LIMIT 1",
                             (season, newest)).fetchone():
            warnings.append(f"{table} has no {season} week {newest} rows")
    if "draft_picks_v2" in have:
        n, ok = con.execute("SELECT COUNT(*), SUM(player_id LIKE '00-%') FROM draft_picks_v2 "
                            "WHERE draft_season = ?", (season,)).fetchone()
        if n and (ok or 0) < 0.5 * n:
            warnings.append(f"only {ok or 0} of {n} {season} draft picks carry an official id")
    return {"failures": failures, "warnings": warnings, "weeks": [int(w) for w in z.index], "zero_share": z}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--db", type=Path, default=None)
    a = ap.parse_args()
    if a.db is None:
        from config.settings import DB_PATH
        a.db = Path(DB_PATH)
    with sqlite3.connect(f"file:{a.db}?mode=ro", uri=True) as con:
        r = check(con, a.season)
    if r["weeks"]:
        print(f"{a.season} weeks {r['weeks'][0]}-{r['weeks'][-1]}: share of teams with a zero weekly sum "
              "(worst week)")
        for col, limit in LIMITS.items():
            print(f"  {col:22s} {r['zero_share'][col].max():5.0%}  (limit {limit:.0%})")
    for w in r["warnings"]:
        print(f"WARNING: {w}")
    for f in r["failures"]:
        print(f"FAIL: {f}")
    print("LIVE INPUT CHECK FAILED" if r["failures"] else "live input check passed")
    return 1 if r["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
