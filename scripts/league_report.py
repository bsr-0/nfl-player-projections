"""Insights for one fantasy team, built from a private ESPN league snapshot.

PHASE 1 -- the join, and nothing else. Every section of the eventual report
rests on one question: does a player ESPN names resolve to a projection this
project produced? `--check-join` answers it out loud, including who does not
resolve and why, because a silent 80% match rate is how a report ends up
confidently ranking two thirds of a roster.

Reads only. Writes nothing, publishes nothing: league data stays under
data/espn_private/ (see tests/test_espn_data_stays_private.py).

Usage:
    python scripts/league_report.py --check-join
    python scripts/league_report.py --check-join --week 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROJECT_ROOT
from src.integrations.league_join import (
    load_projections, load_snapshot, join_players, match_report,
)


def _print_report(label: str, report: dict) -> None:
    matched, total = report["matched"], report["players"]
    rate = f"{matched / total:.0%}" if total else "n/a"
    methods = ", ".join(f"{k} {v}" for k, v in report["by_method"].items())
    print(f"\n{label}: {matched}/{total} matched ({rate})"
          + (f" -- {methods}" if methods else ""))
    for reason, n in report.get("by_reason", {}).items():
        print(f"  {n:>3} {reason}")
        examples = [u for u in report["unmatched"]
                    if u["unmatched_reason"] == reason]
        for u in examples[:8]:
            print(f"        {u['espn_name']:<24}{u['position']:<5}{u['nfl_team']}")
        if len(examples) > 8:
            print(f"        ... and {len(examples) - 8} more")


def check_join(week=None) -> int:
    snapshot = load_snapshot()
    week = snapshot.week if week is None else week
    print(f"Snapshot   : {snapshot.path.relative_to(PROJECT_ROOT)} "
          f"(pulled {snapshot.manifest.get('pulled_at')})")
    print(f"League     : {snapshot.info.get('name')} -- "
          f"season {snapshot.season}, week {week}")

    projections = load_projections(snapshot.season, week)
    mode = projections["projection_mode"].dropna().unique()
    mode = mode[0] if len(mode) else "unknown"
    projected = int(projections["season_total"].notna().sum())
    with_week = int(projections["week_points"].notna().sum())
    intervals = int(projections["week_ci_low"].notna().sum())
    print(f"Projections: {len(projections)} board players, {projected} with a "
          f"season total, {with_week} with a week {week} number")
    print(f"             mode={mode}"
          + (f", {intervals} carry an 80% interval" if intervals else
             ", no intervals published in this mode"))

    _print_report("Rostered", match_report(
        join_players(snapshot.rostered(), projections)))
    _print_report("Free agents", match_report(
        join_players(snapshot.free_agents, projections)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-join", action="store_true",
                    help="report how much of the league resolves to a projection")
    ap.add_argument("--week", type=int, default=None,
                    help="week to project (default: the league's current week)")
    args = ap.parse_args()

    if not args.check_join:
        ap.error("phase 1 only implements --check-join")
    return check_join(args.week)


if __name__ == "__main__":
    sys.exit(main())
