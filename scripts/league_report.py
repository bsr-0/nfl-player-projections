"""Insights for one fantasy team, built from a private ESPN league snapshot.

Writes the report as JSON under data/espn_private/reports/, which the
gitignore already covers by location -- the same rule that keeps the snapshot
itself out of the repo (tests/test_espn_data_stays_private.py). Nothing here
touches docs/.

`--check-join` reports the join the whole thing rests on: whether a player
ESPN names resolves to a projection this project produced, and who does not,
because a silent 80% match rate is how a report ends up confidently ranking
two thirds of a roster.

Which team is yours comes from ESPN_TEAM_ID, or --team with an id or any part
of the name.

Usage:
    ESPN_TEAM_ID=1 python scripts/league_report.py
    python scripts/league_report.py --team "Baby Back Gibbs" --week 3
    python scripts/league_report.py --check-join
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ESPN_PRIVATE_DIR, PROJECT_ROOT
from src.integrations.league_insights import build_report
from src.integrations.league_join import (
    load_projections, load_snapshot, join_players, match_report,
)

REPORT_DIR = ESPN_PRIVATE_DIR / "reports"


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


def _summarise(report: dict) -> None:
    m = report["matchup"]
    print(f"{report['team']['name']} -- week {report['week']} vs "
          f"{m['opponent']}")
    print(f"  projected {m['projected_total']} to "
          f"{m['opponent_projected_total']} ({m['edge']:+})")
    print(f"  mode={report['projection_mode']} -- {report['mode_note']}")
    print("\n  Starters")
    for p in report["starters"]:
        flag = "" if p["points_source"] == "model" else "  [espn]"
        print(f"    {p['slot']:<9}{p['name']:<24}{p['points']:>6}{flag}")
    for label, key in (("Tough calls", "tough_calls"), ("Waivers", "waivers")):
        rows = report[key]
        print(f"\n  {label}: {len(rows)}")
        for r in rows[:5]:
            if key == "tough_calls":
                print(f"    {r['slot']:<9}{r['starting']['name']:<20}"
                      f"{r['starting']['points']:>6}  vs  "
                      f"{r['benched']['name']:<20}{r['benched']['points']:>6}"
                      f"   gap {r['gap']}")
            else:
                print(f"    {r['position']:<4}{r['name']:<24}{r['points']:>6}"
                      f"   over {r['instead_of']['name']} by {r['over']}")
    trades = report["trades"]
    print(f"\n  Proposed trades (valued over "
          f"{trades['horizon_weeks']} weeks)")
    for t in trades["proposals"]:
        give, get = t["give"], t["get"]
        print(f"    with {t['with']}")
        print(f"      give {give['name']} ({give['position']}) "
              f"-> get {get['name']} ({get['position']})")
        print(f"      you +{t['our_gain_per_week']}/wk "
              f"(+{t['our_gain_over_horizon']} over {t['horizon_weeks']}), "
              f"them +{t['their_gain_per_week_espn']}/wk by ESPN's numbers")
        change = t["lineup_change"]
        moves = ([f"{c['name']} starts at {c['slot']}" for c in change["in"]]
                 + [f"{c['name']} to the bench" for c in change["out"]]
                 + [f"{c['name']} {c['from']}->{c['to']}"
                    for c in change["moved"]])
        if moves:
            print(f"      lineup: {'; '.join(moves)}")
    if not trades["proposals"]:
        print("    none that improve both lineups")
    print(f"\n  Disagreements ({trades['basis']}): "
          f"{len(trades['buy_low'])} buy-low, {len(trades['sell_high'])} sell-high")
    for kind in ("buy_low", "sell_high"):
        for r in trades[kind][:5]:
            pos, owner = r["position"], (r["fantasy_team"] or "")[:22]
            label = "buy" if kind == "buy_low" else "sell"
            print(f"    {label:<5}{r['name']:<22}{owner:<24}"
                  f"ours {pos}{r['model_rank']:<4} espn {pos}{r['espn_rank']:<4}"
                  f" {r['model_ppg']:>6} vs {r['espn_ppg']:>6}")


def build(team, week=None, write=True, horizon=None) -> int:
    snapshot = load_snapshot()
    week = snapshot.week if week is None else week
    projections = load_projections(snapshot.season, week)
    report = build_report(snapshot, projections, team, week, horizon=horizon)
    _summarise(report)

    if not write:
        return 0
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"report_{report['season']}_wk{report['week']}.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False, default=str))
    latest = REPORT_DIR / "latest.json"
    latest.write_text(out.read_text())
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)} ({out.stat().st_size:,} bytes)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-join", action="store_true",
                    help="report how much of the league resolves to a projection")
    ap.add_argument("--team", default=os.environ.get("ESPN_TEAM_ID"),
                    help="your team, by id or part of its name "
                         "(default: $ESPN_TEAM_ID)")
    ap.add_argument("--week", type=int, default=None,
                    help="week to project (default: the league's current week)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="weeks a trade is valued over "
                         "(default: the rest of the fantasy regular season)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the report without writing it")
    args = ap.parse_args()

    if args.check_join:
        return check_join(args.week)
    return build(args.team, args.week, write=not args.dry_run,
                 horizon=args.horizon)


if __name__ == "__main__":
    sys.exit(main())
