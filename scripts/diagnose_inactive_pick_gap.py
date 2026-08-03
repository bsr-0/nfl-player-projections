#!/usr/bin/env python3
"""
Diagnose the model-vs-opponent inactive-pick gap.

Background: the 2026-04-23 wide-mode symmetric replay measured the
model picking inactive players on **22.4 %** of slots vs the
opponent's **15.0 %**. That 7.4 pp gap survives fresh per-week
predictions (i.e. is not a stale-prediction proxy artifact). The
chairman's verdict mandates diagnosing this gap before any
user-facing product work resumes.

This script runs five diagnostics on the wide-mode artifacts:

1. **By position.** Is the gap concentrated at rest-candidate
   positions (RB/WR) vs rarely-benched QB?
2. **By prediction tier.** For each position, does the inactive
   rate rise at the top ranks (load-managed stars) or is it flat?
3. **Injury-report cross-tab.** Were the model's inactive picks
   flagged as Questionable/Out/Doubtful at lock time by the
   `player_injuries` cache? If yes, the signal exists but the
   feature pipeline under-weights it.
4. **Counterfactual: drop inactive picks.** If the model dropped
   any pick it would have known was inactive (via injury report)
   before lock, does symmetric win rate recover?
5. **Qualitative top offenders.** Who are the players the model
   keeps picking when they sit?

Usage:
    python scripts/diagnose_inactive_pick_gap.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "nfl_data.db"

DEFAULT_RUNS: Tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "backtest_results" / "ts_backtest_2024_20260423_055841.json",
    PROJECT_ROOT / "data" / "backtest_results" / "ts_backtest_2025_20260423_055829.json",
)

DEFAULT_ROSTER_SLOTS: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}


def load_pred_rows(run_json: Path) -> Tuple[int, List[Dict]]:
    """Load the companion _predictions.csv for a run JSON."""
    csv_path = run_json.with_name(run_json.stem + "_predictions.csv")
    rows: List[Dict] = []
    season = None
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            s = int(r["season"])
            season = s
            rows.append({
                "season": s,
                "week": int(r["week"]),
                "player_id": r["player_id"],
                "name": r.get("name", ""),
                "position": r["position"],
                "team": r.get("team", ""),
                "predicted": float(r["predicted"]) if r.get("predicted") else None,
                "actual": float(r["actual"]) if r.get("actual") else None,
                "is_active": int(r.get("is_active", 1)) if r.get("is_active") not in (None, "") else 1,
            })
    return season, rows


def load_injury_map(conn: sqlite3.Connection) -> Dict[Tuple[str, int, int], str]:
    """(player_id, season, week) -> report_status."""
    rows = conn.execute(
        "SELECT player_id, season, week, report_status FROM player_injuries"
    ).fetchall()
    return {
        (pid, int(s), int(w)): (status or "None")
        for pid, s, w, status in rows
    }


def model_picks_per_week(
    pred_rows: List[Dict],
    slots: Dict[str, int] = DEFAULT_ROSTER_SLOTS,
) -> List[Dict]:
    """For each (season, week) in pred_rows, return the model's
    top-N-per-position picks from the wide pool (active + phantom).

    Each returned record: {season, week, slot_position, rank_within_pos,
    player_id, name, team, predicted, is_active, actual_or_zero}.
    """
    picks: List[Dict] = []
    by_sw = defaultdict(list)
    for r in pred_rows:
        by_sw[(r["season"], r["week"])].append(r)
    for (season, week), rows in sorted(by_sw.items()):
        for pos, n in slots.items():
            pool = sorted(
                (r for r in rows if r["position"] == pos and r["predicted"] is not None),
                key=lambda x: x["predicted"],
                reverse=True,
            )
            for rank, r in enumerate(pool[:n], start=1):
                picks.append({
                    "season": season,
                    "week": week,
                    "slot_position": pos,
                    "rank_within_pos": rank,
                    "player_id": r["player_id"],
                    "name": r["name"],
                    "team": r["team"],
                    "predicted": r["predicted"],
                    "is_active": r["is_active"],
                    "actual_or_zero": r["actual"] if r["actual"] is not None else 0.0,
                })
    return picks


def opponent_picks_per_week(
    conn: sqlite3.Connection,
    season: int,
    weeks: List[int],
    slots: Dict[str, int] = DEFAULT_ROSTER_SLOTS,
) -> List[Dict]:
    """Replicates scripts/prospective_opponent_replay.py's construction."""
    picks: List[Dict] = []
    cur = conn.cursor()
    for week in weeks:
        rows = cur.execute(
            """
            SELECT DISTINCT pws.player_id, p.position, p.name, pws.team
            FROM player_weekly_stats pws
            JOIN players p ON pws.player_id = p.player_id
            WHERE pws.season = ? AND pws.week < ?
            """,
            (season, week),
        ).fetchall()
        if not rows:
            continue
        pids = {r[0] for r in rows}
        placeholders = ",".join(["?"] * len(pids))
        prior = {
            pid: float(fp or 0.0)
            for pid, fp in cur.execute(
                f"SELECT player_id, fantasy_points FROM player_weekly_stats "
                f"WHERE season=? AND week=? AND player_id IN ({placeholders})",
                (season, week - 1, *pids),
            ).fetchall()
        }
        this_actual = {
            pid: float(fp or 0.0)
            for pid, fp in cur.execute(
                f"SELECT player_id, fantasy_points FROM player_weekly_stats "
                f"WHERE season=? AND week=? AND player_id IN ({placeholders})",
                (season, week, *pids),
            ).fetchall()
        }
        by_pos: Dict[str, List[Dict]] = {}
        for pid, pos, name, team in rows:
            if pid not in prior:
                continue
            by_pos.setdefault(pos, []).append({
                "player_id": pid,
                "name": name,
                "team": team or "",
                "prior_actual": prior[pid],
                "this_actual": this_actual.get(pid, None),
            })
        for pos, n in slots.items():
            pool = sorted(by_pos.get(pos, []), key=lambda x: x["prior_actual"], reverse=True)
            for rank, r in enumerate(pool[:n], start=1):
                picks.append({
                    "season": season,
                    "week": week,
                    "slot_position": pos,
                    "rank_within_pos": rank,
                    "player_id": r["player_id"],
                    "name": r["name"],
                    "team": r["team"],
                    "prior_actual": r["prior_actual"],
                    "is_active": 0 if r["this_actual"] is None else 1,
                    "actual_or_zero": r["this_actual"] if r["this_actual"] is not None else 0.0,
                })
    return picks


def pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d else "n/a"


def wilson_interval(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, default=list(DEFAULT_RUNS))
    ap.add_argument("--top-offenders", type=int, default=15, help="Number of top repeat-offender players to print.")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    injury_map = load_injury_map(conn)

    all_model_picks: List[Dict] = []
    all_opp_picks: List[Dict] = []
    for run_json in args.runs:
        season, pred_rows = load_pred_rows(run_json)
        if season is None:
            continue
        all_model_picks.extend(model_picks_per_week(pred_rows))
        weeks = sorted({r["week"] for r in pred_rows})
        all_opp_picks.extend(opponent_picks_per_week(conn, season, weeks))

    print("=" * 70)
    print("DIAGNOSTIC 1 — Inactive-pick rate by POSITION (load-management test)")
    print("=" * 70)
    print(f"{'pos':>6}  {'model inactive / total':>24}  {'opp inactive / total':>24}  {'gap':>8}")
    model_by_pos: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    opp_by_pos: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for p in all_model_picks:
        pos = p["slot_position"]
        w, t = model_by_pos[pos]
        model_by_pos[pos] = (w + (0 if p["is_active"] else 1), t + 1)
    for p in all_opp_picks:
        pos = p["slot_position"]
        w, t = opp_by_pos[pos]
        opp_by_pos[pos] = (w + (0 if p["is_active"] else 1), t + 1)
    for pos in ["QB", "RB", "WR", "TE"]:
        mi, mt = model_by_pos.get(pos, (0, 0))
        oi, ot = opp_by_pos.get(pos, (0, 0))
        mrate = (mi / mt * 100) if mt else 0
        orate = (oi / ot * 100) if ot else 0
        print(f"  {pos:<4}  {mi:>4} / {mt:<4}  ({mrate:>5.1f}%)    {oi:>4} / {ot:<4}  ({orate:>5.1f}%)    {mrate - orate:>+6.1f}pp")

    print()
    print("=" * 70)
    print("DIAGNOSTIC 2 — Inactive-pick rate BY MODEL RANK (does the gap concentrate at top-1?)")
    print("=" * 70)
    print(f"{'position':>8}  {'rank':>5}  {'inactive / picks':>20}  {'rate':>6}")
    model_by_rank: Dict[Tuple[str, int], Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for p in all_model_picks:
        key = (p["slot_position"], p["rank_within_pos"])
        w, t = model_by_rank[key]
        model_by_rank[key] = (w + (0 if p["is_active"] else 1), t + 1)
    for pos in ["QB", "RB", "WR", "TE"]:
        n_slots = DEFAULT_ROSTER_SLOTS[pos]
        for rank in range(1, n_slots + 1):
            i, t = model_by_rank.get((pos, rank), (0, 0))
            print(f"  {pos:>6}  {rank:>3}     {i:>4} / {t:<4}         {pct(i, t):>6}")

    print()
    print("=" * 70)
    print("DIAGNOSTIC 3 — Injury-report status at lock time for the model's INACTIVE picks")
    print("=" * 70)
    inactive_model_picks = [p for p in all_model_picks if not p["is_active"]]
    status_counts: Counter = Counter()
    for p in inactive_model_picks:
        status = injury_map.get((p["player_id"], p["season"], p["week"]), "(no report row)")
        status_counts[status] += 1
    print(f"  {'status':<22}  {'count':>6}  {'% of model inactive picks':>28}")
    total = sum(status_counts.values()) or 1
    for status, n in status_counts.most_common():
        print(f"  {status:<22}  {n:>6}  {n / total * 100:>26.1f}%")
    print(f"  {'TOTAL':<22}  {total:>6}")

    print()
    print("=" * 70)
    print("DIAGNOSTIC 4 — Counterfactual: drop picks with Out/Doubtful/IR at lock time")
    print("=" * 70)
    # If the production pipeline perfectly filtered out any pick with
    # a "Out" / "Doubtful" / "IR" / "IR-R" status, how many of the 55
    # inactive picks would have been avoided?
    avoidable_statuses = {"Out", "Doubtful", "IR", "IR-R"}
    avoidable_picks = [
        p for p in inactive_model_picks
        if injury_map.get((p["player_id"], p["season"], p["week"]), "") in avoidable_statuses
    ]
    questionable_picks = [
        p for p in inactive_model_picks
        if injury_map.get((p["player_id"], p["season"], p["week"]), "") == "Questionable"
    ]
    unflagged_picks = [
        p for p in inactive_model_picks
        if injury_map.get((p["player_id"], p["season"], p["week"]), "(no report row)") in ("(no report row)", "None", "Probable")
    ]
    print(f"  Avoidable via strict Out/Doubtful/IR filter: {len(avoidable_picks)} / {len(inactive_model_picks)} ({pct(len(avoidable_picks), len(inactive_model_picks))})")
    print(f"  Additionally Questionable (probabilistic): {len(questionable_picks)} / {len(inactive_model_picks)} ({pct(len(questionable_picks), len(inactive_model_picks))})")
    print(f"  Unflagged by any injury report:            {len(unflagged_picks)} / {len(inactive_model_picks)} ({pct(len(unflagged_picks), len(inactive_model_picks))})")
    # Net inactive-pick rate if strict filter is perfect:
    n_model_picks = sum(mt for _, mt in model_by_pos.values())
    adjusted_inactive = len(inactive_model_picks) - len(avoidable_picks)
    opp_total_inactive = sum(oi for oi, _ in opp_by_pos.values())
    opp_total_slots = sum(ot for _, ot in opp_by_pos.values())
    print()
    print(f"  Model inactive rate today:                 {pct(len(inactive_model_picks), n_model_picks)}")
    print(f"  Model inactive rate after strict filter:   {pct(adjusted_inactive, n_model_picks)}")
    print(f"  Opponent inactive rate (baseline):         {pct(opp_total_inactive, opp_total_slots)}")
    gap_now = (len(inactive_model_picks) / n_model_picks - opp_total_inactive / opp_total_slots) * 100
    gap_after = (adjusted_inactive / n_model_picks - opp_total_inactive / opp_total_slots) * 100
    print(f"  Gap today:                                 {gap_now:+.1f}pp")
    print(f"  Gap after strict filter:                   {gap_after:+.1f}pp")

    print()
    print("=" * 70)
    print(f"DIAGNOSTIC 5 — Top {args.top_offenders} repeat-offender players (model picked them inactive)")
    print("=" * 70)
    offender_counter: Counter = Counter()
    offender_details: Dict[str, Dict] = {}
    for p in inactive_model_picks:
        key = p["player_id"]
        offender_counter[key] += 1
        if key not in offender_details:
            offender_details[key] = {"name": p["name"], "team": p["team"], "position": p["slot_position"]}
    print(f"  {'player':<24}  {'pos':>4}  {'team':>4}  {'inactive picks':>14}  {'mostly':<20}")
    for pid, cnt in offender_counter.most_common(args.top_offenders):
        info = offender_details[pid]
        # what was the most common injury status across those weeks?
        picks = [p for p in inactive_model_picks if p["player_id"] == pid]
        status_hist = Counter(
            injury_map.get((pid, p["season"], p["week"]), "(no report row)")
            for p in picks
        )
        mostly = status_hist.most_common(1)[0]
        print(f"  {info['name']:<24}  {info['position']:>4}  {info['team']:>4}  {cnt:>14}  {mostly[0]}({mostly[1]})")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for pos in ["QB", "RB", "WR", "TE"]:
        mi, mt = model_by_pos.get(pos, (0, 0))
        oi, ot = opp_by_pos.get(pos, (0, 0))
        mrate = (mi / mt * 100) if mt else 0
        orate = (oi / ot * 100) if ot else 0
        print(f"  {pos}: model {mrate:.1f}% vs opp {orate:.1f}% (gap {mrate - orate:+.1f}pp)")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
