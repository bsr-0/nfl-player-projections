#!/usr/bin/env python3
"""
Wide-mode symmetric prospective replay — closes the residual from
``docs/SYMMETRIC_PROSPECTIVE_REPLAY.md``.

The earlier symmetric replay used a last-known-prediction proxy for
inactive players because the walk-forward only emitted predictions
for players who actually played each week.  That proxy was
structurally biased against the model (26.4 % inactive picks vs the
opponent's 14.3 %, since stale predictions kept ranking sat-out
players high).

The ``--emit-inactive-predictions`` flag added to the walk-forward
produces a fresh per-week prediction for every cumulative-active-
pool player, including players who ended up inactive (``is_active=0``
in the CSV, ``actual=NaN``).  With that CSV, the model can draft
from the same pool the opponent does, and each pick's
this-week score is 0 for inactive picks — truly symmetric.

**Run this AFTER generating a wide CSV, e.g.:**

```
python scripts/run_ts_backtest.py --season 2024 --alpha 10000 \
    --emit-inactive-predictions
python scripts/run_ts_backtest.py --season 2025 --alpha 10000 \
    --emit-inactive-predictions
python scripts/wide_symmetric_replay.py \
    --runs data/backtest_results/ts_backtest_2024_<ts>.json \
           data/backtest_results/ts_backtest_2025_<ts>.json
```
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "nfl_data.db"

DEFAULT_ROSTER_SLOTS: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
BREAKEVEN_WIN_RATE = 110.0 / 210.0  # -110 break-even


def wilson_interval(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence, 1.960)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def bootstrap(flags: List[int], trials: int, seed: int) -> List[float]:
    import random
    rng = random.Random(seed)
    n = len(flags)
    return [sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(trials)]


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def load_wide_predictions(run_json: Path) -> Tuple[int, List[Dict]]:
    """Return (season, rows) from the companion _predictions.csv.

    Rows include is_active ∈ {0, 1}.  Inactive rows have actual=NaN.
    """
    csv_path = run_json.with_name(run_json.stem + "_predictions.csv")
    rows: List[Dict] = []
    season = None
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            season = int(r["season"])
            rows.append({
                "season": season,
                "week": int(r["week"]),
                "player_id": r["player_id"],
                "position": r["position"],
                "predicted": float(r["predicted"]) if r.get("predicted") else None,
                "actual": float(r["actual"]) if r.get("actual") else None,
                "is_active": int(r.get("is_active", 1) or 1),
            })
    return season, rows


def symmetric_model_actual(
    all_rows: List[Dict],
    week: int,
    slots: Dict[str, int] = DEFAULT_ROSTER_SLOTS,
    active_ids: Optional[set] = None,
) -> Tuple[float, int]:
    """Pick model lineup from the cumulative-active pool this week,
    ranked by this-week predicted.  Inactive picks (those with no
    observed actual) score 0.

    If ``active_ids`` is provided, the model's pool is first filtered
    to players whose (player_id, season, week) carries
    ``status == 'ACT'`` in the weekly_rosters cache.  This simulates
    the production harness's pre-lock active-roster gate described in
    ``docs/INACTIVE_PICK_GAP_DIAGNOSIS.md``.
    """
    week_rows = [r for r in all_rows if r["week"] == week and r["predicted"] is not None]
    if active_ids is not None:
        week_rows = [r for r in week_rows if r["player_id"] in active_ids]
    total = 0.0
    n_inactive = 0
    for pos, n_starters in slots.items():
        pool = sorted(
            (r for r in week_rows if r["position"] == pos),
            key=lambda r: r["predicted"],
            reverse=True,
        )
        if len(pool) < n_starters:
            continue
        for r in pool[:n_starters]:
            if r["is_active"] == 1 and r["actual"] is not None:
                total += r["actual"]
            else:
                n_inactive += 1
    return total, n_inactive


def load_active_roster_ids(
    conn: sqlite3.Connection, season: int, week: int
) -> set:
    """(player_id) set for players with status='ACT' in weekly_rosters.

    Returns an empty set if the cache has no rows for (season, week) —
    callers should treat that as "no filter, use the original pool"
    (not "filter out everyone")."""
    rows = conn.execute(
        "SELECT DISTINCT player_id FROM weekly_rosters "
        "WHERE season = ? AND week = ? AND status = 'ACT'",
        (int(season), int(week)),
    ).fetchall()
    return {r[0] for r in rows}


def build_prospective_opponent_pool(
    conn: sqlite3.Connection, season: int, week: int
) -> List[Tuple[str, str, float, float]]:
    """Same construction as scripts/prospective_opponent_replay.py:
    cumulative-active pool, rank by prior-week actual, 0 for inactives."""
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT DISTINCT pws.player_id, p.position "
        "FROM player_weekly_stats pws JOIN players p ON pws.player_id = p.player_id "
        "WHERE pws.season = ? AND pws.week < ?",
        (season, week),
    ).fetchall()
    player_ids = {r[0] for r in rows}
    if not player_ids:
        return []
    placeholders = ",".join(["?"] * len(player_ids))
    prior = {
        pid: float(fp or 0.0)
        for pid, fp in cur.execute(
            f"SELECT player_id, fantasy_points FROM player_weekly_stats "
            f"WHERE season = ? AND week = ? AND player_id IN ({placeholders})",
            (season, week - 1, *player_ids),
        ).fetchall()
    }
    this_wk = {
        pid: float(fp or 0.0)
        for pid, fp in cur.execute(
            f"SELECT player_id, fantasy_points FROM player_weekly_stats "
            f"WHERE season = ? AND week = ? AND player_id IN ({placeholders})",
            (season, week, *player_ids),
        ).fetchall()
    }
    out: List[Tuple[str, str, float, float]] = []
    for pid, pos in rows:
        if pid not in prior:
            continue
        out.append((pid, pos, prior[pid], this_wk.get(pid, 0.0)))
    return out


def opponent_prospective_actual(
    pool: List[Tuple[str, str, float, float]],
    slots: Dict[str, int] = DEFAULT_ROSTER_SLOTS,
    active_ids: Optional[set] = None,
) -> Tuple[float, int]:
    """If ``active_ids`` is provided, pool is filtered to players with
    status='ACT' in weekly_rosters before ranking — keeps the filter
    symmetric (model + opp face the same gate)."""
    if active_ids is not None:
        pool = [r for r in pool if r[0] in active_ids]
    total = 0.0
    n_inactive = 0
    for pos, n_starters in slots.items():
        pos_pool = sorted(
            (r for r in pool if r[1] == pos), key=lambda r: r[2], reverse=True
        )
        if len(pos_pool) < n_starters:
            continue
        for r in pos_pool[:n_starters]:
            total += r[3]
            if r[3] == 0.0:
                n_inactive += 1
    return total, n_inactive


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--breakeven", type=float, default=BREAKEVEN_WIN_RATE)
    ap.add_argument("--bootstrap-trials", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--apply-active-filter",
        action="store_true",
        help=(
            "Apply the weekly_rosters status='ACT' filter to both sides "
            "before ranking. Simulates the production harness's pre-lock "
            "active-roster gate from docs/INACTIVE_PICK_GAP_DIAGNOSIS.md."
        ),
    )
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))

    per_week: List[Dict] = []
    model_inactive_total = 0
    opp_inactive_total = 0

    for run_path in args.runs:
        with run_path.open() as f:
            run_d = json.load(f)
        season, rows = load_wide_predictions(run_path)
        weekly = (run_d.get("decision_quality") or {}).get("weekly_results") or []
        live_hindsight_by_week = {int(w["week"]): (bool(w["won_vs_hindsight"]),
                                                   float(w["hindsight_actual"]),
                                                   float(w["model_actual"])) for w in weekly}
        # Iterate all weeks that have at least one prediction row (active or phantom).
        weeks = sorted({r["week"] for r in rows})
        for week in weeks:
            active_ids = None
            if args.apply_active_filter:
                active_ids = load_active_roster_ids(conn, season, week)
                if not active_ids:
                    # Cache miss — don't silently collapse to "drop
                    # everyone."  Skip the week instead and flag.
                    print(
                        f"  ! skip season={season} week={week}: no "
                        f"weekly_rosters rows (run backfill_weekly_rosters.py)",
                        file=sys.stderr,
                    )
                    continue
            model_sym, model_na = symmetric_model_actual(rows, week, active_ids=active_ids)
            opp_pool = build_prospective_opponent_pool(conn, season, week)
            opp_sym, opp_na = opponent_prospective_actual(opp_pool, active_ids=active_ids)
            if not opp_pool or model_sym == 0.0 and model_na == 0:
                # e.g., week 1 has no prior-week history for anyone
                continue

            won_sym = model_sym > opp_sym

            live_won, live_opp_h, live_model_h = live_hindsight_by_week.get(week, (None, None, None))
            per_week.append({
                "season": season, "week": week,
                "wide_model_sym": round(model_sym, 2),
                "wide_opp_sym": round(opp_sym, 2),
                "won_wide_sym": won_sym,
                "model_inactive": model_na,
                "opp_inactive": opp_na,
                "live_wide_hindsight_won": live_won,
                "live_wide_hindsight_model": live_model_h,
                "live_wide_hindsight_opp": live_opp_h,
            })
            model_inactive_total += model_na
            opp_inactive_total += opp_na

    conn.close()

    if not per_week:
        print("No weeks scored", file=sys.stderr)
        return 2

    n = len(per_week)
    wins = sum(1 for r in per_week if r["won_wide_sym"])
    win_rate = wins / n
    lo, hi = wilson_interval(wins, n)

    print(f"Wide-mode symmetric replay: {n} weeks scored\n")
    print(f"Opponent inactive picks: {opp_inactive_total}/{n * 6} "
          f"({opp_inactive_total / (n * 6) * 100:.1f}%)")
    print(f"Model    inactive picks: {model_inactive_total}/{n * 6} "
          f"({model_inactive_total / (n * 6) * 100:.1f}%)")
    print()
    print(f"Wide-mode active-only hindsight (from the run's own decision_quality):")
    live_wins = sum(1 for r in per_week if r["live_wide_hindsight_won"])
    print(f"  {live_wins}-{n - live_wins}  ({live_wins / n * 100:.2f}%)")
    print()
    print(f"Wide-mode SYMMETRIC (both sides in prospective pool, 0 for inactives):")
    print(f"  Record:        {wins}-{n - wins}  ({win_rate * 100:.2f}%)")
    print(f"  Wilson 95% CI: [{lo * 100:.2f}%, {hi * 100:.2f}%]")

    flags = [1 if r["won_wide_sym"] else 0 for r in per_week]
    draws = bootstrap(flags, args.bootstrap_trials, args.seed)
    p5 = percentile(draws, 0.05)
    p50 = percentile(draws, 0.50)
    p95 = percentile(draws, 0.95)
    mean = sum(draws) / len(draws)
    frac_be = sum(1 for x in draws if x > args.breakeven) / len(draws)
    print(f"  Bootstrap ({args.bootstrap_trials} resamples, seed={args.seed}):")
    print(f"    mean:                  {mean * 100:.2f}%")
    print(f"    5th percentile:        {p5 * 100:.2f}%  <-- kill gate")
    print(f"    median:                {p50 * 100:.2f}%")
    print(f"    95th percentile:       {p95 * 100:.2f}%")
    print(f"    P(resample > {args.breakeven * 100:.1f}%):  {frac_be * 100:.2f}%")
    print()
    if p5 < args.breakeven:
        print(f"Verdict: FAIL — bootstrap p5 ({p5 * 100:.2f}%) below {args.breakeven * 100:.2f}% break-even.")
        return 1
    print(f"Verdict: PASS — bootstrap p5 ({p5 * 100:.2f}%) clears {args.breakeven * 100:.2f}% break-even.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
