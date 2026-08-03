#!/usr/bin/env python3
"""
Prospective-opponent replay — Step 3 of the 2026-04-22 re-council
(``council-transcript-20260422-032550.md``).

The Contrarian peer reviewer flagged a concern with the 30-13 (69.8 %)
cross-season hindsight win rate: the "Hindsight" opponent tier in
``src/evaluation/backtester.py::backtest_lineup_decisions`` picks top-N
per position from the same ``pos_pool`` the model does — and ``pos_pool``
is the set of players who **actually played** that week.  In real H2H,
an opponent locks their lineup BEFORE knowing who ends up inactive;
they don't get to retrospectively prune their picks to players who
played. The current hindsight opponent gets that free prune.

This script rebuilds the opponent with a **prospective pool**:

- All players who appeared in ``player_weekly_stats`` for the season in
  any week strictly before the current week (i.e., every player who
  had been active at least once by lock time of week N).
- Opponent sorts by the player's **prior-week** actual fantasy points
  and picks top-N per position (``DEFAULT_ROSTER_SLOTS`` = QB:1, RB:2,
  WR:2, TE:1 — same as the live backtest).
- Each pick's "this week" fantasy points = the value from
  ``player_weekly_stats`` if a row exists for (player, season, week),
  otherwise **0** (they were on the roster but didn't play — an
  inactive slot the opponent can't swap out retrospectively).

The **model's lineup is left unchanged** — it still picks from the
active-player pool. This isolates the Contrarian's concern: *how much
of the 69.8 % win rate came from the opponent's retrospective ability
to prune inactive picks?* A full prospective replay would move both
sides to the broader pool and would require re-running the
walk-forward backtest to score the model on inactive picks; that's
out of scope for this session and documented in the findings doc.

**Chairman's Step 3 kill criterion:** if the prospective replay win
rate drops more than 3 pp below the 69.8 % hindsight number, or the
Wilson lower bound falls below the 52.38 % -110 break-even, halt
forward deployment and re-council on opponent modeling.

Usage:
    python scripts/prospective_opponent_replay.py
    python scripts/prospective_opponent_replay.py --runs path1.json path2.json
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "nfl_data.db"

# Production-config runs (α=10 000, post-Vegas, post-injury) that the
# 69.8 % cross-season hindsight number was computed on.
DEFAULT_RUNS: Tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "backtest_results" / "ts_backtest_2024_20260422_025527.json",
    PROJECT_ROOT / "data" / "backtest_results" / "ts_backtest_2025_20260422_024024.json",
)

DEFAULT_ROSTER_SLOTS: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

# -110 American odds break-even (chairman's Step 3 reference).
BREAKEVEN_WIN_RATE = 110.0 / 210.0


def wilson_interval(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence, 1.960)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def load_run(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def build_prospective_pool(
    conn: sqlite3.Connection, season: int, week: int
) -> List[Tuple[str, str, float, float]]:
    """Return [(player_id, position, prior_week_actual, this_week_actual)]
    for every player who appeared in PWS for ``season`` in any week < ``week``.

    ``prior_week_actual`` = PWS(season, week-1).fantasy_points (or the
    most recent prior week if the player sat week-1 — NOT what the
    backtester's Hindsight tier does, so we match its shift(1) semantics
    and use strictly week-1; missing => excluded from the pool for
    that position-round, same as the live ``has_prior`` filter).

    ``this_week_actual`` = PWS(season, week).fantasy_points if the row
    exists, else 0 (player was on the roster but inactive).
    """
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT DISTINCT pws.player_id, p.position
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season = ? AND pws.week < ?
        """,
        (season, week),
    ).fetchall()

    player_ids = {row[0] for row in rows}
    if not player_ids:
        return []

    placeholders = ",".join(["?"] * len(player_ids))
    prior_rows = cur.execute(
        f"SELECT player_id, fantasy_points FROM player_weekly_stats "
        f"WHERE season = ? AND week = ? AND player_id IN ({placeholders})",
        (season, week - 1, *player_ids),
    ).fetchall()
    prior_fp = {pid: float(fp or 0.0) for pid, fp in prior_rows}

    this_rows = cur.execute(
        f"SELECT player_id, fantasy_points FROM player_weekly_stats "
        f"WHERE season = ? AND week = ? AND player_id IN ({placeholders})",
        (season, week, *player_ids),
    ).fetchall()
    this_fp = {pid: float(fp or 0.0) for pid, fp in this_rows}

    out: List[Tuple[str, str, float, float]] = []
    for pid, pos in rows:
        if pid not in prior_fp:
            continue  # matches live backtest's has_prior filter
        out.append((pid, pos, prior_fp[pid], this_fp.get(pid, 0.0)))
    return out


def prospective_opponent_actual(
    pool: List[Tuple[str, str, float, float]],
    slots: Dict[str, int] = DEFAULT_ROSTER_SLOTS,
) -> Tuple[float, int]:
    """Sum the opponent lineup's this-week actuals under the prospective
    construction. Returns (total, n_inactive_picks).
    """
    total = 0.0
    n_inactive = 0
    for pos, n_starters in slots.items():
        pos_pool = sorted(
            (row for row in pool if row[1] == pos),
            key=lambda r: r[2],  # prior_week_actual
            reverse=True,
        )
        if len(pos_pool) < n_starters:
            continue
        for row in pos_pool[:n_starters]:
            total += row[3]
            if row[3] == 0.0:
                n_inactive += 1
    return total, n_inactive


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_RUNS),
        help="Walk-forward run JSONs to re-score. Each must carry "
        "decision_quality.weekly_results with model_actual + "
        "hindsight_actual + won_vs_hindsight.",
    )
    ap.add_argument(
        "--breakeven",
        type=float,
        default=BREAKEVEN_WIN_RATE,
        help=f"Break-even win rate for the kill gate (default {BREAKEVEN_WIN_RATE:.4f} = -110).",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))

    rows_out: List[Dict] = []
    n_hindsight_wins = 0
    n_prospective_wins = 0
    n_weeks = 0
    n_inactive_total = 0

    for run_path in args.runs:
        d = load_run(run_path)
        dq = d.get("decision_quality") or {}
        weekly = dq.get("weekly_results") or []
        season = d.get("season")
        if not weekly or season is None:
            print(f"skip {run_path.name}: missing weekly_results or season", file=sys.stderr)
            continue
        for w in weekly:
            week = int(w["week"])
            model_actual = float(w["model_actual"])
            hindsight_actual = float(w["hindsight_actual"])
            won_hindsight = bool(w["won_vs_hindsight"])

            pool = build_prospective_pool(conn, int(season), week)
            prosp_actual, n_inactive = prospective_opponent_actual(pool)
            won_prospective = model_actual > prosp_actual

            rows_out.append({
                "season": int(season),
                "week": week,
                "model_actual": round(model_actual, 2),
                "hindsight_actual": round(hindsight_actual, 2),
                "prospective_actual": round(prosp_actual, 2),
                "delta_opp": round(prosp_actual - hindsight_actual, 2),
                "won_hindsight": won_hindsight,
                "won_prospective": won_prospective,
                "flipped": won_hindsight != won_prospective,
                "n_inactive_picks": n_inactive,
            })
            n_weeks += 1
            if won_hindsight:
                n_hindsight_wins += 1
            if won_prospective:
                n_prospective_wins += 1
            n_inactive_total += n_inactive

    conn.close()

    if n_weeks == 0:
        print("No weeks processed.", file=sys.stderr)
        return 2

    hindsight_rate = n_hindsight_wins / n_weeks
    prospective_rate = n_prospective_wins / n_weeks
    delta = prospective_rate - hindsight_rate

    lo_h, hi_h = wilson_interval(n_hindsight_wins, n_weeks)
    lo_p, hi_p = wilson_interval(n_prospective_wins, n_weeks)

    # Summary
    print(f"Weeks processed: {n_weeks}")
    print(f"Total inactive opp picks (prospective): {n_inactive_total} "
          f"({n_inactive_total / (n_weeks * 6) * 100:.1f}% of opp slots)")
    print()
    print(f"{'':<18}{'W-L':>8}{'Win %':>8}  {'Wilson 95% CI':<22}")
    print(f"  {'Hindsight':<16}{n_hindsight_wins}-{n_weeks - n_hindsight_wins:<5}{hindsight_rate*100:>7.2f}%"
          f"  [{lo_h*100:.2f}%, {hi_h*100:.2f}%]")
    print(f"  {'Prospective':<16}{n_prospective_wins}-{n_weeks - n_prospective_wins:<5}{prospective_rate*100:>7.2f}%"
          f"  [{lo_p*100:.2f}%, {hi_p*100:.2f}%]")
    print(f"  {'Delta':<16}{'':>13}{delta*100:+7.2f}pp")
    print()

    flipped = sum(1 for r in rows_out if r["flipped"])
    print(f"Weeks with flipped outcome: {flipped}/{n_weeks}")
    print(f"Avg opp-total delta (prospective - hindsight): "
          f"{sum(r['delta_opp'] for r in rows_out) / n_weeks:+.2f} FP")
    print()

    # Kill gates
    kill_delta = abs(delta) > 0.03
    kill_wilson = lo_p < args.breakeven
    print("Kill gates:")
    print(f"  |delta| > 3 pp:                   "
          f"{'FAIL' if kill_delta else 'PASS'}  "
          f"(|delta| = {abs(delta)*100:.2f}pp)")
    print(f"  Wilson LB < {args.breakeven*100:.2f}%:          "
          f"{'FAIL' if kill_wilson else 'PASS'}  "
          f"(LB = {lo_p*100:.2f}%)")
    print()
    if kill_delta or kill_wilson:
        print("Verdict: FAIL — hindsight-opponent leakage materially overstated the edge.")
        return 1
    print("Verdict: PASS — prospective construction preserves the cross-season edge within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
