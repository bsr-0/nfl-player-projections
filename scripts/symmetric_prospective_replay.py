#!/usr/bin/env python3
"""
Fully-symmetric prospective replay — residual from Step 3 of the
2026-04-22 re-council (`council-transcript-20260422-032550.md`).

Step 3's one-sided replay moved only the **opponent** to the
cumulative-active pool (players active at least once through week
N-1) and found that the retrospective filter was helping the
opponent dodge their inactive picks. That lift bounded the edge
between **69.8 %** (hindsight; both sides free to dodge inactives)
and **83.7 %** (opponent constrained, model still free). This
script closes the symmetry:

- **Model pool** = every player seen in the walk-forward
  predictions CSV for this season in any week < W.
- **Model ranking** = each player's most recent `predicted` value
  through week N-1 (a pragmatic proxy; see caveats below).
- **Model picks** top-N per position under the same
  ``DEFAULT_ROSTER_SLOTS = {QB:1, RB:2, WR:2, TE:1}``.
- **Model score** per pick = the player's this-week actual if
  present in the CSV (= they played), otherwise **0** (roster slot
  burned on an inactive).
- **Opponent pool** = the Step 3 prospective construction, unchanged.

Both sides now face the same penalty for picking a player who ends
up inactive.

**The approximation (honest caveat):** a proper symmetric replay
would require re-running the walk-forward to generate per-week
predictions for every cumulative-active player, not just the
players who actually played. That is a ~1 h walk-forward re-run
beyond this script's scope. The last-known-prediction proxy
under-ranks players who have sat recently (their prediction is
stale), which slightly biases the model AWAY from stale-inactive
picks — in practice making the model's measured win rate a small
upper bound vs. the true symmetric replay.

Usage:
    python scripts/symmetric_prospective_replay.py

Reads the two production-config walk-forward runs and re-scores
both sides under the symmetric prospective construction.
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

# Production-config (alpha=10 000, post-Vegas, post-injury) runs that
# the 30-13 (69.8 %) cross-season hindsight and 36-7 (83.7 %) one-sided
# prospective numbers were computed on.
DEFAULT_RUNS: Tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "backtest_results" / "ts_backtest_2024_20260422_025527.json",
    PROJECT_ROOT / "data" / "backtest_results" / "ts_backtest_2025_20260422_024024.json",
)

DEFAULT_ROSTER_SLOTS: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

BREAKEVEN_WIN_RATE = 110.0 / 210.0  # -110 H2H break-even


def wilson_interval(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence, 1.960)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def load_predictions_csv(run_path: Path):
    """Return (season, list_of_rows) where each row is a dict with
    season/week/player_id/position/predicted/actual.  Uses stdlib
    csv to avoid a pandas dep — there are ~5600 rows per season."""
    import csv

    csv_path = run_path.with_name(run_path.stem + "_predictions.csv")
    rows: List[Dict] = []
    season = None
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            s = int(r["season"])
            season = s
            rows.append({
                "season": s,
                "week": int(r["week"]),
                "player_id": r["player_id"],
                "position": r["position"],
                "predicted": float(r["predicted"]) if r.get("predicted") else None,
                "actual": float(r["actual"]) if r.get("actual") else None,
            })
    return season, rows


def symmetric_model_actual(
    all_rows: List[Dict],
    week: int,
    slots: Dict[str, int] = DEFAULT_ROSTER_SLOTS,
    freshness: int = 0,
) -> Tuple[float, int]:
    """Compute the model's this-week actual under the symmetric
    prospective construction.  Returns (total, n_inactive_picks).

    ``freshness`` (weeks): if > 0, the model's pool is restricted
    to players who played within the last ``freshness`` weeks
    (week N-1 down to N-freshness).  ``freshness=1`` matches the
    opponent's tight prior-week filter exactly and removes the
    stale-prediction artifact.  Default ``0`` = full cumulative
    history.
    """
    # Last-known prediction per (player_id, position) using any row
    # strictly before this week, optionally filtered to the freshness
    # window.
    cutoff = week - freshness if freshness > 0 else 0
    last_pred: Dict[str, Tuple[str, float]] = {}
    for r in all_rows:
        if r["week"] >= week:
            continue
        if freshness > 0 and r["week"] < cutoff:
            continue
        if r["predicted"] is None:
            continue
        last_pred[r["player_id"]] = (r["position"], r["predicted"])

    # This-week actuals by player_id (only active players have a row).
    this_actual: Dict[str, float] = {}
    for r in all_rows:
        if r["week"] != week:
            continue
        if r["actual"] is None:
            continue
        this_actual[r["player_id"]] = r["actual"]

    total = 0.0
    n_inactive = 0
    for pos, n_starters in slots.items():
        pool = sorted(
            (
                (pid, pred) for pid, (p, pred) in last_pred.items() if p == pos
            ),
            key=lambda t: t[1],
            reverse=True,
        )
        if len(pool) < n_starters:
            continue
        for pid, _ in pool[:n_starters]:
            if pid in this_actual:
                total += this_actual[pid]
            else:
                n_inactive += 1
    return total, n_inactive


def build_prospective_opponent_pool(
    conn: sqlite3.Connection, season: int, week: int
) -> List[Tuple[str, str, float, float]]:
    """Same construction as scripts/prospective_opponent_replay.py."""
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
    player_ids = {r[0] for r in rows}
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
            continue
        out.append((pid, pos, prior_fp[pid], this_fp.get(pid, 0.0)))
    return out


def opponent_prospective_actual(
    pool: List[Tuple[str, str, float, float]],
    slots: Dict[str, int] = DEFAULT_ROSTER_SLOTS,
) -> Tuple[float, int]:
    total = 0.0
    n_inactive = 0
    for pos, n_starters in slots.items():
        pos_pool = sorted(
            (row for row in pool if row[1] == pos),
            key=lambda r: r[2],
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
        help="Walk-forward run JSONs. The companion _predictions.csv must sit beside each.",
    )
    ap.add_argument("--breakeven", type=float, default=BREAKEVEN_WIN_RATE)
    ap.add_argument(
        "--model-freshness",
        type=int,
        default=0,
        help="Weeks of lookback for the model's prediction pool. 1 = tight "
        "prior-week filter (matches opponent); 0 = full cumulative history.",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))

    totals = {"hindsight": 0, "asymmetric": 0, "symmetric": 0}
    weeks = 0
    weekly_rows: List[Dict] = []
    model_inactive_total = 0
    opp_inactive_total = 0

    for run_path in args.runs:
        with run_path.open() as f:
            run_d = json.load(f)
        season, pred_rows = load_predictions_csv(run_path)
        dq = run_d.get("decision_quality") or {}
        weekly = dq.get("weekly_results") or []
        if not weekly:
            print(f"skip {run_path.name}: no decision_quality.weekly_results", file=sys.stderr)
            continue

        for w in weekly:
            week = int(w["week"])
            model_actual_hindsight = float(w["model_actual"])
            hindsight_opp = float(w["hindsight_actual"])
            won_hindsight = bool(w["won_vs_hindsight"])

            # Opponent prospective (from DB)
            opp_pool = build_prospective_opponent_pool(conn, season, week)
            opp_sym, opp_na = opponent_prospective_actual(opp_pool)
            opp_inactive_total += opp_na

            # Model symmetric (from predictions CSV)
            model_sym, model_na = symmetric_model_actual(pred_rows, week, freshness=args.model_freshness)
            model_inactive_total += model_na

            won_asym = model_actual_hindsight > opp_sym          # model unchanged, opp prospective
            won_sym = model_sym > opp_sym                         # both prospective

            weekly_rows.append({
                "season": season,
                "week": week,
                "model_h": model_actual_hindsight,
                "model_sym": round(model_sym, 2),
                "delta_model": round(model_sym - model_actual_hindsight, 2),
                "opp_h": hindsight_opp,
                "opp_sym": round(opp_sym, 2),
                "won_h": won_hindsight,
                "won_asym": won_asym,
                "won_sym": won_sym,
                "model_inactive_picks": model_na,
                "opp_inactive_picks": opp_na,
            })
            weeks += 1
            if won_hindsight:
                totals["hindsight"] += 1
            if won_asym:
                totals["asymmetric"] += 1
            if won_sym:
                totals["symmetric"] += 1

    conn.close()

    if weeks == 0:
        print("No weeks processed.", file=sys.stderr)
        return 2

    print(f"Weeks processed: {weeks}\n")
    print(f"Inactive picks over all weeks:")
    print(f"  Model (symmetric): {model_inactive_total} / {weeks * 6} ({model_inactive_total / (weeks * 6) * 100:.1f} %)")
    print(f"  Opp   (symmetric): {opp_inactive_total} / {weeks * 6} ({opp_inactive_total / (weeks * 6) * 100:.1f} %)")
    print()

    print(f"{'':<18}{'W-L':>10}{'Win %':>10}  {'Wilson 95 %':<22}")
    for label, key in (
        ("Hindsight (live)", "hindsight"),
        ("Asymmetric", "asymmetric"),
        ("Symmetric", "symmetric"),
    ):
        w = totals[key]
        lo, hi = wilson_interval(w, weeks)
        print(
            f"  {label:<16}{w}-{weeks - w:<7}{w/weeks*100:>8.2f} %"
            f"  [{lo*100:.2f} %, {hi*100:.2f} %]"
        )
    print()

    avg_model_drop = sum(r["delta_model"] for r in weekly_rows) / weeks
    print(f"Average model total drop vs hindsight (symmetric): {avg_model_drop:+.2f} FP/week")

    flips_sym_vs_h = sum(1 for r in weekly_rows if r["won_sym"] != r["won_h"])
    print(f"Flips symmetric vs hindsight: {flips_sym_vs_h}/{weeks}")
    print()

    # Apply re-council Step 3 kill criterion to the symmetric number.
    # The chairman's rule is "Wilson LB below 52.38 % break-even" for halt.
    lo_sym, hi_sym = wilson_interval(totals["symmetric"], weeks)
    if lo_sym < args.breakeven:
        print(f"Kill check (Wilson LB < {args.breakeven*100:.2f} %): "
              f"FAIL (LB = {lo_sym*100:.2f} %) — sample non-deployable under symmetric construction.")
        return 1
    print(f"Kill check (Wilson LB < {args.breakeven*100:.2f} %): "
          f"PASS (LB = {lo_sym*100:.2f} %) — symmetric construction preserves a deployable edge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
