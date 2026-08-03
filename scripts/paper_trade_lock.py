#!/usr/bin/env python3
"""
Forward paper-trade harness — lock and score.

Implements the two per-week operations defined in
``docs/PAPER_TRADE_PROTOCOL_20260422.md`` (re-council Step 5):

- ``--lock``: on Wednesday of game week N, generate the model's
  lineup for the week, construct the prospective opponent from the
  same cumulative-active pool used in
  ``scripts/prospective_opponent_replay.py``, and insert a row into
  ``paper_trade_entries`` with the full snapshot + model config +
  git SHA.  Refuses to run if any frozen-path file has been modified
  since the first lock of this paper-trade run (unless
  ``--force-unfreeze`` is passed, which is the explicit "reset the
  clock" action from the protocol).

- ``--score``: on Monday after games, read the row, fetch each
  player's final fantasy points from ``player_weekly_stats``, fill
  in ``model_actual`` / ``opponent_actual`` / ``won`` /
  ``score_timestamp``.

This file is the scaffolding.  Production wiring work that is
**explicitly out of scope** for this skeleton:

- Automatic weekly trigger (cron / GitHub Action / systemd timer).
- DFS salary integration — the current harness ignores salary caps
  and uses top-N-per-position by predicted FP, matching the backtest's
  ``DEFAULT_ROSTER_SLOTS``.  Real DFS deployment would merge a
  salary feed and call ``LineupOptimizer.optimize_lineup``.
- Per-slate opponent calibration from real contest data.
- Alerting on stop-loss tripwires.

All of the above are defensible additions for the late-August 2026
dry run; none block landing the skeleton today.

Usage:
    # Wednesday lock with walk-forward predictions for week N:
    python scripts/paper_trade_lock.py --lock \\
        --season 2026 --week 1 \\
        --predictions-csv data/backtest_results/ts_backtest_2026_*_predictions.csv

    # Monday score:
    python scripts/paper_trade_lock.py --score --season 2026 --week 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "nfl_data.db"

DEFAULT_ROSTER_SLOTS: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}

# From config.settings.DECISION_QUALITY — hard-coded to avoid config import
# reaching into pandas/sklearn during a lightweight run.
DEFAULT_PAYOUT_MULTIPLIER = 1.8
DEFAULT_ENTRY_USD = 10.0

# Frozen paths (from docs/PAPER_TRADE_PROTOCOL_20260422.md).  Any edit
# to these files without --force-unfreeze resets the clock.
FROZEN_PATHS: Tuple[str, ...] = (
    "config/settings.py",
    "src/features/feature_engineering.py",
    "src/evaluation/ts_backtester.py",
    "src/evaluation/backtester.py",
)


# --------------------------------------------------------------------
# Git helpers
# --------------------------------------------------------------------

def current_git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def frozen_paths_modified_since(base_sha: str) -> List[str]:
    """Return the subset of FROZEN_PATHS that differ between ``base_sha``
    and current HEAD (or working tree).  Empty list = nothing changed."""
    try:
        r = subprocess.run(
            ["git", "diff", "--name-only", base_sha, "--"] + list(FROZEN_PATHS),
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        changed_vs_sha = [x for x in r.stdout.splitlines() if x]
    except Exception:
        return []
    try:
        r = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--"] + list(FROZEN_PATHS),
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        changed_working = [x for x in r.stdout.splitlines() if x]
    except Exception:
        changed_working = []
    return sorted(set(changed_vs_sha) | set(changed_working))


def first_lock_sha(conn: sqlite3.Connection) -> Optional[str]:
    """SHA recorded on the earliest row of the current paper-trade run.

    "Current run" = all rows with the same lock_git_sha chain back to
    the first lock.  For this skeleton, "first lock" = the row with
    the smallest (season, week).  A real implementation would also
    support "start a new run" semantics.
    """
    row = conn.execute(
        "SELECT lock_git_sha FROM paper_trade_entries "
        "ORDER BY season ASC, week ASC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


# --------------------------------------------------------------------
# Lineup construction
# --------------------------------------------------------------------

def load_predictions(csv_path: Path, season: int, week: int) -> List[Dict]:
    """Read this-week predictions from a walk-forward-style CSV.

    Expected columns: season, week, player_id, name, position, team,
    predicted.  Filters to the given (season, week).
    """
    out: List[Dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r.get("season", -1)) != season:
                continue
            if int(r.get("week", -1)) != week:
                continue
            pred = r.get("predicted") or r.get("predicted_points")
            if not pred:
                continue
            out.append({
                "player_id": r["player_id"],
                "name": r.get("name", ""),
                "position": r["position"],
                "team": r.get("team", ""),
                "predicted": float(pred),
            })
    return out


def build_model_lineup(
    predictions: List[Dict],
    slots: Dict[str, int] = DEFAULT_ROSTER_SLOTS,
    active_ids: Optional[set] = None,
) -> List[Dict]:
    """Top-N-by-predicted per position, matching the backtest's
    ``DEFAULT_ROSTER_SLOTS`` construction.

    If ``active_ids`` is provided, the pool is filtered to players
    whose (season, week, player_id) carries ``status='ACT'`` in the
    ``weekly_rosters`` cache.  This is the pre-lock active-roster
    filter from docs/INACTIVE_PICK_GAP_DIAGNOSIS.md / docs/
    ACTIVE_ROSTER_FILTER.md — the harness's equivalent of
    "drop players marked OUT / IR / CUT / SUS / INA before the
    optimizer runs."

    Production-deployment note: the pool construction is still
    non-salary-capped; a real DFS lock would additionally route
    through ``LineupOptimizer.optimize_lineup(strategy='cash')`` with
    a salary column from a DFS feed.  Out of scope for this skeleton.
    """
    pool_rows = predictions
    if active_ids is not None:
        pool_rows = [p for p in pool_rows if p["player_id"] in active_ids]
    picks: List[Dict] = []
    for pos, n in slots.items():
        pool = [p for p in pool_rows if p["position"] == pos]
        pool.sort(key=lambda p: p["predicted"], reverse=True)
        for p in pool[:n]:
            picks.append({
                "slot": pos,
                "player_id": p["player_id"],
                "name": p["name"],
                "team": p["team"],
                "position": pos,
                "predicted": p["predicted"],
            })
    return picks


def load_active_roster_ids(
    conn: sqlite3.Connection, season: int, week: int
) -> set:
    """Returns set of player_ids with status='ACT' in weekly_rosters.
    Empty set means the cache has no data — callers should refuse to
    lock with the filter on (rather than silently drop everyone)."""
    rows = conn.execute(
        "SELECT DISTINCT player_id FROM weekly_rosters "
        "WHERE season = ? AND week = ? AND status = 'ACT'",
        (int(season), int(week)),
    ).fetchall()
    return {r[0] for r in rows}


def build_prospective_opponent(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    slots: Dict[str, int] = DEFAULT_ROSTER_SLOTS,
    active_ids: Optional[set] = None,
) -> List[Dict]:
    """Construct the opponent using the prospective pool (players active
    at least once through week N-1; rank by prior-week actual).

    Applies the same active-roster filter as ``build_model_lineup`` when
    ``active_ids`` is provided — both sides face the same gate so the
    measurement stays symmetric."""
    cur = conn.cursor()
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
        return []

    player_ids = {r[0] for r in rows}
    placeholders = ",".join(["?"] * len(player_ids))
    prior = {
        pid: float(fp or 0.0)
        for pid, fp in cur.execute(
            f"SELECT player_id, fantasy_points FROM player_weekly_stats "
            f"WHERE season = ? AND week = ? AND player_id IN ({placeholders})",
            (season, week - 1, *player_ids),
        ).fetchall()
    }

    # Pool = players with a prior-week actual, matched to their position.
    pool_by_pos: Dict[str, List[Dict]] = {}
    for pid, pos, name, team in rows:
        if pid not in prior:
            continue
        if active_ids is not None and pid not in active_ids:
            continue
        pool_by_pos.setdefault(pos, []).append({
            "slot": pos,
            "player_id": pid,
            "name": name,
            "position": pos,
            "team": team or "",
            "prior_week_actual": prior[pid],
        })

    picks: List[Dict] = []
    for pos, n in slots.items():
        pool = sorted(pool_by_pos.get(pos, []), key=lambda x: x["prior_week_actual"], reverse=True)
        picks.extend(pool[:n])
    return picks


# --------------------------------------------------------------------
# Lock / score
# --------------------------------------------------------------------

def lock_week(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    predictions_csv: Path,
    payout_multiplier: float = DEFAULT_PAYOUT_MULTIPLIER,
    entry_usd: float = DEFAULT_ENTRY_USD,
    force_unfreeze: bool = False,
    use_active_filter: bool = True,
    notes: str = "",
) -> int:
    base_sha = first_lock_sha(conn)
    if base_sha is not None:
        changed = frozen_paths_modified_since(base_sha)
        if changed and not force_unfreeze:
            raise SystemExit(
                "refusing to lock: frozen-path files have been modified since "
                f"the first lock of this paper-trade run (SHA {base_sha[:10]}):\n"
                + "\n".join(f"  {p}" for p in changed)
                + "\n\nRun with --force-unfreeze if this is an intentional clock reset, "
                "then the 8-12 week window restarts per "
                "docs/PAPER_TRADE_PROTOCOL_20260422.md."
            )

    predictions = load_predictions(predictions_csv, season, week)
    if not predictions:
        raise SystemExit(
            f"refusing to lock: no predictions for (season={season}, week={week}) "
            f"in {predictions_csv}"
        )

    active_ids: Optional[set] = None
    if use_active_filter:
        active_ids = load_active_roster_ids(conn, season, week)
        if not active_ids:
            raise SystemExit(
                f"refusing to lock: active-roster filter is on but "
                f"weekly_rosters has no ACT rows for (season={season}, "
                f"week={week}). Run scripts/backfill_weekly_rosters.py, "
                "or pass --no-active-filter to bypass (NOT recommended "
                "for live locks — see docs/ACTIVE_ROSTER_FILTER.md)."
            )

    model_lineup = build_model_lineup(predictions, active_ids=active_ids)
    opponent_lineup = build_prospective_opponent(conn, season, week, active_ids=active_ids)
    if len(model_lineup) != sum(DEFAULT_ROSTER_SLOTS.values()):
        raise SystemExit(
            f"refusing to lock: model lineup has {len(model_lineup)} picks, "
            f"expected {sum(DEFAULT_ROSTER_SLOTS.values())}"
        )
    if len(opponent_lineup) != sum(DEFAULT_ROSTER_SLOTS.values()):
        raise SystemExit(
            f"refusing to lock: opponent lineup has {len(opponent_lineup)} picks, "
            f"expected {sum(DEFAULT_ROSTER_SLOTS.values())}"
        )

    model_config = {
        "ridge_alpha_default": 10_000,  # per config.settings.RIDGE_DEFAULT_ALPHA
        "feature_mode": "causal",
        "payout_multiplier": payout_multiplier,
        "entry_usd": entry_usd,
        "active_roster_filter": bool(use_active_filter),
    }

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO paper_trade_entries
            (season, week, lock_timestamp, lock_git_sha, model_config_json,
             model_lineup_json, opponent_lineup_json, opponent_method,
             notional_entry_usd, notional_payout_multiplier, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            season, week,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            current_git_sha(),
            json.dumps(model_config, sort_keys=True),
            json.dumps(model_lineup, sort_keys=True),
            json.dumps(opponent_lineup, sort_keys=True),
            "prospective",
            entry_usd,
            payout_multiplier,
            notes or None,
        ),
    )
    conn.commit()
    return cur.lastrowid


def score_week(conn: sqlite3.Connection, season: int, week: int) -> Dict:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT id, model_lineup_json, opponent_lineup_json FROM paper_trade_entries "
        "WHERE season = ? AND week = ?",
        (season, week),
    ).fetchone()
    if not row:
        raise SystemExit(f"no paper_trade_entries row for (season={season}, week={week})")
    row_id, model_lineup_json, opponent_lineup_json = row
    model_lineup = json.loads(model_lineup_json)
    opp_lineup = json.loads(opponent_lineup_json)

    all_ids = {p["player_id"] for p in model_lineup} | {p["player_id"] for p in opp_lineup}
    placeholders = ",".join(["?"] * len(all_ids))
    actuals = {
        pid: float(fp or 0.0)
        for pid, fp in cur.execute(
            f"SELECT player_id, fantasy_points FROM player_weekly_stats "
            f"WHERE season = ? AND week = ? AND player_id IN ({placeholders})",
            (season, week, *all_ids),
        ).fetchall()
    }

    model_actual = sum(actuals.get(p["player_id"], 0.0) for p in model_lineup)
    opp_actual = sum(actuals.get(p["player_id"], 0.0) for p in opp_lineup)
    won = 1 if model_actual > opp_actual else 0

    cur.execute(
        "UPDATE paper_trade_entries SET model_actual=?, opponent_actual=?, won=?, "
        "score_timestamp=? WHERE id=?",
        (
            model_actual, opp_actual, won,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            row_id,
        ),
    )
    conn.commit()
    return {
        "season": season, "week": week,
        "model_actual": model_actual, "opponent_actual": opp_actual,
        "won": bool(won),
    }


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lock", action="store_true", help="Lock lineup for (season, week).")
    mode.add_argument("--score", action="store_true", help="Score a previously-locked (season, week).")

    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--predictions-csv", type=Path, help="Required for --lock.")
    ap.add_argument("--payout-multiplier", type=float, default=DEFAULT_PAYOUT_MULTIPLIER)
    ap.add_argument("--entry-usd", type=float, default=DEFAULT_ENTRY_USD)
    ap.add_argument(
        "--force-unfreeze",
        action="store_true",
        help="Permit a lock even if frozen-path files have changed. "
        "Resets the 8-12 week paper-trade clock per the protocol.",
    )
    ap.add_argument(
        "--no-active-filter",
        action="store_true",
        help=(
            "Skip the pre-lock weekly_rosters status='ACT' filter. "
            "NOT recommended for live locks — the filter is the fix "
            "for the 7.4 pp model-vs-opponent inactive-pick gap "
            "(docs/ACTIVE_ROSTER_FILTER.md). Provided for A/B testing "
            "and retrospective replays where the filter is applied "
            "externally."
        ),
    )
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    conn = sqlite3.connect(str(args.db))
    try:
        if args.lock:
            if not args.predictions_csv:
                ap.error("--lock requires --predictions-csv")
            row_id = lock_week(
                conn, args.season, args.week,
                args.predictions_csv,
                payout_multiplier=args.payout_multiplier,
                entry_usd=args.entry_usd,
                force_unfreeze=args.force_unfreeze,
                use_active_filter=not args.no_active_filter,
                notes=args.notes,
            )
            print(f"Locked (season={args.season}, week={args.week}); row_id={row_id}")
            return 0
        elif args.score:
            result = score_week(conn, args.season, args.week)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
