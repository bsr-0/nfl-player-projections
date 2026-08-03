#!/usr/bin/env python3
"""
One-time backfill of Vegas `spread_line` and `total_line` into the local
schedule table.  Implements Step C of `docs/PHASE_1_VEGAS_FINDINGS.md`:
without this cache, `_create_vegas_game_script_features` silently
collapses Vegas inputs to constants (implied_team_total=23.0, spread=0.0)
whenever `nfl_data_py.import_schedules()` cannot reach its upstream host.

Source of truth: the canonical `nflverse/nfldata` repository on GitHub,
fetched by URL rather than via `nfl_data_py.import_schedules()` because
the latter points at `http://www.habitatring.com/games.csv`, which is
not always reachable from every environment (notably not from this
sandbox — see `curl -I` output on that host).  GitHub raw is the
redirect target `nfl_data_py` itself follows and is network-policy
friendly.

Usage:
    python scripts/backfill_vegas_lines.py                 # seasons 2018-current
    python scripts/backfill_vegas_lines.py -s 2018 2025    # explicit range
    python scripts/backfill_vegas_lines.py --dry-run       # preview, no writes

Idempotent: relies on the `UNIQUE(season, week, home_team, away_team)`
constraint on the `schedule` table — re-running overwrites existing rows
with the latest upstream values and never duplicates.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NFLVERSE_GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)


def _fetch_games_csv(url: str = NFLVERSE_GAMES_URL, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-projections-backfill"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _parse_float(v: str) -> Optional[float]:
    if v is None or v == "" or v == "NA":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _filter_rows(
    csv_text: str,
    seasons: Optional[Tuple[int, int]],
) -> List[dict]:
    """Parse the games.csv and return rows in the season range with non-null Vegas data."""
    reader = csv.DictReader(io.StringIO(csv_text))
    out: List[dict] = []
    lo = seasons[0] if seasons else None
    hi = seasons[1] if seasons else None
    for row in reader:
        try:
            season = int(row["season"])
        except (KeyError, ValueError, TypeError):
            continue
        if lo is not None and season < lo:
            continue
        if hi is not None and season > hi:
            continue
        week_raw = row.get("week")
        try:
            week = int(week_raw) if week_raw not in (None, "") else None
        except (ValueError, TypeError):
            week = None
        if week is None:
            continue
        home = row.get("home_team") or ""
        away = row.get("away_team") or ""
        if not home or not away:
            continue
        spread_line = _parse_float(row.get("spread_line", ""))
        total_line = _parse_float(row.get("total_line", ""))
        if spread_line is None and total_line is None:
            # Don't write rows that carry no Vegas information at all.
            continue
        out.append({
            "season": season,
            "week": week,
            "home_team": home,
            "away_team": away,
            "game_id": row.get("game_id") or None,
            "game_time": row.get("gameday") or None,
            "venue": row.get("stadium") or None,
            "home_score": _parse_int(row.get("home_score")),
            "away_score": _parse_int(row.get("away_score")),
            "spread_line": spread_line,
            "total_line": total_line,
        })
    return out


def _parse_int(v: Optional[str]) -> Optional[int]:
    if v is None or v == "" or v == "NA":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _coverage_report(rows: List[dict]) -> str:
    if not rows:
        return "  (no rows)"
    seasons: dict = {}
    for r in rows:
        seasons.setdefault(r["season"], []).append(r)
    lines: List[str] = []
    lines.append(f"  {'season':>6}  {'rows':>5}  {'spread_line':>12}  {'total_line':>11}")
    lines.append("  " + "-" * 44)
    for s in sorted(seasons):
        rs = seasons[s]
        sl = sum(1 for r in rs if r["spread_line"] is not None)
        tl = sum(1 for r in rs if r["total_line"] is not None)
        lines.append(f"  {s:>6}  {len(rs):>5}  {sl:>12}  {tl:>11}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seasons", "-s",
        nargs=2,
        type=int,
        metavar=("LO", "HI"),
        default=[2018, 2025],
        help="Inclusive season range to backfill (default: 2018 2025).",
    )
    parser.add_argument(
        "--url",
        default=NFLVERSE_GAMES_URL,
        help=f"Source URL for games.csv (default: {NFLVERSE_GAMES_URL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch + parse + report but do not write to the database.",
    )
    args = parser.parse_args()

    lo, hi = sorted(args.seasons)
    print(f"Fetching {args.url} …")
    csv_text = _fetch_games_csv(args.url)
    print(f"  {len(csv_text):,} bytes")

    rows = _filter_rows(csv_text, (lo, hi))
    print(f"Parsed {len(rows):,} rows in seasons {lo}-{hi} with at least one Vegas value.")
    print(_coverage_report(rows))

    if args.dry_run:
        print("\n--dry-run: no database writes.")
        return 0

    # Use DatabaseManager.insert_schedule for UPSERT via the existing UNIQUE
    # constraint.  This is slower than a single executemany but avoids
    # duplicating the INSERT SQL here.
    from src.utils.database import DatabaseManager

    db = DatabaseManager()
    written = 0
    for row in rows:
        if db.insert_schedule(row):
            written += 1
    print(f"\nWrote {written:,} rows to the schedule table.")

    # Quick verification: count how many non-null spread_lines we now have.
    with db._get_connection() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM schedule WHERE spread_line IS NOT NULL AND season BETWEEN ? AND ?",
            (lo, hi),
        )
        n = cur.fetchone()[0]
    print(f"Verification: {n:,} rows in schedule have spread_line populated for seasons {lo}-{hi}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
