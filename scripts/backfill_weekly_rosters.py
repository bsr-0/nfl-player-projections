#!/usr/bin/env python3
"""
One-time backfill of weekly NFL roster snapshots into the local
`weekly_rosters` table.  Implements action #1 of
`docs/INACTIVE_PICK_GAP_DIAGNOSIS.md`: the pre-lock active-roster
filter.

Source: nflverse weekly_rosters parquet on GitHub (the same data
nfl_data_py's ``import_weekly_rosters`` wraps, fetched directly to
bypass the ``habitatring.com`` gameday-merge call that is blocked
in some sandboxes).

Rows capture point-in-time status per (player, season, week):
ACT (active, eligible to play), INA (inactive / healthy scratch),
RES (reserve / IR / PUP / NFI), CUT (released), RET (retired),
EXE (exempt / suspended), DEV (practice squad).  Only ACT is
"playable"; everything else is dropped by the active-roster filter.

Usage:
    python scripts/backfill_weekly_rosters.py               # 2018-2025
    python scripts/backfill_weekly_rosters.py -s 2024 2025  # explicit range
    python scripts/backfill_weekly_rosters.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ROSTER_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "weekly_rosters/roster_weekly_{year}.parquet"
)


def _fetch_season(year: int) -> pd.DataFrame:
    return pd.read_parquet(ROSTER_URL.format(year=year))


def _coverage_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "  (no rows)"
    lines = [f"  {'season':>6}  {'rows':>6}  {'ACT':>5}  {'INA':>5}  {'RES':>5}  {'CUT':>5}  {'other':>6}"]
    lines.append("  " + "-" * 50)
    known = {"ACT", "INA", "RES", "CUT"}
    for s in sorted(df["season"].unique()):
        sub = df[df["season"] == s]
        counts = sub["status"].fillna("NULL").value_counts().to_dict()
        other = sum(v for k, v in counts.items() if k not in known)
        lines.append(
            f"  {int(s):>6}  {len(sub):>6}  "
            f"{counts.get('ACT', 0):>5}  "
            f"{counts.get('INA', 0):>5}  "
            f"{counts.get('RES', 0):>5}  "
            f"{counts.get('CUT', 0):>5}  "
            f"{other:>6}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--seasons", "-s",
        nargs=2, type=int, metavar=("LO", "HI"),
        default=[2018, 2025],
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lo, hi = sorted(args.seasons)
    frames: List[pd.DataFrame] = []
    for year in range(lo, hi + 1):
        print(f"Fetching weekly rosters for {year} …")
        try:
            frames.append(_fetch_season(year))
        except Exception as e:
            print(f"  SKIP {year}: {type(e).__name__}: {e}", file=sys.stderr)
    if not frames:
        print("No rosters fetched.", file=sys.stderr)
        return 1
    df = pd.concat(frames, ignore_index=True)
    print(f"\n{len(df):,} roster-weeks across seasons {lo}-{hi}.")
    print(_coverage_report(df))

    if args.dry_run:
        print("\n--dry-run: no writes.")
        return 0

    from src.utils.database import DatabaseManager
    db = DatabaseManager()
    written = 0
    skipped = 0
    with db._get_connection() as conn:
        cur = conn.cursor()
        for _, row in df.iterrows():
            pid = row.get("gsis_id") or ""
            if not pid:
                skipped += 1
                continue
            try:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO weekly_rosters
                        (player_id, season, week, team, position, status,
                         status_description_abbr, full_name, game_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(pid),
                        int(row["season"]) if pd.notna(row.get("season")) else None,
                        int(row["week"]) if pd.notna(row.get("week")) else None,
                        row.get("team"),
                        row.get("position"),
                        row.get("status"),
                        row.get("status_description_abbr"),
                        row.get("full_name"),
                        row.get("game_type"),
                    ),
                )
                written += 1
            except (sqlite3.IntegrityError, ValueError, TypeError):
                skipped += 1
        conn.commit()
    print(f"\nWrote {written:,} rows; skipped {skipped:,} (missing gsis_id or malformed).")

    with db._get_connection() as conn:
        for row in conn.execute(
            "SELECT season, COUNT(*), "
            "SUM(CASE WHEN status='ACT' THEN 1 ELSE 0 END) AS act "
            "FROM weekly_rosters WHERE season BETWEEN ? AND ? "
            "GROUP BY season ORDER BY season",
            (lo, hi),
        ).fetchall():
            print(f"  season {row[0]}: {row[1]:,} rows; ACT={row[2]:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
