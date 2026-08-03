#!/usr/bin/env python3
"""
One-time backfill of NFL player injury reports into the local
`player_injuries` table.  Implements Phase 3 of the predictive-ceiling
workstream (`docs/PREDICTIVE_CEILING_PLAN.md`).

Source: ``nfl_data_py.import_injuries([season])`` — pulls the canonical
pre-game official injury report per (player, season, week).  These are
the Friday-before-Sunday reports ("point-in-time"), NOT post-game IR
designations.

Usage:
    python scripts/backfill_injuries.py                # default 2018-2025
    python scripts/backfill_injuries.py -s 2018 2025
    python scripts/backfill_injuries.py --dry-run

Idempotent via ``UNIQUE(player_id, season, week)`` on the table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_seasons(seasons: List[int]) -> "pd.DataFrame":  # noqa: F821
    """Fetch injuries for the given seasons as a single DataFrame."""
    import nfl_data_py as nfl
    return nfl.import_injuries(list(seasons))


def _coverage_report(df) -> str:
    import pandas as pd
    if df.empty:
        return "  (no rows)"
    lines = [
        f"  {'season':>6}  {'rows':>6}  {'Out':>5}  {'Doubtful':>9}  {'Questionable':>12}  {'None/Probable':>14}"
    ]
    lines.append("  " + "-" * 60)
    for s in sorted(df["season"].unique()):
        sub = df[df["season"] == s]
        counts = sub["report_status"].fillna("None").value_counts().to_dict()
        lines.append(
            f"  {int(s):>6}  {len(sub):>6}  "
            f"{counts.get('Out', 0):>5}  "
            f"{counts.get('Doubtful', 0):>9}  "
            f"{counts.get('Questionable', 0):>12}  "
            f"{counts.get('None', 0) + counts.get('Probable', 0):>14}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seasons", "-s",
        nargs=2,
        type=int,
        metavar=("LO", "HI"),
        default=[2018, 2025],
        help="Inclusive season range (default: 2018 2025).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch + report but do not write to the database.",
    )
    args = parser.parse_args()

    lo, hi = sorted(args.seasons)
    seasons = list(range(lo, hi + 1))
    print(f"Fetching injuries for seasons {seasons} via nfl_data_py …")
    df = _load_seasons(seasons)
    print(f"  {len(df):,} rows total")
    print(_coverage_report(df))

    if args.dry_run:
        print("\n--dry-run: no database writes.")
        return 0

    # Upsert loop — sqlite3 INSERT OR REPLACE against the UNIQUE constraint.
    import sqlite3
    from src.utils.database import DatabaseManager

    db = DatabaseManager()
    written = 0
    skipped_no_id = 0
    with db._get_connection() as conn:
        cur = conn.cursor()
        for _, row in df.iterrows():
            player_id = row.get("gsis_id") or ""
            if not player_id:
                skipped_no_id += 1
                continue
            try:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO player_injuries
                    (player_id, season, week, team, full_name, position,
                     report_status, practice_status, report_primary_injury,
                     date_modified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(player_id),
                        int(row["season"]) if row.get("season") is not None else None,
                        int(row["week"]) if row.get("week") is not None else None,
                        row.get("team"),
                        row.get("full_name"),
                        row.get("position"),
                        row.get("report_status"),
                        row.get("practice_status"),
                        row.get("report_primary_injury"),
                        str(row.get("date_modified") or ""),
                    ),
                )
                written += 1
            except (sqlite3.IntegrityError, ValueError, TypeError):
                continue
        conn.commit()

    print(
        f"\nWrote {written:,} rows; "
        f"skipped {skipped_no_id:,} rows with missing gsis_id."
    )

    # Verification
    with db._get_connection() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM player_injuries WHERE season BETWEEN ? AND ?",
            (lo, hi),
        ).fetchone()[0]
        by_season = conn.execute(
            "SELECT season, COUNT(*) FROM player_injuries "
            "WHERE season BETWEEN ? AND ? GROUP BY season ORDER BY season",
            (lo, hi),
        ).fetchall()
    print(f"Verification: {n:,} rows persisted for seasons {lo}-{hi}.")
    for s, c in by_season:
        print(f"  {s}: {c:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
