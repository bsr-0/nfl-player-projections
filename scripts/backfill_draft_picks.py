#!/usr/bin/env python3
"""
One-time backfill of NFL draft picks into the local ``draft_picks``
table.  Prerequisite for Phase 4C of Step 4 — rookie priors.

Source: nflverse draft_picks parquet on GitHub (same provenance as
``scripts/backfill_weekly_rosters.py``; reachable from sandbox via
``raw.githubusercontent.com`` / releases).

Rows map nflverse column → local schema:

    nflverse         →  local draft_picks
    gsis_id          →  player_id
    pfr_player_name  →  player_name
    position         →  position
    college          →  college
    season           →  draft_season
    round            →  draft_round
    pick             →  draft_pick
    team             →  draft_team

Usage:
    python scripts/backfill_draft_picks.py                  # 2006-2025
    python scripts/backfill_draft_picks.py -s 2018 2025
    python scripts/backfill_draft_picks.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DRAFT_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "draft_picks/draft_picks.parquet"
)


def _coverage_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "  (no rows)"
    lines = [f"  {'season':>6}  {'rows':>6}  {'with_gsis':>10}"]
    lines.append("  " + "-" * 30)
    for s in sorted(df["season"].unique()):
        sub = df[df["season"] == s]
        with_id = sub["gsis_id"].notna().sum()
        lines.append(f"  {int(s):>6}  {len(sub):>6}  {with_id:>10}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--seasons", "-s",
        nargs=2, type=int, metavar=("LO", "HI"),
        default=[2006, 2025],
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lo, hi = sorted(args.seasons)
    print(f"Fetching {DRAFT_URL} …")
    df = pd.read_parquet(DRAFT_URL)
    print(f"Loaded {len(df):,} total draft-pick rows (seasons "
          f"{df['season'].min()}-{df['season'].max()}).")

    df = df[df["season"].between(lo, hi)].copy()
    print(f"Filtered to {len(df):,} rows for seasons {lo}-{hi}.")
    print(_coverage_report(df))

    if args.dry_run:
        print("\n--dry-run: no writes.")
        return 0

    from src.utils.database import DatabaseManager
    db = DatabaseManager()  # __init__ runs _init_database idempotently

    written = 0
    skipped = 0
    with db._get_connection() as conn:
        cur = conn.cursor()
        # Purge rows in the target range first so we get a clean
        # rewrite (draft picks are immutable history; this is a
        # one-time backfill).
        cur.execute(
            "DELETE FROM draft_picks WHERE draft_season BETWEEN ? AND ?",
            (lo, hi),
        )
        for _, row in df.iterrows():
            try:
                cur.execute(
                    """
                    INSERT INTO draft_picks
                        (player_id, player_name, position, college,
                         draft_season, draft_round, draft_pick, draft_team)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(row["gsis_id"]) if pd.notna(row.get("gsis_id")) else None,
                        row.get("pfr_player_name"),
                        row.get("position"),
                        row.get("college"),
                        int(row["season"]) if pd.notna(row.get("season")) else None,
                        int(row["round"]) if pd.notna(row.get("round")) else None,
                        int(row["pick"]) if pd.notna(row.get("pick")) else None,
                        row.get("team"),
                    ),
                )
                written += 1
            except (sqlite3.IntegrityError, ValueError, TypeError):
                skipped += 1
        conn.commit()

    print(f"\nWrote {written:,} rows; skipped {skipped:,}.")
    with db._get_connection() as conn:
        for row in conn.execute(
            "SELECT draft_season, COUNT(*), "
            "SUM(CASE WHEN player_id IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM draft_picks WHERE draft_season BETWEEN ? AND ? "
            "GROUP BY draft_season ORDER BY draft_season",
            (lo, hi),
        ).fetchall():
            print(f"  season {row[0]}: {row[1]:,} picks; {row[2]:,} with gsis_id")
    return 0


if __name__ == "__main__":
    sys.exit(main())
