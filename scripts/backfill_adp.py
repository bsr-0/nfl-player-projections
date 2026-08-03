#!/usr/bin/env python3
"""
One-time backfill of historical ADP (Average Draft Position) into
the local ``adp_history`` table.

Source: the dynastyprocess/data GitHub mirror of FantasyPros ECR
(Expert Consensus Rank) — ``db_fpecr.parquet`` on
raw.githubusercontent.com.  FantasyPros ECR is the public proxy
for ADP that most draft research uses; the parquet contains weekly
snapshots back to 2019-12-27.

For the PPR redraft snake-draft simulator we keep the
``page_type='redraft-overall'`` slice (the overall PPR draft
board).  Per-position pages are available too but aren't needed
for v1 — the simulator drafts from the overall pool with position
constraints.

Season tagging: FantasyPros rescrapes continuously; a scrape dated
2024-09-06 is the 2024-season draft board.  We derive season from
scrape month: Jan-Feb → previous season (post-season rankings),
Mar onward → current NFL season.  For the draft simulator only
Jun-Sep scrapes are useful (pre-season draft boards).

Usage:
    python scripts/backfill_adp.py                      # 2024+2025 by default
    python scripts/backfill_adp.py --seasons 2023 2025
    python scripts/backfill_adp.py --dry-run
    python scripts/backfill_adp.py --parquet-cache /tmp/db_fpecr.parquet
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import urllib.request
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FPECR_URL = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/"
    "files/db_fpecr.parquet"
)

# We keep redraft pages the simulator needs.  Per-position redraft
# pages (redraft-qb/rb/wr/te) are kept so the sim can do position
# tiebreakers later; redraft-overall is the primary draft board.
KEEP_PAGE_TYPES = {
    "redraft-overall",
    "redraft-qb",
    "redraft-rb",
    "redraft-wr",
    "redraft-te",
}


def _download_parquet(dest: Path) -> None:
    print(f"Downloading {FPECR_URL} → {dest} (~36 MB) …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FPECR_URL, dest)


def _infer_season(scrape_date: pd.Timestamp) -> int:
    # Pre-March scrapes describe the prior NFL season; from March
    # onward the FantasyPros board is already reoriented toward
    # the upcoming season's draft.
    if scrape_date.month <= 2:
        return scrape_date.year - 1
    return scrape_date.year


def _prepare_frame(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    df = df[df["page_type"].isin(KEEP_PAGE_TYPES)].copy()
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    df["season"] = df["scrape_date"].map(_infer_season).astype(int)
    df = df[df["season"].isin(seasons)].copy()
    return df


def _coverage_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "  (no rows)"
    lines = [f"  {'season':>6}  {'scrapes':>7}  {'rows':>8}  {'players':>8}"]
    lines.append("  " + "-" * 38)
    for s in sorted(df["season"].unique()):
        sub = df[df["season"] == s]
        lines.append(
            f"  {int(s):>6}  "
            f"{sub['scrape_date'].nunique():>7}  "
            f"{len(sub):>8,}  "
            f"{sub['player'].nunique():>8,}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--seasons", "-s",
        nargs=2, type=int, metavar=("LO", "HI"),
        default=[2024, 2025],
    )
    ap.add_argument(
        "--parquet-cache",
        type=Path,
        default=Path("/tmp/db_fpecr.parquet"),
        help="Local cache path for the parquet (avoids redownload).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lo, hi = sorted(args.seasons)
    seasons = list(range(lo, hi + 1))

    if not args.parquet_cache.exists():
        _download_parquet(args.parquet_cache)
    else:
        print(f"Using cached parquet at {args.parquet_cache}")

    df = pd.read_parquet(args.parquet_cache)
    print(f"Loaded {len(df):,} total ECR rows.")
    df = _prepare_frame(df, seasons)
    print(f"Filtered to {len(df):,} rows for seasons {seasons}, "
          f"page_types {sorted(KEEP_PAGE_TYPES)}.")
    print(_coverage_report(df))

    if args.dry_run:
        print("\n--dry-run: no writes.")
        return 0

    from src.utils.database import DatabaseManager
    # DatabaseManager().__init__ calls _init_database() which runs
    # idempotent CREATE TABLE IF NOT EXISTS for every table including
    # the new adp_history.
    db = DatabaseManager()

    written = 0
    skipped = 0
    with db._get_connection() as conn:
        cur = conn.cursor()
        for _, row in df.iterrows():
            try:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO adp_history
                        (season, scrape_date, fp_player_id, player_name,
                         mergename, position, team, ecr, sd, best, worst,
                         ecr_type, page_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(row["season"]),
                        row["scrape_date"].strftime("%Y-%m-%d"),
                        str(row["id"]) if pd.notna(row.get("id")) else None,
                        str(row["player"]) if pd.notna(row.get("player")) else None,
                        str(row["mergename"]) if pd.notna(row.get("mergename")) else None,
                        row.get("pos"),
                        row.get("team"),
                        float(row["ecr"]) if pd.notna(row.get("ecr")) else None,
                        float(row["sd"]) if pd.notna(row.get("sd")) else None,
                        float(row["best"]) if pd.notna(row.get("best")) else None,
                        float(row["worst"]) if pd.notna(row.get("worst")) else None,
                        row.get("ecr_type"),
                        row.get("page_type"),
                    ),
                )
                written += 1
            except (sqlite3.IntegrityError, ValueError, TypeError):
                skipped += 1
        conn.commit()
    print(f"\nWrote {written:,} rows; skipped {skipped:,}.")

    with db._get_connection() as conn:
        for row in conn.execute(
            "SELECT season, COUNT(*), COUNT(DISTINCT scrape_date), "
            "       COUNT(DISTINCT player_name) "
            "  FROM adp_history "
            " WHERE season BETWEEN ? AND ? "
            " GROUP BY season ORDER BY season",
            (lo, hi),
        ).fetchall():
            print(f"  season {row[0]}: {row[1]:,} rows; "
                  f"{row[2]} scrape dates; {row[3]:,} players.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
