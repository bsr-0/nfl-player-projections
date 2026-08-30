"""Backfill the PBP-derived columns of team_stats for historical seasons.

All eleven PBP-derived columns were EXACTLY ZERO for 100% of rows in every
season 2006-2024, with zero NULLs, because only
load_current_season_stats_from_pbp() ever ran and it does just the current
season. Two of them (pace_sec_per_play, neutral_pass_rate_oe) feed
CAUSAL_FEATURES for all four positions, so models saw a constant across the
training window and real values only in the newest season -- both a dead
feature and a train/serve discontinuity. See GAPS.md 2026-08-29.

Deliberately a targeted UPDATE of those eleven columns, NOT
DatabaseManager.insert_team_stats. That method's upsert reads

    points_scored = COALESCE(excluded.points_scored, team_stats.points_scored)

while its parameter binding uses `stats.get("points_scored", 0)`. A dict
carrying only PBP columns therefore binds 0, and COALESCE(0, existing) is 0 --
so routing this backfill through it would silently zero out points, yards and
turnovers for every row touched. An UPDATE naming only the eleven columns
cannot do that.

Usage:
    python scripts/backfill_team_pbp_stats.py --dry-run
    python scripts/backfill_team_pbp_stats.py --start 2006 --end 2024
"""
import argparse
import shutil
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DB_PATH
from src.data.pbp_stats_aggregator import get_team_stats_from_pbp

PBP_COLS = [
    "neutral_pass_plays", "neutral_run_plays", "neutral_pass_rate",
    "neutral_pass_rate_lg", "neutral_pass_rate_oe", "drive_count",
    "drive_success_rate", "avg_drive_epa", "points_per_drive",
    "pace_sec_per_play",
]

# Sanity bounds. pace_sec_per_play = 0 is the defect being fixed, so a backfill
# that writes another impossible value must fail loudly rather than replace one
# fabrication with another.
BOUNDS = {
    "pace_sec_per_play": (5.0, 60.0),
    "neutral_pass_rate": (0.0, 1.0),
    "neutral_pass_rate_lg": (0.0, 1.0),
    "neutral_pass_rate_oe": (-1.0, 1.0),
    "drive_success_rate": (0.0, 1.0),
    "drive_count": (1, 30),
}


def backup_db() -> Path:
    dest = DB_PATH.with_name(
        f"{DB_PATH.stem}.prebackfill_{datetime.now():%Y%m%d_%H%M%S}.db"
    )
    shutil.copy2(DB_PATH, dest)
    return dest


def check_bounds(df: pd.DataFrame, season: int) -> list:
    problems = []
    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        if s.min() < lo or s.max() > hi:
            problems.append(
                f"{season} {col}: [{s.min():.3f}, {s.max():.3f}] outside "
                f"[{lo}, {hi}]"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2006)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        dest = backup_db()
        print(f"DB backed up -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)\n")

    con = sqlite3.connect(DB_PATH)
    set_clause = ", ".join(f"{c} = ?" for c in PBP_COLS)
    sql = (f"UPDATE team_stats SET {set_clause} "
           f"WHERE team = ? AND season = ? AND week = ?")

    total_rows = total_matched = 0
    for season in range(args.start, args.end + 1):
        try:
            df = get_team_stats_from_pbp(season)
        except Exception as e:
            print(f"{season}: FAILED to aggregate ({e}) -- skipped")
            continue
        if df is None or df.empty:
            print(f"{season}: no PBP rows -- skipped")
            continue

        problems = check_bounds(df, season)
        if problems:
            print(f"{season}: BOUNDS VIOLATION, refusing to write")
            for p in problems:
                print(f"    {p}")
            continue

        missing = [c for c in PBP_COLS if c not in df.columns]
        if missing:
            print(f"{season}: aggregator missing {missing} -- skipped")
            continue

        matched = 0
        if not args.dry_run:
            cur = con.cursor()
            for _, r in df.iterrows():
                vals = [None if pd.isna(r[c]) else float(r[c]) for c in PBP_COLS]
                cur.execute(sql, vals + [r["team"], int(r["season"]),
                                         int(r["week"])])
                matched += cur.rowcount
            con.commit()

        total_rows += len(df)
        total_matched += matched
        pace = pd.to_numeric(df["pace_sec_per_play"], errors="coerce")
        print(f"{season}: {len(df):4d} team-weeks, {matched:4d} rows updated, "
              f"pace median {pace.median():.1f}s")

    # Rows the aggregator did not produce (the DB carries ~30 more team-weeks
    # per season than PBP yields) would otherwise keep their fabricated zeros,
    # which is a half-fix: the column would be real for most rows and fake for
    # the rest, with nothing distinguishing them.
    #
    # The fabrication signature is all ten PBP columns being EXACTLY zero at
    # once. A real team-week cannot have zero drives, zero neutral plays and
    # zero seconds per play simultaneously, so this cannot match a genuine row.
    zero_pred = " AND ".join(f"{c} = 0" for c in PBP_COLS)
    null_clause = ", ".join(f"{c} = NULL" for c in PBP_COLS)
    count_sql = (f"SELECT COUNT(*) FROM team_stats WHERE season BETWEEN ? AND ? "
                 f"AND {zero_pred}")
    n_fab = con.execute(count_sql, (args.start, args.end)).fetchone()[0]
    if args.dry_run:
        print(f"\nwould NULL {n_fab} rows still all-zero after backfill")
    elif n_fab:
        con.execute(
            f"UPDATE team_stats SET {null_clause} "
            f"WHERE season BETWEEN ? AND ? AND {zero_pred}",
            (args.start, args.end),
        )
        con.commit()
        print(f"\nNULLed {n_fab} rows the aggregator could not produce "
              f"(absence is now recorded as absence)")

    print(f"\n{'DRY RUN -- nothing written' if args.dry_run else 'done'}: "
          f"{total_rows} aggregated, {total_matched} rows updated")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
