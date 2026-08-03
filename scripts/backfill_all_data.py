#!/usr/bin/env python3
"""Bulk backfill all high-value datasets from nfl-data-py into the local DB.

Downloads: draft picks, combine data, snap counts, injuries (nfl-data-py
version), NGS stats, QBR, depth charts, contracts, weekly rosters.

Usage:
    python scripts/backfill_all_data.py
    python scripts/backfill_all_data.py --only draft_picks combine
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "nfl_data.db"
from datetime import datetime
SEASONS = list(range(2018, datetime.now().year + 1))


def _save_df(df: pd.DataFrame, table: str, conn: sqlite3.Connection,
             if_exists: str = "replace"):
    """Save DataFrame to SQLite, handling numpy types."""
    # Convert numpy types to native Python for SQLite compatibility
    for col in df.columns:
        if df[col].dtype in (np.int64, np.int32):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        elif df[col].dtype in (np.float64, np.float32):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    df.to_sql(table, conn, if_exists=if_exists, index=False)
    print(f"    → {table}: {len(df)} rows")


def backfill_draft_picks(conn):
    """NFL draft picks with player IDs and draft capital."""
    import nfl_data_py as nfl
    print("  Loading draft picks...")
    df = nfl.import_draft_picks()
    # Keep relevant columns
    cols = ["season", "round", "pick", "team", "gsis_id", "pfr_player_id",
            "player_name", "position", "age", "college",
            "cfb_player_id", "category"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    # Also update the existing draft_picks table format
    legacy = df.rename(columns={
        "gsis_id": "player_id", "player_name": "player_name",
        "season": "draft_season", "round": "draft_round",
        "pick": "draft_pick", "team": "draft_team",
    })
    legacy_cols = ["player_id", "player_name", "position", "college",
                   "draft_season", "draft_round", "draft_pick", "draft_team"]
    legacy_cols = [c for c in legacy_cols if c in legacy.columns]
    legacy = legacy[legacy_cols].dropna(subset=["player_id"])
    _save_df(legacy, "draft_picks_v2", conn)
    # Also backfill the original draft_picks table
    conn.execute("DELETE FROM draft_picks")
    for _, r in legacy.iterrows():
        try:
            conn.execute(
                "INSERT OR IGNORE INTO draft_picks (player_id, player_name, position, college, draft_season, draft_round, draft_pick, draft_team) VALUES (?,?,?,?,?,?,?,?)",
                (r.get("player_id"), r.get("player_name"), r.get("position"),
                 r.get("college"), r.get("draft_season"), r.get("draft_round"),
                 r.get("draft_pick"), r.get("draft_team")),
            )
        except Exception:
            pass
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM draft_picks").fetchone()[0]
    print(f"    → draft_picks (legacy): {n} rows")


def backfill_combine(conn):
    """NFL combine data (40-yard, bench, vertical, etc.)."""
    import nfl_data_py as nfl
    print("  Loading combine data...")
    df = nfl.import_combine_data()
    _save_df(df, "combine_data_v2", conn)


def backfill_snap_counts(conn):
    """Weekly snap counts per player (offensive/defensive/ST)."""
    import nfl_data_py as nfl
    print("  Loading snap counts (2018-2025)...")
    df = nfl.import_snap_counts(SEASONS)
    _save_df(df, "snap_counts", conn)


def backfill_ngs(conn):
    """Next Gen Stats: passing, rushing, receiving."""
    import nfl_data_py as nfl
    for stat_type in ["passing", "rushing", "receiving"]:
        print(f"  Loading NGS {stat_type}...")
        try:
            df = nfl.import_ngs_data(stat_type=stat_type, years=SEASONS)
            _save_df(df, f"ngs_{stat_type}", conn)
        except Exception as e:
            print(f"    FAILED: {e}")


def backfill_qbr(conn):
    """ESPN QBR ratings."""
    import nfl_data_py as nfl
    print("  Loading QBR...")
    try:
        df = nfl.import_qbr(SEASONS)
        _save_df(df, "qbr", conn)
    except Exception as e:
        print(f"    FAILED: {e}")


def backfill_injuries_nflpy(conn):
    """Injury reports from nfl-data-py (complements our existing backfill)."""
    import nfl_data_py as nfl
    print("  Loading injuries (nfl-data-py)...")
    df = nfl.import_injuries(SEASONS)
    _save_df(df, "injuries_nflpy", conn)


def backfill_depth_charts(conn):
    """Weekly depth charts (starter/backup designation)."""
    import nfl_data_py as nfl
    print("  Loading depth charts (2024-2025)...")
    try:
        df = nfl.import_depth_charts(range(2024, 2026))
        # Compact: keep key columns
        cols = ["season", "club_code", "week", "depth_team",
                "last_name", "first_name", "football_name",
                "position", "jersey_number", "gsis_id",
                "depth_position", "full_name"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        _save_df(df, "depth_charts", conn)
    except Exception as e:
        print(f"    FAILED: {e}")


def backfill_contracts(conn):
    """Player contract data (APY, guaranteed money)."""
    import nfl_data_py as nfl
    print("  Loading contracts...")
    df = nfl.import_contracts()
    _save_df(df, "contracts", conn)


def backfill_weekly_rosters(conn):
    """Weekly 53-man roster snapshots."""
    import nfl_data_py as nfl
    print("  Loading weekly rosters (2024-2025)...")
    try:
        df = nfl.import_weekly_rosters(range(2024, 2026))
        cols = ["season", "team", "position", "depth_chart_position",
                "status", "player_name", "player_id", "gsis_it_id",
                "week", "game_type", "headshot_url"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        _save_df(df, "weekly_rosters_v2", conn)
    except Exception as e:
        print(f"    FAILED: {e}")


ALL_BACKFILLS = {
    "draft_picks": backfill_draft_picks,
    "combine": backfill_combine,
    "snap_counts": backfill_snap_counts,
    "ngs": backfill_ngs,
    "qbr": backfill_qbr,
    "injuries": backfill_injuries_nflpy,
    "depth_charts": backfill_depth_charts,
    "contracts": backfill_contracts,
    "weekly_rosters": backfill_weekly_rosters,
}


def main():
    parser = argparse.ArgumentParser(description="Backfill all datasets from nfl-data-py")
    parser.add_argument("--only", nargs="+", choices=list(ALL_BACKFILLS.keys()),
                        default=None, help="Only backfill these datasets")
    args = parser.parse_args()

    to_run = args.only or list(ALL_BACKFILLS.keys())

    print(f"Backfilling {len(to_run)} datasets into {DB_PATH}")
    print(f"Datasets: {', '.join(to_run)}\n")

    conn = sqlite3.connect(DB_PATH)
    for name in to_run:
        try:
            ALL_BACKFILLS[name](conn)
        except Exception as e:
            print(f"  {name} FAILED: {e}")
    conn.close()

    print("\nDone.")
    # Summary
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for table in ["draft_picks", "draft_picks_v2", "combine_data_v2",
                   "snap_counts", "ngs_passing", "ngs_rushing", "ngs_receiving",
                   "qbr", "injuries_nflpy", "depth_charts", "contracts",
                   "weekly_rosters_v2"]:
        try:
            n = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {n} rows")
        except Exception:
            print(f"  {table}: not found")
    conn.close()


if __name__ == "__main__":
    main()
