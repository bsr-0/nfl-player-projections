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

from config.settings import CURRENT_NFL_SEASON

# Floor for every season-ranged pull here. Was 2018, which was never an
# upstream limit -- just where this script happened to start -- and it
# silently truncated tables that other scripts had backfilled earlier.
MIN_SEASON = 2013

# Upper bound is the current NFL season, NOT datetime.now().year: for most of
# the calendar year the latter names a season that has not been played, so
# every pull requested a nonexistent year (in Aug 2026 it asked for 2026
# while CURRENT_NFL_SEASON was 2025).
SEASONS = list(range(MIN_SEASON, CURRENT_NFL_SEASON + 1))

# Where upstream genuinely starts later than MIN_SEASON. Asking below these
# returns an empty frame or raises, so each dataset is clamped to its own
# floor rather than the global one. Verified against nflverse 2026-08-19;
# datasets absent from this map use MIN_SEASON.
DATASET_MIN_SEASON = {
    "ngs": 2016,          # Next Gen Stats begin 2016
    "snap_counts": 2013,  # PFR feed; the 2012 release file exists but is empty
}

# depth_charts changed schema for 2025: a daily ESPN-style feed with no
# season/week/depth_team column. Anything at or beyond this needs
# scripts/backfill_depth_charts_2025.py, not a straight append.
DEPTH_CHART_NEW_SCHEMA_SEASON = 2025


def seasons_for(dataset: str) -> list:
    """SEASONS clamped to a dataset's real upstream floor."""
    floor = DATASET_MIN_SEASON.get(dataset, MIN_SEASON)
    return [s for s in SEASONS if s >= floor]


def _assert_no_history_lost(df: pd.DataFrame, table: str, conn: sqlite3.Connection):
    """Refuse a `replace` that would drop seasons the incoming frame lacks.

    Every loader here writes with `if_exists="replace"` -- it drops the table
    and recreates it -- so any season the incoming frame doesn't cover is
    silently deleted. `SEASONS` now starts at MIN_SEASON (2013) rather than
    2018, which removes the original cause, but the guard stays: per-dataset
    floors mean a pull can still be narrower than what's stored (NGS starts
    2016), and depth_charts holds a 2025 season this script cannot fetch.

    Raising here rather than switching to `append` is deliberate: append
    would duplicate every existing row instead. The caller in main() catches
    and reports per-dataset, so one guarded table doesn't abort the rest.
    """
    if "season" not in df.columns:
        return
    try:
        existing = pd.read_sql(
            f"SELECT DISTINCT season FROM {table} WHERE season IS NOT NULL", conn)
    except Exception:
        return  # table doesn't exist yet -- nothing to lose
    if existing.empty:
        return

    have = {int(s) for s in existing.season.dropna()}
    incoming = {int(s) for s in pd.to_numeric(df["season"], errors="coerce").dropna()}
    lost = sorted(have - incoming)
    if lost:
        raise RuntimeError(
            f"refusing to replace `{table}`: would delete seasons {lost}, "
            f"which this script does not re-fetch (it pulls {min(incoming)}-"
            f"{max(incoming)}). Widen SEASONS, or use the dedicated "
            f"append-only backfill script for the earlier seasons."
        )


def _save_df(df: pd.DataFrame, table: str, conn: sqlite3.Connection,
             if_exists: str = "replace"):
    """Save DataFrame to SQLite, handling numpy types."""
    if if_exists == "replace":
        _assert_no_history_lost(df, table, conn)
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
    # pfr_player_id / cfb_player_id are RETAINED, not dropped. A GSIS id is
    # minted only once a player appears in official game data, so every pick
    # who has not yet debuted arrives here with a null one -- 7 of the 80
    # skill-position picks in the 2026 class. Without an alternate identifier
    # those rows are anonymous (this table has no name column), and
    # cfb_player_id is effectively the name ("dezhaun-stribling-1"). Keeping
    # them means such a pick is identifiable now and can be reconciled to a
    # GSIS id later. See scripts/backfill_draft_pick_identity.py.
    legacy_cols = ["player_id", "player_name", "position", "college",
                   "draft_season", "draft_round", "draft_pick", "draft_team",
                   "pfr_player_id", "cfb_player_id"]
    legacy_cols = [c for c in legacy_cols if c in legacy.columns]
    # Deliberately NOT dropna(subset=["player_id"]) -- that silently discarded
    # ~1,846 real draft picks whose GSIS id is absent, which is precisely the
    # not-yet-debuted population a rookie projection needs most.
    legacy = legacy[legacy_cols]
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
    seasons = seasons_for("snap_counts")
    print(f"  Loading snap counts ({seasons[0]}-{seasons[-1]})...")
    df = nfl.import_snap_counts(seasons)
    _save_df(df, "snap_counts", conn)


def backfill_ngs(conn):
    """Next Gen Stats: passing, rushing, receiving."""
    import nfl_data_py as nfl
    for stat_type in ["passing", "rushing", "receiving"]:
        print(f"  Loading NGS {stat_type}...")
        try:
            df = nfl.import_ngs_data(stat_type=stat_type, years=seasons_for("ngs"))
            _save_df(df, f"ngs_{stat_type}", conn)
        except Exception as e:
            print(f"    FAILED: {e}")


def backfill_qbr(conn):
    """ESPN QBR ratings."""
    import nfl_data_py as nfl
    print("  Loading QBR...")
    try:
        df = nfl.import_qbr(seasons_for("qbr"))
        _save_df(df, "qbr", conn)
    except Exception as e:
        print(f"    FAILED: {e}")


def backfill_injuries_nflpy(conn):
    """Injury reports from nfl-data-py (complements our existing backfill)."""
    import nfl_data_py as nfl
    print("  Loading injuries (nfl-data-py)...")
    df = nfl.import_injuries(seasons_for("injuries"))
    _save_df(df, "injuries_nflpy", conn)


def backfill_depth_charts(conn):
    """Weekly depth charts (starter/backup designation).

    Handles only the classic weekly schema (2013-2024). 2025+ ships a daily
    ESPN-style feed with no season/week/depth_team column at all; loading it
    here previously kept whatever columns happened to match -- which was
    `gsis_id` alone -- and wrote 554,215 rows with every key NULL. Those were
    later deleted as "junk" and the season written off as unbackfillable. Use
    scripts/backfill_depth_charts_2025.py, which bridges the schemas.
    """
    import nfl_data_py as nfl
    cols = ["season", "club_code", "week", "depth_team",
            "last_name", "first_name", "football_name",
            "position", "jersey_number", "gsis_id",
            "depth_position", "full_name"]
    seasons = [s for s in seasons_for("depth_charts")
               if s < DEPTH_CHART_NEW_SCHEMA_SEASON]
    print(f"  Loading depth charts ({seasons[0]}-{seasons[-1]})...")
    try:
        df = nfl.import_depth_charts(seasons)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"depth chart feed is missing {missing} -- upstream changed "
                f"schema. Do NOT let this fall through to a partial-column "
                f"write; see scripts/backfill_depth_charts_2025.py."
            )
        df = df.dropna(subset=["week", "gsis_id"])[cols].drop_duplicates()
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
    print("  Loading weekly rosters...")
    try:
        df = nfl.import_weekly_rosters(seasons_for("weekly_rosters"))
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
