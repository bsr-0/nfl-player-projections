#!/usr/bin/env python3
"""Backfill PBP-derived features (EPA, success rate, sack rate, scramble rate)
into the player_weekly_stats table from nfl-data-py play-by-play data.

Usage:
    python scripts/backfill_pbp_features.py
    python scripts/backfill_pbp_features.py --seasons 2024 2025
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nfl_data.db"


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    """Load play-by-play data from nfl-data-py."""
    import nfl_data_py as nfl

    frames = []
    for yr in seasons:
        print(f"  Loading PBP for {yr}...", end=" ", flush=True)
        try:
            pbp = nfl.import_pbp_data([yr])
            frames.append(pbp)
            print(f"{len(pbp)} plays")
        except Exception as e:
            print(f"FAILED ({e})")
    if not frames:
        raise ValueError("No PBP data loaded")
    return pd.concat(frames, ignore_index=True)


def aggregate_player_weekly(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-level PBP data to player-week level."""
    records = []

    # --- QB stats (passer + scrambles) ---
    pass_plays = pbp[
        (pbp["play_type"] == "pass") & pbp["passer_player_id"].notna()
    ].copy()

    qb_weekly = (
        pass_plays.groupby(["season", "week", "passer_player_id"])
        .agg(
            pass_epa=("epa", "mean"),
            pass_success_rate=("success", "mean"),
            pass_wpa=("wpa", "sum"),
            n_dropbacks=("play_id", "count"),
            n_sacks=("sack", "sum"),
            n_scrambles=("qb_scramble", "sum"),
        )
        .reset_index()
    )
    qb_weekly = qb_weekly.rename(columns={"passer_player_id": "player_id"})
    qb_weekly["sack_rate"] = qb_weekly["n_sacks"] / qb_weekly["n_dropbacks"].clip(lower=1)
    qb_weekly["scramble_rate"] = qb_weekly["n_scrambles"] / qb_weekly["n_dropbacks"].clip(lower=1)

    # Also get QB rush EPA from designed runs (not scrambles)
    qb_rush = pbp[
        (pbp["play_type"] == "run")
        & pbp["rusher_player_id"].notna()
        & (pbp["qb_scramble"] == 0)
    ].copy()
    qb_rush_agg = (
        qb_rush.groupby(["season", "week", "rusher_player_id"])
        .agg(rush_epa=("epa", "mean"), rush_wpa=("wpa", "sum"), rush_success_rate=("success", "mean"))
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )

    qb_stats = qb_weekly.merge(
        qb_rush_agg, on=["season", "week", "player_id"], how="left"
    )
    records.append(qb_stats)

    # --- RB stats (rusher) ---
    rush_plays = pbp[
        (pbp["play_type"] == "run") & pbp["rusher_player_id"].notna()
    ].copy()
    rb_weekly = (
        rush_plays.groupby(["season", "week", "rusher_player_id"])
        .agg(rush_epa=("epa", "mean"), rush_success_rate=("success", "mean"), rush_wpa=("wpa", "sum"))
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )
    records.append(rb_weekly)

    # --- WR/TE stats (receiver) ---
    recv_plays = pbp[
        (pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna()
    ].copy()
    recv_weekly = (
        recv_plays.groupby(["season", "week", "receiver_player_id"])
        .agg(recv_epa=("epa", "mean"), recv_success_rate=("success", "mean"), recv_wpa=("wpa", "sum"))
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id"})
    )
    records.append(recv_weekly)

    # Merge all — a player can appear as both rusher and receiver
    combined = records[0]  # QB stats
    for df in records[1:]:
        combined = combined.merge(
            df, on=["season", "week", "player_id"], how="outer",
            suffixes=("", "_dup"),
        )
        # Resolve duplicates by preferring non-null
        for col in combined.columns:
            if col.endswith("_dup"):
                base = col[:-4]
                if base in combined.columns:
                    combined[base] = combined[base].fillna(combined[col])
                combined = combined.drop(columns=[col])

    return combined


def update_db(stats: pd.DataFrame, db_path: Path) -> int:
    """Write PBP features into player_weekly_stats."""
    conn = sqlite3.connect(db_path)

    # Map nfl-data-py gsis_id format to our player_id format
    # nfl-data-py uses "00-0012345", our DB uses the same format
    update_cols = [
        "pass_epa", "rush_epa", "recv_epa",
        "pass_wpa", "rush_wpa", "recv_wpa",
        "pass_success_rate", "rush_success_rate", "recv_success_rate",
    ]

    updated = 0
    for _, row in stats.iterrows():
        pid = row["player_id"]
        season = int(row["season"])
        week = int(row["week"])

        sets = []
        vals = []
        for col in update_cols:
            val = row.get(col)
            if pd.notna(val):
                sets.append(f"{col} = ?")
                vals.append(float(val))

        if not sets:
            continue

        vals.extend([pid, season, week])
        sql = f"UPDATE player_weekly_stats SET {', '.join(sets)} WHERE player_id = ? AND season = ? AND week = ?"
        cursor = conn.execute(sql, vals)
        updated += cursor.rowcount

    conn.commit()
    conn.close()
    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill PBP features into DB")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2018, 2026)))
    args = parser.parse_args()

    print(f"Backfilling PBP features for seasons: {args.seasons}")
    print(f"DB: {DB_PATH}")

    pbp = load_pbp(args.seasons)
    print(f"\nAggregating to player-week level...")
    stats = aggregate_player_weekly(pbp)
    print(f"  {len(stats)} player-week rows")

    print(f"\nUpdating database...")
    n = update_db(stats, DB_PATH)
    print(f"  {n} rows updated")

    # Verify
    conn = sqlite3.connect(DB_PATH)
    for col in ["pass_epa", "rush_epa", "recv_epa", "pass_success_rate"]:
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM player_weekly_stats WHERE {col} IS NOT NULL AND {col} != 0 AND season >= 2018"
        )
        print(f"  {col}: {cursor.fetchone()[0]} non-zero rows")
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
