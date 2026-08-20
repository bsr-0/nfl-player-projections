#!/usr/bin/env python3
"""Backfill static player bio fields (birth_date, college, height, weight).

These are all destroyed by the same root cause: `insert_player` used
INSERT OR REPLACE, which rewrites the entire row, so every weekly ingest
NULLed any column the caller did not supply. `_store_weekly_data` supplies
only player_id/name/position, so `college` ended up NULL for all 2,985
players and `height` for 757 (GAPS.md 2026-08-20). `insert_player` is now
a real upsert, so these values stay put once restored.

birth_date specifically: without this, coverage was 68% of players and only 50-71% of
player-weeks depending on position (QB worst). That is thin enough that the
age fix in src/features/player_age.py would still fall back to a position
constant for a third of the training set -- which is the bug it exists to
remove, just quieter.

nfl_data_py.import_players() carries birth_date for 24,998 of 25,046
players, keyed by gsis_id (identical format to our player_id). Backfilling
takes row-weighted coverage to ~100% at every position.

Only fills NULL/empty values; never overwrites an existing birth_date.
Spot-checked before writing this: of the 2,016 players where both sources
have a value, exactly 2 disagree -- so the existing column is consistent
with the feed and there is no case for a wholesale overwrite.

Usage:
    python scripts/backfill_player_birth_dates.py            # dry-run (default)
    python scripts/backfill_player_birth_dates.py --write    # real update, backs up DB first
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH


# DB column -> feed column. Only static biographical facts belong here;
# anything that changes with time (team, position, status) must come from
# the roster snapshots instead.
BIO_FIELDS = {"birth_date": "birth_date", "college": "college_name",
              "height": "height", "weight": "weight"}


def load_source() -> pd.DataFrame:
    import nfl_data_py as nfl
    players = nfl.import_players()
    if "gsis_id" not in players.columns:
        raise RuntimeError(f"import_players() lacks gsis_id; got {list(players.columns)[:20]}")

    available = {db_col: feed_col for db_col, feed_col in BIO_FIELDS.items()
                 if feed_col in players.columns}
    missing = sorted(set(BIO_FIELDS) - set(available))
    if missing:
        print(f"  (feed has no column for: {', '.join(missing)} -- skipped)")

    src = players[["gsis_id"] + list(available.values())].copy()
    src.columns = ["player_id"] + list(available.keys())
    src = src[src.player_id.notna() & (src.player_id != "")]
    if "birth_date" in src.columns:
        # Normalise to YYYY-MM-DD, matching what the column already holds.
        src["birth_date"] = pd.to_datetime(src.birth_date, errors="coerce").dt.strftime("%Y-%m-%d")
    # height/weight are stored as bare integer strings ("73"), but the feed
    # supplies floats (73.0). Writing those raw would leave the column in two
    # formats at once, so coerce before it can happen.
    for col in ("height", "weight"):
        if col in src.columns:
            nums = pd.to_numeric(src[col], errors="coerce")
            src[col] = nums.round().astype("Int64").astype(str).replace("<NA>", None)
    return src.replace({"": None})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="apply the update (default: dry-run)")
    args = ap.parse_args()

    src = load_source()
    fields = [c for c in src.columns if c != "player_id"]
    print(f"Source: {len(src)} players from nfl_data_py, fields: {', '.join(fields)}")

    conn = sqlite3.connect(str(DB_PATH))
    current = pd.read_sql(f"SELECT player_id, {', '.join(fields)} FROM players", conn)
    merged = current.merge(src, on="player_id", how="left", suffixes=("", "_new"))

    plan = {}
    for f in fields:
        blank = merged[f].isna() | (merged[f].astype(str).str.strip() == "")
        candidate = merged[f + "_new"]
        fillable = merged[blank & candidate.notna()]
        plan[f] = fillable
        have = (~blank).sum()
        disagree = merged[~blank & candidate.notna()]
        n_dis = (disagree[f].astype(str).str[:10] != disagree[f + "_new"].astype(str).str[:10]).sum()
        print(f"  {f:11s} have={have:5d}  fillable={len(fillable):5d}  "
              f"still missing={int((blank & candidate.isna()).sum()):4d}  "
              f"disagreeing (left alone)={n_dis}")

    if not args.write:
        print("\nDRY RUN -- nothing written. Re-run with --write to apply.")
        conn.close()
        return

    backup = DB_PATH.parent / f"nfl_data.db.bak-bio-{datetime.now():%Y%m%d%H%M%S}"
    shutil.copy2(DB_PATH, backup)
    print(f"\nBacked up database to {backup}")

    for f, fillable in plan.items():
        if fillable.empty:
            continue
        conn.executemany(
            f"UPDATE players SET {f} = ?, updated_at = CURRENT_TIMESTAMP "
            f"WHERE player_id = ? AND ({f} IS NULL OR TRIM({f}) = '')",
            list(zip(fillable[f + "_new"], fillable["player_id"])),
        )
        print(f"  wrote {len(fillable)} {f}")
    conn.commit()

    after = pd.read_sql(
        "SELECT " + ", ".join(
            f"SUM(CASE WHEN {f} IS NOT NULL AND TRIM({f}) != '' THEN 1 ELSE 0 END) AS {f}"
            for f in fields
        ) + ", COUNT(*) AS total FROM players", conn
    ).iloc[0]
    print("\nCoverage now: " + ", ".join(f"{f}={after[f]}/{after['total']}" for f in fields))
    conn.close()


if __name__ == "__main__":
    main()
