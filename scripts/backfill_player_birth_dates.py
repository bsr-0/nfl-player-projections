#!/usr/bin/env python3
"""Backfill `players.birth_date` from nfl_data_py, so age is derivable.

Without this, `players.birth_date` covers 68% of players and only 50-71% of
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


def load_source() -> dict:
    import nfl_data_py as nfl
    players = nfl.import_players()
    if "gsis_id" not in players.columns or "birth_date" not in players.columns:
        raise RuntimeError(
            f"import_players() lacks gsis_id/birth_date; got {list(players.columns)[:20]}"
        )
    src = players[["gsis_id", "birth_date"]].dropna()
    src = src[(src.gsis_id != "") & (src.birth_date != "")]
    # Normalise to YYYY-MM-DD, matching what the column already holds.
    dates = pd.to_datetime(src.birth_date, errors="coerce")
    src = src.assign(birth_date=dates.dt.strftime("%Y-%m-%d")).dropna(subset=["birth_date"])
    return dict(zip(src.gsis_id, src.birth_date))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="apply the update (default: dry-run)")
    args = ap.parse_args()

    bd_map = load_source()
    print(f"Source: {len(bd_map)} players with a birth_date from nfl_data_py")

    conn = sqlite3.connect(str(DB_PATH))
    current = pd.read_sql("SELECT player_id, position, birth_date FROM players", conn)
    missing = current.birth_date.isna() | (current.birth_date == "")
    current["candidate"] = current.player_id.map(bd_map)

    fillable = current[missing & current.candidate.notna()]
    still_missing = current[missing & current.candidate.isna()]
    print(f"players table: {len(current)} rows, {(~missing).sum()} already have a birth_date")
    print(f"  fillable now:  {len(fillable)}")
    print(f"  still missing: {len(still_missing)} (no gsis_id match in the feed)")

    both = current[~missing & current.candidate.notna()]
    disagree = both[both.birth_date.astype(str).str[:10] != both.candidate.astype(str).str[:10]]
    print(f"  existing values disagreeing with the feed: {len(disagree)} of {len(both)} "
          f"(left untouched)")

    if not args.write:
        print("\nDRY RUN -- nothing written. Re-run with --write to apply.")
        conn.close()
        return

    backup = DB_PATH.parent / f"nfl_data.db.bak-birthdate-{datetime.now():%Y%m%d%H%M%S}"
    shutil.copy2(DB_PATH, backup)
    print(f"\nBacked up database to {backup}")

    conn.executemany(
        "UPDATE players SET birth_date = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE player_id = ? AND (birth_date IS NULL OR birth_date = '')",
        [(row.candidate, row.player_id) for row in fillable.itertuples(index=False)],
    )
    conn.commit()

    after = pd.read_sql(
        "SELECT COUNT(*) n FROM players WHERE birth_date IS NOT NULL AND birth_date != ''", conn
    ).n.iloc[0]
    print(f"Wrote {len(fillable)} birth dates. players.birth_date now populated for {after}/{len(current)}.")
    conn.close()


if __name__ == "__main__":
    main()
