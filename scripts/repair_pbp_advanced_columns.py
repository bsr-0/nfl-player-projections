#!/usr/bin/env python3
"""Repair PBP-derived columns of player_weekly_stats rows stored as defaults.

2026 rows were stored with these columns at 0 (recv_epa non-zero on 0.1% of
rows; 80% in 2024/2025) because the current-season weekly merge zero-filled
them before the PBP values could be merged in (fixed in
`NFLDataLoader._merge_advanced_pbp_features`). This re-derives them with the
fixed loader and updates the stored rows.

Only existing rows are touched, only these columns, and a stored non-zero value
is never overwritten. Default is a dry run; --write copies the database first.

    python scripts/repair_pbp_advanced_columns.py --seasons 2026
    python scripts/repair_pbp_advanced_columns.py --seasons 2026 --write
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import DB_PATH

KEY = ["player_id", "season", "week"]
ADV_COLS = [
    "pass_epa", "rush_epa", "recv_epa", "pass_wpa", "rush_wpa", "recv_wpa",
    "pass_success_rate", "rush_success_rate", "recv_success_rate",
    "neutral_targets", "neutral_rushes", "third_down_targets", "short_yardage_rushes",
    "redzone_targets", "goal_line_touches", "two_minute_targets", "high_leverage_touches",
]


def plan_updates(stored: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    """Long frame (key, column, value) of stored defaults that the derived frame fills.

    A cell is updated only if the stored value is NULL or 0, the derived value is
    present and different. Rows absent from `stored` are never created.
    """
    cols = [c for c in ADV_COLS if c in stored.columns and c in derived.columns]
    d = derived[KEY + cols].drop_duplicates(KEY)
    j = stored[KEY + cols].merge(d, on=KEY, suffixes=("_old", "_new"))
    parts = []
    for c in cols:
        old, new = j[c + "_old"], j[c + "_new"]
        take = new.notna() & (old.isna() | (old == 0)) & ~np.isclose(old.fillna(0).astype(float), new.fillna(0).astype(float), atol=1e-12)
        if take.any():
            parts.append(j.loc[take, KEY].assign(column=c, value=new[take].astype(float)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY + ["column", "value"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    from src.data.nfl_data_loader import NFLDataLoader
    loader = NFLDataLoader()
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
        stored = pd.read_sql(
            f"SELECT {','.join(KEY + ADV_COLS)} FROM player_weekly_stats "
            f"WHERE season IN ({','.join('?' * len(a.seasons))})", con, params=a.seasons)
    derived = loader.load_weekly_data(a.seasons, store_in_db=False)
    plan = plan_updates(stored, derived)
    rows = plan[KEY].drop_duplicates()
    print(f"\nstored rows {len(stored)}; rows to update {len(rows)}; cells {len(plan)}")
    print(plan.groupby("column").size().to_string())
    if not a.write:
        print("dry run: nothing written (use --write)")
        return
    if plan.empty:
        return
    backup = DB_PATH.parent / "backups" / f"nfl_data_pre_pbp_repair_{datetime.now():%Y%m%d_%H%M%S}.db"
    backup.parent.mkdir(exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as con, sqlite3.connect(str(backup)) as dst:
        con.backup(dst)
        print(f"backup: {backup}")
        con.execute("BEGIN")
        for column, g in plan.groupby("column"):
            con.executemany(
                f'UPDATE player_weekly_stats SET "{column}" = ? WHERE player_id = ? AND season = ? AND week = ?',
                [(float(v), p, int(s), int(w)) for p, s, w, v in zip(g.player_id, g.season, g.week, g.value)])
        con.commit()
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
        after = pd.read_sql(
            f"SELECT {','.join(KEY + ADV_COLS)} FROM player_weekly_stats "
            f"WHERE season IN ({','.join('?' * len(a.seasons))})", con, params=a.seasons)
    left = plan_updates(after, derived)
    print(f"after write: cells still differing {len(left)}")
    if len(left):
        raise SystemExit("repair incomplete")


if __name__ == "__main__":
    main()
