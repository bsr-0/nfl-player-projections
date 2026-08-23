#!/usr/bin/env python
"""Attach alternate player identifiers to `draft_picks_v2`.

Why this exists: 7 of the 80 skill-position picks in the 2026 class carry no
GSIS `player_id`, which made them anonymous rows -- position, college, round,
pick and nothing else, since the table has no name column either.

That is NOT a local ingestion bug. `nfl.import_draft_picks([2026])` has no
GSIS id for the same 7 picks. GSIS ids are minted when a player first appears
in official game data, and those players have not debuted. The id cannot be
recovered because it does not yet exist anywhere.

What CAN be fixed is identity. Upstream carries `pfr_player_id` (e.g.
`StriDe01`) and `cfb_player_id` (e.g. `dezhaun-stribling-1`, which is
effectively the name), and this repo already matches PFR data by id
elsewhere. Backfilling both means those picks are identifiable now and
joinable to a GSIS id later, once nflverse assigns one.

Matched on `(draft_season, draft_round, draft_pick)` -- verified unique in
both the local table and upstream, so the update cannot fan out.

Usage:
    python scripts/backfill_draft_pick_identity.py            # apply
    python scripts/backfill_draft_pick_identity.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DB_PATH

NEW_COLUMNS = {"pfr_player_id": "TEXT", "cfb_player_id": "TEXT"}
KEY = ["draft_season", "draft_round", "draft_pick"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    local = pd.read_sql("SELECT rowid, * FROM draft_picks_v2", conn)
    print(f"local draft_picks_v2: {len(local)} rows")
    if local.duplicated(KEY).any():
        print("ABORT: (season, round, pick) is not unique locally; the update could fan out.")
        return 1

    import nfl_data_py as nfl
    up = nfl.import_draft_picks()
    up = up.rename(columns={"season": "draft_season", "round": "draft_round",
                            "pick": "draft_pick"})
    keep = KEY + [c for c in NEW_COLUMNS if c in up.columns]
    up = up[keep]
    if up.duplicated(KEY).any():
        print("ABORT: (season, round, pick) is not unique upstream.")
        return 1
    print(f"upstream: {len(up)} rows ({len(up) - len(local):+d} vs local)")

    for col, decl in NEW_COLUMNS.items():
        if col not in local.columns:
            print(f"  adding column {col} {decl}")
            if not args.dry_run:
                conn.execute(f"ALTER TABLE draft_picks_v2 ADD COLUMN {col} {decl}")

    merged = local[["rowid"] + KEY].merge(up, on=KEY, how="left")
    assert len(merged) == len(local), "merge changed row count -- key is not unique"

    for col in NEW_COLUMNS:
        if col not in merged.columns:
            continue
        filled = merged[col].notna().sum()
        print(f"  {col}: matched {filled}/{len(merged)} rows ({filled / len(merged):.1%})")
        if args.dry_run:
            continue
        rows = [(v, int(r)) for r, v in zip(merged["rowid"], merged[col]) if pd.notna(v)]
        conn.executemany(f"UPDATE draft_picks_v2 SET {col} = ? WHERE rowid = ?", rows)

    if not args.dry_run:
        conn.commit()

    after = pd.read_sql("SELECT COUNT(*) n FROM draft_picks_v2", conn).n[0]
    print(f"\nrows after: {after} (must equal {len(local)})")

    if args.dry_run:
        print("\n--dry-run: no columns added, nothing written.")
        conn.close()
        return 0

    check = pd.read_sql(
        "SELECT draft_round, draft_pick, position, college, pfr_player_id, cfb_player_id "
        "FROM draft_picks_v2 WHERE draft_season = 2026 "
        "AND (player_id IS NULL OR player_id = '') "
        "AND position IN ('QB','RB','WR','TE') ORDER BY draft_pick", conn)
    print("\nthe 7 previously-anonymous 2026 skill picks:")
    print(check.to_string(index=False) if not check.empty else "  (none found)")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
